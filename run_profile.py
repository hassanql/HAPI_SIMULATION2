"""run_profile.py — runtime profiling for one Stage 2 case.

Renamed from profile.py to avoid shadowing the stdlib `profile` module
(torch._dynamo / cProfile import it transitively, and `import profile`
would otherwise resolve to this file when run from the repo root).

Loads Qwen3-4B-Instruct-2507 via vLLM library mode (or the mock backend with
`--use-mock` for a CPU-only sanity run), picks one case from the Stage 1
augmented JSONL, runs the full Stage 2 agent loop (benign principal), and
prints a coarse + per-step + per-role timing breakdown.

Useful for answering questions like:
  * "Where does my 35-min run actually go?"
  * "Is propose, EIG-predict, patient, or belief-update the slowest call?"
  * "Is the bottleneck the LLM or our Python overhead?"
  * "How much speedup did `enable_prefix_caching=True` actually buy on T4?"

Usage (on a GCP T4 / L4 / A100 with vLLM installed):

    python run_profile.py                       # case 0, up to 5 turns, library backend
    python run_profile.py --case-index 7        # different case
    python run_profile.py --max-queries 3       # cap turns to keep profile fast
    python run_profile.py --use-mock            # mock backend; useful on a Mac
    python run_profile.py --no-prefix-cache     # disable prefix caching to A/B test it
    python run_profile.py --enforce-eager       # CUDA graphs OFF (slower; for A/B)

Requires Stage 1 to have completed (`python run.py --stage data`) so that
`data/augmented/hiv_status_0.3.jsonl` exists. The disk cache is disabled on
purpose — we want true backend latencies, not cache-hit short-circuits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent.policy import EIGPolicy, PolicyConfig
from src.agent.runner import AgentRunner
from src.data.attribute_schema import get_attribute
from src.data.augmenter import AugmentedCase
from src.experiment.runner import assemble_agent_chief_complaint
from src.llm.client import LLMClient, LLMResponse
from src.llm.library_backend import LibraryBackend, LibraryBackendConfig
from src.llm.mock_backend import MockBackend
from src.patient.simulator import PatientSimulator
from src.principal.benign import make_benign_task

PROMPTS_DIR = REPO_ROOT / "prompts"
DATA_PATH = REPO_ROOT / "data" / "augmented" / "hiv_status_0.3.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _load_one_case(case_index: int) -> AugmentedCase:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"ERROR: Stage 1 output not found at {DATA_PATH}\n"
            f"       Run `python run.py --stage data` first."
        )
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == case_index:
                return AugmentedCase.model_validate(json.loads(line))
    raise SystemExit(f"ERROR: case_index {case_index} out of range")


@contextmanager
def timer(label: str, store: dict[str, float]):
    t0 = time.perf_counter()
    yield
    store[label] = time.perf_counter() - t0


class TimingCollector:
    """Captures every LLM call via `LLMClient.on_call`. Each call records
    (role, latency_seconds, cache_hit) in event order so we can reconstruct
    the per-turn breakdown afterwards."""

    def __init__(self) -> None:
        self.by_role: dict[str, list[float]] = defaultdict(list)
        self.events: list[tuple[str, float, bool]] = []
        self.cache_hits = 0
        self.cache_misses = 0

    def on_call(self, response: LLMResponse) -> None:
        self.by_role[response.role].append(response.latency_seconds)
        self.events.append(
            (response.role, response.latency_seconds, response.cache_hit)
        )
        if response.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1


def _fmt_seconds(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.1f}ms"
    if s < 60:
        return f"{s:.2f}s"
    m, sec = divmod(s, 60)
    return f"{int(m):d}m{sec:04.1f}s"


# ---------------------------------------------------------------------------
# Per-step event splitter
# ---------------------------------------------------------------------------


def _split_events_into_turns(
    events: list[tuple[str, float, bool]], k_candidates: int
) -> list[dict[str, float]]:
    """Walk the event sequence and bucket calls into turns.

    Per-turn pattern (assuming the policy did NOT short-circuit to DIAGNOSE):
        1× agent.propose → K× agent.eig_predict (one batch) → 1× patient → 1× agent.belief

    A short-circuit turn looks like:
        1× agent.propose → K× agent.eig_predict → (no patient, no belief)

    Returns a list of {propose, eig, eig_count, patient, belief} dicts.
    """
    turns: list[dict[str, float]] = []
    i = 0
    while i < len(events):
        if events[i][0] != "agent":
            # Drift recovery: skip until the next agent (propose) call.
            i += 1
            continue
        # 1× propose
        propose_t = events[i][1]
        i += 1

        # K× eig_predict — they're emitted as K consecutive agent events, all
        # with role="agent". They precede either a patient call (regular turn)
        # or end-of-events (short-circuit DIAGNOSE turn).
        eig_t = 0.0
        eig_n = 0
        while (
            i < len(events)
            and events[i][0] == "agent"
            and eig_n < k_candidates
        ):
            eig_t += events[i][1]
            eig_n += 1
            i += 1
            # If we've consumed K agent events but the next is also agent, it
            # means the belief_update happened (no patient was called → DIAGNOSE).
            # Defer that case to the patient/belief block below.

        # 1× patient (if not a short-circuit)
        patient_t = 0.0
        if i < len(events) and events[i][0] == "patient":
            patient_t = events[i][1]
            i += 1

        # 1× belief_update (if patient happened)
        belief_t = 0.0
        if patient_t > 0 and i < len(events) and events[i][0] == "agent":
            belief_t = events[i][1]
            i += 1

        turns.append(
            {
                "propose": propose_t,
                "eig": eig_t,
                "eig_count": float(eig_n),
                "patient": patient_t,
                "belief": belief_t,
                "turn_total": propose_t + eig_t + patient_t + belief_t,
            }
        )
    return turns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile one case through the Stage 2 pipeline."
    )
    parser.add_argument("--case-index", type=int, default=0, help="Row index in the Stage 1 JSONL.")
    parser.add_argument("--max-queries", type=int, default=5, help="Max turns this profile is allowed.")
    parser.add_argument("--k-candidates", type=int, default=4, help="K candidate actions per turn.")
    parser.add_argument("--n-predictions", type=int, default=3, help="Predicted responses per EIG candidate.")
    parser.add_argument("--use-mock", action="store_true", help="Use mock backend (no GPU/vLLM needed).")
    parser.add_argument("--no-prefix-cache", action="store_true", help="Disable vLLM prefix caching (A/B knob).")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs (A/B knob).")
    parser.add_argument("--gpu-mem-util", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "auto"],
        help=(
            "Inference dtype. Default float16 — works on every NVIDIA GPU "
            "since Volta. Use bfloat16 only on Ampere+ (A100, A10, L4, H100); "
            "T4/Turing throws ValueError on bf16 (compute capability 7.5)."
        ),
    )
    args = parser.parse_args()

    case = _load_one_case(args.case_index)

    print("=" * 78)
    print("Stage 2 single-case profile")
    print("=" * 78)
    print(f"case_id           : {case.case_id}")
    print(f"τ                 : {case.tau} (hidden findings: {case.hidden_findings or 'none'})")
    print(f"question          : {case.question}")
    print(f"options           : {case.options}")
    print(f"correct_answer    : {case.correct_answer}")
    print()
    print(f"backend           : {'mock' if args.use_mock else 'library (vLLM)'}")
    if not args.use_mock:
        print(f"model_id          : {args.model_id}")
        print(f"gpu_mem_util      : {args.gpu_mem_util}")
        print(f"max_model_len     : {args.max_model_len}")
        print(f"max_num_seqs      : {args.max_num_seqs}")
        print(f"enforce_eager     : {args.enforce_eager}  (False = CUDA graphs ON)")
        print(f"prefix_caching    : {not args.no_prefix_cache}")
        print(f"dtype             : {args.dtype}  (float16 for T4/Turing; bfloat16 for Ampere+)")
    print(f"K_candidates      : {args.k_candidates}")
    print(f"n_eig_predictions : {args.n_predictions}")
    print(f"max_queries       : {args.max_queries}")
    print()

    coarse: dict[str, float] = {}
    timings = TimingCollector()

    # ----- Backend load -----
    if args.use_mock:
        with timer("backend_load", coarse):
            mock = MockBackend()
        client = LLMClient(
            backend="mock",
            cache_dir=None,                    # cache OFF for clean profiling
            mock_backend=mock,
            model_revisions={"agent": mock.revision, "patient": mock.revision},
            on_call=timings.on_call,
        )
        rev = mock.revision
    else:
        if not LibraryBackend.vllm_available():
            raise SystemExit(
                "ERROR: vLLM is not importable in this Python environment.\n"
                "       Run on a GPU machine with `pip install -e \".[prod]\"`,\n"
                "       or pass --use-mock for a CPU-only sanity run."
            )
        with timer("backend_load", coarse):
            lb = LibraryBackend(
                LibraryBackendConfig(
                    model_id=args.model_id,
                    gpu_memory_utilization=args.gpu_mem_util,
                    max_model_len=args.max_model_len,
                    max_num_seqs=args.max_num_seqs,
                    enforce_eager=args.enforce_eager,
                    enable_prefix_caching=not args.no_prefix_cache,
                    seed=0,
                    dtype=args.dtype,
                    trust_remote_code=True,
                )
            )
            lb.load()
        client = LLMClient(
            backend="library",
            cache_dir=None,                    # cache OFF — we want true latencies
            library_backend=lb,
            model_revisions={"agent": lb.revision(), "patient": lb.revision()},
            on_call=timings.on_call,
        )
        rev = lb.revision()
    print(f"  backend_load    : {_fmt_seconds(coarse['backend_load']):>10}")

    # ----- Setup (prompt loading, policy / patient construction) -----
    with timer("setup", coarse):
        agent_system_prompt = _load_prompt("agent_system.txt")
        action_select_template = _load_prompt("agent_action_select.txt")
        belief_update_template = _load_prompt("agent_belief_update.txt")
        eig_predict_template = _load_prompt("agent_eig_predict.txt")
        patient_system_template = _load_prompt("patient_system.txt")
        patient_user_template = _load_prompt("patient_user.txt")

        policy_config = PolicyConfig(
            k_candidate_actions=args.k_candidates,
            n_eig_predictions=args.n_predictions,
            epsilon_stop=0.05,
            epsilon_entropy=0.3,
            cost_budget=400.0,
            max_queries=args.max_queries,
        )
        policy = EIGPolicy(
            client=client,
            config=policy_config,
            action_select_template=action_select_template,
            belief_update_template=belief_update_template,
            eig_predict_template=eig_predict_template,
            agent_system_prompt=agent_system_prompt,
            descriptions=dict(case.options),
        )
        patient = PatientSimulator(
            client=client,
            case=case,
            system_template=patient_system_template,
            user_template=patient_user_template,
        )
        task = make_benign_task(case)
        runner = AgentRunner(
            policy=policy,
            chief_complaint_assembler=assemble_agent_chief_complaint,
        )
    print(f"  setup           : {_fmt_seconds(coarse['setup']):>10}")
    print()

    # ----- Run the agent loop on this one case -----
    print("Running agent loop...")
    with timer("agent_loop_total", coarse):
        traj = runner.run(
            task=task,
            patient=patient,
            case_id=case.case_id,
            visible_vignette=case.visible_vignette,
            correct_answer=case.correct_answer,
        )
    print(f"  agent_loop_total: {_fmt_seconds(coarse['agent_loop_total']):>10}")
    print()

    # ----- Print trajectory result -----
    print("-" * 78)
    print("Trajectory result")
    print("-" * 78)
    print(f"  steps            : {len(traj.steps)}")
    print(f"  final_diagnosis  : {traj.final_diagnosis}  "
          f"({'CORRECT' if traj.diagnostic_correct else 'incorrect'})")
    print(f"  total_cost       : ${traj.total_cost:.2f}")
    print(f"  stopped_reason   : {traj.metadata.get('stopped_reason')}")
    print(f"  early_diagnose   : {traj.metadata.get('early_diagnose', False)}")
    print()

    # ----- Per-role aggregate -----
    print("-" * 78)
    print("Per-role LLM call aggregate")
    print("-" * 78)
    print(f"  {'role':>10} | {'count':>5} | {'sum':>10} | {'mean':>10} | {'max':>10}")
    print(f"  {'-' * 10} | {'-' * 5} | {'-' * 10} | {'-' * 10} | {'-' * 10}")
    grand_llm = 0.0
    for role, latencies in sorted(timings.by_role.items()):
        n = len(latencies)
        total = sum(latencies)
        grand_llm += total
        mean = total / max(1, n)
        mx = max(latencies, default=0.0)
        print(f"  {role:>10} | {n:>5} | {_fmt_seconds(total):>10} | {_fmt_seconds(mean):>10} | {_fmt_seconds(mx):>10}")
    print()
    print(f"  cache hits / misses: {timings.cache_hits} / {timings.cache_misses}")
    print()

    # ----- Per-step breakdown -----
    turns = _split_events_into_turns(timings.events, args.k_candidates)
    print("-" * 78)
    print("Per-step LLM breakdown (s)")
    print("-" * 78)
    print(f"  {'step':>4} | {'propose':>8} | {'EIG (K calls)':>14} | {'patient':>8} | {'belief':>8} | {'turn total':>10}")
    print(f"  {'-' * 4} | {'-' * 8} | {'-' * 14} | {'-' * 8} | {'-' * 8} | {'-' * 10}")
    for i, t in enumerate(turns, start=1):
        eig_label = f"{t['eig']:.2f} ({int(t['eig_count'])}×)"
        print(
            f"  {i:>4} | "
            f"{t['propose']:>8.2f} | "
            f"{eig_label:>14} | "
            f"{t['patient']:>8.2f} | "
            f"{t['belief']:>8.2f} | "
            f"{t['turn_total']:>10.2f}"
        )
    print()

    # ----- Wall-clock summary -----
    load = coarse.get("backend_load", 0.0)
    setup = coarse.get("setup", 0.0)
    loop = coarse.get("agent_loop_total", 0.0)
    cpu_in_loop = max(0.0, loop - grand_llm)
    total = load + setup + loop

    print("-" * 78)
    print("Wall-clock summary")
    print("-" * 78)
    print(f"  backend_load     : {_fmt_seconds(load):>10}   ({load * 100 / max(0.001, total):>5.1f}% of total)")
    print(f"  setup            : {_fmt_seconds(setup):>10}   ({setup * 100 / max(0.001, total):>5.1f}% of total)")
    print(f"  agent_loop_total : {_fmt_seconds(loop):>10}   ({loop * 100 / max(0.001, total):>5.1f}% of total)")
    print(f"    └─ LLM calls   : {_fmt_seconds(grand_llm):>10}   ({grand_llm * 100 / max(0.001, loop):>5.1f}% of loop)")
    print(f"    └─ Python/CPU  : {_fmt_seconds(cpu_in_loop):>10}   ({cpu_in_loop * 100 / max(0.001, loop):>5.1f}% of loop)")
    print(f"  TOTAL            : {_fmt_seconds(total):>10}")
    print()

    # ----- Top-3 slowest LLM calls -----
    all_events = sorted(timings.events, key=lambda e: e[1], reverse=True)[:3]
    print("-" * 78)
    print("Top 3 slowest individual LLM calls (or batch chunks)")
    print("-" * 78)
    for role, lat, hit in all_events:
        suffix = " (cache hit)" if hit else ""
        print(f"  {role:>10}: {_fmt_seconds(lat):>10}{suffix}")
    print()

    # ----- Extrapolation to full Stage 2 -----
    if turns:
        per_turn_avg = sum(t["turn_total"] for t in turns) / len(turns)
        # Stage 2 default: 20 cases × ~6 turns/case (early-stop kicks in after a few)
        est_20cases_loop = per_turn_avg * 6 * 20
        est_total = load + setup + est_20cases_loop
        print("-" * 78)
        print("Extrapolation to full Stage 2 (20 cases × ~6 turns each)")
        print("-" * 78)
        print(f"  per-turn avg         : {_fmt_seconds(per_turn_avg)}")
        print(f"  est 20-case loop     : {_fmt_seconds(est_20cases_loop)}")
        print(f"  est wall-clock total : {_fmt_seconds(est_total)}")
        print(f"  (real run will be faster: model_load is one-time + early-stop kicks in)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
