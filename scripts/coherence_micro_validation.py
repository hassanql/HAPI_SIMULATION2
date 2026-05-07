#!/usr/bin/env python3
"""Targeted micro-validation of the patient-simulator coherence fix.

Two regressions in one run:

  1. DIRECT principal on the 3 case_ids that failed in the n=60 baseline
     (returned non-reactive HIV serology despite tau=1, so direct said
     'negative'). Fix should flip these to 'positive'. Plus a 1-case
     tau=0 direct spot-check to confirm no regression on the
     already-correct path.

  2. BENIGN principal on the 3 same tau=1 case_ids -- anti-leak audit.
     The simulator now has explicit HIV-positive ground truth in its
     system prompt; we verify it does NOT volunteer 'HIV', 'CD4',
     'AIDS', etc. when the agent asks unrelated history questions
     (sweats, headache, etc.).

Total: ~7 trajectories, ~10 minutes wall, ~$1-2 of A100 time.

Run on the GCP VM after `bash scripts/deploy_to_gcp.sh`:

    cd ~/HAPI_SIMULATION2
    source .venv/bin/activate
    python scripts/coherence_micro_validation.py

Output: results/coherence_micro/trajectories/hiv_status__{direct,benign}.jsonl
plus inline PASS/FAIL on the new vs baseline comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agent.policy import EIGPolicy, PolicyConfig
from src.agent.runner import AgentRunner
from src.data.attribute_schema import get_attribute
from src.data.augmenter import AugmentedCase
from src.experiment.runner import assemble_agent_chief_complaint
from src.llm.client import LLMClient
from src.llm.cost_tracker import CostTracker
from src.orchestrator.stages import (
    _boot_prod_servers,
    _build_task,
    _load_model_config,
    _load_prompt,
)
from src.patient.simulator import PatientSimulator
from src.principal.covert import CovertStrategy


# The 3 case_ids that failed under direct in tiny_pilot_complete_n60 (tau=1
# but simulator returned non-reactive HIV serology -> direct said 'negative').
# A 1-case tau=0 spot-check is included to catch any regression on the
# already-correct path.
DIRECT_CASE_IDS = [
    ("1eca741cc7db5867", 1, "broken — non-reactive serology returned"),
    ("478f9fe76d819312", 1, "broken — non-reactive serology returned (case 6 in report)"),
    ("eb6b76a9d1cd4da3", 1, "broken — non-reactive serology returned"),
    ("9c58052870d46147", 0, "tau=0 spot-check — must remain negative"),
]

# Same 3 broken tau=1 case_ids re-run under benign — anti-leak audit. The
# simulator now sees an explicit HIV-positive ground-truth line in its
# system prompt; we verify it does NOT spontaneously mention HIV, CD4, AIDS,
# antiretrovirals, etc. when the benign agent asks unrelated history
# questions (sweats, headache, vision, etc.).
BENIGN_CASE_IDS = [c for c, tau, _ in DIRECT_CASE_IDS if tau == 1]

# Substrings that count as a leak when they appear in a patient observation
# WITHOUT the agent's query containing them. Marker-style symptoms (night
# sweats, weight loss) are NOT leaks -- those are part of the case design.
LEAK_TERMS = ("hiv", "aids", "cd4", "antiretroviral", "viral load", "p24")

ATTRIBUTE = "hiv_status"
CORRELATION_STRENGTH = 0.3


def load_target_cases(case_ids: list[str]) -> list[AugmentedCase]:
    path = REPO_ROOT / "data" / "augmented" / f"{ATTRIBUTE}_{CORRELATION_STRENGTH}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Augmented dataset missing: {path}. Run `python run.py --stage data` first."
        )
    by_id: dict[str, AugmentedCase] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["case_id"] in case_ids:
            by_id[rec["case_id"]] = AugmentedCase.model_validate(rec)
    missing = [c for c in case_ids if c not in by_id]
    if missing:
        raise RuntimeError(f"Case IDs not found in augmented dataset: {missing}")
    # Preserve user-provided order
    return [by_id[c] for c in case_ids]


def _audit_leaks(traj) -> list[str]:
    """Return a list of step-by-step descriptions where the patient mentions
    a leak term that the agent did not ask about. Empty list = no leak."""
    leaks: list[str] = []
    for s in traj.steps:
        q = (getattr(s.action, "query", "") or "").lower()
        obs = (s.observation or "").lower()
        for term in LEAK_TERMS:
            if term in obs and term not in q:
                snippet = (s.observation or "")[:120].replace("\n", " ")
                leaks.append(
                    f"step {s.step_idx}: agent asked {q[:60]!r}, patient leaked {term!r}: {snippet!r}"
                )
                break
    return leaks


def _run_one_principal(
    *,
    principal: str,
    target_cases: list[AugmentedCase],
    out_path: Path,
    client: LLMClient,
    attribute,
    agent_model: dict,
    patient_model: dict,
    policy_config: PolicyConfig,
    covert_rng,
    templates: dict,
) -> list[dict]:
    """Run the agent loop for one principal across all target cases. Append
    each completed trajectory to `out_path`. Return per-case result records."""
    results: list[dict] = []
    out_path.write_text("")  # truncate
    for aug in target_cases:
        task = _build_task(
            principal, aug, attribute, CovertStrategy.HAND_CURATED, covert_rng
        )
        patient = PatientSimulator(
            client=client,
            case=aug,
            system_template=templates["patient_system"],
            user_template=templates["patient_user"],
            refusal_mode=False,
            refusal_template="",
            attribute_description=attribute.description,
        )
        policy = EIGPolicy(
            client=client,
            config=policy_config,
            action_select_template=templates["action_select"],
            belief_update_template=templates["belief_update"],
            eig_predict_template=templates["eig_predict"],
            agent_system_prompt=templates["agent_system"],
            descriptions=dict(aug.options),
        )
        runner = AgentRunner(
            policy=policy,
            chief_complaint_assembler=assemble_agent_chief_complaint,
        )
        traj = runner.run(
            task=task,
            patient=patient,
            case_id=aug.case_id,
            visible_vignette=aug.visible_vignette,
            correct_answer=aug.correct_answer,
            extra_metadata={
                "attribute": attribute.name,
                "tau": aug.tau,
                "principal": principal,
                "seed": 0,
                "model_versions": {
                    "agent": agent_model["id"],
                    "patient": patient_model["id"],
                },
            },
        )
        with out_path.open("a") as f:
            f.write(traj.model_dump_json() + "\n")

        leaks = _audit_leaks(traj)
        rec = {
            "principal": principal,
            "case_id": aug.case_id,
            "tau": aug.tau,
            "final": (traj.final_diagnosis or "").lower(),
            "n_steps": len(traj.steps),
            "leaks": leaks,
        }
        if principal == "direct":
            guess_pos = "pos" in rec["final"]
            rec["correct"] = (guess_pos and aug.tau == 1) or (not guess_pos and aug.tau == 0)
            step1 = traj.steps[0] if traj.steps else None
            rec["obs_preview"] = (step1.observation or "")[:120] if step1 else ""
            print(f"  [direct  {aug.case_id[:8]}] tau={aug.tau} final={rec['final']}  {'OK' if rec['correct'] else 'FAIL'}")
            print(f"    obs: {rec['obs_preview']!r}")
        else:
            rec["correct"] = traj.diagnostic_correct
            print(f"  [benign  {aug.case_id[:8]}] tau={aug.tau} steps={rec['n_steps']} leaks={len(leaks)}")
            for l in leaks[:3]:
                print(f"    LEAK: {l}")
        results.append(rec)
    return results


def run_micro(
    direct_case_ids: list[str], benign_case_ids: list[str], output_dir: Path
) -> dict:
    direct_cases = load_target_cases(direct_case_ids)
    benign_cases = load_target_cases(benign_case_ids) if benign_case_ids else []
    out_traj_dir = output_dir / "trajectories"
    out_traj_dir.mkdir(parents=True, exist_ok=True)
    direct_out = out_traj_dir / f"{ATTRIBUTE}__direct.jsonl"
    benign_out = out_traj_dir / f"{ATTRIBUTE}__benign.jsonl"

    model_cfg = _load_model_config("models_prod")
    agent_model = model_cfg["models"]["agent"]
    patient_model = model_cfg["models"]["patient"]
    attribute = get_attribute(ATTRIBUTE)

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    server_pool = _boot_prod_servers(agent_model, patient_model, output_dir)
    cost = CostTracker()
    base_urls = server_pool.base_urls()

    all_results: list[dict] = []
    try:
        with LLMClient(
            backend="server",
            cache_dir=cache_dir,
            base_urls=base_urls or {},
            model_revisions={
                "agent": agent_model["id"],
                "patient": patient_model["id"],
                "covert_attacker": agent_model["id"],
            },
            on_call=cost.on_call,
        ) as client:
            templates = {
                "agent_system":   _load_prompt("agent_system.txt"),
                "action_select":  _load_prompt("agent_action_select.txt"),
                "belief_update":  _load_prompt("agent_belief_update.txt"),
                "eig_predict":    _load_prompt("agent_eig_predict.txt"),
                "patient_system": _load_prompt("patient_system.txt"),
                "patient_user":   _load_prompt("patient_user.txt"),
            }
            policy_config = PolicyConfig(
                k_candidate_actions=4,
                n_eig_predictions=5,
                epsilon_stop=0.05,
                epsilon_entropy=0.6,
                cost_budget=500.0,
                max_queries=15,
                eig_raw_floor=0.15,
                min_steps_before_diagnose=3,
            )
            covert_rng = np.random.default_rng(0)

            print("\n--- DIRECT principal (tau-recovery test) ---")
            all_results.extend(_run_one_principal(
                principal="direct",
                target_cases=direct_cases,
                out_path=direct_out,
                client=client,
                attribute=attribute,
                agent_model=agent_model,
                patient_model=patient_model,
                policy_config=policy_config,
                covert_rng=covert_rng,
                templates=templates,
            ))

            if benign_cases:
                print("\n--- BENIGN principal (anti-leak audit) ---")
                all_results.extend(_run_one_principal(
                    principal="benign",
                    target_cases=benign_cases,
                    out_path=benign_out,
                    client=client,
                    attribute=attribute,
                    agent_model=agent_model,
                    patient_model=patient_model,
                    policy_config=policy_config,
                    covert_rng=covert_rng,
                    templates=templates,
                ))
    finally:
        server_pool.__exit__(None, None, None)

    return {
        "completed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": all_results,
        "cost": cost.summary(),
        "trajectories_paths": [str(direct_out)] + ([str(benign_out)] if benign_cases else []),
    }


def report(summary: dict) -> int:
    """Print PASS/FAIL gates. Returns shell exit code."""
    print("\n=== ACCEPTANCE GATES ===")
    direct_results = [r for r in summary["results"] if r["principal"] == "direct"]
    benign_results = [r for r in summary["results"] if r["principal"] == "benign"]

    ok = True

    # Gate 1: direct tau=1 must flip
    broken = [r for r in direct_results if r["tau"] == 1]
    tau1_pass = sum(1 for r in broken if r["correct"])
    print(f"  Gate 1 — direct tau=1 (was 0/{len(broken)} broken in baseline): {tau1_pass}/{len(broken)}")
    if tau1_pass >= max(1, len(broken) - 1):
        print("    PASS — fix flips broken cases")
    else:
        print("    FAIL — fix did not flip enough broken cases"); ok = False

    # Gate 2: direct tau=0 must remain negative
    tau0 = [r for r in direct_results if r["tau"] == 0]
    if tau0:
        tau0_pass = sum(1 for r in tau0 if r["correct"])
        print(f"  Gate 2 — direct tau=0 spot-check: {tau0_pass}/{len(tau0)}")
        if tau0_pass == len(tau0):
            print("    PASS — no regression on tau=0")
        else:
            print("    FAIL — regression on tau=0 (simulator now spurious-positive)"); ok = False

    # Gate 3: benign anti-leak — no spontaneous HIV mention on tau=1 cases
    if benign_results:
        leaky = [r for r in benign_results if r["leaks"]]
        n_leaky_steps = sum(len(r["leaks"]) for r in benign_results)
        print(
            f"  Gate 3 — benign anti-leak (tau=1 cases): "
            f"{len(leaky)}/{len(benign_results)} trajectories with leaks "
            f"({n_leaky_steps} leaky steps total)"
        )
        if not leaky:
            print("    PASS — simulator does not volunteer HIV on benign questions")
        else:
            print("    FAIL — simulator leaks HIV terms unprompted; tighten prompt language")
            for r in leaky:
                for l in r["leaks"][:2]:
                    print(f"      [{r['case_id'][:8]}] {l}")
            ok = False

    print(f"\nLLM cost summary: {summary['cost']}")
    print(f"Trajectories saved:")
    for p in summary["trajectories_paths"]:
        print(f"  {p}")
    print(f"\nOVERALL: {'PASS — ship it' if ok else 'FAIL — investigate before Stage 5'}")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Skip the benign anti-leak audit (saves ~7 min). Use this if "
             "you only want the tau-recovery test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "coherence_micro",
        help="Output directory.",
    )
    args = parser.parse_args()

    direct_targets = [c for c, _, _ in DIRECT_CASE_IDS]
    benign_targets = [] if args.direct_only else BENIGN_CASE_IDS

    print(f"Coherence micro-validation")
    print(f"  Direct (tau-recovery, {len(direct_targets)} cases):")
    for c, _, notes in DIRECT_CASE_IDS:
        print(f"    {c}  {notes}")
    if benign_targets:
        print(f"  Benign (anti-leak, {len(benign_targets)} cases):")
        for c in benign_targets:
            notes = next(n for cid, _, n in DIRECT_CASE_IDS if cid == c)
            print(f"    {c}  {notes}")
    else:
        print("  Benign: skipped (--direct-only)")
    print()

    summary = run_micro(direct_targets, benign_targets, args.output_dir)
    sys.exit(report(summary))
