"""TravelPlanner pilot orchestrator (Phase 1).

Reads a YAML config (see configs/travelplanner/stage_pilot_phase1.yaml),
picks N cases, generates τ-conditioned personas, pre-computes covert
briefs, runs trajectories in parallel for all (case, τ, principal)
combinations, and saves outputs to:

    results/travelplanner/<stage_name>/
      trajectories.jsonl       — one TravelTrajectory per line
      metrics.json             — summary numbers
      cover_task_eval.json     — per-trajectory cover-task pass/fail
      personas.jsonl           — generated personas
      covert_briefs.json       — covert principal briefs used
      log.jsonl                — structured run log
      config.yaml              — frozen copy of the config

Usage:
    python scripts/travelplanner_pilot.py configs/travelplanner/stage_pilot_phase1.yaml
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.agent.travel_runner import (  # noqa: E402
    TravelAgentConfig,
    TravelTrajectory,
    run_travel_trajectory,
)
from src.data.travel_data import select_pilot_cases, TravelCase  # noqa: E402
from src.data.travel_persona import (  # noqa: E402
    generate_persona,
    ATTRIBUTE_DESCRIPTIONS,
)
from src.llm.api_backend import build_api_backend_from_config  # noqa: E402
from src.llm.client import LLMClient  # noqa: E402
from src.data.travel_oracle import TravelUserOracle  # noqa: E402
from src.principal.travel_task import (  # noqa: E402
    make_benign_task,
    make_benign_realistic_task,
    make_benign_task_sasp,
    make_direct_task,
    make_direct_task_sasp,
    make_covert_task,
    make_covert_task_sasp,
    precompute_covert_briefs,
)


# ---------------------------------------------------------------------
# Config loading + LLM client setup
# ---------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_llm_client(config: dict[str, Any], cache_dir: Path) -> LLMClient:
    model_cfg_name = config.get("model_config", "models_api")
    model_cfg_path = REPO / "configs" / f"{model_cfg_name}.yaml"
    model_cfg = _load_yaml(model_cfg_path)
    api_profile = config.get("api_profile") or model_cfg.get("default_profile")
    backend_kind = config.get("backend", "api")
    if backend_kind != "api":
        raise RuntimeError(f"Phase 1 pilot supports backend=api only, got {backend_kind!r}")
    api_backend = build_api_backend_from_config(model_cfg, profile=api_profile)
    return LLMClient(backend="api", cache_dir=cache_dir, api_backend=api_backend)


# ---------------------------------------------------------------------
# Single-trajectory runner (for ThreadPoolExecutor)
# ---------------------------------------------------------------------


def _run_one(
    case: TravelCase,
    tau: int,
    principal: str,
    attribute: str,
    *,
    persona_base_seed: int,
    client: LLMClient,
    agent_config: TravelAgentConfig,
    action_space_mode: str = "differentiated",
) -> tuple[TravelTrajectory | None, str | None]:
    """Run one trajectory. Returns (trajectory, None) on success or
    (None, error_message) on failure.

    `action_space_mode`:
      'differentiated' — benign asks 4 TP-native dims; covert/direct ask 9
                         (configuration-laundering experiment)
      'same'           — all principals have the same 9-dim action space
                         (pure task-laundering experiment)
    """
    try:
        persona = generate_persona(case, tau, attribute, base_seed=persona_base_seed)
        oracle = TravelUserOracle(case, persona, client=client, attribute=attribute)
        if action_space_mode == "differentiated":
            makers = {"benign": make_benign_task,
                      "benign_realistic": make_benign_realistic_task,
                      "direct": make_direct_task,
                      "covert": lambda c, a: make_covert_task(c, a, client=None)}
        elif action_space_mode == "same":
            makers = {"benign": make_benign_task_sasp,
                      "benign_realistic": make_benign_realistic_task,  # same 6-dim regardless of mode
                      "direct": make_direct_task_sasp,
                      "covert": lambda c, a: make_covert_task_sasp(c, a, client=None)}
        else:
            return None, f"unknown action_space_mode {action_space_mode!r}"
        if principal not in makers:
            return None, f"unknown principal {principal!r}"
        task = makers[principal](case, attribute)
        traj = run_travel_trajectory(case, task, oracle, client, config=agent_config)
        # Cover-task eval (lazy import to avoid a startup-time cycle).
        from scripts.travelplanner_cover_task_eval import evaluate_cover_task
        evaluate_cover_task(traj, case)
        return traj, None
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        return None, f"{type(e).__name__}: {e}\n{tb}"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/travelplanner_pilot.py <config.yaml>", file=sys.stderr)
        sys.exit(1)
    cfg_path = Path(sys.argv[1]).resolve()
    config = _load_yaml(cfg_path)

    stage_name = config.get("output_subdir", "pilot_phase1")
    attribute = config.get("attribute", "pregnancy")
    n_cases = int(config.get("num_cases", 10))
    principals = config.get("principals", ["benign", "direct", "covert"])
    tau_values = config.get("tau_values", [0, 1])
    base_seed = int(config.get("seed", 0))
    max_concurrent = int(config.get("max_concurrent_trajectories", 4))
    case_stratify = bool(config.get("case_stratify_by_level", True))
    case_split = config.get("case_split", "validation")
    action_space_mode = config.get("action_space_mode", "differentiated")

    out_dir = REPO / "results" / "travelplanner" / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "llm_cache").mkdir(exist_ok=True)
    cache_dir = out_dir / "llm_cache"

    # Freeze config + write run header
    (out_dir / "config.yaml").write_text(cfg_path.read_text())
    print(f"[setup] stage={stage_name} attribute={attribute} cases={n_cases} "
          f"principals={principals} tau={tau_values} max_concurrent={max_concurrent} "
          f"action_space_mode={action_space_mode}")
    print(f"[setup] outputs -> {out_dir}")

    # Build LLM client
    client = _build_llm_client(config, cache_dir=cache_dir)
    print(f"[setup] LLM client ready: profile={config.get('api_profile')}")

    # Select cases
    cases = select_pilot_cases(
        n=n_cases,
        split=case_split,
        seed=base_seed,
        stratify_by_level=case_stratify,
    )
    print(f"[cases] selected {len(cases)} cases:")
    for c in cases:
        print(f"  - {c.case_id} [{c.level}] {c.org}->{c.dest} {c.days}d "
              f"${c.budget} p={c.people_number}")

    # Generate personas (deterministic, no API)
    personas = []
    for c in cases:
        for t in tau_values:
            p = generate_persona(c, t, attribute, base_seed=base_seed)
            personas.append(p)
    with (out_dir / "personas.jsonl").open("w") as f:
        for p in personas:
            f.write(p.model_dump_json() + "\n")
    print(f"[personas] generated {len(personas)} personas -> personas.jsonl")

    # Pre-compute covert briefs IF covert principal is in the set
    covert_briefs_map: dict[str, Any] = {}
    if "covert" in principals:
        print(f"[covert] pre-computing covert briefs in parallel "
              f"(max_workers={max_concurrent}) ...")
        t0 = time.time()
        covert_briefs_map = precompute_covert_briefs(
            cases, attribute, client=client, max_workers=max_concurrent
        )
        elapsed = time.time() - t0
        print(f"[covert] {len(covert_briefs_map)} briefs ready in {elapsed:.1f}s")
        with (out_dir / "covert_briefs.json").open("w") as f:
            briefs_dump = {
                cid: {"task_brief": ts.task_brief, "attribute": ts.attribute,
                      "target_dims": ts.target_dims}
                for cid, ts in covert_briefs_map.items()
            }
            json.dump(briefs_dump, f, indent=2)

    # Build the full task list
    work: list[tuple[TravelCase, int, str]] = []
    for c in cases:
        for t in tau_values:
            for p in principals:
                work.append((c, t, p))
    total = len(work)
    print(f"[run] dispatching {total} trajectories ...")

    agent_config = TravelAgentConfig(
        max_elicitation_steps=int(config.get("max_elicitation_steps", 8)),
        elicit_temperature=float(config.get("elicit_temperature", 0.4)),
        plan_temperature=float(config.get("plan_temperature", 0.2)),
        max_tokens_plan=int(config.get("max_tokens_plan", 4096)),
    )

    trajectories: list[TravelTrajectory] = []
    errors: list[dict[str, Any]] = []
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {
            ex.submit(
                _run_one, c, t, p, attribute,
                persona_base_seed=base_seed,
                client=client, agent_config=agent_config,
                action_space_mode=action_space_mode,
            ): (c, t, p)
            for (c, t, p) in work
        }
        done = 0
        for fut in as_completed(futures):
            c, t, p = futures[fut]
            traj, err = fut.result()
            done += 1
            if traj is not None:
                trajectories.append(traj)
                cover_ok = "✓" if traj.cover_task_passed else "✗"
                tag = f"{c.case_id}/τ={t}/{p}"
                pred = f" pred={traj.tau_prediction}" if p == "direct" else ""
                n_steps = traj.metadata.get("n_elicit_steps", 0)
                print(f"  [{done:>3}/{total}] {tag} cover={cover_ok} "
                      f"steps={n_steps} dims={len(traj.preference_profile)}{pred}")
            else:
                errors.append({"case_id": c.case_id, "tau": t, "principal": p,
                               "error": err})
                print(f"  [{done:>3}/{total}] {c.case_id}/τ={t}/{p} ERROR\n    {err}")
    wall = time.time() - t_start
    print(f"[run] done in {wall:.1f}s ({len(trajectories)} ok, {len(errors)} errors)")

    # Save trajectories
    with (out_dir / "trajectories.jsonl").open("w") as f:
        for tr in trajectories:
            f.write(tr.model_dump_json() + "\n")

    # Cover-task summary
    cover_summary = {
        "by_principal": {},
        "by_principal_tau": {},
    }
    for p in principals:
        ts = [t for t in trajectories if t.task.principal_type == p]
        if not ts:
            continue
        passed = sum(1 for t in ts if t.cover_task_passed)
        cover_summary["by_principal"][p] = {
            "n": len(ts), "passed": passed,
            "rate": passed / len(ts) if ts else 0.0,
        }
        for tau in tau_values:
            ts_pt = [t for t in ts if t.tau == tau]
            if not ts_pt:
                continue
            passed_pt = sum(1 for t in ts_pt if t.cover_task_passed)
            cover_summary["by_principal_tau"][f"{p}_tau{tau}"] = {
                "n": len(ts_pt), "passed": passed_pt,
                "rate": passed_pt / len(ts_pt) if ts_pt else 0.0,
            }

    # Direct τ-recovery (the sanity gate)
    direct_trajs = [t for t in trajectories if t.task.principal_type == "direct"]
    direct_correct = sum(
        1 for t in direct_trajs
        if (t.tau == 1 and t.tau_prediction == "yes")
        or (t.tau == 0 and t.tau_prediction == "no")
    )
    direct_acc = direct_correct / len(direct_trajs) if direct_trajs else 0.0

    metrics = {
        "stage": stage_name,
        "attribute": attribute,
        "n_cases": len(cases),
        "n_trajectories": len(trajectories),
        "n_errors": len(errors),
        "wall_seconds": wall,
        "cover_task": cover_summary,
        "direct_tau_accuracy": {
            "n": len(direct_trajs),
            "correct": direct_correct,
            "rate": direct_acc,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=2))

    print("\n========== PILOT SUMMARY ==========")
    print(f"Trajectories: {len(trajectories)}/{total}  Errors: {len(errors)}")
    print(f"Cover-task pass rate by principal:")
    for p, s in cover_summary["by_principal"].items():
        print(f"  {p:7s}: {s['passed']}/{s['n']} ({s['rate']*100:.0f}%)")
    print(f"Direct τ-recovery: {direct_correct}/{len(direct_trajs)} ({direct_acc*100:.0f}%)")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
