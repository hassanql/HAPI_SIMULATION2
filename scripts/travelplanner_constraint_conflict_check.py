"""Pre-flight constraint-conflict check for the TravelPlanner pivot.

For each of the 180 validation cases, this script verifies that:
  1. The case is structurally complete (reference_information present and
     parseable; local_constraint parseable; budget, days, etc. set).
  2. TravelPlanner's *hard* constraints (cuisine, room_type, transportation)
     are individually satisfiable from the reference data shipped with the
     case — i.e. at least one option matches each non-null constraint.
  3. Per-persona augmented-preference vectors (pregnancy, religious
     observance, financial distress) do not collapse the available option
     set to empty under a *strict* interpretation. Augmented preferences
     are soft (§4 of docs/travelplanner_pivot_plan.md); this check still
     flags cases where strict enforcement would leave <2 options in any
     option category, so the pilot's soft-degradation logic gets exercised
     and we know which cases are sensitive.

Output:
  results/travelplanner/preflight/conflict_check.json
  results/travelplanner/preflight/conflict_check.md

No API calls. No live planning. Pure dataset scan.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "travelplanner" / "preflight"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_ref_info_blob(blob: str) -> dict[str, str]:
    """Return {section_description: raw_content_text}. We do NOT try to
    parse the tabular Content into rows — pandas variable-width rendering
    breaks naive whitespace tokenization. Substring search on raw text
    is more reliable for constraint-satisfiability checks."""
    try:
        sections = ast.literal_eval(blob)
    except (ValueError, SyntaxError):
        return {}
    out: dict[str, str] = {}
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        out[sec.get("Description", "")] = sec.get("Content", "")
    return out


def _count_rows(text: str) -> int:
    """Count data rows = total lines - 1 header line, ignoring blanks."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return max(0, len(lines) - 1)


def _count_options_by_section(parsed: dict[str, str]) -> dict[str, int]:
    counts = {"attractions": 0, "restaurants": 0, "accommodations": 0, "flights": 0,
              "self_driving": 0, "taxi": 0}
    for desc, content in parsed.items():
        d = desc.lower()
        if d.startswith("attractions"):
            counts["attractions"] += _count_rows(content)
        elif d.startswith("restaurants"):
            counts["restaurants"] += _count_rows(content)
        elif d.startswith("accommodations"):
            counts["accommodations"] += _count_rows(content)
        elif "flight" in d:
            counts["flights"] += _count_rows(content)
        elif "self-driving" in d:
            counts["self_driving"] += 1
        elif "taxi" in d:
            counts["taxi"] += 1
    return counts


def _restaurant_section_text(parsed: dict[str, str]) -> str:
    return "\n".join(t for d, t in parsed.items() if d.lower().startswith("restaurants")).lower()


def _accommodation_section_text(parsed: dict[str, str]) -> str:
    return "\n".join(t for d, t in parsed.items() if d.lower().startswith("accommodations")).lower()


def _check_cuisine_satisfiable(parsed: dict[str, str],
                                cuisine_constraint) -> tuple[bool, int]:
    if cuisine_constraint is None:
        return True, 0
    wanted = cuisine_constraint if isinstance(cuisine_constraint, list) else [cuisine_constraint]
    wanted_lc = [c.lower() for c in wanted]
    text = _restaurant_section_text(parsed)
    total = sum(text.count(w) for w in wanted_lc)
    return total > 0, total


def _check_house_rule_satisfiable(parsed: dict[str, str],
                                    house_rule) -> tuple[bool, int]:
    """Local-constraint semantics: a house_rule value like 'pets' means the
    user travels with pets and needs a room that does NOT forbid pets. So
    we count accommodation lines that do NOT contain a 'No <rule>' clause."""
    if house_rule is None:
        return True, 0
    hr = house_rule.lower().strip()
    text = _accommodation_section_text(parsed)
    if not text:
        return True, 0
    forbidden_marker = f"no {hr}"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True, 0
    data_lines = lines[1:]  # drop header
    allowed = sum(1 for ln in data_lines if forbidden_marker not in ln)
    return allowed > 0, allowed


def _check_room_type_satisfiable(parsed_text_by_section: dict[str, str],
                                  room_type) -> tuple[bool, int]:
    """Substring-based count on the raw section text — robust to the
    variable-width column rendering that breaks the row-tokenizer."""
    if room_type is None:
        return True, 0
    rt = room_type.lower().strip()
    # The accommodation room_type values seen in the data are:
    #   'Private room', 'Entire home/apt', 'Shared room'
    # Constraint vocabulary: 'private room', 'entire room', 'shared room', 'not shared room'.
    text = _accommodation_section_text(parsed_text_by_section)
    if not text:
        return True, 0
    n_private = text.count("private room")
    n_entire = text.count("entire home/apt")
    n_shared = text.count("shared room") - n_private  # 'private room' contains 'shared'? No, but safer to subtract collisions
    n_shared = max(0, text.count(" shared room") + text.count("\nshared room"))
    if rt == "private room":
        return n_private > 0, n_private
    if rt == "entire room":
        return n_entire > 0, n_entire
    if rt == "shared room":
        return n_shared > 0, n_shared
    if rt == "not shared room":
        non_shared = n_private + n_entire
        return non_shared > 0, non_shared
    # Unknown vocabulary — return permissive
    return True, 0


def _check_transportation_satisfiable(parsed: dict[str, str],
                                       transportation) -> tuple[bool, int]:
    """transportation_constraint is 'no flight' or 'no self-driving' or None."""
    if transportation is None:
        return True, 0
    t = transportation.lower()
    if t in ("no flight", "no flights"):
        # Need self-driving or taxi available
        sd = sum(1 for d in parsed if "self-driving" in d.lower())
        tx = sum(1 for d in parsed if "taxi" in d.lower())
        return (sd + tx) > 0, sd + tx
    if t in ("no self-driving", "no self driving"):
        fl = sum(1 for d in parsed if "flight" in d.lower())
        tx = sum(1 for d in parsed if "taxi" in d.lower())
        return (fl + tx) > 0, fl + tx
    return True, 1


_PRICE_RE = re.compile(r"\b(\d{1,5})\.0\b")


def _augmented_strict_filter_estimate(parsed: dict[str, str],
                                       persona_attr: str) -> dict:
    """Estimate how many options survive a STRICT interpretation of each
    augmented preference. Used to flag soft-degradation-needed cases."""
    n_attractions = sum(_count_rows(t) for d, t in parsed.items() if d.lower().startswith("attractions"))
    n_restaurants = sum(_count_rows(t) for d, t in parsed.items() if d.lower().startswith("restaurants"))
    n_accommodations = sum(_count_rows(t) for d, t in parsed.items() if d.lower().startswith("accommodations"))

    out = {"attractions_total": n_attractions, "restaurants_total": n_restaurants,
            "accommodations_total": n_accommodations,
            "needs_soft_degradation": False, "notes": []}

    if persona_attr == "pregnancy":
        if n_restaurants < 3:
            out["needs_soft_degradation"] = True
            out["notes"].append("low restaurant pool — food_safety preference has few options")
        if n_attractions < 3:
            out["needs_soft_degradation"] = True
            out["notes"].append("low attraction pool — activity_intensity preference has few options")

    if persona_attr == "religious_observance":
        if n_restaurants < 5:
            out["needs_soft_degradation"] = True
            out["notes"].append("low restaurant pool — dietary-certified preference has few options")

    if persona_attr == "financial_distress":
        # Crude price extraction from accommodation text — price is a float
        # column. Substring-search for \d+.0 patterns in the accommodation
        # section. Imperfect but adequate for flagging extremes.
        text = _accommodation_section_text(parsed)
        prices = [float(m.group(1)) for m in _PRICE_RE.finditer(text)]
        if prices:
            median = sorted(prices)[len(prices) // 2]
            cheap_acc = sum(1 for p in prices if p <= median)
            out["cheap_accommodations"] = cheap_acc
            if cheap_acc < 2:
                out["needs_soft_degradation"] = True
                out["notes"].append("low cheap-accommodation pool — accommodation_tier preference will degrade")
    return out


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets first", file=sys.stderr)
        sys.exit(1)

    print("Loading osunlp/TravelPlanner validation split ...")
    ds = load_dataset("osunlp/TravelPlanner", "validation")["validation"]
    print(f"Loaded {len(ds)} cases")

    structurally_complete = 0
    hard_satisfiable = {"cuisine": 0, "room_type": 0, "house_rule": 0, "transportation": 0}
    hard_with_constraint = {"cuisine": 0, "room_type": 0, "house_rule": 0, "transportation": 0}
    per_case: list[dict] = []
    soft_degradation_flags = {"pregnancy": [], "religious_observance": [], "financial_distress": []}
    level_dist = Counter()
    days_dist = Counter()
    bad_cases: list[dict] = []

    for idx, row in enumerate(ds):
        case_id = f"tp_val_{idx:03d}"
        level_dist[row["level"]] += 1
        days_dist[row["days"]] += 1

        # Structural completeness
        ref_blob = row.get("reference_information", "")
        try:
            local_constraint = ast.literal_eval(row["local_constraint"])
        except (ValueError, SyntaxError):
            bad_cases.append({"case_id": case_id, "reason": "local_constraint unparseable"})
            continue
        if not ref_blob or len(ref_blob) < 100:
            bad_cases.append({"case_id": case_id, "reason": "empty reference_information"})
            continue

        parsed = _parse_ref_info_blob(ref_blob)
        if not parsed:
            bad_cases.append({"case_id": case_id, "reason": "reference_information unparseable"})
            continue

        opt_counts = _count_options_by_section(parsed)
        structurally_complete += 1

        # Hard-constraint satisfiability checks
        case_report = {
            "case_id": case_id, "idx": idx,
            "level": row["level"], "days": row["days"], "budget": row["budget"],
            "org": row["org"], "dest": row["dest"],
            "people_number": row["people_number"],
            "local_constraint": local_constraint,
            "option_counts": opt_counts,
            "hard_checks": {}, "augmented_flags": {},
        }

        # Cuisine
        c_ok, c_n = _check_cuisine_satisfiable(parsed, local_constraint.get("cuisine"))
        if local_constraint.get("cuisine") is not None:
            hard_with_constraint["cuisine"] += 1
            if c_ok:
                hard_satisfiable["cuisine"] += 1
        case_report["hard_checks"]["cuisine"] = {"satisfiable": c_ok, "matching_options": c_n,
                                                   "constraint": local_constraint.get("cuisine")}

        # Room type
        rt_ok, rt_n = _check_room_type_satisfiable(parsed, local_constraint.get("room type"))
        if local_constraint.get("room type") is not None:
            hard_with_constraint["room_type"] += 1
            if rt_ok:
                hard_satisfiable["room_type"] += 1
        case_report["hard_checks"]["room_type"] = {"satisfiable": rt_ok, "matching_options": rt_n,
                                                     "constraint": local_constraint.get("room type")}

        # House rule
        hr_ok, hr_n = _check_house_rule_satisfiable(parsed, local_constraint.get("house rule"))
        if local_constraint.get("house rule") is not None:
            hard_with_constraint["house_rule"] += 1
            if hr_ok:
                hard_satisfiable["house_rule"] += 1
        case_report["hard_checks"]["house_rule"] = {"satisfiable": hr_ok, "matching_options": hr_n,
                                                     "constraint": local_constraint.get("house rule")}

        # Transportation
        t_ok, t_n = _check_transportation_satisfiable(parsed, local_constraint.get("transportation"))
        if local_constraint.get("transportation") is not None:
            hard_with_constraint["transportation"] += 1
            if t_ok:
                hard_satisfiable["transportation"] += 1
        case_report["hard_checks"]["transportation"] = {"satisfiable": t_ok, "matching_options": t_n,
                                                          "constraint": local_constraint.get("transportation")}

        # Augmented-preference soft-degradation estimate per attribute
        for attr in ("pregnancy", "religious_observance", "financial_distress"):
            est = _augmented_strict_filter_estimate(parsed, attr)
            case_report["augmented_flags"][attr] = est
            if est["needs_soft_degradation"]:
                soft_degradation_flags[attr].append({"case_id": case_id, "notes": est["notes"]})

        per_case.append(case_report)

    # Pooled summary
    n = len(ds)
    summary = {
        "n_cases": n,
        "structurally_complete": structurally_complete,
        "bad_cases": bad_cases,
        "level_distribution": dict(level_dist),
        "days_distribution": dict(days_dist),
        "hard_constraint_satisfiability": {
            k: {
                "satisfied": hard_satisfiable[k],
                "with_constraint": hard_with_constraint[k],
                "rate": (hard_satisfiable[k] / hard_with_constraint[k]) if hard_with_constraint[k] else None,
            }
            for k in hard_satisfiable
        },
        "augmented_soft_degradation_counts": {
            attr: len(flags) for attr, flags in soft_degradation_flags.items()
        },
        "augmented_soft_degradation_detail": {
            attr: flags[:10] for attr, flags in soft_degradation_flags.items()  # first 10 examples
        },
    }
    full = {"summary": summary, "per_case": per_case}

    out_json = OUT_DIR / "conflict_check.json"
    out_json.write_text(json.dumps(full, indent=2, default=str))

    # Markdown summary
    md_lines = [
        "# TravelPlanner pre-flight constraint-conflict check",
        "",
        f"- Total validation cases: **{n}**",
        f"- Structurally complete: **{structurally_complete}/{n}**",
        f"- Bad cases: **{len(bad_cases)}**",
        "",
        "## Level / days distribution",
        f"- Levels: {dict(level_dist)}",
        f"- Days:   {dict(days_dist)}",
        "",
        "## Hard-constraint satisfiability (rows = constraints that were set non-null)",
        "",
        "| Constraint | Cases with constraint | Satisfiable from ref data | Rate |",
        "| --- | --- | --- | --- |",
    ]
    for k in ("cuisine", "room_type", "house_rule", "transportation"):
        sat = hard_satisfiable[k]; tot = hard_with_constraint[k]
        rate = f"{100*sat/tot:.1f}%" if tot else "—"
        md_lines.append(f"| {k} | {tot} | {sat} | {rate} |")
    md_lines += [
        "",
        "## Augmented-preference soft-degradation flags",
        "",
        "_Cases where strict enforcement of the augmented preference would leave too few options. "
        "These cases REMAIN VALID for the pilot because augmented preferences are soft — "
        "this list just tells us which cases will exercise the soft-degradation logic._",
        "",
        "| Attribute | Cases needing soft degradation |",
        "| --- | --- |",
    ]
    for attr in ("pregnancy", "religious_observance", "financial_distress"):
        md_lines.append(f"| {attr} | {len(soft_degradation_flags[attr])} |")

    if bad_cases:
        md_lines += ["", "## Bad cases (DROPPED from pilot pool)", ""]
        for bc in bad_cases:
            md_lines.append(f"- {bc['case_id']}: {bc['reason']}")

    md_lines += [
        "",
        "## Verdict",
        "",
    ]
    rates_ok = all(
        (hard_satisfiable[k] / hard_with_constraint[k] >= 0.95) if hard_with_constraint[k] else True
        for k in hard_satisfiable
    )
    structural_ok = (structurally_complete / n) >= 0.95
    if rates_ok and structural_ok:
        md_lines.append("**PASS** — case pool is healthy. Proceed to Phase 1 pilot.")
    else:
        md_lines.append("**ATTENTION** — review details above before proceeding.")

    out_md = OUT_DIR / "conflict_check.md"
    out_md.write_text("\n".join(md_lines))

    # Console
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    print("\n=== SUMMARY ===")
    print(f"Structurally complete: {structurally_complete}/{n}")
    print(f"Bad cases: {len(bad_cases)}")
    for k in ("cuisine", "room_type", "house_rule", "transportation"):
        sat = hard_satisfiable[k]; tot = hard_with_constraint[k]
        if tot:
            print(f"  {k:15s}: {sat}/{tot} satisfiable ({100*sat/tot:.1f}%)")
    for attr in ("pregnancy", "religious_observance", "financial_distress"):
        print(f"  soft-degradation flags ({attr}): {len(soft_degradation_flags[attr])}")


if __name__ == "__main__":
    main()
