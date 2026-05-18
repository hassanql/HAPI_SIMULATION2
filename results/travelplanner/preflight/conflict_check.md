# TravelPlanner pre-flight constraint-conflict check

- Total validation cases: **180**
- Structurally complete: **180/180**
- Bad cases: **0**

## Level / days distribution
- Levels: {'easy': 60, 'medium': 60, 'hard': 60}
- Days:   {3: 60, 5: 60, 7: 60}

## Hard-constraint satisfiability (rows = constraints that were set non-null)

| Constraint | Cases with constraint | Satisfiable from ref data | Rate |
| --- | --- | --- | --- |
| cuisine | 48 | 48 | 100.0% |
| room_type | 64 | 64 | 100.0% |
| house_rule | 77 | 77 | 100.0% |
| transportation | 51 | 51 | 100.0% |

## Augmented-preference soft-degradation flags

_Cases where strict enforcement of the augmented preference would leave too few options. These cases REMAIN VALID for the pilot because augmented preferences are soft — this list just tells us which cases will exercise the soft-degradation logic._

| Attribute | Cases needing soft degradation |
| --- | --- |
| pregnancy | 0 |
| religious_observance | 0 |
| financial_distress | 0 |

## Verdict

**PASS** — case pool is healthy. Proceed to Phase 1 pilot.