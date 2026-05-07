"""Posterior over diagnostic options (spec §7.2).

The agent maintains `Belief` — a probability distribution over
`TaskSpec.target_options`. Bayes update applies multiplicative likelihoods
elicited from the LLM via the belief-update prompt.

Invariants enforced at every mutation:
  - All probabilities in [0, 1].
  - Sum is approximately 1 (tolerance 1e-6).
  - Entropy is non-negative.
"""

from __future__ import annotations

import math


_EPS = 1e-12
_TOL = 1e-6


class Belief:
    def __init__(self, options: list[str], prior: dict[str, float] | None = None) -> None:
        if not options:
            raise ValueError("Belief requires at least one option.")
        if len(set(options)) != len(options):
            raise ValueError(f"Belief options must be unique: {options}")
        self.options: list[str] = list(options)
        if prior is None:
            n = len(options)
            self._p: dict[str, float] = {o: 1.0 / n for o in options}
        else:
            if set(prior) != set(options):
                raise ValueError(
                    f"Prior keys {sorted(prior)} != options {sorted(options)}"
                )
            self._p = dict(prior)
            self._normalise(strict=True)

    # ---- introspection ------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        return dict(self._p)

    def __getitem__(self, option: str) -> float:
        return self._p[option]

    def __repr__(self) -> str:  # pragma: no cover
        items = ", ".join(f"{o}={p:.3f}" for o, p in self._p.items())
        return f"Belief({items})"

    def entropy(self) -> float:
        """Shannon entropy in nats."""
        h = 0.0
        for p in self._p.values():
            if p > _EPS:
                h -= p * math.log(p)
        return max(0.0, h)

    def map_estimate(self) -> str:
        return max(self._p, key=self._p.__getitem__)

    # ---- mutation -----------------------------------------------------

    def update(self, observation: str, likelihoods: dict[str, float]) -> None:
        """Bayes update: posterior[o] ∝ prior[o] * likelihood[o].

        `observation` is informational (not used in math) — it's recorded by
        the caller in trajectory metadata.
        """
        if set(likelihoods) != set(self.options):
            raise ValueError(
                f"likelihood keys {sorted(likelihoods)} != options {sorted(self.options)}"
            )
        for o in self.options:
            v = likelihoods[o]
            if v < 0:
                raise ValueError(f"Negative likelihood for {o!r}: {v}")
            self._p[o] = self._p[o] * v
        # If everything became zero (degenerate), back off to uniform — better
        # than NaN cascading through downstream metrics.
        total = sum(self._p.values())
        if total <= _EPS:
            n = len(self.options)
            for o in self.options:
                self._p[o] = 1.0 / n
        else:
            self._normalise(strict=False)

    def _normalise(self, *, strict: bool) -> None:
        total = sum(self._p.values())
        if total <= 0:
            raise ValueError(f"Cannot normalise belief with non-positive total: {total}")
        for o in self.options:
            self._p[o] = self._p[o] / total
        s = sum(self._p.values())
        if abs(s - 1.0) > _TOL:
            if strict:
                raise ValueError(f"Belief failed to normalise: sum={s}")

    # ---- helpers ------------------------------------------------------

    @classmethod
    def uniform(cls, options: list[str]) -> "Belief":
        return cls(options)

    def copy(self) -> "Belief":
        return Belief(self.options, self.to_dict())


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) for two same-keyed distributions, in nats."""
    if set(p) != set(q):
        raise ValueError("KL inputs must share keys.")
    out = 0.0
    for k in p:
        pi, qi = p[k], q[k]
        if pi <= _EPS:
            continue
        if qi <= _EPS:
            return float("inf")
        out += pi * math.log(pi / qi)
    return out
