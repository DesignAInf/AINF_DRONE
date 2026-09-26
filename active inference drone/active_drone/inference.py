"""Structured blanket inversion and a learned intervention model."""

from __future__ import annotations

from collections import deque
import numpy as np

from .structure import ACTIVE, EXTERNAL, INTERNAL, IE_MASKS, NODES, PARTITIONS, SENSORY, TEMPLATE_MASKS, TEMPLATES
from .world import DesignAction


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


class InterventionModel:
    """Beta-Bernoulli model learned from causal changes caused by interventions."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.counts = {"shield": np.array([1.0, 1.0]), "swap": np.array([1.0, 1.0])}

    def success_probability(self, kind: str) -> float:
        if kind == "noop":
            return 0.0
        c = self.counts[kind]
        return float(c[1] / c.sum())

    def uncertainty(self, kind: str) -> float:
        if kind == "noop":
            return 0.0
        a, b = self.counts[kind][1], self.counts[kind][0]
        return float(a * b / ((a + b) ** 2 * (a + b + 1)))

    def update(self, action: DesignAction, changed: bool | None) -> None:
        # ``changed`` must be inferred from learned dynamics; None means that
        # there is not yet enough post-intervention evidence to update.
        if action.kind != "noop" and changed is not None:
            self.counts[action.kind][int(bool(changed))] += 1


class StructuredBlanketLearner:
    """Posterior on all complete I/S/A/E partitions plus learned dynamics."""

    def __init__(self, window: int = 34, hazard: float = 0.018):
        self.window, self.hazard = window, hazard
        self.buffer = deque(maxlen=window)
        self.interventions = InterventionModel()
        self.reset()

    def reset(self) -> None:
        self.logits = np.zeros(len(PARTITIONS), float)
        self.posterior = np.full(len(PARTITIONS), 1 / len(PARTITIONS))
        self.a_hat = np.zeros((NODES, NODES), float)
        self.residual_var = np.ones(NODES)
        self.buffer.clear()
        self.interventions.reset()
        self.steps = 0
        self._marginals = np.full((NODES, 4), 0.25)
        self._pending_interventions: list[tuple[DesignAction, np.ndarray, int]] = []

    def predict(self) -> None:
        self.posterior = (1 - self.hazard) * self.posterior + self.hazard / len(self.posterior)

    def observe_transition(self, x: np.ndarray, x_next: np.ndarray, control: float,
                           action: DesignAction, changed: bool | None = None) -> None:
        # The learner never reads the simulator's ground-truth causal-change
        # flag.  Instead, it stores the pre-intervention learned dynamics and
        # evaluates efficacy only after enough new transitions have accrued.
        if action.kind != "noop":
            self._pending_interventions.append((action, self.a_hat.copy(), self.steps))
        self.buffer.append((np.asarray(x, float), np.asarray(x_next, float), float(control)))
        self.steps += 1
        if len(self.buffer) >= 12 and self.steps % 2 == 0:
            self._fit()
            self._resolve_pending_interventions()

    def _resolve_pending_interventions(self) -> None:
        keep = []
        for action, before, start in self._pending_interventions:
            if self.steps - start < 6:
                keep.append((action, before, start)); continue
            if action.kind == "shield":
                i, j = action.first, action.second
                pre = abs(before[i, j]) + abs(before[j, i])
                post = abs(self.a_hat[i, j]) + abs(self.a_hat[j, i])
                inferred = (pre - post) > 0.025
            else:
                inferred = float(np.linalg.norm(self.a_hat - before, ord="fro")) > 0.060
            self.interventions.update(action, inferred)
        self._pending_interventions = keep

    def _fit(self) -> None:
        x = np.asarray([r[0] for r in self.buffer])
        y = np.asarray([r[1] for r in self.buffer])
        u = np.asarray([r[2] for r in self.buffer])[:, None]
        design = np.column_stack([x, u, np.ones(len(x))])
        ridge = 0.18 * np.eye(design.shape[1])
        beta = np.linalg.pinv(design.T @ design + ridge) @ (design.T @ y)
        self.a_hat = beta[:NODES].T
        resid = y - design @ beta
        self.residual_var = np.mean(resid * resid, axis=0) + 1e-4

        observed = np.clip(abs(self.a_hat), 0, 0.55)
        # Score causal topology rather than exact edge amplitude.  This makes
        # the learner robust to strong but blanket-compatible couplings and
        # prevents coupling magnitude alone from identifying a breach.
        presence = np.clip(observed / 0.14, 0.0, 1.0)
        topology = -18.0 * np.mean((TEMPLATE_MASKS.astype(float) - presence[None, :, :]) ** 2, axis=(1, 2))
        amplitude = -12.0 * np.mean((TEMPLATES - observed[None, :, :]) ** 2, axis=(1, 2))
        structural = topology + amplitude
        control_gain = abs(beta[NODES])
        innovation = self.residual_var / max(np.mean(self.residual_var), 1e-9)
        semantic = np.zeros(len(PARTITIONS))
        for role, signal, sign in ((ACTIVE, control_gain, 1.0), (EXTERNAL, innovation, 0.55)):
            mask = PARTITIONS == role
            semantic += sign * np.mean(mask * signal[None, :], axis=1)
        # S and I orientation follows from the direction of the learned matrix.
        score = structural + 1.7 * semantic
        # Sequential evidence accumulation.  V5 replaced the posterior by a
        # fresh softmax at every refit, leaving q(M) almost maximally diffuse.
        # Here each refit contributes a tempered log-likelihood increment to
        # the predicted posterior.  The 0.25 tempering avoids counting heavily
        # overlapping rolling windows as independent samples.
        self.logits = np.log(self.posterior + 1e-15) + 0.50 * score
        shifted = self.logits - np.max(self.logits)
        q = np.exp(shifted)
        self.posterior = q / q.sum()
        self._update_marginals()

    def _update_marginals(self) -> None:
        for role in range(4):
            self._marginals[:, role] = self.posterior @ (PARTITIONS == role)

    @property
    def marginals(self) -> np.ndarray:
        return self._marginals

    @property
    def map_roles(self) -> np.ndarray:
        return PARTITIONS[int(np.argmax(self.posterior))].copy()

    @property
    def partition_entropy(self) -> float:
        return entropy(self.posterior)

    def expected_gap(self) -> float:
        # Vectorized posterior expectation of direct I--E coupling energy.
        sq = self.a_hat * self.a_hat
        pair_prob = np.tensordot(self.posterior, IE_MASKS, axes=(0, 0))
        return float(np.sum(pair_prob * sq))

    def action_features(self, action: DesignAction, quality: np.ndarray,
                        marginals: np.ndarray | None = None) -> tuple[float, float, float]:
        """Predicted structural benefit, pragmatic benefit, and epistemic value."""
        if action.kind == "noop":
            return 0.0, 0.0, 0.0
        a, b = action.first, action.second
        marg = self.marginals if marginals is None else np.asarray(marginals, float)
        if action.kind == "shield":
            ie = marg[a, INTERNAL] * marg[b, EXTERNAL] + marg[b, INTERNAL] * marg[a, EXTERNAL]
            strength = self.a_hat[a, b] ** 2 + self.a_hat[b, a] ** 2
            structural = float(ie * (0.15 + 18 * strength))
            pragmatic = 0.0
        else:
            boundary_a = marg[a, SENSORY] + marg[a, ACTIVE]
            boundary_b = marg[b, SENSORY] + marg[b, ACTIVE]
            # Precision crafting: route the uniquely precise module into a
            # likely boundary port and route the degraded boundary module out.
            ext_a = marg[a, EXTERNAL]
            ext_b = marg[b, EXTERNAL]
            gain_ab = boundary_a * ext_b * (quality[b] - quality[a])
            gain_ba = boundary_b * ext_a * (quality[a] - quality[b])
            # No global min/max shortcut: a high-quality module is useful only
            # insofar as the inferred role structure predicts that routing it
            # into the boundary will improve sensing or action.
            pragmatic = float(max(gain_ab, gain_ba))
            structural = 0.04 * abs(float(boundary_a - boundary_b))
        uncertainty = self.interventions.uncertainty(action.kind)
        local_h = entropy(marg[a]) + entropy(marg[b])
        epistemic = float(uncertainty * local_h)
        return structural, pragmatic, epistemic
