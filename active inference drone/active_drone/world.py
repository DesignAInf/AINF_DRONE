"""A causal drone world in which design actions change the actual boundary."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .structure import ACTIVE, EXTERNAL, INTERNAL, SENSORY, NODES, architectural_gap, edge_template

TASKS = ("construct", "stabilize", "transform")


@dataclass(frozen=True)
class DesignAction:
    kind: str = "noop"
    first: int = -1
    second: int = -1

    @property
    def label(self) -> str:
        if self.kind == "noop":
            return "noop"
        if self.kind == "shield":
            return f"shield({self.first},{self.second})"
        return f"swap({self.first},{self.second})"


class ConstitutiveWorld:
    """Linear stochastic process plus a simple cross-track viability state."""

    horizon = 104
    dt = 0.18

    def __init__(self, rng: np.random.Generator, task: str):
        if task not in TASKS:
            raise ValueError(task)
        self.rng, self.task = rng, task
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        self.x = self.rng.normal(0, 0.15, NODES)
        self.y = 0.0
        self.wind = 0.0
        self.collision = False
        self.roles = np.repeat(np.arange(4), 2).astype(int)
        # Randomise module identities without changing role counts.
        self.roles = self.roles[self.rng.permutation(NODES)]
        self.initial_roles = self.roles.copy()
        self.quality = self.rng.uniform(0.72, 0.94, NODES)
        self.context_time = int(self.rng.integers(50, 59))
        self.breach_time = int(self.rng.integers(27, 39))
        self.leaks: dict[tuple[int, int], float] = {}
        self.distractors: dict[tuple[int, int], float] = {}
        if self.task == "construct":
            self._seed_leaks(3)
        if self.task in ("construct", "stabilize"):
            self._seed_lawful_distractors(4)
        self.transform_pair = self._choose_transform_pair()
        self.transform_decoys = self._choose_transform_decoys()
        self.target_roles = self.roles.copy()
        self._context_changed = False
        self._breached = False
        self.matrix = self._matrix()
        return self.state()

    def _choose_transform_pair(self) -> tuple[int, int]:
        boundary = np.flatnonzero(np.isin(self.roles, (SENSORY, ACTIVE)))
        external = np.flatnonzero(self.roles == EXTERNAL)
        return int(boundary[0]), int(external[0])

    def _choose_transform_decoys(self) -> tuple[int, int]:
        bad, good = self.transform_pair
        internal = [int(i) for i in np.flatnonzero(self.roles == INTERNAL) if i not in (bad, good)]
        boundary = [int(i) for i in np.flatnonzero(np.isin(self.roles, (SENSORY, ACTIVE))) if i not in (bad, good)]
        return internal[0], boundary[0]

    def _seed_lawful_distractors(self, count: int = 4) -> None:
        # Strong but blanket-compatible edges prevent raw coupling magnitude
        # from identifying a breach.  The true leak can be selected reliably
        # only by combining learned dynamics with the inferred role partition.
        allowed = []
        for src, dst in ((EXTERNAL, SENSORY), (SENSORY, INTERNAL),
                         (INTERNAL, ACTIVE), (ACTIVE, EXTERNAL)):
            for j in np.flatnonzero(self.roles == src):
                for i in np.flatnonzero(self.roles == dst):
                    if i != j:
                        allowed.append((int(i), int(j)))
        self.rng.shuffle(allowed)
        for edge in allowed[:count]:
            self.distractors[edge] = float(self.rng.uniform(0.46, 0.53))

    def _seed_leaks(self, count: int = 1) -> None:
        i = np.flatnonzero(self.roles == INTERNAL)
        e = np.flatnonzero(self.roles == EXTERNAL)
        pairs = [(int(a), int(b)) for a in i for b in e]
        self.rng.shuffle(pairs)
        for pair in pairs[:count]:
            self.leaks[pair] = float(self.rng.uniform(0.34, 0.48))

    def _matrix(self) -> np.ndarray:
        matrix = edge_template(self.roles)
        for (i, j), strength in self.distractors.items():
            matrix[i, j] = strength
        for (i, e), strength in self.leaks.items():
            matrix[i, e] = strength
            matrix[e, i] = 0.72 * strength
        return matrix

    @property
    def factorization_gap(self) -> float:
        return architectural_gap(self.matrix, self.roles)

    @property
    def sensory_precision(self) -> float:
        return float(np.mean(self.quality[self.roles == SENSORY]))

    @property
    def active_precision(self) -> float:
        return float(np.mean(self.quality[self.roles == ACTIVE]))

    def state(self) -> dict:
        return {"t": self.t, "x": self.x.copy(), "y": float(self.y),
                "roles": self.roles.copy(), "target_roles": self.target_roles.copy(),
                "quality": self.quality.copy(), "matrix": self.matrix.copy(),
                "factorization_gap": self.factorization_gap,
                "sensory_precision": self.sensory_precision,
                "active_precision": self.active_precision,
                "collision": bool(self.collision)}

    def candidate_actions(self) -> list[DesignAction]:
        actions = [DesignAction()]
        # Any pair can be shielded; only genuine unscreened I--E edges benefit.
        if self.task == "construct" or (self.task == "stabilize" and self._breached):
            for i in range(NODES):
                for j in range(i + 1, NODES):
                    actions.append(DesignAction("shield", i, j))
        # Re-routing is available only once the transform context has made a
        # precision mismatch observable. It preserves exactly two nodes/role.
        if self.task == "transform" and self._context_changed:
            for i in range(NODES):
                for j in range(i + 1, NODES):
                    actions.append(DesignAction("swap", i, j))
        return actions

    def _apply_design(self, action: DesignAction, constitutive: bool) -> bool:
        if action.kind == "noop" or not constitutive:
            return False
        before = self.matrix.copy()
        if action.kind == "shield":
            key = (action.first, action.second)
            rev = (action.second, action.first)
            self.leaks.pop(key, None)
            self.leaks.pop(rev, None)
        elif action.kind == "swap":
            a, b = action.first, action.second
            self.roles[a], self.roles[b] = self.roles[b], self.roles[a]
            # Leaks are physical links and stay attached to module identities.
        self.matrix = self._matrix()
        return bool(np.max(abs(self.matrix - before)) > 1e-10)

    def _exogenous_events(self) -> None:
        if self.task == "stabilize" and self.t == self.breach_time and not self._breached:
            self._seed_leaks(1)
            self._breached = True
            self.matrix = self._matrix()
        if self.task == "transform" and self.t == self.context_time and not self._context_changed:
            bad, good = self.transform_pair
            decoy_low, decoy_high = self.transform_decoys
            # The globally most extreme qualities are deliberately the wrong
            # pair.  The useful swap is identifiable only from role inference.
            self.quality[bad] = 0.20
            self.quality[good] = 0.90
            self.quality[decoy_low] = 0.10
            self.quality[decoy_high] = 0.99
            self.target_roles = self.roles.copy()
            self.target_roles[bad], self.target_roles[good] = self.target_roles[good], self.target_roles[bad]
            self._context_changed = True

    def step(self, control: float, design: DesignAction, constitutive: bool = True) -> dict:
        self._exogenous_events()
        changed = self._apply_design(design, constitutive)
        active = (self.roles == ACTIVE).astype(float)
        external = (self.roles == EXTERNAL).astype(float)
        sensory = (self.roles == SENSORY).astype(float)
        self.wind = 0.88 * self.wind + self.rng.normal(0, 0.23)
        innovation = self.rng.normal(0, 0.12, NODES)
        innovation += external * self.rng.normal(0, 0.24, NODES)
        self.x = self.matrix @ self.x + 0.24 * control * active + 0.20 * self.wind * external + innovation
        # Better routed sensory states suppress estimation noise; better active
        # states transmit control. Leaks add an irreducible viability penalty.
        sensed_wind = self.wind + self.rng.normal(0, 0.20 / max(self.sensory_precision, 0.12))
        drift = sensed_wind + 1.22 * self.active_precision * control
        self.y = 0.91 * self.y + self.dt * drift + self.rng.normal(0, 0.025)
        self.y += 0.004 * self.factorization_gap * np.sign(self.wind + 1e-9)
        self.t += 1
        self.collision = bool(abs(self.y) > 2.35)
        out = self.state()
        out.update({"wind": float(self.wind), "design_changed_causal_matrix": changed,
                    "design_action": design.label, "control": float(control),
                    "done": bool(self.t >= self.horizon or self.collision)})
        return out
