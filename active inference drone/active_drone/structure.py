"""Exact four-role partitions and path-space factorisation diagnostics."""

from __future__ import annotations

from itertools import combinations
import numpy as np

INTERNAL, SENSORY, ACTIVE, EXTERNAL = range(4)
ROLE_NAMES = ("internal", "sensory", "active", "external")
ROLE_SHORT = ("I", "S", "A", "E")
NODES = 8


def enumerate_partitions() -> np.ndarray:
    """All 8!/(2!)^4=2520 labelled partitions with two nodes per role."""
    rows = []
    universe = set(range(NODES))
    for internal in combinations(range(NODES), 2):
        rest_i = universe.difference(internal)
        for sensory in combinations(sorted(rest_i), 2):
            rest_s = rest_i.difference(sensory)
            for active in combinations(sorted(rest_s), 2):
                roles = np.full(NODES, EXTERNAL, int)
                roles[list(internal)] = INTERNAL
                roles[list(sensory)] = SENSORY
                roles[list(active)] = ACTIVE
                rows.append(roles)
    return np.asarray(rows, dtype=np.int8)


PARTITIONS = enumerate_partitions()


def edge_template(roles: np.ndarray) -> np.ndarray:
    """Directed blanket template; columns are causes and rows are effects."""
    r = np.asarray(roles, int)
    # Disallowed channels are exactly absent. This makes the clean template a
    # genuine path boundary rather than merely a weak graphical bottleneck.
    out = np.zeros((NODES, NODES), float)
    np.fill_diagonal(out, 0.32)
    # The directed blanket cycle: E -> S -> I -> A -> E.
    for source, target in ((EXTERNAL, SENSORY), (SENSORY, INTERNAL),
                           (INTERNAL, ACTIVE), (ACTIVE, EXTERNAL)):
        out[np.ix_(r == target, r == source)] = 0.24
    # Weak within-role coupling makes the two members exchangeable.
    for role in range(4):
        idx = np.flatnonzero(r == role)
        out[np.ix_(idx, idx)] = 0.18
        out[idx, idx] = 0.32
    radius = max(abs(np.linalg.eigvals(out)))
    return out * min(1.0, 0.82 / max(radius, 1e-9))


TEMPLATES = np.asarray([edge_template(p) for p in PARTITIONS])
TEMPLATE_MASKS = TEMPLATES > 0.05

# Mask of directed I<->E pairs for every complete partition.  This supports
# vectorized posterior expectations of leak energy.
IE_MASKS = np.zeros((len(PARTITIONS), NODES, NODES), dtype=float)
for _k, _roles in enumerate(PARTITIONS):
    _i = _roles == INTERNAL; _e = _roles == EXTERNAL
    IE_MASKS[_k][np.ix_(_i, _e)] = 1.0
    IE_MASKS[_k][np.ix_(_e, _i)] = 1.0



def assignment_accuracy(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Role accuracy; exchange of the two nodes inside a role is immaterial."""
    return float(np.mean(np.asarray(estimate) == np.asarray(truth)))


def blanket_set_accuracy(estimate: np.ndarray, truth: np.ndarray) -> float:
    e = np.isin(estimate, (SENSORY, ACTIVE))
    t = np.isin(truth, (SENSORY, ACTIVE))
    return float(np.mean(e == t))


def architectural_gap(matrix: np.ndarray, roles: np.ndarray, noise: float = 0.12) -> float:
    """Linear-Gaussian path-law KL rate induced by unscreened I<->E drift.

    This is the quadratic Girsanov/Foellmer control-energy surrogate for the
    conditional path-space mutual information. It is zero exactly when the
    simulated architecture has no direct interior--exterior drift channel.
    """
    a = np.asarray(matrix, float)
    i = np.flatnonzero(np.asarray(roles) == INTERNAL)
    e = np.flatnonzero(np.asarray(roles) == EXTERNAL)
    leak = np.concatenate([a[np.ix_(i, e)].ravel(), a[np.ix_(e, i)].ravel()])
    return float(0.5 * np.sum(leak * leak) / (noise * noise))
