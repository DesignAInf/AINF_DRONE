"""Paired construct/stabilise/transform experiments."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .agent import CONDITIONS, ConstitutiveDrone
from .structure import assignment_accuracy, blanket_set_accuracy
from .world import ConstitutiveWorld, TASKS


@dataclass
class MissionResult:
    condition: str
    task: str
    success: int
    collision: int
    viability: float
    mean_gap: float
    final_gap: float
    role_accuracy: float
    blanket_accuracy: float
    exact_partition: float
    architecture_score: float
    boundary_precision: float
    causal_changes: int
    recovery_delay: float
    records: list[dict]


def make_system(seed: int, condition: str, task: str):
    idx = TASKS.index(task)
    seeds = np.random.SeedSequence([seed, idx, 505]).spawn(2)
    return (ConstitutiveWorld(np.random.default_rng(seeds[0]), task),
            ConstitutiveDrone(np.random.default_rng(seeds[1]), condition))


def run_mission(world: ConstitutiveWorld, drone: ConstitutiveDrone) -> MissionResult:
    world.reset(); drone.reset(); records = []; changes = 0
    initial_gap = world.factorization_gap
    while True:
        before = world.x.copy()
        drone.learner.predict()
        policy, _ = drone.select(world)
        state = world.step(policy.control, policy.design, constitutive=drone.constitutive)
        # Ground-truth causal change is recorded for evaluation only.  It is
        # never passed to the learner or its intervention model.
        drone.learner.observe_transition(before, state["x"], policy.control, policy.design)
        changes += int(state["design_changed_causal_matrix"])
        roles = drone.learner.map_roles
        rec = {**state, "map_roles": roles, "role_marginals": drone.learner.marginals.copy(),
               "partition_entropy": drone.learner.partition_entropy,
               "estimated_gap": drone.learner.expected_gap(), "efe": policy.efe,
               "efe_risk": policy.risk, "efe_ambiguity": policy.ambiguity,
               "efe_structural_value": policy.structural_value,
               "efe_epistemic_value": policy.epistemic_value,
               "role_accuracy": assignment_accuracy(roles, world.roles),
               "blanket_accuracy": blanket_set_accuracy(roles, world.roles)}
        records.append(rec)
        if state["done"]:
            break

    y = np.array([r["y"] for r in records])
    gaps = np.array([r["factorization_gap"] for r in records])
    acc = np.array([r["role_accuracy"] for r in records])
    bacc = np.array([r["blanket_accuracy"] for r in records])
    target_match = np.array_equal(world.roles, world.target_roles)
    if world.task == "construct":
        task_ok = gaps[-1] < 0.10 * max(initial_gap, 1e-9)
        event = 0
    elif world.task == "stabilize":
        task_ok = gaps[-1] < 0.25
        event = world.breach_time
    else:
        task_ok = target_match
        event = world.context_time
    success = int(task_ok and not world.collision)
    recovery = np.nan
    for t in range(min(event, len(records)), len(records)):
        if gaps[t] < 0.25 and abs(y[t]) < 1.0:
            recovery = float(t - event); break
    # Behavioral viability is deliberately independent of architecture
    # metrics.  Factorization and routed precision remain in Q_arch rather
    # than being built into the outcome they are meant to explain.
    viability = float(np.exp(-np.mean(y * y) / 1.7) * (0.25 if world.collision else 1.0))
    precision = np.array([min(r["sensory_precision"], r["active_precision"]) for r in records])
    boundary_precision = float(np.mean(precision[-15:]))
    architecture_score = float(np.exp(-gaps[-1] / 8.0) * boundary_precision)
    return MissionResult(drone.condition, world.task, success, int(world.collision), viability,
                         float(gaps.mean()), float(gaps[-1]), float(acc.mean()), float(bacc.mean()),
                         float(np.array_equal(drone.learner.map_roles, world.roles)), architecture_score,
                         boundary_precision, changes,
                         recovery, records)
