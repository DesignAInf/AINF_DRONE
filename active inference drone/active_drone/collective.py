"""Distributed inference and constitutive design of a two-drone group blanket.

Author: Luca M. Possati
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

from .inference import InterventionModel, entropy
from .structure import (ACTIVE, EXTERNAL, INTERNAL, NODES, PARTITIONS, SENSORY,
                        TEMPLATES, TEMPLATE_MASKS, architectural_gap, assignment_accuracy,
                        blanket_set_accuracy, edge_template)
from .world import DesignAction

COLLECTIVE_TASKS = ("assemble", "repair", "reconfigure")
COLLECTIVE_CONDITIONS = ("collective_constitutive", "collective_uniform_posterior",
                         "collective_diagnostic", "individual_constitutive",
                         "local_shared_policy", "pooled_independent_policy",
                         "fixed_team", "communication_only", "random_joint",
                         "oracle_collective")
OWNER = np.array([0, 0, 0, 0, 1, 1, 1, 1], int)


class CollectiveWorld:
    """Two drones carrying one payload through a jointly designed interface."""

    horizon = 112
    dt = 0.17

    def __init__(self, rng: np.random.Generator, task: str):
        if task not in COLLECTIVE_TASKS:
            raise ValueError(task)
        self.rng, self.task = rng, task
        self.reset()

    def reset(self) -> dict:
        self.t = 0
        # Each vehicle initially owns one module of every role; identities are
        # permuted locally, so role labels are not recoverable from ownership.
        local_roles = np.arange(4)
        self.roles = np.concatenate([local_roles[self.rng.permutation(4)],
                                     local_roles[self.rng.permutation(4)]]).astype(int)
        self.initial_roles = self.roles.copy()
        self.target_roles = self.roles.copy()
        self.x = self.rng.normal(0, .12, NODES)
        self.y = np.array([-.16, .16], float)
        self.wind = np.zeros(2)
        self.quality = self.rng.uniform(.74, .94, NODES)
        self.leaks: dict[tuple[int, int], float] = {}
        self.distractors: dict[tuple[int, int], float] = {}
        self.breach_time = int(self.rng.integers(29, 41))
        self.context_time = int(self.rng.integers(56, 66))
        self._breached = False; self._context_changed = False
        if self.task == "assemble":
            self._seed_cross_leaks(1)
        if self.task in ("assemble", "repair"):
            self._seed_lawful_distractors(4)
        self.transform_pair = self._choose_cross_transform()
        self.transform_decoys = self._choose_cross_decoys()
        self.matrix = self._matrix()
        self.collision = False
        return self.state()

    def _seed_lawful_distractors(self, count: int = 4) -> None:
        allowed = []
        for src, dst in ((EXTERNAL, SENSORY), (SENSORY, INTERNAL),
                         (INTERNAL, ACTIVE), (ACTIVE, EXTERNAL)):
            for j in np.flatnonzero(self.roles == src):
                for i in np.flatnonzero(self.roles == dst):
                    if OWNER[i] != OWNER[j]:
                        allowed.append((int(i), int(j)))
        self.rng.shuffle(allowed)
        for edge in allowed[:count]:
            self.distractors[edge] = float(self.rng.uniform(.46, .53))

    def _cross_pairs(self, role_a: int, role_b: int) -> list[tuple[int, int]]:
        left = np.flatnonzero(self.roles == role_a)
        right = np.flatnonzero(self.roles == role_b)
        return [(int(i), int(j)) for i in left for j in right if OWNER[i] != OWNER[j]]

    def _seed_cross_leaks(self, count: int) -> None:
        all_pairs = self._cross_pairs(INTERNAL, EXTERNAL)
        # At least one local learner must observe both endpoints. The group
        # covers all such links only by pooling the two complementary views.
        mask_a=set([0,1,2,3,4,5]); mask_b=set([2,3,4,5,6,7])
        pairs=[p for p in all_pairs if set(p).issubset(mask_a) or set(p).issubset(mask_b)] or all_pairs
        self.rng.shuffle(pairs)
        for pair in pairs[:count]:
            self.leaks[tuple(sorted(pair))] = float(self.rng.uniform(.34, .47))

    def _choose_cross_transform(self) -> tuple[int, int]:
        boundary = np.flatnonzero(np.isin(self.roles, (SENSORY, ACTIVE)))
        external = np.flatnonzero(self.roles == EXTERNAL)
        pairs = [(int(a), int(b)) for a in boundary for b in external if OWNER[a] != OWNER[b]]
        return pairs[int(self.rng.integers(len(pairs)))]

    def _choose_cross_decoys(self) -> tuple[int, int]:
        bad, good = self.transform_pair
        candidates = []
        for low in np.flatnonzero(self.roles == INTERNAL):
            for high in np.flatnonzero(np.isin(self.roles, (SENSORY, ACTIVE))):
                if OWNER[low] != OWNER[high] and low not in (bad, good) and high not in (bad, good):
                    candidates.append((int(low), int(high)))
        if candidates:
            return candidates[0]
        # Fallback remains role-incompatible even if it is not the globally
        # most adversarial cross-owner pair.
        low = int(np.flatnonzero(self.roles == INTERNAL)[0])
        high = int([i for i in np.flatnonzero(np.isin(self.roles, (SENSORY, ACTIVE))) if i != bad][0])
        return low, high

    def _matrix(self) -> np.ndarray:
        matrix = edge_template(self.roles)
        for (i, j), strength in self.distractors.items():
            matrix[i, j] = strength
        for (i, e), strength in self.leaks.items():
            matrix[i, e] = strength; matrix[e, i] = .72 * strength
        return matrix

    @property
    def factorization_gap(self) -> float:
        return architectural_gap(self.matrix, self.roles)

    def precision(self, owner: int, role: int) -> float:
        mask = (OWNER == owner) & (self.roles == role)
        return float(np.mean(self.quality[mask])) if np.any(mask) else .12

    @property
    def group_boundary_precision(self) -> float:
        return float(min(np.mean(self.quality[self.roles == SENSORY]),
                         np.mean(self.quality[self.roles == ACTIVE])))

    @property
    def payload(self) -> float:
        return float(self.y.mean())

    @property
    def formation_error(self) -> float:
        return float(self.y[0] - self.y[1])

    def state(self) -> dict:
        return {"t": self.t, "x": self.x.copy(), "y": self.y.copy(),
                "payload": self.payload, "formation_error": self.formation_error,
                "roles": self.roles.copy(), "target_roles": self.target_roles.copy(),
                "quality": self.quality.copy(), "matrix": self.matrix.copy(),
                "factorization_gap": self.factorization_gap,
                "group_boundary_precision": self.group_boundary_precision,
                "collision": bool(self.collision)}

    def candidate_actions(self) -> list[DesignAction]:
        actions = [DesignAction()]
        # In repair, a nonspecific integrity alarm opens the structural action
        # gate after the breach; it reveals neither the affected link nor its
        # roles and is available identically in every condition.
        if self.task == "assemble" or (self.task == "repair" and self._breached):
            for i in range(4):
                for j in range(4, 8):
                    actions.append(DesignAction("shield", i, j))
        if self.task == "reconfigure" and self._context_changed:
            for i in range(4):
                for j in range(4, 8):
                    actions.append(DesignAction("swap", i, j))
        return actions

    def _events(self) -> None:
        if self.task == "repair" and self.t == self.breach_time and not self._breached:
            self._seed_cross_leaks(1); self._breached = True; self.matrix = self._matrix()
        if self.task == "reconfigure" and self.t == self.context_time and not self._context_changed:
            bad, good = self.transform_pair
            decoy_low, decoy_high = self.transform_decoys
            self.quality[bad] = .20; self.quality[good] = .90
            self.quality[decoy_low] = .10; self.quality[decoy_high] = .99
            self.target_roles = self.roles.copy()
            self.target_roles[bad], self.target_roles[good] = self.target_roles[good], self.target_roles[bad]
            self._context_changed = True

    def _apply_agreed_design(self, first: DesignAction, second: DesignAction, constitutive: bool) -> tuple[bool, bool]:
        agreement = first == second and first.kind != "noop"
        if not agreement or not constitutive:
            return agreement, False
        before = self.matrix.copy()
        action = first
        if action.kind == "shield":
            self.leaks.pop(tuple(sorted((action.first, action.second))), None)
        elif action.kind == "swap":
            i, j = action.first, action.second
            self.roles[i], self.roles[j] = self.roles[j], self.roles[i]
        self.matrix = self._matrix()
        return True, bool(np.max(abs(self.matrix-before)) > 1e-10)

    def step(self, controls: np.ndarray, proposals: tuple[DesignAction, DesignAction],
             constitutive: bool = True) -> dict:
        self._events()
        agreement, changed = self._apply_agreed_design(proposals[0], proposals[1], constitutive)
        common = self.rng.normal(0, .20)
        self.wind = .86*self.wind + common + self.rng.normal(0, .12, 2)
        innovation = self.rng.normal(0, .12, NODES)
        for owner in (0, 1):
            innovation += ((OWNER == owner) & (self.roles == EXTERNAL)) * self.rng.normal(0, .22, NODES)
        injection = np.zeros(NODES)
        for owner in (0, 1):
            injection += .24*float(controls[owner]) * ((OWNER == owner) & (self.roles == ACTIVE))
            injection += .18*float(self.wind[owner]) * ((OWNER == owner) & (self.roles == EXTERNAL))
        self.x = self.matrix @ self.x + injection + innovation
        next_y = self.y.copy()
        for owner in (0, 1):
            sensory = self.precision(owner, SENSORY); active = self.precision(owner, ACTIVE)
            sensed = self.wind[owner] + self.rng.normal(0, .17/max(sensory, .12))
            tension = -.32*(self.y[owner]-self.y[1-owner])
            next_y[owner] = .90*self.y[owner] + self.dt*(sensed + 1.22*active*controls[owner] + tension)
            next_y[owner] += self.rng.normal(0, .022)
        next_y += .0035*self.factorization_gap*np.sign(self.wind + 1e-9)
        self.y = next_y; self.t += 1
        self.collision = bool(abs(self.payload) > 2.35 or abs(self.formation_error) > 1.55)
        out = self.state()
        out.update({"wind": self.wind.copy(), "controls": np.asarray(controls, float).copy(),
                    "proposal_a": proposals[0].label, "proposal_b": proposals[1].label,
                    "agreement": agreement, "design_changed_causal_matrix": changed,
                    "done": bool(self.t >= self.horizon or self.collision)})
        return out


class LocalRoleLearner:
    """Exact partition posterior from one drone's partially observed paths."""

    def __init__(self, owner: int, observed: np.ndarray, window: int = 36):
        self.owner = owner; self.observed = np.asarray(observed, int); self.window = window
        self.buffer = deque(maxlen=window); self.interventions = InterventionModel(); self.reset()

    def reset(self) -> None:
        self.posterior = np.full(len(PARTITIONS), 1/len(PARTITIONS))
        self.logits = np.zeros(len(PARTITIONS)); self.a_hat = np.zeros((NODES, NODES))
        self.seen = np.zeros((NODES, NODES), bool); self.residual_var = np.ones(NODES)
        self.buffer.clear(); self.steps = 0; self.interventions.reset()

    def predict(self) -> None:
        self.posterior = .982*self.posterior + .018/len(self.posterior)

    @property
    def marginals(self) -> np.ndarray:
        out = np.zeros((NODES, 4))
        for role in range(4): out[:, role] = self.posterior @ (PARTITIONS == role)
        return out

    def observe(self, x: np.ndarray, x_next: np.ndarray, controls: np.ndarray,
                action: DesignAction, changed: bool) -> None:
        self.buffer.append((np.asarray(x)[self.observed], np.asarray(x_next)[self.observed],
                            np.asarray(controls, float)))
        self.steps += 1
        if len(self.buffer) >= 12 and self.steps % 2 == 0: self._fit()

    def _fit(self) -> None:
        x = np.asarray([r[0] for r in self.buffer]); y = np.asarray([r[1] for r in self.buffer])
        u = np.asarray([r[2] for r in self.buffer])
        design = np.column_stack([x, u, np.ones(len(x))])
        beta = np.linalg.pinv(design.T@design + .18*np.eye(design.shape[1])) @ (design.T@y)
        idx = self.observed
        local_a = beta[:len(idx)].T
        self.a_hat[:] = 0; self.a_hat[np.ix_(idx, idx)] = local_a
        self.seen[:] = False; self.seen[np.ix_(idx, idx)] = True
        resid = y-design@beta; rv = np.mean(resid*resid, axis=0)+1e-4
        self.residual_var[:] = np.mean(rv); self.residual_var[idx] = rv
        observed_presence=np.clip(abs(local_a)/.14,0,1)
        template_presence=TEMPLATE_MASKS[:,idx][:,:,idx].astype(float)
        topology=-20*np.mean((template_presence-observed_presence[None])**2,axis=(1,2))
        amp_templates=TEMPLATES[:,idx][:,:,idx]
        amplitude=-10*np.mean((amp_templates-np.clip(abs(local_a),0,.55)[None])**2,axis=(1,2))
        structural=topology+amplitude
        control_gain = abs(beta[len(idx)+self.owner])
        innovation = rv/max(np.mean(rv), 1e-9)
        semantic = np.zeros(len(PARTITIONS))
        local_owned = OWNER[idx] == self.owner
        for k, node in enumerate(idx):
            if local_owned[k]: semantic += 1.8*control_gain[k]*(PARTITIONS[:, node] == ACTIVE)
            semantic += .50*innovation[k]*(PARTITIONS[:, node] == EXTERNAL)
        score = structural + semantic/max(len(idx), 1)
        self.logits = np.log(self.posterior + 1e-15) + 0.50*score
        z = self.logits-self.logits.max(); q = np.exp(z); self.posterior = q/q.sum()


@dataclass
class CollectivePolicy:
    controls: np.ndarray
    proposal_a: DesignAction
    proposal_b: DesignAction
    efe: float
    risk: float
    structural_value: float
    precision_value: float
    epistemic_value: float


class CollaborativeTeam:
    controls = np.array([-1.05, 0.0, 1.05])

    def __init__(self, rng: np.random.Generator, condition: str):
        if condition not in COLLECTIVE_CONDITIONS: raise ValueError(condition)
        self.rng, self.condition = rng, condition
        self.learners = (LocalRoleLearner(0, np.array([0,1,2,3,4,5])),
                         LocalRoleLearner(1, np.array([2,3,4,5,6,7])))
        self.reset()

    def reset(self) -> None:
        for learner in self.learners: learner.reset()
        self.swap_used = False
        self.design_attempts = 0
        self.group_buffer = deque(maxlen=38)
        self.group_a_hat = np.zeros((NODES,NODES))
        self.reference_a_hat = None
        self.group_logits = np.zeros(len(PARTITIONS))
        self.group_steps = 0
        self.failed_actions: set[str] = set()
        self.interventions = InterventionModel()
        self._pending_interventions: list[tuple[DesignAction, np.ndarray, int]] = []

    @property
    def pooled_posterior(self) -> np.ndarray:
        # Local evidences are combined with sufficient statistics reconstructed
        # from the union of their complementary messages. No raw observation is
        # available to both drones individually.
        logq = (np.log(self.learners[0].posterior+1e-15)+
                np.log(self.learners[1].posterior+1e-15)+self.group_logits)
        logq -= logq.max(); q = np.exp(logq); return q/q.sum()

    @property
    def pooled_marginals(self) -> np.ndarray:
        q = self.pooled_posterior; out = np.zeros((NODES,4))
        for role in range(4): out[:,role] = q@(PARTITIONS==role)
        return out

    @property
    def pooled_roles(self) -> np.ndarray:
        return PARTITIONS[int(np.argmax(self.pooled_posterior))].copy()

    @property
    def pooled_entropy(self) -> float:
        return entropy(self.pooled_posterior)

    def _combined_a(self) -> np.ndarray:
        return self.group_a_hat

    def _fit_group_statistics(self) -> None:
        if len(self.group_buffer)<12: return
        x=np.asarray([r[0] for r in self.group_buffer]); y=np.asarray([r[1] for r in self.group_buffer]); u=np.asarray([r[2] for r in self.group_buffer])
        design=np.column_stack([x,u,np.ones(len(x))])
        beta=np.linalg.pinv(design.T@design+.22*np.eye(design.shape[1])) @ (design.T@y)
        self.group_a_hat=beta[:NODES].T
        if self.group_steps>=26 and self.reference_a_hat is None:
            self.reference_a_hat=self.group_a_hat.copy()
        observed=np.clip(abs(self.group_a_hat),0,.55)
        presence=np.clip(observed/.14,0,1)
        topology=-20*np.mean((TEMPLATE_MASKS.astype(float)-presence[None])**2,axis=(1,2))
        amplitude=-10*np.mean((TEMPLATES-observed[None])**2,axis=(1,2))
        structural=topology+amplitude
        control_gain=abs(beta[NODES:NODES+2]).T
        semantic=np.zeros(len(PARTITIONS))
        for owner in (0,1):
            nodes=np.flatnonzero(OWNER==owner)
            for node in nodes: semantic+=1.15*control_gain[node,owner]*(PARTITIONS[:,node]==ACTIVE)
        # Group evidence is an additional tempered likelihood term.  Local
        # posteriors already accumulate evidence sequentially.
        self.group_logits=.50*(structural+semantic/4)

    def _wind_estimate(self, world: CollectiveWorld, marg: np.ndarray, owner: int) -> float:
        weights = marg[:,EXTERNAL]*(OWNER==owner)
        return float(np.sum(weights*world.x)/max(weights.sum(),1e-9))

    def _controls_for(self, world: CollectiveWorld, marg: np.ndarray | None = None) -> np.ndarray:
        # Common role-neutral low-level controller for every collective
        # condition.  This removes the previous energy confound between the
        # communication and blanket conditions and isolates structural design.
        selected = []
        for owner in (0,1):
            local = world.x[OWNER == owner]
            wind = float(np.mean(local))
            scores=[]
            for u in self.controls:
                payload_next=.90*world.y[owner]+world.dt*(wind+float(u)-.32*(world.y[owner]-world.y[1-owner]))
                scores.append(payload_next**2+.32*(payload_next-world.y[1-owner])**2+.035*u*u)
            selected.append(float(self.controls[int(np.argmin(scores))]))
        return np.asarray(selected)

    def _action_value(self, action: DesignAction, marg: np.ndarray, a_hat: np.ndarray,
                      quality: np.ndarray, intervention: InterventionModel) -> tuple[float,float,float,float]:
        if action.kind=="noop": return 0.,0.,0.,0.
        i,j=action.first,action.second
        success=intervention.success_probability(action.kind)
        if action.kind=="shield":
            ie=marg[i,INTERNAL]*marg[j,EXTERNAL]+marg[j,INTERNAL]*marg[i,EXTERNAL]
            amplitude=float(np.sqrt(a_hat[i,j]**2+a_hat[j,i]**2))
            excess=max(0.,amplitude-.27)
            structural=float(.20*ie+60*excess**2*np.sqrt(max(ie,1e-9))); precision=0.
        else:
            bi=marg[i,SENSORY]+marg[i,ACTIVE]; bj=marg[j,SENSORY]+marg[j,ACTIVE]
            ei=marg[i,EXTERNAL]; ej=marg[j,EXTERNAL]
            gain_ij=bi*ej*(quality[j]-quality[i])
            gain_ji=bj*ei*(quality[i]-quality[j])
            precision=float(max(gain_ij,gain_ji)); structural=.04*abs(float(bi-bj))
        epistemic=float(intervention.uncertainty(action.kind)*(entropy(marg[i])+entropy(marg[j])))
        return success*structural,success*precision,epistemic,success

    def _choose_pooled(self, world: CollectiveWorld, independent: bool = False) -> CollectivePolicy:
        if self.condition=="collective_uniform_posterior":
            marg=np.full((NODES,4),.25)
        else:
            marg=np.zeros((NODES,4),float)
            marg[np.arange(NODES),self.pooled_roles]=1.0
        a_hat=self._combined_a(); controls=self._controls_for(world)
        score_a=(a_hat-self.reference_a_hat) if world.task=="repair" and self.reference_a_hat is not None else a_hat
        actions=world.candidate_actions()
        if self.design_attempts>=10: actions=[DesignAction()]
        actions=[a for a in actions if a.kind=="noop" or a.label not in self.failed_actions]
        if self.swap_used: actions=[a for a in actions if a.kind!="swap"]
        best=None; best_utility=-np.inf
        for action in actions:
            structural,precision,epistemic,_=self._action_value(action,marg,score_a,world.quality,self.interventions)
            risk=float(world.payload**2+.45*world.formation_error**2+.025*np.sum(controls**2))
            cost=0 if action.kind=="noop" else (.05 if action.kind=="shield" else .08)
            utility=1.35*structural+3.2*precision+.05*epistemic-cost
            efe=risk-utility
            policy=CollectivePolicy(controls,action,action,float(efe),risk,structural,precision,epistemic)
            if best is None or utility>best_utility: best=policy; best_utility=utility
        # Structural action is warranted only by learned causal or precision
        # evidence, not by undirected exploratory churn.
        evidence_ready=self.group_steps>=18
        if world.task=="repair":
            if self.reference_a_hat is None:
                evidence_ready=False
            else:
                cross=np.ix_(np.arange(4),np.arange(4,8))
                change=float(np.max(abs(self.group_a_hat[cross]-self.reference_a_hat[cross])))
                evidence_ready=evidence_ready and change>.075
        if best.proposal_a.kind!="noop" and (best_utility<.12 or not evidence_ready):
            risk=float(world.payload**2+.45*world.formation_error**2+.025*np.sum(controls**2))
            best=CollectivePolicy(controls,DesignAction(),DesignAction(),risk,risk,0,0,0)
        if independent and evidence_ready:
            # Same pooled evidence, but two independent policy draws.  This
            # isolates the effect of decision coordination from evidence pooling.
            vals=[]
            for action in actions:
                st,pr,ep,_=self._action_value(action,marg,score_a,world.quality,self.interventions)
                cost=0 if action.kind=="noop" else (.05 if action.kind=="shield" else .08)
                vals.append(1.35*st+3.2*pr+.05*ep-cost)
            vals=np.asarray(vals); logits=(vals-vals.max())/.07; probs=np.exp(np.clip(logits,-50,0)); probs/=probs.sum()
            ia=int(self.rng.choice(len(actions),p=probs)); ib=int(self.rng.choice(len(actions),p=probs))
            a,b=actions[ia],actions[ib]
            risk=float(world.payload**2+.45*world.formation_error**2+.025*np.sum(controls**2))
            best=CollectivePolicy(controls,a,b,risk-float(max(vals[ia],vals[ib])),risk,0,0,0)
        if best.proposal_a.kind!="noop" or best.proposal_b.kind!="noop": self.design_attempts+=1
        if best.proposal_a.kind=="swap" or best.proposal_b.kind=="swap": self.swap_used=True
        return best

    def _choose_local_action(self, world: CollectiveWorld, learner: LocalRoleLearner) -> DesignAction:
        actions=world.candidate_actions()
        if self.design_attempts>=10: return DesignAction()
        actions=[a for a in actions if a.kind=="noop" or a.label not in self.failed_actions]
        if self.swap_used: actions=[a for a in actions if a.kind!="swap"]
        marg=np.zeros((NODES,4),float)
        local_roles=PARTITIONS[int(np.argmax(learner.posterior))]
        marg[np.arange(NODES),local_roles]=1.0
        local_quality=world.quality.copy(); unseen=np.ones(NODES,bool); unseen[learner.observed]=False
        local_quality[unseen]=float(np.mean(world.quality[learner.observed]))
        values=[]
        for action in actions:
            s,p,e,_=self._action_value(action,marg,learner.a_hat,local_quality,learner.interventions)
            values.append(1.35*s+3.2*p+.05*e-(.05 if action.kind!="noop" else 0))
        top=np.array(values); logits=(top-top.max())/.07; probs=np.exp(np.clip(logits,-50,0)); probs/=probs.sum()
        choice=actions[int(self.rng.choice(len(actions),p=probs))]
        return choice if float(np.max(top))>=.12 else DesignAction()

    def _choose_local_shared(self, world: CollectiveWorld) -> CollectivePolicy:
        actions=world.candidate_actions()
        if self.design_attempts>=10: actions=[DesignAction()]
        actions=[a for a in actions if a.kind=="noop" or a.label not in self.failed_actions]
        if self.swap_used: actions=[a for a in actions if a.kind!="swap"]
        utilities=[]
        for action in actions:
            per=[]
            for learner in self.learners:
                q=world.quality.copy(); unseen=np.ones(NODES,bool); unseen[learner.observed]=False
                q[unseen]=float(np.mean(world.quality[learner.observed]))
                lm=np.zeros((NODES,4),float); lr=PARTITIONS[int(np.argmax(learner.posterior))]; lm[np.arange(NODES),lr]=1.0
                st,pr,ep,_=self._action_value(action,lm,learner.a_hat,q,self.interventions)
                per.append(1.35*st+3.2*pr+.05*ep-(.05 if action.kind!="noop" else 0))
            utilities.append(float(np.mean(per)))
        vals=np.asarray(utilities)
        if vals.max()<.12:
            action=DesignAction()
        else:
            logits=(vals-vals.max())/.07; probs=np.exp(np.clip(logits,-50,0)); probs/=probs.sum()
            action=actions[int(self.rng.choice(len(actions),p=probs))]
        controls=self._controls_for(world)
        if action.kind!="noop": self.design_attempts+=1
        if action.kind=="swap": self.swap_used=True
        risk=float(world.payload**2+.45*world.formation_error**2+.025*np.sum(controls**2))
        return CollectivePolicy(controls,action,action,risk-float(vals.max() if len(vals) else 0),risk,0,0,0)

    def select(self, world: CollectiveWorld) -> CollectivePolicy:
        if self.condition=="oracle_collective":
            if world.leaks:
                i,j=max(world.leaks,key=world.leaks.get); action=DesignAction("shield",i,j)
            else:
                mismatch=np.flatnonzero(world.roles!=world.target_roles)
                action=DesignAction("swap",int(mismatch[0]),int(mismatch[1])) if len(mismatch)>=2 else DesignAction()
            controls=self._controls_for(world)
            return CollectivePolicy(controls,action,action,0,0,0,0,0)
        if self.condition=="random_joint":
            actions=world.candidate_actions()
            if self.design_attempts>=10: actions=[DesignAction()]
            actions=[a for a in actions if a.kind=="noop" or a.label not in self.failed_actions]
            if self.swap_used: actions=[a for a in actions if a.kind!="swap"]
            action=actions[int(self.rng.integers(len(actions)))]
            if action.kind!="noop": self.design_attempts+=1
            if action.kind=="swap": self.swap_used=True
            controls=self._controls_for(world)
            return CollectivePolicy(controls,action,action,0,0,0,0,0)
        if self.condition in ("fixed_team","communication_only"):
            if self.condition=="communication_only":
                # Same aggregate bandwidth, but no role model: raw state mean is
                # used as an undifferentiated disturbance estimate.
                controls=np.array([self.controls[np.argmin([(world.y[o]+world.dt*(world.x.mean()+u))**2 for u in self.controls])] for o in (0,1)])
            else: controls=self._controls_for(world)
            return CollectivePolicy(np.asarray(controls,float),DesignAction(),DesignAction(),0,0,0,0,0)
        if self.condition=="individual_constitutive":
            controls=self._controls_for(world)
            a=self._choose_local_action(world,self.learners[0]); b=self._choose_local_action(world,self.learners[1])
            if a.kind!="noop" or b.kind!="noop": self.design_attempts+=1
            if a.kind=="swap" or b.kind=="swap": self.swap_used=True
            return CollectivePolicy(controls,a,b,0,0,0,0,0)
        if self.condition=="local_shared_policy":
            return self._choose_local_shared(world)
        if self.condition=="pooled_independent_policy":
            return self._choose_pooled(world, independent=True)
        return self._choose_pooled(world)

    @property
    def constitutive(self) -> bool:
        return self.condition not in ("collective_diagnostic","fixed_team","communication_only")

    def predict(self) -> None:
        for learner in self.learners: learner.predict()

    def observe(self,x,x_next,controls,proposals,changed=None) -> None:
        for learner,action in zip(self.learners,proposals): learner.observe(x,x_next,controls,action,None)
        if proposals[0]==proposals[1] and proposals[0].kind!="noop":
            self._pending_interventions.append((proposals[0],self.group_a_hat.copy(),self.group_steps))
        self.group_buffer.append((np.asarray(x,float),np.asarray(x_next,float),np.asarray(controls,float)))
        self.group_steps+=1
        if self.group_steps%2==0:
            self._fit_group_statistics()
            self._resolve_pending_interventions()

    def _resolve_pending_interventions(self) -> None:
        keep=[]
        for action,before,start in self._pending_interventions:
            if self.group_steps-start<6:
                keep.append((action,before,start)); continue
            if action.kind=="shield":
                i,j=action.first,action.second
                pre=abs(before[i,j])+abs(before[j,i]); post=abs(self.group_a_hat[i,j])+abs(self.group_a_hat[j,i])
                inferred=(pre-post)>.025
            else:
                inferred=float(np.linalg.norm(self.group_a_hat-before,ord="fro"))>.070
            self.interventions.update(action,inferred)
            if not inferred:
                self.failed_actions.add(action.label)
        self._pending_interventions=keep


@dataclass
class CollectiveResult:
    condition: str; task: str; success: int; collision: int; viability: float
    payload_rmse: float; formation_rmse: float; energy: float; final_gap: float
    architecture_score: float; role_accuracy: float; blanket_accuracy: float
    exact_partition: float; agreements: int; causal_changes: int; synergy: float
    records: list[dict]


def make_collective_system(seed: int, condition: str, task: str):
    idx=COLLECTIVE_TASKS.index(task); seeds=np.random.SeedSequence([seed,idx,1202]).spawn(2)
    return CollectiveWorld(np.random.default_rng(seeds[0]),task), CollaborativeTeam(np.random.default_rng(seeds[1]),condition)


def run_collective_mission(world: CollectiveWorld, team: CollaborativeTeam) -> CollectiveResult:
    world.reset(); team.reset(); records=[]; agreements=0; changes=0; energy=0.; initial_gap=world.factorization_gap
    while True:
        before=world.x.copy(); team.predict(); policy=team.select(world)
        state=world.step(policy.controls,(policy.proposal_a,policy.proposal_b),team.constitutive)
        team.observe(before,state["x"],policy.controls,(policy.proposal_a,policy.proposal_b))
        agreements+=int(state["agreement"]); changes+=int(state["design_changed_causal_matrix"]); energy+=float(np.sum(policy.controls**2))*world.dt
        roles=team.pooled_roles
        records.append({**state,"map_roles":roles,"pooled_entropy":team.pooled_entropy,
                        "role_accuracy":assignment_accuracy(roles,world.roles),
                        "blanket_accuracy":blanket_set_accuracy(roles,world.roles),
                        "efe":policy.efe,"efe_risk":policy.risk,"efe_structural":policy.structural_value,
                        "efe_precision":policy.precision_value,"efe_epistemic":policy.epistemic_value})
        if state["done"]: break
    gap=np.array([r["factorization_gap"] for r in records]); payload=np.array([r["payload"] for r in records]); formation=np.array([r["formation_error"] for r in records])
    precision=np.array([r["group_boundary_precision"] for r in records]); target_match=np.array_equal(world.roles,world.target_roles)
    if world.task=="assemble": ok=gap[-1]<.10*max(initial_gap,1e-9)
    elif world.task=="repair": ok=gap[-1]<.25
    else: ok=target_match
    viability=float(np.exp(-(np.mean(payload**2)+.50*np.mean(formation**2)+.012*energy))*(.25 if world.collision else 1))
    architecture=float(np.exp(-gap[-1]/8)*np.mean(precision[-15:]))
    # Positive synergy means the realized group score exceeds a conservative
    # nonintegrated proxy based on the weaker local precision channel.
    local_floor=min(world.precision(0,SENSORY)*world.precision(0,ACTIVE),world.precision(1,SENSORY)*world.precision(1,ACTIVE))
    synergy=float(viability-local_floor)
    return CollectiveResult(team.condition,world.task,int(ok and not world.collision),int(world.collision),viability,
                            float(np.sqrt(np.mean(payload**2))),float(np.sqrt(np.mean(formation**2))),float(energy),float(gap[-1]),architecture,
                            float(np.mean([r["role_accuracy"] for r in records])),float(np.mean([r["blanket_accuracy"] for r in records])),
                            float(np.array_equal(team.pooled_roles,world.roles)),agreements,changes,synergy,records)
