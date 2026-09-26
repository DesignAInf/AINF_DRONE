# Frozen V5 Model Specification

## Ontological separation

The physical organization is

\[
C_t=(r_t,W_t,\lambda_t,q_t^{\mathrm{phys}}),
\]

where `r_t` assigns each of eight modules to one of four roles, `W_t` is the directed drift matrix, `lambda_t` is the set of direct interior--exterior leaks, and `q_phys` stores module precision. The agent's structural belief is a different object:

\[
q_t(M),\qquad M\in\mathcal M,qquad |\mathcal M|=\frac{8!}{(2!)^4}=2520.
\]

No result is counted as boundary design unless an action changes `C_t`.

## Constitutive criterion

For a design action `d_t`, V5 requires

\[
C_{t+1}=T(C_t,d_t,\omega_t),\qquad
T(C_t,d_t,\omega_t)\ne T(C_t,d'_t,\omega_t)
\]

for at least one admissible pair `d_t != d'_t` under the same exogenous realization `omega_t`. Unit tests verify this condition for shielding and verify its failure by construction in `diagnostic_only`.

## Path-space factorization gap

The simulator uses a linear Gaussian process. Direct `I <-> E` drift is collected in `W_IE` and `W_EI`. With innovation scale `sigma`, the quadratic Girsanov/Foellmer rate is

\[
\Delta_{\mathrm{fac}}(C_t)=
\frac{1}{2\sigma^2}
\left(\lVert W_{IE}\rVert_F^2+\lVert W_{EI}\rVert_F^2\right).
\]

It is zero for the clean directed cycle `E -> S -> I -> A -> E`, and positive when an unscreened interior--exterior drift channel exists.

## Structured inversion

The agent learns a regularized transition matrix from a rolling window and assigns a probability to every complete four-role partition. The likelihood score combines template fit, the learned response to control, and learned innovation variance. No role is fixed and no table of probe outcomes conditional on roles is provided.

## Learned intervention model

For each intervention family `k` in `{shield, swap}`, the model maintains a Beta posterior on whether the action changed the causal matrix. Counts start at `Beta(1,1)` and are updated only from observed pre/post causal effects. Its posterior mean and variance enter policy evaluation.

## Joint EFE policy

Every candidate policy is a pair `pi=(u,d)` of a control value and a design action. The implementation scores expected pragmatic risk, ambiguity, design cost, structural improvement, precision improvement, and epistemic value in a single objective. `factorized` is the explicit ablation that chooses control first and then structure.

## Frozen tasks

- `construct`: three randomly selected direct `I--E` links are open initially.
- `stabilize`: one link opens at a seed-random time; no role changes exogenously.
- `transform`: a seed-random context time degrades one boundary module and improves one external module. Only a drone-selected port swap changes their roles.

## Primary hypothesis

Across paired seeds and all three tasks, `constitutive` must outperform `diagnostic_only` in task success and final architecture score. Task-specific predictions are:

- construct: lower final factorization gap;
- stabilize: lower final gap and finite recovery;
- transform: higher boundary precision and viability; no predicted gap difference, because both pre- and post-transform architectures can factorize.

## Claim boundary

The benchmark supports constitutive boundary design only inside a declared eight-module state space and a declared action repertoire. Hardware transfer, open-ended variable discovery, thermodynamic calibration, and unconstrained individuation remain outside scope.

## Collective extension

Part II distributes the eight modules across two owners. Each drone observes six modules, with a four-module overlap, and maintains a local posterior. Fixed-size sufficient-statistic messages are pooled into a group posterior over the same 2,520 valid partitions.

A collective design action changes the group organization only if both drones propose the same non-null cross-owner shield or routing swap. Unit tests enforce both sides of this invariant: unilateral proposals are inert and matching bilateral proposals can be constitutive.

The collective tasks are:

- `assemble`: close initially open cross-owner interior--exterior links;
- `repair`: localize and close a randomly timed cross-owner breach;
- `reconfigure`: agree on a cross-owner port swap after a precision change.

The main controls are collective diagnostic action, local individual blankets, a fixed team, equal-bandwidth communication without a role model, random bilateral action, and an oracle. The collective claim is operational and task-relative; it does not establish unrestricted ontological identity.
