

**Dynamic Markov-Blanket Discovery and Active Inference in Individual and Collective Drones**  
Luca M. Possati — University of Twente

This repository is the computational companion to the paper on **inversion-guided boundary design**.

The central question is whether an artificial agent can do more than *infer* a Markov blanket: can it use that inferred organization to choose interventions that **construct, stabilize, or transform the causal boundary itself**?

The model distinguishes:

- `q_t(M)`: the agent's posterior over all **2,520** complete assignments of eight modules to internal, sensory, active, and external roles;
- `C_t`: the physical causal organization that actually generates trajectories.

A change in `q_t(M)` is epistemic. A constitutive intervention changes `C_t`.

The core loop tested here is:

```text
paths → q_t(M) → d_t → C_{t+1} → new paths
```
---

## Central result

The decisive test is a **uniform-posterior ablation**. The ablated agents retain the same learned dynamics, intervention model, motor controller, and structural action repertoire, but structural design no longer receives the inferred Markov-blanket posterior.

### Part I — individual boundary design

| Task | Inferred posterior | Uniform posterior | Random design | Oracle |
|---|---:|---:|---:|---:|
| Construct | **60%** | 25% | 0% | 100% |
| Stabilize | **90%** | 50% | 45% | 100% |
| Transform | **75%** | 5% | 5% | 100% |

The strongest contrast is **Transform**: removing structural information reduces success from **75% to 5%**.

### Part II — collective boundary design

| Task | Pooled posterior + shared policy | Uniform posterior | Random joint | Oracle |
|---|---:|---:|---:|---:|
| Assemble | **65%** | 35% | **65%** | 100% |
| Repair | **85%** | 50% | 45% | 100% |
| Reconfigure | **90%** | 0% | 0% | 100% |

The strongest contrast is **Reconfigure**: the full collective model succeeds in **90%** of missions, while the uniform-posterior control succeeds in **0%**.

`Assemble` is an important limiting case: random joint action reaches the same 65% success rate as the full controller. The paper therefore does **not** claim that structural inference is necessary for every form of boundary modification. The evidence is strongest when successful intervention depends on inferred role organization rather than on generic structural change.

![Uniform-posterior ablation](publication_v6/v6_uniform_posterior_ablation.png)

---

## What the model establishes

### 1. Complete role inference

The structural hypothesis space contains all complete assignments of eight modules to four roles, with two modules per role:

```text
|M| = 8! / (2!)^4 = 2520
```

The learner maintains a posterior over complete partitions rather than four independent role classifiers.

### 2. Constitutive rather than merely diagnostic intervention

Structural actions can physically modify the process that generates future trajectories:

- **shield** an unscreened internal–external coupling;
- **swap** the physical modules occupying causal ports.

Diagnostic controls may update beliefs and evaluate nominal interventions, but they cannot alter the physical causal organization.

### 3. No ground-truth shortcut

The V6 learner does **not** receive:

- the true role map;
- the true breach location;
- shield candidates generated from the true leak set;
- a ground-truth intervention-success bit.

Candidate structural actions are generated independently of the true breach, and intervention efficacy is estimated from learned dynamics.

### 4. Harder discriminating tasks

V6 includes:

- **lawful high-magnitude distractor couplings**, which make “shield the largest edge” unreliable;
- **precision decoys**, which make a naïve min–max precision swap unreliable.

The lower absolute success rates relative to earlier development versions are intentional: the tasks are harder because successful design must depend on inferred functional organization rather than privileged simulator cues.

### 5. Behavioral viability is separated from architecture

Behavioral viability is computed from task/flight performance rather than from factorization gap or routed precision. Architectural quality is reported separately.

---

## Collective experiment: evidence pooling vs policy coordination

The two-drone experiment separates **evidence pooling** from **policy coordination** with a 2×2 design.

| Evidence | Policy | Assemble | Repair | Reconfigure |
|---|---|---:|---:|---:|
| Local | Independent | 0% | 0% | 15% |
| Local | Shared | 50% | 40% | 35% |
| Pooled | Independent | 20% | 30% | 80% |
| Pooled | Shared | **65%** | **85%** | **90%** |

The result is task-dependent. In **Reconfigure**, evidence pooling carries most of the gain; in **Assemble** and **Repair**, shared policy coordination also contributes substantially.

![Collective 2x2](publication_v6/v6_collective_2x2.png)

---

## Conceptual contribution

Most active-inference models optimize behavior **through an already specified interface**. This project treats the interface itself as a design variable.

> **Interactive inference:** How should information and action be organized within an interface?  
> **Inversion-guided boundary design:** Which causal organization should constitute the interface?

The computational model operationalizes three related design ideas:

- **Precision crafting** — selecting, routing, gating, and calibrating sensory and active channels;
- **Curiosity sculpting** — using uncertainty-sensitive structural interventions;
- **Prediction embedding** — allowing inferred structural expectations to guide persistent changes in the physical organization that generates future evidence.

The strongest claim is not that one controller wins every benchmark. It is that a Markov blanket can function as an **inferred, actionable, and revisable causal organization**.

---

## Quick start

Requirements:

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Fast smoke runs:

```bash
python run_experiment.py --quick --workers 4 --output quick_results
python run_collective_experiment.py --quick --workers 4 --output collective_quick_results
```

---

## Reproduce the V6 experiments

### Individual experiment

```bash
python run_experiment.py \
  --seeds 20 \
  --seed-offset 1000 \
  --workers 6 \
  --output results_v6
```

Conditions:

```text
constitutive
uniform_posterior
diagnostic_only
factorized
random_design
passive
oracle
```

Tasks:

```text
construct
stabilize
transform
```

### Collective experiment

```bash
python run_collective_experiment.py \
  --seeds 20 \
  --seed-offset 2000 \
  --workers 6 \
  --output collective_results_v6
```

Conditions:

```text
collective_constitutive
collective_uniform_posterior
collective_diagnostic
individual_constitutive
local_shared_policy
pooled_independent_policy
fixed_team
communication_only
random_joint
oracle_collective
```

Tasks:

```text
assemble
repair
reconfigure
```

Runs are deterministic conditional on seed. The reported 20 seeds are computational replications under the fixed simulator distribution, not samples from a population of real drones.

---

## Repository structure

```text
active_drone/
  agent.py                    Individual policy and structural design
  collective.py               Two-drone collective model and 2x2 controls
  inference.py                Structural and intervention inference
  structure.py                Role partitions and architecture measures
  world.py                    Individual simulator and task manipulations
  simulation.py               Mission execution
  graphics.py                 Plotting utilities

run_experiment.py             Part I benchmark
run_collective_experiment.py  Part II benchmark
analyze_results.py            Analysis and figure generation
make_experiment_animation.py  Experiment animation

tests/                        Tests and invariants
publication_v6/               Main V6 figures
paper/v6_release/             V6 manuscript, PDF, figures, and released CSVs

v6_primary_individual.csv
v6_primary_collective.csv
v6_uniform_individual.csv
v6_uniform_collective.csv
v6_2x2_collective.csv
v6_primary_summary.json
```

### Authoritative V6 artifacts

Use:

```text
paper/v6_release/
publication_v6/
v6_primary_individual.csv
v6_primary_collective.csv
v6_uniform_individual.csv
v6_uniform_collective.csv
v6_2x2_collective.csv
```

Some files elsewhere in the repository retain legacy `v5` names because they are part of the development history. They should not be used as the evidential basis for the V6 claims.

---

## Compile the V6 paper

From the repository root:

```bash
pdflatex -interaction=nonstopmode \
  -output-directory=paper/v6_release \
  paper/v6_release/The_Boundary_That_Learns_Itself_V6.tex
```

Run `pdflatex` again if cross-references need another pass.

---

## Scope of the claim

The experiments support the following bounded claim:

> **An inferred Markov-blanket organization can guide structural interventions that change the causal boundary generating future trajectories.**

The repository does **not** establish:

- open-ended self-individuation;
- autonomous discovery of an unrestricted ontology of components;
- hardware-level robustness;
- a general equivalence between sparse coupling and Markov-blanket structure;
- that structural inference is necessary for every boundary-modification task;
- that two drones literally become a single organism or person.

The collective claim is operational and task-relative: under the modeled conditions, a causal boundary can cut across two physical chassis and can require pooled evidence and bilateral structural action to maintain or transform.

---

## Paper

The current manuscript is:

**Luca M. Possati (2026). _The Boundary That Learns Itself: Dynamic Markov-Blanket Discovery and Active Inference in an Autonomous Drone_.**

The manuscript and compiled PDF are in:

```text
paper/v6_release/
```

---

## Citation

See [`CITATION.cff`](CITATION.cff).

For the current pre-publication version:

> Possati, Luca M. (2026). *The Boundary That Learns Itself: Dynamic Markov-Blanket Discovery and Active Inference in an Autonomous Drone*. University of Twente. Computational repository, V6.

A final archival citation can replace this entry when the paper receives a DOI.

---

## License

MIT License. See [`LICENSE`](LICENSE).

Copyright © 2026 Luca M. Possati.
DME (1).md…]()

