# System Identification on POD-DEIM ROMs

Reduced-order and full-order system identification for nonlinear reaction-diffusion-like PDEs.

This repository is released under the [MIT License](LICENSE). See [CITATION.cff](CITATION.cff) for software citation metadata.

## Abstract

This project learns unknown local nonlinearities in time-dependent PDEs from trajectory data. It trains on a full-order model (FOM), or builds a reduced-order model (ROM) from trajectory data, then trains on the ROM. 

## Background

### System identification

System identification (SysID) infers a model from observed system behavior. Here, the known linear/spatial part of a PDE is retained and a neural network or polynomial learns the unknown local nonlinear term. The optimization target is the trajectory mismatch between the learned model and a reference solve.

### POD ROMs and DEIM

We would like to model the trajectory of a time-dependent PDE with some initial condition: 
$$\frac{d}{dt}u(t)= F(u(t))$$ 
given $u(0).$ We discretize in time, say, via 
$$u(t+1)= u(t) + \Delta t \frac{d}{dt}u(t) = u(t) + \Delta t F(u(t)) .$$ 

To accelerate this algorithm, we attempt to exploit the symmetries of a PDE solution trajectory: we build some reduced version of the state variable, $\tilde{u}$, which encodes the most important properties of $u$. Proper orthogonal decomposition (POD) reduces a state trajectory to a low-dimensional basis. Given a snapshot matrix $S$ of trajectory states, its singular value decomposition supplies spatial POD modes $\phi_1,...,\phi_r.$ A rank-`r` state basis, concatenated into a matrix as $\Phi$, represents a full state approximately as $u \approx \Phi \tilde{u}.$ If $F$ is linear, we can precompute $\tilde{F} = \Phi^T F \Phi$, arriving at the fast approximation $\frac{d}{dt}\tilde{u}(t) = \tilde{F}(\tilde{u}(t)).$

In this project, we deal with nonlinear PDEs; specifically, PDEs of the form $F(u) = L(u) + N(u)$, a sum of linear and nonlinear parts. We can apply the nice reduction technique above to the linear part, but to reduce the nonlinear part we need *hyperreduction*; in this case, we employ the discrete empirical interpolation method (DEIM). 

DEIM builds a separate rank-$m$ nonlinear basis as the singular vectors of a snapshot matrix of nonlinear function evaluations, $S_N \mapsto \psi_1,...,\psi_m.$ We then choose $m$ spatial points at which to evaluate our nonlinearity at every timestep (these points are chosen via a greedy algorithm), and at each time we do 
$$\frac{d}{dt}\tilde{u}(t) = \tilde{L}(\tilde{u}(t)) + BN(u_p(t)),$$ 
where $\tilde{L} = \Phi^T L \Phi$, $u_p$ is our reduced $\tilde{u}$ projected up into the full state but only at $m$ interpolation points, and $B$ is the matrix that interpolates those $m$ evaluations of $N$ based on our snapshot-produced basis $\psi_1,...,\psi_m \equiv \Psi$ and our projector $\Phi.$

We employ a Petrov-Galerkin ROM for the Cahn-Hilliard equation, as well as a gradient flow-preserving version of DEIM. 

### Equations

The reference models use the following nonlinear terms:

- **Allen-Cahn (AC):** $u_t = ε² Δu - k(u³ - u).$ The learner approximates the scalar reaction $-k(u³-u).$
- **Cahn-Hilliard (CH):** $c_t = Δ(-ε² Δc + c³-c) - σ(c-mean(c)).$ The learner approximates $c³-c.$
- **Reaction-diffusion (RD):**
  $$
  \begin{aligned}
  \frac{\partial v_1}{\partial t} &= D_1 \Delta v_1 + v_1 - v_1^3 - v_2 - 0.005, \\
  \frac{\partial v_2}{\partial t} &= D_2 \Delta v_2 + 10(v_1-v_2).
  \end{aligned}
  $$
  The first reaction is fixed; the learner approximates the two-input second reaction.

For each run, the workflow is: generate a reference trajectory, construct a FOM or ROM with the chosen learner, train it over scheduled windows, validate on the full trajectory after each stage, and save the result under `Data/`.

## Requirements and First Setup

The pinned Julia environment is [Julia/Project.toml](Julia/Project.toml) and [Julia/Manifest.toml](Julia/Manifest.toml). The cluster launcher expects Julia 1.12.6 under `Julia/julia-1.12.6/` and uses the repository-local depot `Julia/depot/`.

On an x86_64 Linux cluster, run once from the repository root:

```bash
bash src/Tools/Julia/setup_julia.sh
```

This downloads Julia if necessary, instantiates `Julia/`, and precompiles dependencies into the repository-local depot. The depot and downloaded runtime are intentionally ignored by Git.

For local development, use an installed Julia compatible with the project:

```bash
julia --project=Julia --startup-file=no src/Tools/Tests/runtests.jl
```

## How To Run

All cluster commands below are intended to run from the repository root on the Slurm login node.

### Direct FOM or ROM Submission

Direct submission is useful for a single diagnostic run, but sweeps are the preferred interface because they retain an explicit parameter file and integrate with the virtual queue.

`src/Tools/Slurm/run.slurm` is the direct cluster launcher. Set `MODE` to `fom` or `rom`, set `EQUATION` to `ac`, `ch`, or `rd`, and provide the required `ETAS` and `ITERS` schedules. For example:

```bash
INSTANTIATE=false \
MODE=fom EQUATION=ac \
N=64 DIMENSION=1 BOUNDARY_CONDITION=dirichlet \
TFINAL=0.2 N_OBS=10 H=8 \
ETAS=1e-3,3e-4 ITERS=100,200 \
WINDOW_T=0.05,0.2 WINDOW_N_OBS=2,10 \
WINDOW_START_POLICY=random,beginning \
RUN_NAME=ac_direct_example \
sbatch src/Tools/Slurm/run.slurm
```

Set `INSTANTIATE=true` or omit it for the first run after changing `Julia/Project.toml`; subsequent cluster runs should normally use `INSTANTIATE=false`.

### Sweep Submission

A sweep file is plain text with `KEY = VALUE` assignments. Store project sweep files under `Data/Sweeps/`. The supplied examples are [fom.txt](Data/Sweeps/examples/fom.txt) and [rom.txt](Data/Sweeps/examples/rom.txt).

Inspect a sweep before submission:

```bash
python3 src/Tools/Slurm/Sweeps/sweep_params.py count Data/Sweeps/my_sweep.txt
python3 src/Tools/Slurm/Sweeps/sweep_params.py list Data/Sweeps/my_sweep.txt
```

Submit directly to Slurm when enough submission slots are available:

```bash
MAX_SUBMITTED=20 MAX_CONCURRENT=5 \
bash src/Tools/Slurm/Sweeps/submit_sweep.sh Data/Sweeps/my_sweep.txt
```

For a no-submit check of the generated array command:

```bash
DRY_RUN=true CURRENT_SUBMITTED=0 MAX_SUBMITTED=20 MAX_CONCURRENT=5 \
bash src/Tools/Slurm/Sweeps/submit_sweep.sh Data/Sweeps/my_sweep.txt
```

`MAX_SUBMITTED` is the maximum number of the user's Slurm array tasks that the submit helper may occupy. `MAX_CONCURRENT` is the `%` limit in the submitted array. If the sweep has more combinations than available submission slots, each array worker processes multiple combinations sequentially.

#### Sweep Parameters

Use the uppercase underscore spelling shown below. The parser emits each selected value as an environment variable, and [run.slurm](src/Tools/Slurm/run.slurm) forwards the supported parameters to `src/run.jl`.

| Key | Meaning |
| --- | --- |
| `TARGET` | Required: `fom` or `rom`. |
| `RUN_NAME` | Base output name. Each sweep combination receives an index and parameter label. |
| `SWEEP_NAME` | Optional fallback base name when `RUN_NAME` is absent. |
| `CASE_NAME` | Row-wise-case label used in the output directory name. |
| `EQUATION` | `ac`, `ch`, or `rd`; defaults to `ac` when omitted. |
| `N` | Grid points per spatial dimension. A 2D state has `N × N` spatial unknowns per field. |
| `L` | Physical domain length; default `1.0`. |
| `DIMENSION` | `1` or `2`; each equation supplies a default. |
| `BOUNDARY_CONDITION` | Exactly `periodic`, `dirichlet`, or `neumann`. |
| `INITIAL_CONDITION` | Equation-specific named initial condition; `default` selects the standard state. |
| `TFINAL` | Full reference-trajectory end time. |
| `N_OBS` | Number of full-trajectory validation observations. This is distinct from per-window observations. |
| `EPS2`, `K` | Allen-Cahn interface parameter `ε²` and cubic-reaction coefficient `k`. |
| `SIGMA`, `MEAN_C` | Cahn-Hilliard relaxation coefficient and requested mean initial concentration. |
| `D1`, `D2` | Reaction-diffusion diffusion coefficients. |
| `R`, `M` | ROM state POD rank and nonlinear DEIM rank. They matter only for `TARGET = rom`. |
| `FORCED_DEIM_SPLIT` | Reaction-diffusion ROM only. `true` assigns `M ÷ 2` DEIM points to each reaction function (so an odd `M` uses `2(M ÷ 2)` points); default `false` retains combined DEIM. |
| `LEARNER` | `nn` for the Lux tanh network or `polynomial` for a polynomial nonlinearity. Default: `nn`. |
| `H` | Hidden width of the neural network. The default architecture is `(input, H, H, 1)`. |
| `POLYNOMIAL_DEGREE` | Degree used when `LEARNER = polynomial`. |
| `SEED` | Neural learner initialization and seeded equation initial conditions. |
| `REFERENCE_DT_FACTOR` | Equation-specific reference save-grid scale factor. |
| `ETAS` | Required comma-separated Adam learning-rate schedule, one value per stage. |
| `ITERS` | Required comma-separated optimizer-iteration schedule, one value per stage. |
| `WINDOW_T` | Physical trajectory-window length per stage. Default: the full `TFINAL` for every stage. |
| `WINDOW_N_OBS` | Number of loss observations inside each training window per stage. Default: `N_OBS`. The window initial state is not counted because it is prescribed. |
| `WINDOW_START_POLICY` | `beginning` or `random` per stage. `random` precomputes deterministic starts from `WINDOW_SEED`. Default: `beginning`. |
| `WINDOW_SEED` | Seed used to precompute random training-window starts. Default: `SEED`. |
| `LOSS_NORMALIZATION` | `mean` or `sum`; default `mean`. `mean` keeps the scale less sensitive to the number of observations. |
| `LOSS_SPACE` | ROM only: `FULL` (default) compares reconstructed fields; `REDUCED` compares state POD coordinates before reconstruction. |
| `BETA` | Adam coefficients as `β1,β2`; default `0.9,0.99`. |
| `WARMUP` | `true` or `false`; default `true`. A one-update warmup compiles the differentiated path before measured optimization. |
| `SAVE_FREQUENCY` | Parameter-snapshot interval in optimizer iterations. |
| `LEARNED_FUNCTION_ERROR` | `true` saves and logs an L2 error between the learned and reference reaction at each saved parameter snapshot; default `false`. |
| `LEARNED_FUNCTION_ERROR_BOUNDS` | Comma-separated lower and upper integration bounds for `LEARNED_FUNCTION_ERROR`; default `-1.0,1.0`. R-D integrates over the resulting square. |
| `PRINT_FREQUENCY` | Progress-log interval in optimizer iterations. |
| `JULIA_NUM_THREADS` | Julia thread count requested by the Slurm launcher; default `8`. |
| `JULIA_BLAS_THREADS` | BLAS thread count; default `1`. |
| `INSTANTIATE` | `true` runs dependency instantiation/precompilation before the job; default `true`. Use `false` after setup. |

The current training path intentionally has no batch-size argument. One optimizer update uses one precomputed window; use `WINDOW_N_OBS`, `WINDOW_T`, and the staged schedules to control training cost.

All stage schedules must have equal lengths: `ETAS`, `ITERS`, `WINDOW_T`, `WINDOW_N_OBS`, and `WINDOW_START_POLICY`. `ETAS` and `ITERS` are always required.

#### Cartesian-Product Sweeps

Assignments with several values produce a Cartesian product. Nested bracket lists preserve a complete staged schedule as one choice. This example produces four jobs: two grid sizes times two learning-rate schedules.

```text
TARGET = fom
RUN_NAME = ac_cartesian
EQUATION = ac
N = 64,128
DIMENSION = 1
BOUNDARY_CONDITION = dirichlet
TFINAL = 0.2
N_OBS = 10
H = 8
ETAS = [[1e-3,3e-4],[3e-3,1e-3]]
ITERS = [100,200]
WINDOW_T = [0.05,0.2]
WINDOW_N_OBS = [2,10]
WINDOW_START_POLICY = [random,beginning]
LOSS_NORMALIZATION = mean
WARMUP = true
```


#### Explicit `[[case]]` Sweeps

Use `[[case]]` when each candidate is a complete, readable recipe rather than a Cartesian combination. Shared scalar settings must appear before the first case. Each case value must be singular; a staged schedule is written in one bracketed list.

```text
TARGET = rom
RUN_NAME = ch_cases
EQUATION = ch
N = 128
DIMENSION = 2
BOUNDARY_CONDITION = periodic
TFINAL = 2.0
N_OBS = 50
H = 8
LOSS_NORMALIZATION = mean

[[case]]
CASE_NAME = conservative
R = 10
M = 4
ETAS = [1e-3,3e-4,1e-4]
ITERS = [200,300,500]
WINDOW_T = [0.1,0.5,2.0]
WINDOW_N_OBS = [2,5,10]
WINDOW_START_POLICY = [random,random,beginning]

[[case]]
CASE_NAME = longer_final_stage
R = 15
M = 6
ETAS = [1e-3,2e-4,5e-5]
ITERS = [200,400,1000]
WINDOW_T = [0.1,1.0,2.0]
WINDOW_N_OBS = [2,8,20]
WINDOW_START_POLICY = [random,random,beginning]
```

Unknown sweep keys may appear in the worker log, but they do not change a run unless they are listed in the parameter table above.

### Virtual Queue

The virtual queue is the recommended way to submit many independent sweep combinations when Slurm limits the number of jobs a user may have queued. It expands a sweep into one pending entry per combination, starts a detached daemon, and submits combinations only when both the user-wide submission limit and the virtual-queue concurrency limit permit another job.

```bash
VQ_CLUSTER_MAX_SUBMITTED=20 VQ_MAX_CONCURRENT=5 \
bash src/Tools/Slurm/Sweeps/virtual_sweep_queue.sh Data/Sweeps/my_sweep.txt

bash src/Tools/Slurm/Sweeps/virtual_sweep_queue.sh status
bash src/Tools/Slurm/Sweeps/virtual_sweep_queue.sh stop
```

The daemon uses `nohup`, so it continues after the SSH session ends. Its queue table and activity log live under `src/Tools/Misc/VirtualQueue/`; its transient state directory is ignored by Git.

## Training and Reproducibility

The reference trajectory is first solved with the fixed PDE nonlinearity. Training then performs staged Adam optimization of the neural-network or polynomial parameters. Each stage has its own learning rate, iteration count, trajectory duration, observation count, and start policy.

For a training window, the reference state at `t_start` is used as the exact model initial condition. The loss compares only later observation times in `(t_start, t_start + WINDOW_T]`. `random` windows are generated before optimization from `WINDOW_SEED`, so the same configuration and seed produce the same window order. The loss is spatially weighted and either averaged (`mean`) or accumulated (`sum`) over the comparisons. Loss is calculated based on an approximation of the $L^2$ norm over snapshots; the number of snapshots per window is passed in as `WINDOW_N_OBS`.

After initialization and after each stage, the pipeline evaluates a full-trajectory validation loss. The current differentiated forward solve uses `TRBDF2(autodiff=AutoFiniteDiff())`; sensitivities use `GaussAdjoint(MooncakeVJP())`, with `Optimization.AutoZygote()` as the outer optimization differentiation backend.

## Saved Results and Logs

Every successful run creates `Data/<RUN_NAME>/`. A run name collision is an error: the directory must not already exist. Sweeps derive a unique name from the base `RUN_NAME`, combination index, and either `CASE_NAME` or the swept parameter label. Re-running the same sweep therefore requires a different base name or removal of the old output directory.

FOM runs save:

```text
Data/<RUN_NAME>/
  metadata.txt
  run_params.jls
  parameter_history.jls
  window_history.jls
  validation_history.jls
  log.out, log.err                 
```

ROM runs save the same histories and metadata, plus `rom_data.jls`, which contains the saved configuration and ROM artifacts: POD modes, DEIM modes and indices, singular values, reduced operators, and reaction-diffusion component/index metadata when applicable.

`parameter_history.jls` stores saved trainable-parameter snapshots and, when enabled, their learned-function L2 errors. `window_history.jls` stores every precomputed training-window specification. `validation_history.jls` stores initial and per-stage full-trajectory losses. `metadata.txt` is human-readable; the `.jls` files use Julia serialization.

Slurm initially writes direct-job and array-worker output to `Data/_Logs/`. For a successful sweep combination, the per-combination logs are moved into its run directory as `log.out` and `log.err`; failed-combination logs remain in `Data/_Logs/` for diagnosis. The Slurm array worker logs also remain in `Data/_Logs/`.

## Local Visualization Notebooks

`Untracked/Visualize_results/A-C/` and `Untracked/Visualize_results/R-D/` contain local notebooks for replaying saved FOM and ROM runs. Set `input_path` to either one `Data/<RUN_NAME>` directory or, with `dir_of_dirs = true`, a directory containing runs. Each notebook solves its selected trajectories once into `trajectory_data`, so titles, axes, legends, colorbars, and saved figures can be adjusted without re-solving. Rendered Allen--Cahn/Cahn--Hilliard figures go to `Untracked/Visualize_results/A-C/Data/`; reaction--diffusion figures go to `Untracked/Visualize_results/R-D/Data/`.

The stability notebooks in `Untracked/Tests/` construct fresh FOM and ROM trajectories for Allen--Cahn, Cahn--Hilliard, or reaction--diffusion from notebook-global problem settings. Their plotting and replay functions live in `src/Tools/Visualizations/`, which is loaded by including `visualizations.jl` after activating `Julia/`. The visualization dependency is part of the project environment, so instantiate once after pulling this change before running a notebook.

`Untracked/Tests/timing_tests.ipynb` uses `src/Tools/Tests/timing_tests.jl` to benchmark repeated post-warm-up 2D Allen--Cahn forward solves, losses, and complete Adam steps over FOM or ROM parameter schedules. Give each schedule a required test name; its cases and readable/serialized results are saved under `Untracked/Tests/timing_test_results/<test name>/`. The notebook is opt-in: set `RUN_BENCHMARKS = true` only after choosing a schedule sized for the available machine.

## Repository Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for the concise layout summary.

## Julia Environment

`Julia/Project.toml` declares the direct dependencies; `Julia/Manifest.toml` pins their resolved versions. The main numerical dependencies are SciML/OrdinaryDiffEq, SciMLSensitivity with Mooncake, Lux, ComponentArrays, and Optimization.jl.

The Slurm launcher sets `JULIA_DEPOT_PATH` to `Julia/depot:` so cluster jobs use the repository-local compiled environment rather than an accidental default depot. It also exports `JULIA_NUM_THREADS` and sets BLAS/OpenBLAS/MKL threads from `JULIA_BLAS_THREADS`.

Run the test suite locally after code changes:

```bash
julia --project=Julia --startup-file=no src/Tools/Tests/runtests.jl
```

## GitHub Automation

- [`.github/workflows/test.yml`](.github/workflows/test.yml) runs on pushes and pull requests. It installs Julia 1.12, instantiates `Julia/`, and executes the direct Julia test suite.
- [`.github/dependabot.yml`](.github/dependabot.yml) opens monthly update PRs for GitHub Actions dependencies.
- [LICENSE](LICENSE) declares the MIT license, and [CITATION.cff](CITATION.cff) supplies GitHub citation metadata.


## A Note on AI Usage

Most of the code in this repo was generated with AI, primarily GPT 5.5 and 5.6. Research-critical code was checked line-by-line. CLI tooling / scripting was not checked thoroughly. Other files were half-checked.
