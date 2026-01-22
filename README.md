WIP! I need to organize the repo first


# Quantum-assisted-Optimal-Control-examples
This repository contains two benchmark LQR case studies supporting the thesis. The corresponding plots are shown in Section V, while the full description of the examples is provided here.

## Benchmark instances

This repository summarises the two benchmark LQR instances from numerical results reported in Section V. The inverted-pendulum model follows the standard linearisation around the upright equilibrium (see Brunton & Kutz), while the platooning model uses a leader–follower error formulation commonly adopted in the literature (see Zhu et al., 2022).

### Inverted pendulum on a cart (linearised model)

We consider the linear state-space model $\dot{x}=Ax+Bu$ with $x\in\mathbb{R}^4$ and scalar input $u\in\mathbb{R}$. For the experiments, we use the following numerical linearisation:

$$
A=
\begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & -0.2 & -2 & 0 \\
0 & 0 & 0 & 1 \\
0 & -0.1 & 6 & 0
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
0 \\
0.2 \\
0 \\
0.1
\end{bmatrix}.
$$

The open-loop dynamics are unstable (i.e., $A$ has an eigenvalue with positive real part), motivating feedback design. Controllability is assessed via the standard controllability matrix, and the pair $(A,B)$ is controllable (see Åström & Murray, 2008).

We formulate a finite-horizon LQR with weights $Q=I_4$, terminal weight $S=\mathrm{diag}(0,\,10,\,50,\,10)\succeq 0$, and $R=0.01>0$. To track a nonzero reference $x_{\mathrm{ref}}$, we work in error coordinates $e(t)=x(t)-x_{\mathrm{ref}}$, for which the Hamiltonian system reads:

$$
\frac{d}{dt}\begin{bmatrix}e\\ \lambda\end{bmatrix}
=
\underbrace{\begin{bmatrix}
A & -BR^{-1}B^\top\\
-\,Q & -A^\top
\end{bmatrix}}_{C}
\begin{bmatrix}e\\ \lambda\end{bmatrix},
\qquad
\begin{cases}
e(0)=e_0,\\
\lambda(T)=S\,e(T).
\end{cases}
$$

### Vehicle platooning (leader–follower errors)

We consider a longitudinal platoon consisting of one leader (index $0$) and $N$ identical followers ($i=1,\dots,N$). Each vehicle is modelled as a point mass with actuator lag $\tau>0$:

$$
\dot{s}_i=v_i,\qquad
\dot{v}_i=a_i,\qquad
\dot{a}_i=-\frac{1}{\tau}a_i+\frac{1}{\tau}u_i.
$$

In state-space form, $\dot{x}_i=A_0x_i+b_0u_i$ with

$$
A_0=\begin{bmatrix}
0 & 1 & 0\\
0 & 0 & 1\\
0 & 0 & -1/\tau
\end{bmatrix},
\qquad
b_0=\begin{bmatrix}0\\0\\1/\tau\end{bmatrix},
\qquad
x_i=\begin{bmatrix}s_i\\v_i\\a_i\end{bmatrix}\in\mathbb{R}^3.
$$

The leader input $u_0(t)$ is treated as exogenous (not optimised). Under a constant-spacing policy with desired gap $d>0$, we define leader-relative errors:

$$
e_{s,i}=s_i-s_0+i\,d,\quad
e_{v,i}=v_i-v_0,\quad
e_{a,i}=a_i-a_0.
$$

Stacking $E=[e_1^\top\ \cdots\ e_N^\top]^\top\in\mathbb{R}^{3N}$ and defining differential inputs $\tilde{u}_i:=u_i-u_0$ (stacked as $\tilde{U}$), the follower error dynamics become:

$$
\dot{E}=A_fE+B_f\tilde{U},
\qquad
A_f=I_N\otimes A_0,\quad
B_f=I_N\otimes b_0.
$$

We consider a finite-horizon LQR on $[0,T]$ with Kronecker-lifted weights $Q=I_N\otimes Q_0$ and $R=I_N\otimes R_0$, and terminal weight $S$ (in the reported experiments we use $S=0$, hence $\lambda(T)=0$).

#### References 

## References

- Steven L. Brunton and J. Nathan Kutz, *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control*. Cambridge University Press, 2019. Online companion (Databook): https://databookuw.com/databook.pdf

- Karl Johan Åström and Richard M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton University Press, 2008. ISBN: 978-0-691-13576-2. https://fbsbook.org/

- Yongxin Zhu, Yongfu Li, Simon Hu, and Shuyou Yu, “Optimal Control for Vehicle Platoon Considering External Disturbances,” in *25th IEEE International Conference on Intelligent Transportation Systems (ITSC 2022)*, Macau, China, Oct. 8–12, 2022, pp. 453–458. DOI: 10.1109/ITSC55140.2022.9922486 (https://doi.org/10.1109/ITSC55140.2022.9922486)










Since this is an LQR-type problem, the correctness of the implementation can be verified by comparison with the Riccati-based solution."










# Quantum-assisted-Optimal-Control-examples

Example repository for **optimal control** with a focus on LQR (Linear Quadratic Regulator), comparing:

- **Riccati reference solutions** (ARE: infinite-horizon, DRE: finite-horizon).
- Two alternative methods based on **Hamiltonian evolution**:
  - **Method M** (closed form with a dense inversion).
  - **Method V** (a sparse “stacked”/discretized linear system).
- A **QSVT** (Quantum Singular Value Transformation) analysis block: construction of an **optimal (Chebyshev-type) polynomial** to approximate \(1/x\), which is relevant for matrix inversion (conditioning).

This repo is organized to be run mainly from **notebooks**, which generate reproducible figures and results.

---

## Quick install

### Conda (recommended)

File: `tfm_env.yml`

```bash
conda env create -f tfm_env.yml
conda activate tfm_env
pip install -e .
```

### Optional: install quantum extras (Qiskit)

```bash
pip install -e ".[quantum]"
```

> Note: Qiskit is provided as an optional extra `[quantum]`. The current codebase does not import Qiskit directly; it is included to support future extensions/experiments and quantum-related workflows.

> Run these commands from the repository root (the folder containing `pyproject.toml`).

---

## Minimal mathematical summary (to orient the files)

Continuous-time LQR problem:

$$
\dot{x}(t) = A x(t) + B u(t)
$$

Cost:

$$
J = \int_0^T \left( e(t)^T Q e(t) + u(t)^T R u(t)\right),dt + e(T)^T H e(T)
$$

where (e(t)=x(t)-x_{\text{ref}}(t)) (here it is usually a constant reference state).

Associated Hamiltonian matrix (the one used in the code as `C`):

$$
C=\begin{bmatrix}
A & -B R^{-1} B^T \
-Q & -A^T
\end{bmatrix}
$$

and the linear Hamiltonian system in the extended state (y=[e;\lambda]):

$$
\dot{y}(t)= C,y(t)
$$

with typical boundary conditions:

* (e(0)=e_0)
* (\lambda(T)=H,e(T))

---

## Repository tree (high level)

* `README.md` (this document)
* `pyproject.toml` (package + dependencies)
* `tfm_env.yml` (conda environment)
* `src/` (Python code)
* `notebooks/` (reproducible experiments)
* `figures/` (PNGs generated by notebooks)
* `results/` (NPZ/JSON/ZIP outputs generated by notebooks)

---

# Detailed file description

## Repo root

### `README.md`

Main project document.

### `pyproject.toml`

Defines the `quantum-assisted-optimal-control-examples` package and minimal dependencies:

* `numpy`, `scipy`, `matplotlib`, `qiskit`

It also configures `setuptools` to use `src/` as the package root (`package-dir`).

### `tfm_env.yml`

Conda environment named `tfm_env` with:

* Python 3.11
* numpy, scipy, matplotlib
* jupyter + ipykernel
* qiskit>=1.0

---

## `src/` (code)

> Note: this repository uses a “src layout”. For `import models...` etc. to work, the standard approach is `pip install -e .`.

### `src/lqr/`

#### `src/lqr/__init__.py`

Empty. Only marks `lqr` as a subpackage.

#### `src/lqr/riccati.py`

Implements **LQR reference solutions** based on Riccati equations:

* `AREOutput`, `DREOutput` (dataclasses): result containers.
* `lqr_gain_are(A,B,Q,R)`:

  * Solves the **ARE** (Algebraic Riccati Equation) and returns constant (P) and gain (K).
* `lqr_gain_dre(A,B,Q,R,T,H_T,t_eval,method,...)`:

  * Integrates the **DRE** (Differential Riccati Equation) backward from (P(T)=H).
  * Returns sampled (P(t)) on `t_eval` and time-varying gains.
* `simulate_closed_loop_are(A,B,K,t_eval,e0,...)`:

  * Integrates ( \dot{e}=(A-BK)e ) and produces an error trajectory.
* `simulate_closed_loop_dre(A,B,R,t_eval,e0,P_t,...)`:

  * Integrates dynamics using control (u(t)=-R^{-1}B^T P(t),e(t)).

**Typical use**: notebooks `trajectories_riccati.ipynb` and `conditioning_comparison.ipynb` to obtain (\lambda_0 = P(0)e_0) as a reference.

---

### `src/methods/`

#### `src/methods/__init__.py`

Exports the main functions of the subpackage:

* `build_hamiltonian_C`, `four_partition`, `inverse_check_cond`, `estimate_kappa`
* `solve_m_method`, `assemble_stack`, `solve_v_method`

and the associated dataclasses.

#### `src/methods/common.py`

Utilities shared by M and V:

* `build_hamiltonian_C(A,B,Q,R)`:

  * Builds `C` (Hamiltonian matrix) and also returns `Rinv_BT = R^{-1}B^T` using `scipy.linalg.solve` (stable).
* `four_partition(C,T)`:

  * Computes (S=\exp(C T)) and returns blocks `U00,U01,U10,U11` (each (n\times n)).
* `inverse_check_cond(M, warn_thresh=1e16)`:

  * Computes `cond(M)` and warns if it is ill-conditioned.
* `estimate_kappa(A, dense_threshold=2500, tol=1e-6, maxiter=20000)`:

  * Estimates ( \sigma_\min, \sigma_\max, \kappa) for dense or sparse matrices (uses dense SVD if small, and `svds`/alternative strategies if sparse/large).

#### `src/methods/method_m.py`

Implements **Method M** (closed form):

* `MMethodResult`:

  * `lamb0`: (\lambda(0))
  * `M`, `N`: intermediate matrices
  * `condM`: condition number of `M`
  * `S`: full matrix `expm(C*T)`
* `solve_m_method(C,H,e0,T)`:

  * Computes (S=\exp(CT)), partitions it, and applies:

    * (M = U_{11} - H U_{01})
    * (N = H U_{00} - U_{10})
    * (\lambda_0 = M^{-1} N e_0)

Used in `trajectories_m&v.ipynb` and in `sweep_horizon_mv.py`.

#### `src/methods/method_v.py`

Implements **Method V** (discretization + sparse system):

* `VSystem`:

  * `V`: CSR sparse system matrix
  * `rhs`: right-hand-side vector
  * `S_step`: `expm(C*dt)` for one step
  * `dt`: time step
* `VMethodResult`:

  * `z`: stacked solution
  * `Z`: reshape `(m, 2n)`
  * `E`, `LAM`: split extended state components
  * `lamb0`, `lambT`
  * `condV`: estimated condition number of `V`
* `assemble_stack(C,H,b,T,m)`:

  * Builds the equations:

    * initial condition (e_0=b)
    * recurrence (y_k - S y_{k-1}=0)
    * terminal condition ([-H \ I] y_{m-1}=0)
  * Returns `VSystem`.
* `solve_v_method(C,H,e0,T,m)`:

  * Solves `V z = rhs` with `spsolve` and packages results.

#### `src/methods/sweep_horizon_mv.py`

“Horizon sweep” experiment (M vs V vs DRE comparison):

* Helper functions:

  * `rel_norm`, `_safe_norm`, `_safe_maxabs`
* `SweepMVOutput`: container for arrays (condition numbers, lambdas, residuals, etc.)
* `sweep_MV_over_T(make_model_for_T, T_vals, m_grids, dre_method)`:

  * For each (T):

    1. builds a model
    2. computes DRE reference: (\lambda_0=P(0)e_0)
    3. computes M and V (for multiple `m`)
    4. stores conditioning and relative errors
* Plotting:

  * `plot_kappa_vs_T`
  * `plot_rel_lambda0_vs_T_DRE`
  * `plot_residuals2_vs_T`

Used from `notebooks/conditioning_comparison.ipynb`.

---

### `src/models/`

#### `src/models/__init__.py`

Commented out (does not actively export anything). It acts as a “reminder” for possible exports.

#### `src/models/lqr_model.py`

Dataclass `LQRModel`: **standard structure for an LQR problem** in this repo.

Relevant fields:

* `A`, `B` (dynamics)
* `Q`, `R`, `H` (weights)
* `x_ref`, `x0`, `e0` (reference and initial state)
* `T`, `dt`, `t_eval` (time grid)
* `C` (precomputed Hamiltonian), `Rinv_BT` (for control)
* optional extras: `leader_at`, `leader_traj`, `leader_t`, `params`

Properties:

* `N`: number of samples in `t_eval`
* `t_span`: `(0,T)`

#### `src/models/cart_pendulum.py`

Factory `make_cart_pendulum(...) -> LQRModel`:

* Builds the linearized **inverted pendulum on a cart** model:

  * Defines `A` (4×4) and `B` (4×1)
* Allows overrides:

  * direct `Q`, `R`, `H`
  * or `R_scalar`, `H_diag`
* Defines `x_ref`, `x0` and computes `e0=x0-x_ref`
* Builds `t_eval` from `T` and `dt`
* Precomputes `C` and `Rinv_BT` via `build_hamiltonian_C`

Includes usage examples in comments at the end.

#### `src/models/vehicle_platoon.py`

Factory `make_vehicle_platoon(...) -> LQRModel`:

**Platooning** model in leader–follower error coordinates.

* Key parameters:

  * `n`: number of followers (state dimension `3n`)
  * `d`: desired spacing
  * `tau`: actuator time constant
* Builds per-vehicle block `A0` (3×3) and `b0` (3×1), then:

  * `A = kron(I_n, A0)` (3n×3n)
  * `B = kron(I_n, b0)` (3n×n)
* Weights:

  * `Q = kron(I_n, Q0)`
  * `R = kron(I_n, R0)`
  * default `H = I` (3n×3n)
* Initialization:

  * defines `s_abs`, `v_abs`, `a_abs` (leader + n followers)
  * computes `e0` with leader-relative errors (including the distance offset)
* Leader:

  * configurable `u0(t)`
  * optionally integrates leader trajectory (`integrate_leader=True`) with `solve_ivp`
  * builds `leader_at(t)` via interpolation (for absolute-state reconstruction)
* Precomputes `C`, `Rinv_BT` and returns `LQRModel`.

---

### `src/qsvt/`

#### `src/qsvt/__init__.py`

Empty. Marks `qsvt` as a subpackage.

#### `src/qsvt/qsvt_optimal_poly.py`

Large QSVT analysis module: optimal polynomial approximating (1/x) on

$$
S(a)=[-1,-a]\cup[a,1],\quad a=1/\kappa
$$

Main functions:

* `helper_Lfrac`, `helper_P`, `poly`: build the (Chebyshev-type) polynomial used for approximation.
* `error_for_degree(d,kappa)`, `mindegree_for_error(eps,kappa)`:

  * error evaluation and minimum-degree selection for a target.
* `optimal_poly_for_kappa(kappa, target_epsilon)`:

  * returns (degree, polynomial, actual error bound, `a`)
* Plots:

  * `plot_optimal_poly_for_kappa`
  * `plot_poly_degree_comparison`
  * `plot_error_vs_degree`
  * `plot_error_vs_degree_subplots`
  * `plot_error_vs_degree_comparison`
* Export:

  * `save_qsvt_figures_and_results(...)`:

    * computes, plots, and saves `.png`, `.json`, `.npz` to `figures/` and `results/`.

Used from `notebooks/qsvt_analysis.ipynb`.

---

### `src/utils/`

#### `src/utils/__init__.py`

Empty. Marks `utils` as a subpackage.

#### `src/utils/export.py`

Utilities to **save reproducible outputs**:

* `find_repo_root(start=None)`:

  * finds the repository root (detects `src/`).
* `make_run_id(prefix)`:

  * creates IDs like `prefix_YYYYMMDD_HHMMSS_microseconds`.
* `ensure_out_dirs(root)`:

  * creates `figures/` and `results/` if missing.
* `save_json(path, obj)`:

  * JSON serialization with support for numpy scalars/arrays.
* `save_npz(path, **arrays)`:

  * `np.savez_compressed`.

#### `src/utils/integrate.py`

Integrates the Hamiltonian system given (\lambda_0):

* `integrate_with_lamb0_yielded(C,e0,lamb0,T,t_eval,...)`:

  * integrates (\dot{y}=Cy) with `solve_ivp`
  * returns:

    * `sol` (OdeResult)
    * `E(t)` and `LAM(t)`
    * optionally reconstructs `X(t)` if `model_name` is:

      * `"cart_pendulum"` (requires `x_ref`)
      * `"vehicle_platoon"` (requires `leader_traj`, `d`, `n_followers`)
    * optionally computes control `U(t)=-R^{-1}B^T \lambda(t)` if `Rinv_BT` is provided

#### `src/utils/metrics.py`

* `rel_norm(a,b,eps=1e-15)`:

  * L2 relative error: `||a-b|| / max(||a||,||b||,eps)`.

#### `src/utils/plotting.py`

“Nice” plots for `TrajectoryResult` (cart and vehicle):

* `_infer_kind_from_model_name`:

  * decides whether the model is “cart” or “vehicle” based on `model_name`.
* `_parse_cart_plot_only`:

  * filters variables to plot by name/regex.
* `plot_result(tr, *, panels=("X","LAM","E","U"), ...)`:

  * for **cart**: plots variables using `cart_labels`, supports `cart_plot_only`, colormap, etc.
  * for **vehicle**:

    * can plot one component across all followers (`overlay=False`)
    * or plot s/v/a with different styles (`vehicle_overlay_components=True`)
    * option `vehicle_u_mode="tilde"` or `"absolute"` (for inputs)

#### `src/utils/riccati_maps.py`

Post-processing maps from Riccati solutions:

* `lam_of_e_are(E,P)` / `lam_of_e_dre(E,P_t)`:

  * (\lambda(t)=P e(t))
* `u_of_e_are(E,K)` / `u_of_e_dre(E,Rinv_BT,P_t)`:

  * (u(t)=-K e(t)) or (u(t)=-R^{-1}B^T P(t)e(t))

#### `src/utils/trajectory.py`

Reconstructions and standard container:

* `x_of_e_cart(E,x_ref)`:

  * reconstructs `X = E + x_ref`.
* `x_of_e_vehicle(E,X0_traj,d,n_followers)`:

  * reconstructs followers’ absolute states using the leader trajectory.
* `compute_control_from_lam(LAM,Rinv_BT)`:

  * computes `U = -(Rinv_BT @ LAM^T)^T`.
* `TrajectoryResult` (dataclass):

  * `model_name`, `method_name`, `t`, `E`, `LAM`, `X`, `U`, `meta`, etc.
* `make_trajectory_result(...)`:

  * helper to build the dataclass consistently.

---

### `src/**/__pycache__/*.pyc`

Examples included in the zip:

* `src/lqr/__pycache__/...`
* `src/methods/__pycache__/...`
* `src/models/__pycache__/...`
* `src/qsvt/__pycache__/...`
* `src/utils/__pycache__/...`

These are **compiled Python bytecode**, automatically generated when running/importing modules. They are **not source code** and can be safely deleted.

---

## `notebooks/`

### `notebooks/conditioning_comparison.ipynb`

Notebook to compare **conditioning** and accuracy (vs DRE) of M and V as the horizon `T` varies.

* Uses `methods/sweep_horizon_mv.py`.
* Generates:

  * plots of (\kappa(M)) and (\kappa(V)) vs `T`
  * relative error of (\lambda_0) vs DRE
  * residual norms of the linear systems
* Saves to `figures/` and `results/`, and optionally packs a zip into `results/*_outputs.zip`.

### `notebooks/trajectories_m&v.ipynb`

Notebook to generate full **trajectories** using:

* `solve_m_method` and `solve_v_method` (compute (\lambda_0))
* `integrate_with_lamb0_yielded` (integrates ( \dot{y}=Cy ))
* `plot_result` to plot E/LAM/X/U
* Saves trajectories as `.npz` (`*_M_traj.npz`, `*_V_traj.npz`) + figures.

### `notebooks/trajectories_riccati.ipynb`

Notebook to produce reference trajectories with:

* `lqr_gain_are` (ARE)
* `lqr_gain_dre` (DRE)
* mappings in `riccati_maps.py`
* plots via `plot_result`
* exports `.npz` and `.png` + optional zip.

### `notebooks/qsvt_analysis.ipynb`

Notebook for QSVT optimal polynomial analysis:

* optimal polynomial for a given (\kappa) and target (\varepsilon)
* degree comparison
* error vs degree (single (\kappa) and multiple (\kappa))
* saves `.png` + `.npz` + `.json` and optional zip.

---

## `figures/` (already-generated PNGs)

> These figures are **artifacts from previous notebook runs** (included in your zip). They can be deleted and regenerated.

### Conditioning sweeps

* `figures/Conditioning_20260107_015336_160795_kappa_vs_T.png`
  Conditioning plot (\kappa(M)) and (\kappa(V)) vs (T) (cart-pendulum instance).

* `figures/Conditioning_20260107_015336_160795_rel_lambda0_vs_DRE.png`
  Relative error of (\lambda_0) (M and V) against DRE vs (T).

* `figures/Conditioning_20260107_015336_160795_residuals2_vs_T.png`
  2-norm residuals for the linear systems in M and V vs (T).

* `figures/Conditioning_20260107_015530_367649_kappa_vs_T.png`
  Same, but for the vehicle-platoon instance.

* `figures/Conditioning_20260107_015530_367649_rel_lambda0_vs_DRE.png`

* `figures/Conditioning_20260107_015530_367649_residuals2_vs_T.png`

### M/V trajectories

* `figures/methods_20260106_125005_886049_M_cart_pendulum.png`
  Typical panels (E, LAM, X, U) for Method M on cart-pendulum.

* `figures/methods_20260106_125005_886049_V_cart_pendulum.png`
  Same for Method V on cart-pendulum.

* `figures/methods_20260106_125005_886049_M_vehicle_platoon.png`

* `figures/methods_20260106_125005_886049_V_vehicle_platoon.png`

### Riccati (ARE/DRE)

* `figures/riccati_20260106_125017_168377_ARE_cart_pendulum.png`

* `figures/riccati_20260106_125017_168377_DRE_cart_pendulum.png`

* `figures/riccati_20260106_125017_168377_ARE_vehicle_platoon.png`

* `figures/riccati_20260106_125017_168377_DRE_vehicle_platoon.png`

* `figures/riccati_20260107_004812_722738_ARE_cart_pendulum.png`

* `figures/riccati_20260107_004812_722738_DRE_cart_pendulum.png`

* `figures/riccati_20260107_004812_722738_ARE_vehicle_platoon.png`

* `figures/riccati_20260107_004812_722738_DRE_vehicle_platoon.png`

* `figures/riccati_20260107_005013_913624_ARE_cart_pendulum.png`

* `figures/riccati_20260107_005013_913624_DRE_cart_pendulum.png`

* `figures/riccati_20260107_005013_913624_ARE_vehicle_platoon.png`

> Note: you will also see:

* `figures/mmmriccati_20260107_004812_722738_DRE_vehicle_platoon.png`
* `figures/mmmriccati_20260107_005013_913624_DRE_vehicle_platoon.png`
  These are vehicle-platoon DRE figures saved with a different prefix (“mmmriccati”), likely due to an accidental change in the `run_id`.

### QSVT

* `figures/qsvt_20260106_125431_140529_optimal_poly_kappa6_eps0.47.png`
  Visualizes the optimal polynomial vs (1/x) on the target interval.

* `figures/qsvt_20260106_125431_140529_poly_degree_comparison_kappa6.png`
  Polynomial comparison for multiple degrees.

* `figures/qsvt_20260106_125431_140529_error_vs_degree_comparison_dmax1001.png`
  Error vs degree for multiple (\kappa).

* `figures/qsvt_20260106_125644_838045_optimal_poly_kappa6_eps0.47.png`

* `figures/qsvt_20260106_125644_838045_poly_degree_comparison_kappa6.png`

* `figures/qsvt_20260106_125644_838045_error_vs_degree_comparison_dmax1001.png`

* `figures/qsvt_20260106_125644_838045_error_vs_degree_subplots.png`

* `figures/qsvt_20260106_125644_838045_error_vs_degree_kappa5000_dmax100001.png`

---

## `results/` (generated data)

### Conditioning sweeps

* `results/Conditioning_20260107_015336_160795_sweep_meta.json`
  Metadata: `run_id`, `T_vals`, `m_grids`, `condM`, `condV`, relative errors vs DRE, residuals, notes.

* `results/Conditioning_20260107_015336_160795_sweep_arrays.npz`
  Compressed arrays with keys:

  * `T_vals (38,)`, `m_grids (3,)`
  * `condM (38,)`, `condV_mat (38,3)`
  * `rel_M_vs_DRE (38,)`, `relV_mat (38,3)`
  * `resM_2 (38,)`, `resV2_mat (38,3)`
  * `lamb0_M`, `lamb0_DRE`, `lamb0_V_m5`, `lamb0_V_m15`, `lamb0_V_m50` (here state dimension 4)

* `results/Conditioning_20260107_015336_160795_outputs.zip`
  Zip with the 3 figures + the 2 result files above.

* `results/Conditioning_20260107_015530_367649_sweep_meta.json`

* `results/Conditioning_20260107_015530_367649_sweep_arrays.npz`
  Same structure, but here `lamb0_*` have dimension 15 (vehicle-platoon).

* `results/Conditioning_20260107_015530_367649_outputs.zip`

### M/V trajectories

* `results/methods_20260106_125005_886049_meta.json`
  Run metadata (`run_id`, `model_name`, and estimated conditioning for M and V).

* `results/methods_20260106_125005_886049_M_traj.npz`
  Method M trajectory with keys:

  * `t`, `E`, `LAM`, `X`, `U`

* `results/methods_20260106_125005_886049_V_traj.npz`
  Same for Method V (typically sampled at `m` points).

* `results/methods_20260106_125005_886049_outputs.zip`
  Zip with figures + NPZ/JSON (note: some files are duplicated inside certain zips).

### Riccati ARE/DRE

For each `run_id` you get:

* `*_meta.json`
* `*_ARE_traj.npz` (keys `t,E,LAM,X,U`)
* `*_DRE_traj.npz` (keys `t,E,LAM,X,U`)
* `*_outputs.zip` (figures + results; duplicates appear in some zips)

Specifically:

* `riccati_20260106_125017_168377_*`
* `riccati_20260107_004812_722738_*`
* `riccati_20260107_005013_913624_*`

### QSVT

Run `qsvt_20260106_125431_140529_*`:

* `*_optimal_poly_meta.json`
  Includes `kappa`, `target_epsilon`, `a`, `degree`, `eps_bound`, `chebyshev_coef`.

* `*_poly_degree_comparison_meta.json`
  Includes tested degrees and bounds.

* `*_error_vs_degree_kappa{...}_dmax{...}.npz`
  Keys: `ds`, `eps`.

Run `qsvt_20260106_125644_838045_*` (more complete, includes a zip):

* `qsvt_20260106_125644_838045_optimal_poly_meta.json`
* `qsvt_20260106_125644_838045_poly_degree_comparison_meta.json`
* `qsvt_20260106_125644_838045_error_vs_degree_kappa5000.npz`
* `qsvt_20260106_125644_838045_error_vs_degree_kappa{5,50,500}_dmax1001.npz`
* `qsvt_20260106_125644_838045_error_vs_degree_kappa{5,50,500}_subplot.npz`
* `qsvt_20260106_125644_838045_outputs.zip`

---

## How to run the experiments

1. Start Jupyter:

```bash
jupyter lab
```

2. Run notebooks in this recommended order:

* `notebooks/trajectories_riccati.ipynb` (references)
* `notebooks/trajectories_m&v.ipynb` (M/V trajectories)
* `notebooks/conditioning_comparison.ipynb` (sweeps)
* `notebooks/qsvt_analysis.ipynb` (QSVT analysis)

The notebooks use `utils/export.py` to automatically save outputs into `figures/` and `results/`.
