# Quantum assisted Optimal Control examples

This repository contains **two benchmark case studies** based on **LQR (Linear Quadratic Regulator)** supporting a thesis on **quantum-assisted optimal control**.

The **numerical results** (and figures) are obtained by running the **notebooks**, where you can **modify parameters** to study how trajectories, conditioning, and errors change.



## Benchmark instances

This repository summarizes **two LQR instances** reported in the thesis:

1. **Inverted pendulum on a cart**: linearized model around the upright equilibrium.
2. **Vehicle platooning**: leader–follower formulation in error coordinates.



## LQR formulation 

We consider a continuous-time linear system

$$
\dot{x}(t)=Ax(t)+Bu(t),
$$

and a quadratic cost over a finite horizon $ [0,T] $

$$
J=\int_0^T \left(e(t)^\top Q\,e(t)+u(t)^\top R\,u(t)\right)\,dt \;+\; e(T)^\top S\,e(T),
\quad e(t)=x(t)-x_{\mathrm{ref}}.
$$

The boundary-value formulation is expressed through the Hamiltonian system in the extended variable
$$
y(t)=\begin{bmatrix} e(t)\\ \lambda(t)\end{bmatrix},
\quad
\dot{y}(t)=Cy(t),
$$

with Hamiltonian matrix

$$
C=
\begin{bmatrix}
A & -BR^{-1}B^\top\\
-\,Q & -A^\top
\end{bmatrix},
\qquad
\begin{cases}
e(0)=e_0,\\
\lambda(T)=S\,e(T).
\end{cases}
$$



## Case study 1 — Inverted pendulum on a cart (linearized model)

We consider $\dot{x}=Ax+Bu$ with $x\in\mathbb{R}^4$ and scalar input $u\in\mathbb{R}$. In the experiments, we use the following numerical linearization:

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
0\\
0.2\\
0\\
0.1
\end{bmatrix}.
$$

The open-loop dynamics are **unstable** (there exists at least one eigenvalue with positive real part), motivating feedback design.

In the reported experiments we use a finite-horizon LQR with:
- $Q=I_4$,
- $S=\mathrm{diag}(0,\,10,\,50,\,10)\succeq 0$,
- $R=0.01>0$.

To track a constant reference $x_{\mathrm{ref}}$, we work in error coordinates $e(t)=x(t)-x_{\mathrm{ref}}$.



## Case study 2 — Vehicle platooning (leader–follower relative deviations)

We consider a longitudinal platoon consisting of one leader (index $0$) and $N$ identical followers ($i=1,\dots,N$). Each vehicle is modeled as a point mass with actuator lag $\tau>0$:

$$
\dot{s}_i=v_i,\qquad
\dot{v}_i=a_i,\qquad
\dot{a}_i=-\frac{1}{\tau}a_i+\frac{1}{\tau}u_i.
$$

In state-space form, $\dot{x}_i=A_0x_i+b_0u_i$, with

$$
A_0=
\begin{bmatrix}
0 & 1 & 0\\
0 & 0 & 1\\
0 & 0 & -1/\tau
\end{bmatrix},
\qquad
b_0=
\begin{bmatrix}
0\\0\\1/\tau
\end{bmatrix},
\qquad
x_i=
\begin{bmatrix}s_i\\v_i\\a_i\end{bmatrix}\in\mathbb{R}^3.
$$

The leader input $u_0(t)$ is treated as exogenous (not optimized). Under a constant-spacing policy with desired gap $d>0$, we define leader-relative errors:

$$
e_{s,i}=s_i-s_0+i\,d,\quad
e_{v,i}=v_i-v_0,\quad
e_{a,i}=a_i-a_0.
$$

Stacking $E=[e_1^\top\ \cdots\ e_N^\top]^\top\in\mathbb{R}^{3N}$ and defining differential inputs
$\tilde{u}_i:=u_i-u_0$ (stacked as $\tilde{U}$), the follower error dynamics become:

$$
\dot{E}=A_fE+B_f\tilde{U},
\qquad
A_f=I_N\otimes A_0,\quad
B_f=I_N\otimes b_0.
$$

We consider a finite-horizon LQR with Kronecker-lifted weights:
- $Q=I_N\otimes Q_0$,
- $R=I_N\otimes R_0$,
- terminal weight $S$ (in the reported experiments we use $S=0$, hence $\lambda(T)=0$).



> Since this are LQR-type problem, correctness can be validated by comparison against Riccati-based solutions.

---

## Reproducibility: 

- Experiments are run from `notebooks/`.
- Notebook parameters (such as time horizon $T$, discretizations, weights $Q,R,S$, initial conditions, number of vehicles $N$, etc.) can be modified to explore how results change.
- Generated figures are saved to `figures/`.
- Notebooks call the required helper code from `src/`.

---

## References

- Steven L. Brunton and J. Nathan Kutz, *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control*. Cambridge University Press, 2019.
- Karl Johan Åström and Richard M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton University Press, 2008.
- Yongxin Zhu, Yongfu Li, Simon Hu, and Shuyou Yu, “Optimal Control for Vehicle Platoon Considering External Disturbances,” *IEEE ITSC 2022*. DOI: 10.1109/ITSC55140.2022.9922486


