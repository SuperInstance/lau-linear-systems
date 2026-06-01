# lau-linear-systems

> Linear dynamical systems — state-space models, observability, controllability, Kalman filtering, LQR, and system interconnection

A Rust library for modeling, analyzing, and controlling linear dynamical systems. It provides a complete control theory toolkit: from state-space representations through transfer functions, controllability/observability analysis, state estimation (Kalman filter), optimal control (LQR), system interconnection, discretization, and H₂/H∞ norms.

## What This Does

This crate implements the core machinery of modern control theory:

1. **State-space models** — continuous-time (dx/dt = Ax + Bu) and discrete-time (x[k+1] = Ax[k] + Bu[k])
2. **Transfer functions** — polynomial representations, SS↔TF conversion via Leverrier's algorithm and controllable canonical form
3. **Structural analysis** — controllability and observability matrices with SVD-based rank tests
4. **State estimation** — Luenberger observer (pole-placement) and discrete Kalman filter (predict/update cycle)
5. **Optimal control** — pole placement via Ackermann's formula and LQR via Kleinman iteration
6. **System interconnection** — series, parallel, and feedback connections with proper state augmentation
7. **Discretization** — zero-order hold (matrix exponential) and bilinear (Tustin) transform
8. **System norms** — H₂ norm (via Lyapunov equation) and H∞ norm (frequency sweep approximation)
9. **Agent tracking** — Kalman-filter-based position/velocity/acceleration belief tracker

## Key Idea

Every linear time-invariant (LTI) system can be written in state-space form:

```
dx/dt = Ax + Bu    (state equation)
y     = Cx + Du    (output equation)
```

The matrices A, B, C, D fully characterize the system's dynamics. From these four matrices, you can compute stability (eigenvalues of A), controllability (can we reach any state?), observability (can we infer the state from outputs?), optimal controllers (LQR), and frequency-domain properties (transfer function, norms).

## Install

```toml
[dependencies]
lau-linear-systems = "0.1.0"
```

Or clone directly:

```bash
git clone https://github.com/SuperInstance/lau-linear-systems.git
cd lau-linear-systems
cargo build
```

### Dependencies

- `nalgebra` 0.33 (with `serde-serialize`) — linear algebra
- `num-complex` 0.4 — complex number arithmetic for eigenvalues and transfer function evaluation
- `serde` 1 (with `derive`) — serialization
- `approx` 0.5 (dev) — approximate equality in tests
- `serde_json` 1 (dev) — JSON serialization tests

## Quick Start

### Create and Simulate a System

```rust
use nalgebra::{DMatrix, DVector};
use lau_linear_systems::ss::StateSpaceContinuous;

// Mass-spring-damper: m*x'' + c*x' + k*x = F
// State: [x, x']', Input: [F], Output: [x]
let a = DMatrix::from_row_slice(2, 2, &[
    0.0,  1.0,
    -4.0, -0.5,   // k/m=4, c/m=0.5
]);
let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
let d = DMatrix::zeros(1, 1);

let sys = StateSpaceContinuous::new(a, b, c, d);

// Check stability (all eigenvalues have negative real parts)
println!("Stable: {}", sys.is_stable());
println!("Poles: {:?}", sys.poles());

// Simulate one Euler step
let x = DVector::from_vec(vec![1.0, 0.0]);
let u = DVector::from_vec(vec![0.5]);
let x_next = sys.step(&x, &u, 0.01);
let y = sys.output(&x_next, &u);
```

### Controllability and Observability

```rust
use lau_linear_systems::ctrb_obs::*;

let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -2.0, -3.0]);
let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

// Controllability matrix: [B, AB]
let ctrb = controllability_matrix(&a, &b);
println!("Controllable: {}", is_controllable(&a, &b)); // true

// Observability matrix: [C; CA]
let obs = observability_matrix(&a, &c);
println!("Observable: {}", is_observable(&a, &c)); // true
```

### LQR Controller Design

```rust
use lau_linear_systems::feedback::*;

// Design optimal state feedback: u = -Kx
// Minimizes J = ∫(x'Qx + u'Ru) dt
let q = DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, 1.0]); // State cost
let r = DMatrix::from_row_slice(1, 1, &[1.0]);                   // Control cost

let k = lqr(&sys.a, &sys.b, &q, &r);
println!("LQR gain K = {:?}", k);

// Closed-loop: A_cl = A - BK
let a_cl = &sys.a - &sys.b * &k;
```

### Kalman Filter

```rust
use lau_linear_systems::estimation::*;

// Discretize the system
let sys_d = discretize::zoh_discretize(&sys, 0.01);

// Kalman filter
let q_noise = DMatrix::from_diagonal_element(2, 2, 0.01);  // Process noise
let r_noise = DMatrix::from_diagonal_element(1, 1, 0.1);   // Measurement noise
let x0 = DVector::from_vec(vec![0.0, 0.0]);
let p0 = DMatrix::from_diagonal_element(2, 2, 100.0);

let mut kf = KalmanFilter::new(
    sys_d.a.clone(), sys_d.b.clone(), sys_d.c.clone(), sys_d.d.clone(),
    q_noise, r_noise, x0, p0,
);

// Step: predict + update
let y_meas = DVector::from_vec(vec![1.05]);
let u = DVector::from_vec(vec![0.5]);
let estimate = kf.step(&y_meas, &u);
println!("State estimate: {:?}", estimate);
println!("Uncertainty: {:?}", kf.covariance());
```

### Transfer Function Conversion

```rust
use lau_linear_systems::transfer::*;

// State-space → transfer function
let tf = ss2tf(&sys);
println!("H(s) = {:?} / {:?}", tf.num, tf.den);
println!("Poles: {:?}", tf.poles());
println!("Zeros: {:?}", tf.zeros());

// Evaluate at s = jω
let s = num_complex::Complex64::new(0.0, 1.0);
let h = tf.evaluate(s);
println!("H(j1) = {}", h);

// Transfer function → state-space (controllable canonical form)
let sys_back = tf2ss(&tf);
```

### System Interconnection

```rust
use lau_linear_systems::interconnect::*;

// Series: sys1 → sys2
let series_sys = series(&sys1, &sys2);

// Parallel: y = sys1(u) + sys2(u)
let parallel_sys = parallel(&sys1, &sys2);

// Feedback: sys1 forward, sys2 feedback
let feedback_sys = feedback(&sys1, &sys2);
```

### Agent Belief Tracking

```rust
use lau_linear_systems::agent::AgentBeliefTracker;

// Track an agent with position + velocity state
let mut tracker = AgentBeliefTracker::new_2d(
    0.1,    // dt
    0.01,   // process noise (position)
    0.1,    // process noise (velocity)
    1.0,    // measurement noise
);

// Update with noisy position measurement
tracker.update(5.3, None);
println!("Position: {}", tracker.position_estimate);
println!("Velocity: {}", tracker.velocity_estimate);
println!("Pos uncertainty: {}", tracker.position_uncertainty);
```

## API Reference

### Modules

| Module | Description |
|--------|-------------|
| `ss` | State-space models (continuous & discrete), eigenvalue computation via QR algorithm with Hessenberg reduction |
| `ctrb_obs` | Controllability/observability matrices and SVD-based rank tests |
| `feedback` | Pole placement (Ackermann's formula) and LQR (Kleinman iteration on the CARE) |
| `estimation` | Luenberger observer and discrete Kalman filter (predict/update) |
| `transfer` | Transfer function representation, ss2tf (Leverrier), tf2ss (controllable canonical form) |
| `interconnect` | Series, parallel, and feedback system connections |
| `discretize` | Zero-order hold and bilinear (Tustin) discretization |
| `norms` | H₂ norm (controllability Gramian) and H∞ norm approximation (frequency sweep) |
| `agent` | Kalman-filter-based agent belief tracker (2D and 3D) |

### Core Types

```rust
// State-space
pub struct StateSpaceContinuous { a: DMatrix<f64>, b, c, d }
pub struct StateSpaceDiscrete  { a: DMatrix<f64>, b, c, d, dt: f64 }
pub enum StateSpace { Continuous(StateSpaceContinuous), Discrete(StateSpaceDiscrete) }

// Transfer function
pub struct TransferFunction { num: Vec<f64>, den: Vec<f64> }

// Estimation
pub struct LuenbergerObserver { sys, l: DMatrix<f64>, x_hat: DVector<f64> }
pub struct KalmanFilter { a, b, c, d, q, r, x_hat, p }

// Agent
pub struct AgentBeliefTracker { filter, dt, position_estimate, velocity_estimate, ... }
```

### Key Methods

- `StateSpaceContinuous::new(a, b, c, d)` — create with dimension assertions
- `StateSpaceContinuous::step(x, u, dt)` — Euler integration step
- `StateSpaceContinuous::poles()` — eigenvalues of A via QR + Hessenberg + Wilkinson shifts
- `StateSpaceContinuous::is_stable()` — all eigenvalues have Re < 0 (continuous) or |λ| < 1 (discrete)
- `ss2tf(sys)` — Leverrier's algorithm for C·adj(sI-A)·B + D·det(sI-A)
- `tf2ss(tf)` — controllable canonical form
- `lqr(a, b, q, r)` — Kleinman iteration solving the Continuous Algebraic Riccati Equation
- `KalmanFilter::step(y, u)` — predict then update
- `zoh_discretize(sys, dt)` — matrix exponential via scaling-and-squaring + Taylor series
- `bilinear_discretize(sys, dt)` — Tustin transform
- `h2_norm(sys)` — via controllability Gramian Lyapunov equation
- `hinf_norm_approx(sys, freq_range, n_points)` — frequency sweep for SISO systems

## How It Works

### Eigenvalue Computation

The eigenvalue solver in `ss.rs` implements a three-stage approach:
1. **Hessenberg reduction** via Householder reflections (O(n³))
2. **QR iteration with Wilkinson shifts** (cubic convergence)
3. **Extraction** from quasi-upper-triangular form (2×2 blocks → complex conjugate pairs)

### Transfer Function ↔ State-Space

**SS → TF** uses Leverrier's algorithm to compute:
- Characteristic polynomial: det(sI - A) = sⁿ + c₁sⁿ⁻¹ + ... + cₙ
- Adjugate: adj(sI - A) = M₀sⁿ⁻¹ + M₁sⁿ⁻² + ... + Mₙ₋₁
- Transfer function: H(s) = [C·adj(sI-A)·B + D·det(sI-A)] / det(sI-A)

**TF → SS** uses the **controllable canonical form**:
```
A = [ -a₁  1  0  ...  0  ]
    [ -a₂  0  1  ...  0  ]
    [ ...                 ]
    [ -aₙ  0  0  ...  0  ]

B = [1]    C = [b₁-a₁d₀, b₂-a₂d₀, ..., bₙ-aₙd₀]    D = [d₀]
    [0]
    [...]
```

### LQR via Kleinman Iteration

The Linear Quadratic Regulator solves the Continuous Algebraic Riccati Equation (CARE):

A'P + PA - PBR⁻¹B'P + Q = 0

Kleinman's method iterates:
1. Start with stabilizing gain K₀
2. Solve Lyapunov: (A-BKₖ)'Pₖ + Pₖ(A-BKₖ) = -(Q + Kₖ'RKₖ)
3. Update: Kₖ₊₁ = R⁻¹B'Pₖ
4. Repeat until convergence

### Kalman Filter

The discrete Kalman filter alternates:
- **Predict**: x̂⁻ = Ax̂ + Bu, P⁻ = APᵀA' + Q
- **Update**: K = P⁻C'(CP⁻C' + R)⁻¹, x̂ = x̂⁻ + K(y - ŷ), P = (I - KC)P⁻

### Discretization

**Zero-order hold**: Ad = exp(A·dt), Bd = A⁻¹(Ad - I)·B (or series fallback for singular A)

Matrix exponential uses **scaling and squaring** with Taylor series: choose s such that ‖A·dt/2ˢ‖ < 0.5, compute exp(A·dt/2ˢ) via Taylor, then square s times.

**Bilinear (Tustin)**: s = (2/dt)(z-1)/(z+1), yielding Ad = Φ(I + (dt/2)A), Bd = Φ·dt·B where Φ = (I - (dt/2)A)⁻¹

### H₂ and H∞ Norms

**H₂ norm**: ‖G‖₂ = √(trace(C·P·C')) where P solves the controllability Gramian Lyapunov equation AP + PA' = -BB'.

**H∞ norm**: approximated by max |G(jω)| over a logarithmic frequency sweep, computing G(jω) = C·(jωI - A)⁻¹·B + D via real-imaginary doubled system.

## The Math

### State-Space Representation

```
Continuous:   dx/dt = Ax + Bu,   y = Cx + Du
Discrete:     x[k+1] = Ax[k] + Bu[k],   y[k] = Cx[k] + Du[k]
```

A ∈ ℝⁿˣⁿ (system), B ∈ ℝⁿˣᵐ (input), C ∈ ℝᵖˣⁿ (output), D ∈ ℝᵖˣᵐ (feedthrough)

### Controllability

The pair (A, B) is controllable iff rank([B, AB, A²B, ..., Aⁿ⁻¹B]) = n.

### Observability

The pair (A, C) is observable iff rank([C; CA; CA²; ...; CAⁿ⁻¹]) = n.

Duality: (A, B) controllable ⟺ (A', B') observable.

### Transfer Function

H(s) = C(sI - A)⁻¹B + D

Poles = eigenvalues of A, Zeros = roots of numerator.

### Ackermann's Formula

For single-input controllable systems, the pole placement gain is:

K = eₙ'·C⁻¹·φ_d(A)

where C is the controllability matrix, eₙ = [0, ..., 0, 1], and φ_d is the desired characteristic polynomial evaluated at A.

### Lyapunov Equation

AP + PA' = Q solved via vectorization: (I⊗A + A⊗I)·vec(P) = vec(Q)

## Tests

The crate contains **58 unit tests** across integration tests and inline module tests, covering:

- Continuous/discrete state-space creation and simulation
- Eigenvalue computation and stability analysis
- Controllability/observability rank tests
- Transfer function evaluation, SS↔TF round-trip
- Pole placement and LQR convergence
- Kalman filter predict/update cycle
- Luenberger observer design
- Zero-order hold and bilinear discretization
- H₂ norm and H∞ norm computation
- Series, parallel, and feedback interconnections
- Agent belief tracker convergence

```bash
cargo test
```

## License

MIT
