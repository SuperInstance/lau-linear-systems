//! # lau-linear-systems
//!
//! Linear dynamical systems — state-space models, observability, controllability,
//! Kalman filtering, LQR, and system interconnection.

pub mod ss;
pub mod ctrb_obs;
pub mod estimation;
pub mod feedback;
pub mod transfer;
pub mod interconnect;
pub mod discretize;
pub mod norms;
pub mod agent;

pub use ss::{StateSpace, StateSpaceDiscrete, StateSpaceContinuous};
pub use ctrb_obs::{controllability_matrix, observability_matrix, is_controllable, is_observable};
pub use estimation::{LuenbergerObserver, KalmanFilter};
pub use feedback::{place_poles as pole_placement, lqr};
pub use transfer::{ss2tf, tf2ss, TransferFunction};
pub use interconnect::{series, parallel, feedback};
pub use discretize::{zoh_discretize, bilinear_discretize_disc as bilinear_discretize};
pub use norms::{h2_norm, hinf_norm_approx};
pub use agent::AgentBeliefTracker;
