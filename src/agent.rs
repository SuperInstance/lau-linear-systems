//! Application: agent state estimation using Kalman filtering for belief tracking.
//!
//! Models an agent with position and velocity state, observed through noisy measurements.
//! The Kalman filter tracks the agent's belief (state estimate) and uncertainty.

use nalgebra::{DMatrix, DVector};
use crate::estimation::KalmanFilter;

/// Agent belief tracker using a Kalman filter.
///
/// State: [position, velocity, acceleration]^T (or configurable)
/// Models agent dynamics with process noise and noisy position measurements.
#[derive(Debug, Clone)]
pub struct AgentBeliefTracker {
    filter: KalmanFilter,
    dt: f64,
    pub position_estimate: f64,
    pub velocity_estimate: f64,
    pub position_uncertainty: f64,
    pub velocity_uncertainty: f64,
}

impl AgentBeliefTracker {
    /// Create a new 2D (position, velocity) agent tracker.
    ///
    /// # Arguments
    /// * `dt` - Time step
    /// * `process_noise_pos` - Process noise variance for position
    /// * `process_noise_vel` - Process noise variance for velocity
    /// * `measurement_noise` - Measurement noise variance
    pub fn new_2d(
        dt: f64,
        process_noise_pos: f64,
        process_noise_vel: f64,
        measurement_noise: f64,
    ) -> Self {
        // State: [pos, vel]
        // Model: pos' = vel, vel' = 0 (constant velocity with noise)
        let a = DMatrix::from_row_slice(2, 2, &[
            1.0, dt,
            0.0, 1.0,
        ]);
        let b = DMatrix::zeros(2, 1); // No control input
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]); // Observe position
        let d = DMatrix::zeros(1, 1);
        let q = DMatrix::from_row_slice(2, 2, &[
            process_noise_pos, 0.0,
            0.0, process_noise_vel,
        ]);
        let r = DMatrix::from_row_slice(1, 1, &[measurement_noise]);
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let p0 = DMatrix::from_row_slice(2, 2, &[
            100.0, 0.0,
            0.0, 100.0,
        ]);

        let filter = KalmanFilter::new(a, b, c, d, q, r, x0, p0);

        Self {
            filter,
            dt,
            position_estimate: 0.0,
            velocity_estimate: 0.0,
            position_uncertainty: 100.0,
            velocity_uncertainty: 100.0,
        }
    }

    /// Create a 3D (position, velocity, acceleration) agent tracker.
    pub fn new_3d(
        dt: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> Self {
        let a = DMatrix::from_row_slice(3, 3, &[
            1.0, dt, 0.5 * dt * dt,
            0.0, 1.0, dt,
            0.0, 0.0, 1.0,
        ]);
        let b = DMatrix::zeros(3, 1);
        let c = DMatrix::from_row_slice(1, 3, &[1.0, 0.0, 0.0]);
        let d = DMatrix::zeros(1, 1);
        let q = DMatrix::from_diagonal_element(3, 3, process_noise);
        let r = DMatrix::from_row_slice(1, 1, &[measurement_noise]);
        let x0 = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let p0 = DMatrix::from_diagonal_element(3, 3, 100.0);

        let filter = KalmanFilter::new(a, b, c, d, q, r, x0, p0);

        Self {
            filter,
            dt,
            position_estimate: 0.0,
            velocity_estimate: 0.0,
            position_uncertainty: 100.0,
            velocity_uncertainty: 100.0,
        }
    }

    /// Update belief with a new position measurement.
    pub fn update(&mut self, measured_position: f64, control_input: Option<f64>) {
        let u = DVector::from_vec(vec![control_input.unwrap_or(0.0)]);
        let y = DVector::from_vec(vec![measured_position]);
        self.filter.step(&y, &u);

        self.position_estimate = self.filter.estimate()[0];
        self.velocity_estimate = self.filter.estimate()[1];
        self.position_uncertainty = self.filter.covariance()[(0, 0)];
        self.velocity_uncertainty = self.filter.covariance()[(1, 1)];
    }

    /// Predict forward without a measurement.
    pub fn predict(&mut self, control_input: Option<f64>) {
        let u = DVector::from_vec(vec![control_input.unwrap_or(0.0)]);
        self.filter.predict(&u);

        self.position_estimate = self.filter.estimate()[0];
        self.velocity_estimate = self.filter.estimate()[1];
        self.position_uncertainty = self.filter.covariance()[(0, 0)];
        self.velocity_uncertainty = self.filter.covariance()[(1, 1)];
    }

    /// Get the full state estimate.
    pub fn state(&self) -> &DVector<f64> {
        self.filter.estimate()
    }

    /// Get the full covariance.
    pub fn covariance(&self) -> &DMatrix<f64> {
        self.filter.covariance()
    }

    /// Get the underlying Kalman filter.
    pub fn kalman_filter(&self) -> &KalmanFilter {
        &self.filter
    }
}
