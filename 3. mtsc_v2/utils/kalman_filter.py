import numpy as np
import scipy.linalg


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, w, h, vx, vy, vw, vh

    contains the bounding box center position (x, y), width w, height h, and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location (x, y, w, h) is taken as direct
    observation of the state space (linear observation model).

    https://sharehobby.tistory.com/entry/%EC%B9%BC%EB%A7%8C-%ED%95%84%ED%84%B0Kalman-filter1
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # motion_mat = A
        self.motion_mat = np.eye(2 * ndim, 2 * ndim)
        for idx in range(ndim):
            self.motion_mat[idx, ndim + idx] = dt

        # update_mat = H
        self.update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate.
        # These weights control the amount of uncertainty in the model. This is a bit hacky.
        # 1 / 20, 1 / 160 are better than 1, 1
        self._std_weight_position = 1 / 20
        self._std_weight_velocity = 1 / 160

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.
        Parameters
        measurement : ndarray
            Bounding box coordinates (x, y, w, h) with center position (x, y), width w, and height h.

        Returns
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional) of the new track.
            Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)

        # x
        mean = np.r_[mean_pos, mean_vel]

        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3]]

        # P
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.
        Parameters
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted state.
            Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3]]

        # motion_cov = Q
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # x = Ax
        mean = np.dot(mean, self.motion_mat.T)

        # P = APA' + Q
        covariance = np.linalg.multi_dot((self.motion_mat, covariance, self.motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.
        Parameters
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state estimate.
        """
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               self._std_weight_position * mean[3]]

        # innovation_cov = R
        innovation_cov = np.diag(np.square(std))

        # mean = Hx
        mean = np.dot(self.update_mat, mean)

        # covariance = HPH' + R (at return)
        covariance = np.linalg.multi_dot((self.update_mat, covariance, self.update_mat.T))

        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.

        Parameters
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center position,
            w the width, and h the height of the bounding box.

        Returns
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        # projected_mean = Hx, projected_cov = HPH' + R
        projected_mean, projected_cov = self.project(mean, covariance)

        # K = PH'(HPH' + R)^-1
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self.update_mat.T).T,
                                             check_finite=False).T

        # innovation = z - Hx
        innovation = measurement - projected_mean

        # x = x + K(z - Hx)
        new_mean = mean + np.dot(innovation, kalman_gain.T)

        # Update P
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance


