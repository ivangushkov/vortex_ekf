#!/usr/bin/env python
"""
Dynamic models to be used with eg. EKF.

@author: Lars-Christian Tokle, lars-christian.n.tokle@ntnu.no
"""

## Adapted for use by Vortex NTNU ##

import numpy as np
from numpy import ndarray
from scipy.linalg import expm

class DynamicModel:
    """
    Parent class for dynamic models.

    A model includes the discrete prediction equation f, its Jacobian F, and
    the process noise covariance Q.
    """

    def f(self, x, Ts, **kwargs):
        """Calculate the zero noise Ts time units transition from x.

          Args:
              x (ndarray): state
              Ts (float): time step

          Returns:
              x_kp1 (ndarray): x_k+1, the next state
          """
        raise NotImplementedError

    def F(self, x, Ts, **kwargs):
        """Calculate the transition function jacobian for Ts time units at x.
        Args:
            x (ndarray): state
            Ts (float): time step

        Returns:
            F (ndarray): Discrete ransition function jacobian,
                         for linear systems: x_k+1 = F @ x_k
        """
        raise NotImplementedError

    def Q(self, x, Ts, **kwargs):
        """Calculate the Ts time units transition Covariance.
        Args:
            x (ndarray): state
            Ts (float): time step

        Returns:
            Q (ndarray): covariance matrix
        """
        raise NotImplementedError


class WhitenoiseAcceleration2D(DynamicModel):
    """
    A white noise acceleration model, also known as constan velocity.
    States are position and speed.
    """

    # noise standard deviation
    def __init__(self, sigma_a):
        self.sigma_a = sigma_a

    def f(self, x: ndarray, Ts: float,) -> ndarray:
        """Calculate the zero noise Ts time units transition from x.
        See DynamicModel for variable documentation
        """
        
        F = np.array([
            [1, 0, Ts, 0], [0, 1, 0, Ts], [0, 0, 1, 0], [0, 0, 0, 1]
        ])

        # Model is actually linear here, f and F are the same
        x_kp1 = F @ x

        return x_kp1

    def F(self, x: ndarray, Ts: float,) -> ndarray:
        """Calculate the transition function jacobian for Ts time units at x.
        See DynamicModel for variable documentation"""

        F = np.array([
            [1, 0, Ts, 0], [0, 1, 0, Ts], [0, 0, 1, 0], [0, 0, 0, 1]
        ])
        return F

    def Q(self, x: ndarray, Ts: float,) -> ndarray:
        """Calculate the Ts time units transition Covariance.
        See DynamicModel for variable documentation"""

        Q = np.array([
            [(Ts**3)/3, 0, (Ts**2)/2, 0], [0, (Ts**3)/3, 0, (Ts**2)/2], [(Ts**2)/2, 0, Ts, 0], [0, (Ts**2)/2, 0, Ts]
        ]) * self.sigma_a**2

        return Q


class landmark_gate(DynamicModel):
    """
    Dynamic model for a landmark. Landmarks are assumed time invariant.
    """

    # noise standard deviation array
    def __init__(self, sigmas):
        self.sigma_arr = sigmas

    def f(self, x, Ts):
        """Calculate the zero noise Ts time units transition from x.
        See DynamicModel for variable documentation
        """
        
        n = len(x)
        
        #x_kp1 = np.eye(n) @ x
        x_kp1 = np.matmul(np.eye(n), x)

        return x_kp1

    def F(self, x, Ts):
        """Calculate the transition function jacobian for Ts time units at x.
        See DynamicModel for variable documentation"""


        n = len(x)
        F = np.eye(n)

        return F

    def Q(self, x, Ts):
        """Calculate the Ts time units transition Covariance.
        See(4.64) in the book.
        See DynamicModel for variable documentation"""

        Q = np.diag(self.sigma_arr)*Ts

        return Q
