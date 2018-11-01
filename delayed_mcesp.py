# Driver packages
import numpy as np
from numpy.linalg import norm
import sys
import glob
import re

"""
MCESP for Game-Delayed Reinforcements
"""

class MCESP_D:
    def __init__(self, observations, max_filler, field = np.zeros((1,1))):
        """
        Constructor for MCESP-D. Field integration currently stubbed.

        Parameters
        ----------
        observations : int
            The allowed specificity of stress levels
        max_filler : int
            (Heuristic) The number of call-to-actions to be passed down when a stress is _not_ detected
        field : array, shape (H, W)
            A NumPy grayscale image
        """
        self.actions = 3 # High priority, low priority, none
        self.observations = observations
        self.max_filler = max_filler
        self.q_table = np.zeros((observations,actions))
        self.c = 0
        set_prior(field)

    def set_prior(field):
        """
        Set the initial observation discretization to the dimentionality of observations.
        Initially set discretization factor to uniform. Set discretization learning rate to 1.
        """
        self.observation_thresholds = [i/o for i in range(0,observations)]
        self.observation_samples = 1
        # TODO: For use after integrating image processing with MCESP for Game-Delayed Reinforcements
        # self.norm = field.max()

    def update_reward(observation, action, reward):
        """
        Update the Q-table when a delayed reward is received from a subsequent layer.
        """
        self.q_table[observation,action] = (1 - 1/(1+c)) * self.q_table[action,observation] + (1/(1+c)) * r # Canonical Q-update
        self.c += 1

    def act(observation):
        return(np.argmax(q_table[observation]))