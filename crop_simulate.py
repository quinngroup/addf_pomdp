# Driver packages
import numpy as np
from numpy.linalg import norm
import sys
import glob
import re
from random import randint as rand

# Image packages
import cv2


"""
Crop field simulator, containing the true observation function, which is unobservable to the agents.
"""

class CropField:
    def __init__(self, n_observation, image_path = ''):
        # Create 2 to 4 sectors with whether there is a stress or not
        self.sectors = np.array(np.random.rand((rand(2,4))).round(), dtype=np.int)
        self.create_observation_function(n_observation)
        self.create_transition_function()
        print(self.sectors)
        # TODO: Process an image to generate simulation for bootstrapping
        # image = cv2.imread(image_path,0)
        # self.dimensions = image.shape[:2]

    def create_observation_function(self, n_observation):
        """
        Instantiate the observation function.

        The level of the agent makes stresses harder to distinguish.
        """
        self.o = np.array([self.create_observation_function_level(n_observation, i) for i in range(1,3)])

    def create_observation_function_level(self, n_observation, level):
        true_likelihood = 0.85 - level*0.05 # The likelihood of truly positively identifying the stress
        stress = [(i/n_observation)*(1-true_likelihood) for i in range(1,n_observation)] + [true_likelihood]
        return(np.array([
                list(reversed(stress)), # No stress
                stress # Presence of a stress
            ]))

    def create_transition_function(self):
        """
        Instantiate the transition function.

        Actions don't affect the state transition. States are less likely to transition later in the growing season.
        """
        t = []
        persist = np.array([0.7, 0.3]) # Initial likelihood the state persists
        for i in range(1,24):
            modified = np.array([persist[0]*i,persist[1]])
            modified = modified/modified.sum()
            t.append([modified, list(reversed(modified))])
        self.t = np.array(t)

    def observe():
        print("stub")