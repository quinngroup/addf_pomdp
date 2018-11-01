import numpy as np
from random import randint
from random import random

# Image packages
import cv2


"""
Crop field simulator, containing the true observation function, which is unobservable to the agents.
"""

class CropField:
    """
    -------------------------------------------------------------------------------------------------
    Initialization
    -------------------------------------------------------------------------------------------------
    """
    def __init__(self, n_sector, n_observation, image_path = ''):
        # Build frame
        self.create_observation_function(n_observation)
        self.create_transition_function()

        # Create random initial state for simulated sectors
        self.cur_sectors = np.array(np.random.rand((n_sector)).round(), dtype=np.int)

        self.day = 0

        # TODO: Process an image to generate simulation for bootstrapping
        # image = cv2.imread(image_path,0)
        # self.dimensions = image.shape[:2]

    def create_observation_function(self, n_observation):
        """
        Instantiate the observation function. The layer of the agent makes stresses harder to distinguish.
        """
        self.o = np.array([self.create_observation_function_layer(n_observation, i) for i in range(1,4)])

    def create_observation_function_layer(self, n_observation, layer):
        true_likelihood = 0.85 - layer*0.05 # The likelihood of truly positively identifying the stress
        stress = [(i/n_observation)*(1-true_likelihood) for i in range(1,n_observation)] + [true_likelihood]
        return(np.array([
                list(reversed(stress)), # No stress
                stress # Presence of a stress
            ]))

    def create_transition_function(self):
        """
        Instantiate the transition function. Actions don't affect the state transition. States are less likely to transition later in the growing season.
        """
        t = []
        persist = np.array([0.5, 0.5]) # Initial likelihood the state persists
        for i in range(1,25):
            modified = np.array([persist[0]*i,persist[1]])
            modified = modified/modified.sum()
            t.append([modified, list(reversed(modified))])
        self.t = np.array(t)

    """
    -------------------------------------------------------------------------------------------------
    Observing and transitioning between sets of sectors
    -------------------------------------------------------------------------------------------------
    """
    def observe_sectors(self,layer):
        return([self.observe(s,layer) for s in self.cur_sectors])

    def observe(self,s,l):
        ob = random()
        for i in range(0,len(self.o[l][s])):
            ob -= self.o[l][s][i]
            if ob <= 0:
                return i
        return len(self.o[l][s])-1

    def transition_sectors(self):
        return([self.transition(s) for s in self.cur_sectors])

    def transition(self,s):
        tr = random()
        d = min([self.day,23])
        for i in range(0,len(self.t[d][s])):
            tr -= self.t[d][s][i]
            if tr <= 0:
                return i
        return len(self.t[s])-1

    def iterate_states(self):
        self.cur_sectors = self.transition_sectors()
        self.day += 1
        # new_sectors = self.transition_sectors()
        # self.cur_sectors = new_sectors

    """
    -------------------------------------------------------------------------------------------------
    Testing the CropField simulator
    -------------------------------------------------------------------------------------------------
    """

    def test(self):
        cur = self.cur_sectors
        print("Current State: "+str(cur))
        for i in reversed(range(1,4)):
            print("\tLayer "+str(i)+" observation")
            print("\t"+str(self.observe_sectors(i-1)))
        self.iterate_states()
        print("\tIterated from "+str(cur)+" to "+str(self.cur_sectors))
