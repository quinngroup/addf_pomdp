import numpy as np

"""
MCESP for Game-Delayed Reinforcements
"""

class MCESP_D:
    def __init__(self, observations, max_filler = 0, field = np.zeros((1,1))):
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
        self.actions = 2
        self.observations = observations
        self.max_filler = max_filler
        self.q_table = np.ones((self.observations,self.actions))
        self.c_table = np.zeros((self.observations,self.actions))
        self.set_prior(field)

    def set_prior(self,field):
        """
        Set the initial observation discretization to the dimentionality of observations.
        Initially set discretization factor to uniform. Set discretization learning rate to 1.
        """
        self.observation_thresholds = [i/self.observations for i in range(0,self.observations)]
        self.observation_samples = 1
        # TODO: For use after integrating image processing with MCESP for Game-Delayed Reinforcements
        # self.norm = field.max()

    def update_reward(self, observation, action, reward):
        """
        Update the Q-table when a delayed reward is received from a subsequent layer.
        """
        self.q_table[observation,action] = (1 - self.count(observation,action)) * self.q_table[observation,action] + self.count(observation,action) * reward # Canonical Q-update
        self.increment_count(observation,action)

    def count(self,observation, action):
        """
        Q-learning learning schedule.
        """
        return(1/(1+self.c_table[observation,action]))

    def increment_count(self,observation,action):
        self.c_table[observation,action] += 1

    def act(self,observation):
        """
        Return the current learned max action for this layer. If there's a tie, pick randomly.
        """
        maximum_actions = np.argwhere(self.q_table[observation] == np.amax(self.q_table[observation])).flatten()
        return(np.random.choice(maximum_actions))