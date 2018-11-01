import numpy as np
from pyfiglet import figlet_format

# ADDF Classes
from delayed_mcesp import MCESP_D
from crop_simulate import CropField

"""
Test of the ADDF. We consider a fast and a slow agent, where the fast agent collects all sectors' information once every 3 days, and the slow agent can examine one sector a day.
"""

class ADDF:
    """
    -------------------------------------------------------------------------------------------------
    Initialization
    -------------------------------------------------------------------------------------------------
    """
    def __init__(self, sectors, observations, delays):
        self.fast = MCESP_D(observations)
        self.slow = MCESP_D(observations)
        self.simulator = CropField(sectors, observations)
        self.max_wait_fast = delays[0] # Delay between processing sectors for the fast agent (the slow agent acts once a day)
        self.max_wait_slow = delays[1]
        self.fast_wait = 0 # Current wait period until next observation/action period
        self.slow_wait = 0
        self.slow_queue = [] # Queue of sectors for the slow agent
        self.accuracy = np.zeros((2,4)) # Accuracy metrics for both agents
        print("-------------------------------------------------------------------------------------------------")
        print(figlet_format('ADDF', font='starwars'))
        print("Created a simulated environment with "+str(sectors)+" sectors, "+str(observations)+" observations, and:")
        print("\tThe fast agent acts every "+str(self.max_wait_fast)+" days")
        print("\tThe slow agent acts every "+str(self.max_wait_slow)+" days")
        print("-------------------------------------------------------------------------------------------------")

    """
    -------------------------------------------------------------------------------------------------
    Simulation steps
    -------------------------------------------------------------------------------------------------
    """
    def simulate_day(self):
        day = self.simulator.day
        print("Day "+str(self.simulator.day+1)+" sectors: [ "+' '.join([self.p_s(s) for s in self.simulator.cur_sectors])+" ]")

        if self.fast_wait == 0: # Fast agent acts!
            self.fast_wait = self.max_wait_fast
            print("\tFast agent acts!")

            fast_o = self.simulator.observe_sectors(2)
            print("\t\tFast agent observes sectors: "+str(fast_o))

            a = [self.fast.act(a) for a in fast_o]
            print("\t\tFast agent chooses actions for sectors: [ "+' '.join([self.p_a(act) for act in a])+" ]")

            prioritized_observations = list(sorted(zip(zip(fast_o,a),list(range(0,len(a)))), key=lambda x: -1*x[0][1]))
            ctas = [x for x in prioritized_observations if x[0][1]>0]
            extras = [x for x in prioritized_observations if x[0][1]==0]
            print("\t\tFast agent creates CTAs on sectors "+self.p_c(ctas)+" (omitting "+self.p_c(extras)+")")

            self.slow_queue += ctas

        if self.slow_wait == 0 and len(self.slow_queue)>0: # Slow agent acts!
            self.slow_wait = self.max_wait_slow
            print("\tSlow agent acts!")

            o_a, sector = self.slow_queue.pop(0) # Grab the next sector to examine
            slow_o = self.simulator.observe_sectors(1)[sector] # Pull observation for requested sector
            true_state = self.simulator.cur_sectors[sector]
            print("\t\tSlow agent observes sector "+str(sector)+": "+str(slow_o))

            a = self.slow.act(slow_o)
            print("\t\tSlow agent executes action: "+self.p_a(a))

            # Learning conditions
            if a == 0: # Slow agent considers fast agent incorrect
                print("\t\tSlow agent detects no stress")

                print("\tFast agent learns!")
                reward = 1 - o_a[1]/2
                self.fast.update_reward(o_a[0],o_a[1],reward)
                print("\t\tAction "+self.p_a(o_a[1])+" for observation "+str(o_a[0])+" gets reward "+str(reward))
            else:
                print("\t\tSlow agent predicts a stress")

                print("\tSlow agent learns!")
                if true_state == 0:
                    print("\t\tNo stress exists")
                    reward = 1 - a/2
                    self.slow.update_reward(slow_o,a,reward)
                    print("\t\tAction "+self.p_a(a)+" for observation "+str(slow_o)+" gets reward "+str(reward))

                    print("\tFast agent learns!")
                    reward = 1 - o_a[1]/2
                    self.fast.update_reward(o_a[0],o_a[1],reward)
                    print("\t\tAction "+self.p_a(o_a[1])+" for observation "+str(o_a[0])+" gets reward "+str(reward))
                else:
                    print("\t\tStress identified")
                    reward = a/2
                    self.slow.update_reward(slow_o,a,reward)
                    print("\t\tAction "+self.p_a(a)+" for observation "+str(slow_o)+" gets reward "+str(reward))
                    
                    print("\tFast agent learns!")
                    reward = o_a[1]/2
                    self.fast.update_reward(o_a[0],o_a[1],reward)
                    print("\t\tAction "+self.p_a(o_a[1])+" for observation "+str(o_a[0])+" gets reward "+str(reward))

        self.fast_wait -= 1
        self.slow_wait -= 1
        self.simulator.iterate_states()

    """
    -------------------------------------------------------------------------------------------------
    Pretty printing
    -------------------------------------------------------------------------------------------------
    """
    def p_s(self,s):
        if s == 0:
            return("-")
        else:
            return("S")

    def p_a(self,a):
        if a == 0:
            return("-")
        if a == 1:
            return("L")
        return("H")

    def p_c(self,p):
        if len(p) == 0:
            return('none')
        return("[ "+' '.join([str(x[1]) for x in p])+" ]")
    
    """
    -------------------------------------------------------------------------------------------------
    Testing
    -------------------------------------------------------------------------------------------------
    """
    def test(self):
        for i in range(0,24):
            self.simulate_day()