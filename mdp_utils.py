import mdptoolbox
import numpy as np

def samplefrom(distribution):
    return (np.random.choice(len(distribution), 1, p=distribution))[0]

def playtransition(mdp, state, action):
        nextstate = samplefrom(mdp.P[action][state])
        return nextstate, mdp.R[action][state]
