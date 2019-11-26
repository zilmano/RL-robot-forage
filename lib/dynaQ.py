from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy
import random

class EpsilonGreedyPolicy(Policy):
    def __init__(self,nA,p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)
        self.nA = nA

    def action_prob(self,state,action=None):
        return self.p[state,action]

    def action(self,state):
        if random.random() < .1:
            print("random")
            return random.randrange(0,self.nA)
        else:
            return np.random.choice(len(self.p[state]), p=self.p[state])

def update_policy(q, num_states, num_actions):
    action_prob = np.zeros((num_states,num_actions))
    for s in range(num_states):
        max_q = np.amax(q[s])
        max_not_set = True
        for a in range(num_actions):
            if q[s][a] == max_q and max_not_set:
                action_prob[s][a] = 1
                max_not_set = False
            else:
                action_prob[s][a] = 0
    pi = EpsilonGreedyPolicy(num_actions, action_prob)
    return pi

def tabular_dyna_q(grid_world, init_q, alpha, num_steps, n):
    #References Sutton Book pg. 164

    q = init_q
    num_states = grid_world._env_spec.nS
    num_actions = grid_world._env_spec.nA
    gamma = grid_world._env_spec.gamma
    model = np.zeros((num_states,num_actions, 2))
    previously_visited = []
    pi = update_policy(q, num_states, num_actions)

    for i in range(num_steps):
        S = grid_world.state
        A = pi.action(S)
        previously_visited.append((S,A))
        (SP,R,final) = grid_world.step(A)

        q[S][A] = q[S][A] + alpha * (R + (gamma * np.amax(q[SP])) - q[S][A])
        model[S][A][0] = R
        model[S][A][1] = SP

        for j in range(n):
            (ps,pa) = random.choice(previously_visited)
            pr = model[ps][pa][0]
            psp = int(model[ps][pa][1])
            q[ps][pa] = q[ps][pa] + alpha * (pr + (gamma * np.amax(q[psp])) - q[ps][pa])
        pi = update_policy(q, num_states, num_actions)

        if final:
            grid_world.reset()

    return q, pi
