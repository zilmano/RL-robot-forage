from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy
from policy import NewPolicy
import random
from tqdm import tqdm
import utility as util
from matplotlib.pyplot import *


class EpsilonGreedyPolicy(Policy):
    def __init__(self, nA, p=None, eps=.1):
        self.p = p if p is not None else np.array([1/nA]*nA)
        self.nS = p.shape[0]
        self.nA = nA
        self.eps = eps

    def action_prob(self,state,action=None):
        return self.p[state,action]

    def action(self,state):
        if random.random() < self.eps:
            #print("random")
            return random.randrange(0,self.nA)
        else:
            return np.random.choice(len(self.p[state]), p=self.p[state])

    @property
    def P(self):
        return self.p

def update_policy(q, num_states, num_actions):
    action_prob = np.zeros((num_states, num_actions))
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


def tabular_dyna_q(grid_world, init_q, alpha, num_steps, n, one_episode=False):
    #References Sutton Book pg. 164

    q = init_q
    num_states = grid_world.spec.nS
    num_actions = grid_world.spec.nA
    gamma = grid_world.spec.gamma
    model = np.zeros((num_states, num_actions, 2))
    previously_visited = []
    pi = update_policy(q, num_states, num_actions)
    if grid_world.final:
        return q, pi

    last_ep_step_count = np.zeros(20)
    avg_step_count = 0
    step_count = 0
    episode_count = 0
    episode_steps = []

    for i in tqdm(range(num_steps)):
        S = grid_world.state
        A = pi.action(S)
        previously_visited.append((S, A))
        (SP, R, final) = grid_world.step(A)

        #print(S)
        #print(SP)
        q[S][A] = q[S][A] + alpha * (R + (gamma * np.amax(q[SP])) - q[S][A])
        model[S][A][0] = R
        model[S][A][1] = SP

        for j in range(n):
            (ps,pa) = random.choice(previously_visited)
            pr = model[ps][pa][0]
            psp = int(model[ps][pa][1])
            q[ps][pa] = q[ps][pa] + alpha * (pr + (gamma * np.amax(q[psp])) - q[ps][pa])
        pi = update_policy(q, num_states, num_actions)

        step_count += 1
        avg_step_count += step_count
        while final:
            (test, final) = grid_world.reset(random_start_cell=True)
            last_ep_step_count[episode_count%20] = step_count
            episode_steps.append(step_count)
            step_count = 0
            episode_count += 1
            #print(test)
            #util.visualizeGridTxt(grid_world, grid_world.V)
            if one_episode:
                return q, pi

    avg_step_count = last_ep_step_count.sum()/20
    print( " Dyna Finished. Epsidoes run: {} Average Steps Per Episode {} - Last Episode Steps: {}".format(episode_count,avg_step_count,last_ep_step_count))

    #Steps per episode graph
    eps = range(len(episode_steps))
    plot(eps, episode_steps)
    xlabel('Episodes')
    ylabel('Steps per Episode')
    show()

    return q, pi
