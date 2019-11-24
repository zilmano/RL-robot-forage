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
            return random.randrange(0,self.nA)
        else:
            return np.random.choice(len(self.p[state]), p=self.p[state])

def tabular_dyna_q(grid_world, init_q, alpha, num_steps, n):

    q = init_q
    num_states = grid_world._env_spec.nS
    num_actions = grid_world._env_spec.nA
    gamma = grid_world._env_spec.gamma
    model = np.zeros((num_states,num_actions, 2))
    action_prob = np.zeros((num_states,num_actions))
    grid_world.reset()

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

    for i in range(num_steps):
        S = grid_world.state
        A = pi.action(S)
        (SP,R,final) = grid_world.step(A)

        q[S][A] = q[S][A] + alpha * (R + (gamma * np.amax(q[SP])) - q[S][A])
        model[S][A][0] = R
        model[S][A][1] = SP

        for j in range(n):
            pass


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

        if final:
            grid_world.reset()

    """
    for s in range(env_spec.nS):
        maxQ = np.amax(Q[s])
        maxNotSet = True
        for a in range(env_spec.nA):
            if Q[s][a] == maxQ and maxNotSet:
                action_prob[s][a] = 1
                maxNotSet = False
            else:
                action_prob[s][a] = 0
    pi = EvalPolicy(env_spec.nA, action_prob)

    gamma = env_spec.gamma

    for episode in trajs:
        T = len(episode)
        tau = 0

        for t in range(len(episode)):
            #if tau != T - 1:
                #if t < T:
                    #step = episode[t]
                    #s = step[0]
                    #a = step[1]
                    #r = step[2]
                    #R[t+1] = r
                    #sp = step[3]
                    #S[t+1] = sp
                    #if sp == env_spec.nS - 1:
                        #T = t + 1

            tau = t - n + 1
            if tau >= 0:
                rho = 1.0
                for i in range(tau + 1, min((tau + n), T - 1)):
                    rho = rho * (pi.action_prob(episode[i][0], episode[i][1]) / bpi.action_prob(episode[i][0], episode[i][1]))
                G = 0
                for j in range(tau + 1, min((tau + n), T)):
                    G = G + ((gamma ** (j - tau - 1)) * episode[j][2])
                if tau + n < T:
                    G = G + ((gamma ** (n)) * Q[episode[tau+n][0]][episode[tau+n][1]])

                Q[episode[tau][0]][episode[tau][1]] = Q[episode[tau][0]][episode[tau][1]] + alpha * rho * (G - Q[episode[tau][0]][episode[tau][1]])

                for s in range(env_spec.nS):
                    maxQ = np.amax(Q[s])
                    maxNotSet = True
                    for a in range(env_spec.nA):
                        if Q[s][a] == maxQ and maxNotSet:
                            action_prob[s][a] = 1
                            maxNotSet = False
                        else:
                            action_prob[s][a] = 0
                pi = EvalPolicy(env_spec.nA, action_prob)
    """
    return q, pi
