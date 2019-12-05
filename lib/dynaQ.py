from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy
from policy import NewPolicy
import random
from tqdm import tqdm
import utility as util


class Model:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.trans_mat = np.zeros([nS,nA,nS])
        self.r_mat = np.zeros([nS,nA,nS])
        self.state_and_action_num = np.zeros([nS,nA])
        self.final_state = None
        #print("nA {} nS{} state_and_act {}".format(self.nA,self.nS,self.state_and_action_num.shape))

    def update(self,s,a,s_next,r,is_final):
        #print(" {} {} {} {} {}".format(s,a,s_next,r,is_final))
        self.state_and_action_num[s][a] += 1
        self.trans_mat[s][a][s_next] += 1
        self.r_mat[s][a][s_next] = r
        if is_final:
            assert self.final_state is None or s_next == self.final_state, "there could be only one final state in " \
                                                                           "the environmnet"
            self.final_state = s_next

    def getDistributionForAction(self,s,a):
        return self.trans_mat[s][a]/self.state_and_action_num

    def step(self,s,a):
        s_next =  np.random.choice(self.nS, p=(self.trans_mat[s][a]/self.state_and_action_num[s][a]))
        r = self.r_mat[s][a][s_next]
        is_final = 0
        if s_next == self.final_state:
            is_final = 1
        return s_next, r, is_final

class EpsilonGreedyPolicy(Policy):
    def __init__(self, nA, nS, p=None, eps=.1):
        self.p = p if p is not None else np.ones([nS, nA])*(1/nA)
        self.nS = self.p.shape[0]
        self.nA = nA
        self.eps = eps

    def action_prob(self,state,action=None):
        return self.p[state,action]

    def action(self,state, greedy=False):
        if random.random() < self.eps and not greedy:
            #print("random")
            return random.randrange(0,self.nA)
        else:
            return np.random.choice(len(self.p[state]), p=self.p[state])

    def update(self,state,new_prob_distribution):
        self.p[state] = new_prob_distribution

    @property
    def P(self):
        return self.p

def update_policy(q,s,pi):
    action_prob = np.zeros(pi.nA)
    max_q = np.amax(q[s])
    #max_not_set = False
    n = 0
    for a in range(pi.nA):
        if q[s][a] == max_q:
            n += 1
            action_prob[a] = 1
        else:
            action_prob[a] = 0
    pi.update(s, action_prob/n)


'''def update_policy(q, num_states, num_actions):
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
    return pi'''


def tabular_dyna_q(grid_world, init_q, alpha, num_steps, n, num_of_episodes=None):
    #References Sutton Book pg. 164

    q = init_q
    num_states = grid_world.spec.nS
    num_actions = grid_world.spec.nA
    gamma = grid_world.spec.gamma
    model = Model(num_states, num_actions)
    previously_visited = []
    pi = EpsilonGreedyPolicy(num_actions, num_states,eps=0.3)
    if grid_world.final:
        return q, pi

    last_ep_step_count = np.zeros(20)
    avg_step_count = 0
    step_count = 0
    episode_count = 0
    print("doing episode num 0")
    #for i in tqdm(range(num_steps)):
    for i in range(0,num_steps):
        S = grid_world.state
        A = pi.action(S)
        previously_visited.append((S, A))
        (SP, R, final) = grid_world.step(A)
        model.update(S,A,SP,R,final)
        #print(S)
        #print(SP)
        q[S][A] = q[S][A] + alpha * (R + (gamma * np.amax(q[SP])) - q[S][A])
        update_policy(q,S,pi)
        for j in range(n):
            (sim_state, sim_action) = random.choice(previously_visited)
            sim_next_state,sim_r,_ = model.step(sim_state,sim_action)
            q[sim_state][sim_action] = q[sim_state][sim_action] + alpha * (sim_r + (gamma * np.amax(q[sim_next_state]))

                                       - q[sim_state][
            sim_action])
            update_policy(q, sim_state, pi)
        update_policy(q, SP, pi)

        #pi = update_policy(q, num_states, num_actions)

        step_count += 1
        avg_step_count += step_count
        while final:
            (test, final) = grid_world.reset(random_start_cell=True)
            if not final:
                if episode_count % 100 == 0:
                    print("   episode steps:" + str(step_count))
                    print("doing episode num {}".format(episode_count))
                last_ep_step_count[episode_count%20] = step_count
                step_count = 0
                episode_count += 1
                #print(test)
                #util.visualizeGridTxt(grid_world, grid_world.V)
                if episode_count == num_of_episodes:
                    return q, pi

    avg_step_count = last_ep_step_count.sum()/20
    print( " Dyna Finished. Epsidoes run: {} Average Steps Per Episode {} - Last Episode Steps: {}".format(episode_count,avg_step_count,last_ep_step_count))
    return q, pi
