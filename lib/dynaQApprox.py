from typing import Iterable, Tuple

import numpy as np
from policy import ApproximatePolicy
from TileCoding import TileCodingApproximation
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

def get_next_state_action(next_state_q_vals, greedy=False,eps=.1):
    #print("")
    #print("---------------------------------------------------------------------------------")
    #print("getting next state action...")
    #print("Q values are:" + str(next_state_q_vals))
    max_actions = np.argwhere(next_state_q_vals == next_state_q_vals.max()).flatten()
    non_greedy_actions = np.argwhere(next_state_q_vals != next_state_q_vals.max()).flatten()
    if not greedy:
        if np.random.random() <= eps and len(non_greedy_actions) > 0:
            return np.random.choice(non_greedy_actions)
        else:
            return np.random.choice(max_actions)
    else:
        return np.random.choice(max_actions)


def approx_dyna_q(env, alpha, num_steps, n, num_of_episodes=None):
    #References Sutton Book pg. 164

    num_tilings = 0
    tile_width = np.array(
        [0.0, 0.0])  # Initialize tile width to zero, the tile width will be automatically calculated by
    # TileCoding class with respect to the num of tilings.

    Q0 = TileCodingApproximation(np.array([0, 0]), np.array([env.m, env.n]), num_tilings, tile_width, env.k,
                                   env.n * env.m,
                                   calc_tile_width=True,no_tile_coding=True)
    Q1 = TileCodingApproximation(np.array([0, 0]), np.array([env.m, env.n]), num_tilings, tile_width, env.k,
                                   env.n * env.m,
                                   calc_tile_width=True,no_tile_coding=True)
    Q2 = TileCodingApproximation(np.array([0, 0]), np.array([env.m, env.n]), num_tilings, tile_width, env.k,
                                   env.n * env.m,
                                   calc_tile_width=True,no_tile_coding=True)
    Q3 = TileCodingApproximation(np.array([0, 0]), np.array([env.m, env.n]), num_tilings, tile_width, env.k,
                                   env.n * env.m,
                                   calc_tile_width=True,no_tile_coding=True)

    # Q0 is the approximation for q[s][0], etc..
    Q = [Q0, Q1, Q2, Q3]

    num_states = env.spec.nS
    num_actions = env.spec.nA
    gamma = env.spec.gamma
    model = Model(num_states, num_actions)
    previously_visited = []
    pi = ApproximatePolicy(num_actions, Q, eps=0.3)
    if env.final:
        return q, pi

    last_ep_step_count = np.zeros(20)
    avg_step_count = 0
    step_count = 0
    episode_count = 0
    print("doing episode num 0...")
    #for i in tqdm(range(num_steps)):
    features = (env.grid_cell, env.items_status, env.already_visited, env.final)
    # Choose random action to start
    A = 0
    while True:
        #previously_visited.append((S, A))
        #print("----------------------")
        #print("getting next step...")
        *next_state_features, R = env.step(A)
        #model.update(S,A,SP,R,final)
        #print("Current Cell ({},{}) --> A ({})".format(features[0][1],features[0][0],A))
        #print(S)
        #print(SP)
        next_state_q_vals = np.array([Q0(*next_state_features),
             Q1(*next_state_features),
             Q2(*next_state_features),
             Q3(*next_state_features)])
        q_max_next = next_state_q_vals.max()
        next_A = get_next_state_action(next_state_q_vals, greedy=False,eps=0.4)
        #print("")
        G = R + gamma* q_max_next
        #print(" ***** next state,action is ({},{}), G is {}".format(next_state_features[0][1],next_state_features[
        # 0][0],
                                                                    #next_A, G))
        #print("")
        debug = False
        if A == 5:
            debug = True
        Q[A].update(alpha, G, *features,debug=debug)
        features = next_state_features
        A = next_A
        #q[S][A] = q[S][A] + alpha * (R + (gamma * np.amax(q[SP])) - q[S][A])

        '''for j in range(n):
            (sim_state, sim_action) = random.choice(previously_visited)
            sim_next_state,sim_r,_ = model.step(sim_state,sim_action)
            q[sim_state][sim_action] = q[sim_state][sim_action] + alpha * (sim_r + (gamma * np.amax(q[sim_next_state]))

                                       - q[sim_state][
            sim_action])
            update_policy(q, sim_state, pi)
        update_policy(q, SP, pi)'''

        step_count += 1
        avg_step_count += step_count
        final = features[-1]
        while final:
            features = env.reset(random_start_cell=True)
            final = features[-1]
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
                    avg_step_count = last_ep_step_count.sum() / 20
                    print(
                        " Dyna Finished. Epsidoes run: {} Average Steps Per Episode {} - Last Episode Steps: {}".format(
                            episode_count, avg_step_count, last_ep_step_count))
                    return pi
                    #break

    avg_step_count = last_ep_step_count.sum()/20
    print( " Dyna Finished. Epsidoes run: {} Average Steps Per Episode {} - Last Episode Steps: {}".format(episode_count,avg_step_count,last_ep_step_count))
    return pi
