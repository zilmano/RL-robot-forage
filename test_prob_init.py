import sys
sys.path.append("./lib")
import numpy as np
from TileCoding import TileCodingGridWorldWItems
from dynaQ import tabular_dyna_q
import utility as util
import policy
from GridWorldEnv import GridWorld, Item, Actions
import math
from tqdm import tqdm


def exec_policy_for_episode(env,pi,max_out_steps=math.inf):
    steps = 0
    final = False
    print("start state:{}".format(env.state))
    final = env.final
    while not final and steps <= max_out_steps:
        a = pi.action(env.state)
        (s, r, final) = env.step(a)
        #print("a {} --> s {}".format(a,s))
        steps += 1
    return steps


def execute_policy(env, start_state, pi, num_of_episodes, max_out_steps=math.inf):
    total_steps = 0
    i = 0
    while i < num_of_episodes:
        env.reset(start_state=start_state)
        steps = exec_policy_for_episode(env,pi,max_out_steps)
        total_steps += steps
        #visualizeGrid(env)
        i += 1
    avg_steps = total_steps/num_of_episodes
    return avg_steps

def visualizeGridValueFunc(gridWorldModel, items_status=0):
   util.visualizeGridTxt(gridWorldModel, gridWorldModel.V, items_status)

def visualizeGridProbabilities(gridWorldModel, k, aggregate = False):

    if not aggregate:
        for i in range(0, k):
            util.visualizeGridTxt(gridWorldModel, gridWorldModel.item_loc_probabilities[i])
    else:
        util.visualizeGridTxt(gridWorldModel,np.sum(gridWorldModel.item_loc_probabilities,axis=0))


n = 4
m = 4
k = 2
#gw = GridWorld(m, n, k, debug=False)
#visualizeGridProbabilities(gw, k)

#Q = np.zeros((gridWorldModel._env_spec.nS,gridWorldModel._env_spec.nA))
#dyna_model_training_steps = 50
#learning_rate = 0.1
#q, pi = tabular_dyna_q(gridWorldModel, Q, learning_rate, training_steps, model_training_steps)

sweep_pi = policy.HandMadeSweepPolicy(4, m, n)
episodes_num = 100
start_state = 0
sweep_steps = 0
nn_tour_expected_steps = 0
#for i in tqdm(range(episodes_num)):
gw = GridWorld(m, n, k, debug=True)
visualizeGridProbabilities(gw, k, aggregate=True)
base_line_tour, nn_tour_expected_steps = gw.graph.get_approximate_best_path(start_vertex=m-1)
print("nearest_neighbor_tour:" + str(base_line_tour))

for i in range(0,episodes_num):
    print("inst world model...")
    gw.reset(start_cell=m-1)
    visualizeGridValueFunc(gw)
    print("exec sweep policy for episode...")
    sweep_steps += exec_policy_for_episode(gw,sweep_pi)
    print("get nearest neighbor tour...")
    print("get nn tour cost...")
    #nn_tour_expected_steps += gw.graph.calc_path_cost(base_line_tour)

#$avg_nn_steps = nn_tour_expected_steps/episodes_num
avg_nn_steps = nn_tour_expected_steps
avg_sweep_steps = sweep_steps/episodes_num

print("avg_sweep={} avg_nearest_neigbor={}".format(avg_sweep_steps,avg_nn_steps))



