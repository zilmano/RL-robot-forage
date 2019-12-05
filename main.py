import sys
sys.path.append("./lib")
import numpy as np
import math
import copy
import utility as util
import policy
from TileCoding import TileCodingApproximation
from GridWorldEnv import GridWorld, Item, Actions
from dynaQ import tabular_dyna_q
import MonteCarloControl as mc
from policy import PolicyType

def test1(gridWorldEnv):
    # Oleg: testing the model with some sequence of steps.
    (s, r, *_) = gridWorldEnv.step(3)
    print("s:{} r:{}".format(s, r))
    (s, r, *_) = gridWorldEnv.step(3)
    print("s:{} r:{}".format(s, r))
    (s, r, *_) = gridWorldEnv.step(0)
    print("s:{} r:{}".format(s, r))
    (s, r, *_) = gridWorldEnv.step(2)
    print("s:{} r:{}".format(s, r))
    (s, r, *_) = gridWorldEnv.step(1)
    print("s:{} r:{}".format(s, r))
    util.logmsg(" ")
    util.logmsg("-----   reward funtion at state 5 -----")
    '''for i in range(0, gridWorldEnv.spec.nS):
        util.logmsg("state:{}\n   {}".format(i, gridWorldEnv.r_mat[i]))'''
    util.logmsg("\n")
    (s, r, *_) = gridWorldEnv.step(2)
    print("s:{} r:{}".format(s, r))
    (s, r, *_) = gridWorldEnv.step(3)
    print("s:{} r:{}".format(s, r))
    util.logmsg(" ")
    util.logmsg("-----   reward funtion at state 5 -----")
    '''for i in range(0, gridWorldEnv.spec.nS):
        util.logmsg("state:{}\n   {}".format(i, gridWorldEnv.r_mat[i]))
    util.logmsg("\n")'''
    (s, r, f) = gridWorldEnv.step(0)
    print("s:{} r:{} f:{}".format(s, r, f))

def testRandomPolicy(gridWorldModel):
    # Run two episodes with a random policy
    pi = policy.NewPolicy(gridWorldModel.spec.nA, gridWorldModel.spec.nS)
    i = 0

    visualizeGridValueFunc(gridWorldModel)
    while i < 2:
        a = pi.action(gridWorldModel.state)
        (s,r,final) = gridWorldModel.step(a)
        if final:
            gridWorldModel.reset()
            visualizeGridValueFunc(gridWorldModel)
            i += 1

def exec_policy_for_episode(env,pi,max_out_steps=math.inf):
    steps = 0
    final = False
    #print("start state:{}".format(env.state))
    final = env.final
    while not final and steps <= max_out_steps:
        a = pi.action(env.state,greedy=False)
        (s, r, final) = env.step(a)
        #print("a {} --> s {}".format(a,s))
        steps += 1
    return steps

def testDynaQ(gridWorldModel):
    # Run two episodes with a DynaQ policy
    Q = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    training_steps = 10000000
    model_training_steps = 50
    learning_rate = 0.1
    q, pi = tabular_dyna_q(gridWorldModel, Q, learning_rate, training_steps, model_training_steps, num_of_episodes=1200)
    gridWorldModel.setQ(q,pi)
    #visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n)
    #visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n, item_status=1)
    #visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n, item_status=2)
    #visualizeGridValueFunc(gridWorldModel)
    print(q)
    return pi

def testMonteCarlo(gw):
    Q = np.zeros((gridWorldModel.spec.nS, gridWorldModel.spec.nA))
    randomPi = policy.NewPolicy(gridWorldModel.spec.nA, gridWorldModel.spec.nS)
    sim = util.Simulator(gw, randomPi)
    eps = 0.01
    training_episodes = 10000

    (Q, V, pi) = mc.on_policy_mc_control(Q, eps, sim, training_episodes)
    gw.setQ = Q

    visualizeGridPolicy(pi, gw.m, gw.n, policy_type=PolicyType.e_soft, eps=eps)
    visualizeGridValueFunc(gw)
    return pi

def compareToBaseLine(gw,eval_pi,k):
    sweep_pi = policy.HandMadeSweepPolicy(4, m, n)
    episodes_num = 100
    sweep_steps = 0
    rl_steps = 0

    visualizeGridProbabilities(gw, k, aggregate=True)
    base_line_tour, nn_tour_expected_steps = gw.graph.get_approximate_best_path(start_vertex=m - 1)
    #print("nearest_neighbor_tour:" + str(base_line_tour))

    for i in range(0, episodes_num):
        #print("inst world model...")
        gw.reset(start_cell=(m - 1))
        gw_twin = copy.deepcopy(gw)
        #visualizeGridValueFunc(gw)
        #print("exec sweep policy for episode...")
        sweep_steps += exec_policy_for_episode(gw, sweep_pi)
        rl_steps += exec_policy_for_episode(gw_twin, eval_pi)
        #print("rl steps" + str(rl_steps))
        #print("sweep steps" + str(sweep_steps))
        # nn_tour_expected_steps += gw.graph.calc_path_cost(base_line_tour)
    avg_nn_steps = nn_tour_expected_steps
    avg_sweep_steps = sweep_steps / episodes_num
    avg_rl_steps = rl_steps/episodes_num
    print("avg_sweep={} avg_rl={} avg_nearest_neigbor={}".format(avg_sweep_steps, avg_rl_steps,avg_nn_steps))


def visualizeGridPolicy(pi, m, n, item_status=0,policy_type=PolicyType.greedy,eps=0.1):
    util.visualizePolicyTxt(pi, m, n, item_status,policy_type=policy_type,eps=eps)


def visualizeGridValueFunc(gridWorldModel):
    util.visualizeGridTxt(gridWorldModel,gridWorldModel.V)


def visualizeGridProbabilities(gridWorldModel, k, aggregate=False):

    if not aggregate:
        for i in range(0, k):
            util.visualizeGridTxt(gridWorldModel, gridWorldModel.item_loc_probabilities[i])
    else:
        util.visualizeGridTxt(gridWorldModel,np.sum(gridWorldModel.item_loc_probabilities,axis=0))


if __name__ == "__main__":
    util.openlog('log.txt')

    # Intitalize 4x4 gridworld with 2 items
    n = 8
    m = 8
    k = 2
    # Run for 15 different distributions. Train RL, and then compare on 100 episodes each.
    for i in range(0,15):
        gridWorldModel = GridWorld(m,n,k,debug=False, gamma=1, no_stochastisity=False)
        #visualizeGridValueFunc(gridWorldModel)
        visualizeGridProbabilities(gridWorldModel, k, aggregate=True)

        # Testing
        # testRandomPolicy(gridWorldModel)
        eval_pi = testDynaQ(gridWorldModel)
        #test1(gridWorldModel)
        #mc_pi = testMonteCarlo(gridWorldModel)
        compareToBaseLine(gridWorldModel,eval_pi, k)

    # Example initialization of TileCoding
    '''num_tilings = 6
    tile_width = np.array([0.0,0.0]) # Initialize tile width to zero, the tile width will be automatically calculated by
                                 # TileCoding class with respect to the num of tilings.
    tc = TileCodingGridWorldWItems(np.array([0,0]),np.array([m,n]),num_tilings,tile_width,k,n*m,calc_tile_width=True)'''


