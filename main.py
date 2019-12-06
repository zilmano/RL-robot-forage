import sys
sys.path.append("./lib")
import numpy as np
import math
import copy
import utility as util
import policy
from TileCoding import TileCodingApproximation
from GridWorldEnv import GridWorld, Item, Actions
from ExampleEnv import GridWorldPage60
from dynaQ import tabular_dyna_q
import MonteCarloControl as mc
from policy import PolicyType
import matplotlib.pyplot as plt

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
    Q = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    training_steps = 10000000
    model_training_steps = 50
    learning_rate = 0.1

    q, pi, episode_steps = tabular_dyna_q(gridWorldModel, Q, learning_rate, training_steps, model_training_steps, num_of_episodes=1000, eps=0.3)
    gridWorldModel.setQ(q,pi)
    #visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n)
    #visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n, item_status=1)
    #visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n, item_status=2)
    #visualizeGridValueFunc(gridWorldModel)

    #gridWorldModel.heatMap()

    print(q)
    return pi

def parameterTest(gridWorldModel):
    Q1 = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    Q2 = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    Q3 = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    Q4 = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    Q5 = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    Q6 = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    Q7 = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    Q8 = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    Q9 = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))

    training_steps = 10000000
    #model_training_steps = 50
    learning_rate = 0.1

    q1, pi1, episode_steps1 = tabular_dyna_q(gridWorldModel, Q1, learning_rate, training_steps, 50, num_of_episodes=1000, eps=0.1)
    q2, pi2, episode_steps2 = tabular_dyna_q(gridWorldModel, Q2, learning_rate, training_steps, 50, num_of_episodes=1000, eps=0.2)
    q3, pi3, episode_steps3 = tabular_dyna_q(gridWorldModel, Q3, learning_rate, training_steps, 50, num_of_episodes=1000, eps=0.3)
    #q4, pi4, episode_steps4 = tabular_dyna_q(gridWorldModel, Q4, learning_rate, training_steps, 70, num_of_episodes=100, eps=0.1)
    #q5, pi5, episode_steps5 = tabular_dyna_q(gridWorldModel, Q5, learning_rate, training_steps, 70, num_of_episodes=100, eps=0.2)
    #q6, pi6, episode_steps6 = tabular_dyna_q(gridWorldModel, Q6, learning_rate, training_steps, 70, num_of_episodes=100, eps=0.3)
    #q7, pi7, episode_steps7 = tabular_dyna_q(gridWorldModel, Q7, learning_rate, training_steps, 100, num_of_episodes=100, eps=0.1)
    #q8, pi8, episode_steps8 = tabular_dyna_q(gridWorldModel, Q8, learning_rate, training_steps, 100, num_of_episodes=100, eps=0.2)
    #q9, pi9, episode_steps9 = tabular_dyna_q(gridWorldModel, Q9, learning_rate, training_steps, 100, num_of_episodes=100, eps=0.3)

    eps = range(len(episode_steps1))
    plt.plot(eps, episode_steps1)
    plt.plot(eps, episode_steps2)
    plt.plot(eps, episode_steps3)
    #plt.plot(eps, episode_steps4)
    #plt.plot(eps, episode_steps5)
    #plt.plot(eps, episode_steps6)
    #plt.plot(eps, episode_steps7)
    #plt.plot(eps, episode_steps8)
    #plt.plot(eps, episode_steps9)

    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.show()

    #gridWorldModel.setQ(q,pi)
    #visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n)
    #visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n, item_status=1)
    #visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n, item_status=2)
    #visualizeGridValueFunc(gridWorldModel)
    #gridWorldModel.heatMap()
    #print(q)
    #return pi

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
    return avg_nn_steps, avg_sweep_steps, avg_rl_steps


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

    nn_avgs = []
    sweep_avgs = []
    dyna_avgs = []
    # Run for 10 different distributions. Train RL, and then compare on 100 episodes each.
    for i in range(0,10):
        gridWorldModel = GridWorld(m,n,k,debug=False, gamma=1, no_stochastisity=False)
        #visualizeGridValueFunc(gridWorldModel)
        visualizeGridProbabilities(gridWorldModel, k, aggregate=True)

        # Testing
        # testRandomPolicy(gridWorldModel)
        eval_pi = testDynaQ(gridWorldModel)
        #parameterTest(gridWorldModel)
        #test1(gridWorldModel)
        #mc_pi = testMonteCarlo(gridWorldModel)
        (nn_avg, sweep_avg, dyna_avg) = compareToBaseLine(gridWorldModel,eval_pi, k)
        nn_avgs.append(nn_avg)
        sweep_avgs.append(sweep_avg)
        dyna_avgs.append(dyna_avg)

    experiment_nums = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
    y_pos = np.arange(len(experiment_nums))

    bar_width = 0.2

    rects1 = plt.bar(y_pos, nn_avgs, bar_width,
    color='b',
    label='Nearest Neighbor')

    rects2 = plt.bar(y_pos + bar_width, sweep_avgs, bar_width,
    color='g',
    label='Prioritized Sweep')

    rects3 = plt.bar(y_pos + 2*bar_width, dyna_avgs, bar_width,
    color='r',
    label='DynaQ')

    #averages = [nn_avgs, sweep_avgs, dyna_avgs]

    plt.xticks(y_pos+bar_width, experiment_nums)
    plt.ylabel('Average Number of Steps')
    plt.xlabel('Experiment Number')
    plt.title('Average Number of Steps per Algorithm')
    plt.legend()
    plt.show()

    # Example initialization of TileCoding
    '''num_tilings = 6
    tile_width = np.array([0.0,0.0]) # Initialize tile width to zero, the tile width will be automatically calculated by
                                 # TileCoding class with respect to the num of tilings.
    tc = TileCodingGridWorldWItems(np.array([0,0]),np.array([m,n]),num_tilings,tile_width,k,n*m,calc_tile_width=True)'''
