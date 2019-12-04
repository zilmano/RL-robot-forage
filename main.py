import sys
sys.path.append("./lib")
import numpy as np
import utility as util
import policy
from TileCoding import TileCodingGridWorldWItems
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

def testDynaQ(gridWorldModel):
    # Run two episodes with a DynaQ policy
    Q = np.zeros((gridWorldModel.spec.nS,gridWorldModel.spec.nA))
    training_steps = 20000
    model_training_steps = 50
    learning_rate = 0.1
    q, pi = tabular_dyna_q(gridWorldModel, Q, learning_rate, training_steps, model_training_steps,one_episode=False)
    gridWorldModel.setQ(q,pi)
    visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n)
    visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n, item_status=1)
    visualizeGridPolicy(pi, gridWorldModel.m, gridWorldModel.n, item_status=2)
    visualizeGridValueFunc(gridWorldModel)
    print(q)

def testMonteCarlo(gw):
    Q = np.zeros((gridWorldModel.spec.nS, gridWorldModel.spec.nA))
    training_steps = 10000
    model_training_steps = 50
    learning_rate = 0.3
    randomPi = policy.NewPolicy(gridWorldModel.spec.nA, gridWorldModel.spec.nS)
    # if bPi is None:
    #    bPi = randomPi
    # evalPi = policy.NewPolicy(gridWorldModel.spec.nA, gridWorldModel.spec.nS)

    sim = mc.Simulation(gw, randomPi)
    eps = 0.2
    training_episodes = 1000

    (Q, V, pi) = mc.on_policy_mc_control(Q, eps, sim, training_episodes)
    gw.setQ = Q
    visualizeGridPolicy(pi, gw.m, gw.n, policy_type=PolicyType.e_soft, eps=eps)
    visualizeGridValueFunc(gw)

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
    n = 4
    m = 4
    k = 2
    gridWorldModel = GridWorld(m,n,k,debug=False, gamma=0.99, no_stochastisity=False)
    visualizeGridValueFunc(gridWorldModel)
    visualizeGridProbabilities(gridWorldModel, k, aggregate=True)

    # Testing
   # testRandomPolicy(gridWorldModel)
    testDynaQ(gridWorldModel)
    #test1(gridWorldModel)
    testMonteCarlo(gridWorldModel)

    # Example initialization of TileCoding
    num_tilings = 6
    tile_width = np.array([0.0,0.0]) # Initialize tile width to zero, the tile width will be automatically calculated by
                                 # TileCoding class with respect to the num of tilings.
    tc = TileCodingGridWorldWItems(np.array([0,0]),np.array([m,n]),num_tilings,tile_width,k,calc_tile_width=True)


