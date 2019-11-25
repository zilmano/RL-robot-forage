import sys
sys.path.append("./lib")
import utility as util
import policy
import numpy as np
from GridWorldEnv import GridWorld, Item, Actions
from dynaQ import tabular_dyna_q

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

    visualizeGrid(gridWorldModel)
    while i < 2:
        a = pi.action(gridWorldModel.state)
        (s,r,final) = gridWorldModel.step(a)
        if final:
            gridWorldModel.reset()
            visualizeGrid(gridWorldModel)
            i += 1

def testDynaQ(gridWorldModel):
    # Run two episodes with a DynaQ policy
    Q = np.zeros((gridWorldModel._env_spec.nS,gridWorldModel._env_spec.nA))
    q, pi = tabular_dyna_q(gridWorldModel, Q, 0.1, 1000, 50)
    i = 0

    visualizeGrid(gridWorldModel)
    while i < 2:
        a = pi.action(gridWorldModel.state)
        (s,r,final) = gridWorldModel.step(a)
        if final:
            gridWorldModel.reset()
            visualizeGrid(gridWorldModel)
            i += 1

def visualizeGrid(gridWorldModel):
   util.visualizeGridTxt(gridWorldModel,gridWorldModel.V)


if __name__ == "__main__":
    util.openlog('log.txt')

    # Intitalize 4x4 gridworld with 2 items
    n = 4
    m = 4
    k = 2
    gridWorldModel = GridWorld(m,n,k,debug=True)

    # Testing
    #testRandomPolicy(gridWorldModel)
    testDynaQ(gridWorldModel)
    #test1(gridWorldModel)
