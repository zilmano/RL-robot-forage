import sys
sys.path.append("./lib")
import numpy as np
import utility as util
import policy
from TileCoding import TileCodingGridWorldWItems
from GridWorldEnv import GridWorld, Item, Actions

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
    testRandomPolicy(gridWorldModel)

    # Example initialization of TileCoding
    num_tilings = 6
    tile_width = np.array([0.0,0.0]) # Initialize tile width to zero, the tile width will be automatically calculated by
                                 # TileCoding class with respect to the num of tilings.
    tc = TileCodingGridWorldWItems(np.array([0,0]),np.array([m,n]),num_tilings,tile_width,k,calc_tile_width=True)


