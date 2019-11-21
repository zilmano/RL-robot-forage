from env import Env,EnvSpec
import string
from enum import IntEnum
import numpy as np
import utility as util

class Actions(IntEnum):
    west  = 0
    north = 1
    east  = 2
    south = 3

class Item:
    def __init__(self,mean,variance,name,m,n):
        self.mean = mean
        self.variance = variance
        self.state = None
        self.name = name
        self.m = m
        self.n = n

    def getRowAndCol(self):
        return ((self.mean % m), (item.mean / m))

    def setStateFromRowAndCol(self,row,col):
        self.state = int(col*self.m + row)

class GridWorld(Env):  # MDP introduced at Fig 5.4 in Sutton Book
    def __init__(self,m,n,k,debug=False):
        # 20x20 grid world
        STATE_SPACE_SIZE = m*n
        self.m = m
        self.n = n
        assert self.n*self.m == STATE_SPACE_SIZE, "nS does not match n and m. m*n should give nS"
        # Number of items to search
        self.k = k
        self.items = list()
        self.debug = debug
        self._final_state = False
        self._state = None
        self.trans_mat = None
        self.rmat = None
        # nS states of mxn grid. State numbering is by columns starting from top-left. i.e:
        # top-row, leftmost-col state = 0, bottom-leftmost state = m-1, top-row,second from left col = m
        # 2nd from top row, 2nd from left colunm = m+1.  etc...

        # Actions:
        # 0 -west, 1 -north, 2-east, 3 -south

        # Gamma:
        GAMMA = 0.90
        env_spec = EnvSpec(STATE_SPACE_SIZE, len(Actions), GAMMA)
        super().__init__(env_spec)

        self._V = np.zeros(self.spec.nS)
        self._Q = np.zeros([self.spec.nS, self.spec.nA])

        self._generateItems()
        self.reset()
        '''for i in range(0,self.spec.nS):
            util.logmsg("state:{}\n   {}".format(i,self.r_mat[i]))'''


    def _generateItems(self):
        for i in range(0,self.k):
            mean = np.random.randint(low=0, high=self.spec.nS, size=1)[0]
            variance = np.random.randint(low=2,high=5,size=1)[0]
            self.items.append(Item(mean,variance,string.ascii_lowercase[i],self.m,self.n))

    def _placeItems(self):
        self.item_locations =  [[[] for i in range(0, self.n)] for i in range(0, self.m)]
        for item in self.items:
            (mean_x,mean_y) = self._getRowColFromState(item.mean)
            ((x,y),) = np.random.multivariate_normal([mean_x,mean_y],[[item.variance,0],[0,item.variance]],1)
            x = self.m - 1 if x > self.m - 1 else (0 if x < 0 else x)
            y = self.n - 1 if y > self.n - 1 else (0 if y < 0 else y)
            item.setStateFromRowAndCol(np.rint(x),np.rint(y))
            self.item_locations[int(np.rint(x))][int(np.rint(y))].append(item)

    def _getRowColFromState(self,state):
        return ((state % self. m), int(state / self.m))

    def _getStateFromColRow(self, col, row):
        return col*self.m + row

    def _build_trans_mat(self):
        trans_mat = np.zeros((self.spec.nS, len(Actions), self.spec.nS))
        r_mat = np.zeros((self.spec.nS, len(Actions), self.spec.nS))

        # Build transition matrix
        for state in range(0,self.spec.nS):

                if state < self.m:
                    trans_mat[state][Actions.west][state] = 1
                    r_mat[state][Actions.west][state] = -10
                else:
                    trans_mat[state][Actions.west][state - self.m] = 1
                    r_mat[state][Actions.west][state - self.m] = -1

                if state in range(0,self.spec.nS,self.m):
                    trans_mat[state][Actions.north][state] = 1
                    r_mat[state][Actions.north][state] = -10
                else:
                    trans_mat[state][Actions.north][state - 1] = 1
                    r_mat[state][Actions.north][state - 1] = -1

                if state >= ((self.n)-1) * self.m:
                    trans_mat[state][Actions.east][state] = 1
                    r_mat[state][Actions.east][state] = -10.
                else:
                    trans_mat[state][Actions.east][state + self.m] = 1
                    r_mat[state][Actions.east][state + self.m] = -1

                if state in range(self.m-1, self.spec.nS, self.m):
                    trans_mat[state][Actions.south][state] = 1
                    r_mat[state][Actions.south][state] = -10.
                else:
                    trans_mat[state][Actions.south][state + 1] = 1
                    r_mat[state][Actions.south][state + 1] = -1

        for item in self.items:
            if (item.state < ((self.n)-1) * self.m):
                r_mat[item.state+self.m][Actions.west][item.state] += 20
            if  item.state >= self.m:
                r_mat[item.state-self.m][Actions.east][item.state] += 20
            if  item.state not in range(0,self.spec.nS,self.m):
                r_mat[item.state - 1][Actions.south][item.state] += 20
            if item.state not in range(self.m - 1, self.spec.nS, self.m):
                r_mat[item.state+1][Actions.north][item.state] += 20

        return trans_mat, r_mat

    def _update_trans_mat(self):
        if (self._state < ((self.n) - 1) * self.m):
            self.r_mat[self._state + self.m][Actions.west][self._state] = -5
        if self._state >= self.m:
            self.r_mat[self._state - self.m][Actions.east][self._state] = -5
        if self._state not in range(0, self.spec.nS, self.m):
            self.r_mat[self._state - 1][Actions.south][self._state] = -5
        if self._state not in range(self.m - 1, self.spec.nS, self.m):
            self.r_mat[self._state + 1][Actions.north][self._state] = -5


    def _items_found(self, row, col):
        self.items_to_go -= len(self.item_locations[row][col])
        self.item_locations[row][col] = []

    def reset(self,random_state = False):
        if not random_state:
            self._state = 0
        else:
            self._state = np.random.randint(low=0, high=self.spec.nS, size=1)[0]

        self.items_to_go = self.k
        self._placeItems()
        if self.debug:
            util.logmsg("")
            util.logmsg("resetting environment...")
            util.logmsg("")
            util.logmsg("###################")
            util.logmsg("## Generated Items: ")
            for item in self.items:
                util.printobj(item)



        (row, col) = self._getRowColFromState(self._state)
        if len(self.item_locations[row][col]) > 0:
            self._items_found(row, col)

        self.trans_mat, self.r_mat = self._build_trans_mat()
        self._update_trans_mat()

        self.already_visited = self.spec.nS * [0]
        self.already_visited[self._state] = 1

        if self.items_to_go == 0:
            self._final_state = True
        else:
            self._final_state = False

        if self.debug:
            util.logmsg("")
            util.logmsg("")
            util.logmsg("reset complete.")
            util.logmsg("")

        return self._state

    def step(self, action):
        assert action >=Actions.west and action < self.spec.nA, "Invalid Action!"
        assert self._state >= 0 and self._state < self.spec.nS , "Invalid State!"
        assert not self._final_state, "Episode has already finished. Please restart!"

        prev_state = self._state
        self._state = np.random.choice(self.spec.nS, p=self.trans_mat[self._state, action])
        r = self.r_mat[prev_state, action, self._state]

        if self.debug:
            util.logmsg("taking step with a:{}...".format(action))
            util.logmsg("s:{} r:{}".format(self._state, r))

        (row,col) = self._getRowColFromState(self._state)
        if len(self.item_locations[row][col]) > 0:
            if self.debug:
                util.logmsg("items found!")
            self._items_found(row,col)

        self._update_trans_mat()
        self.already_visited[self._state] = 1
        if self.items_to_go == 0:
            self._final_state = True
            return self._state, r, True
        else:
            return self._state, r, False

    def step_using_policy(self,pi):
        action = pi.action(self._state)
        return self.step(action) + (action,)

    '''def next_reward_and_state(self,state,action):
        ##
        ## Simulate a step from a given state without actually moving (i.e just return s_next and r)

        curr_state = self._state
        is_at_final = self._final_state
        self._state = state
        (next_state,reward,final) = self.step(action)
        self.reset()
        self._state = curr_state
        self._is_final = is_at_final
        return  (next_state,reward,final)'''

    def heatMap(self):
        pass

    @property
    def state(self):
        return self._state

    @property
    def V(self):
        return self._V

    @property
    def Q(self):
        return self._Q

    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat


