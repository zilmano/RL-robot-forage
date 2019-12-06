from env import Env,EnvSpec
import string
from enum import IntEnum
import numpy as np
import utility as util
from scipy import stats
import math
from graph import GridGraphWithItems
import sys


class Actions(IntEnum):
    west = 0
    north = 1
    east = 2
    south = 3


class Item:
    def __init__(self, mean, variance, name, m, n):
        self.mean = mean
        self.variance = variance
        self.state = None
        self.name = name
        self.m = m
        self.n = n
        mean_y, mean_x = self.getRowAndColFromState()
        self.mean_y = m - 0.5 - mean_y
        self.mean_x = mean_x+ 0.5

    def getRowAndColFromState(self):
        return (self.mean % self.m), (int(self.mean/self.m))

    def setStateFromRowAndCol(self, row, col):
        self.state = int(col*self.m + row)


class GridWorldApproxModel(Env):  # MDP introduced at Fig 5.4 in Sutton Book
    def __init__(self, m, n, k, gamma=1, debug=False, no_stochastisity=False):
        # (m x n) grid world
        self.grid_size = m*n

        # Total state size is grid size multiplied by all combinations of items found status.
        # For example grid cell zero would be a different state when none items are found, and when item ony 1 is
        # found. Then, its multiplied by 16, to represent the visit status of the neighbors cells. Each
        # combination of
        # visited neighbors is a different state. 4 neigbors, so 16 combinations.
        # To conclude, state space is: grid_size*|combinations of found items|*|combination of neighboors visit status|
        STATE_SPACE_SIZE = m*n
        self.m = m  # rows
        self.n = n  # columns
        self.k = k  # Number of items to search
        self.items = list()
        self.debug = debug
        self._final_state = False
        self._grid_cell = None
        self.trans_mat = None
        self.r_mat = None
        self.item_loc_probabilities = None
        self.items_to_go = None
        self.items_status = None
        self.already_visited = None
        self.no_stochastisity = no_stochastisity
        # nS states of mxn grid. State numbering is by columns starting from top-left. i.e:
        # top-row, leftmost-col state = 0, bottom-leftmost state = m-1, top-row,second from left col = m
        # 2nd from top row, 2nd from left column = m+1.  etc...

        # Actions:
        # 0 -west, 1 -north, 2-east, 3 -south

        env_spec = EnvSpec(STATE_SPACE_SIZE, len(Actions), gamma)
        super().__init__(env_spec)

        self._V = np.zeros(self.spec.nS)
        self._Q = np.zeros([self.spec.nS, self.spec.nA])

        self._generateItems()
        self._initItemsProbabilities()
        #for i in range(0, k):
        #    print(self.item_loc_probabilities[i])
        #    print(self.item_loc_probabilities[i].sum())

        self.graph_rep = GridGraphWithItems(m, n, k, self.item_loc_probabilities)
        self.reset()
        '''for i in range(0,self.spec.nS):
            util.logmsg("state:{}\n   {}".format(i,self.r_mat[i]))'''

    def _generateItems(self):
        for i in range(0, self.k):
            mean = np.random.randint(low=0, high=self.grid_size, size=1)[0]
            variance = np.random.randint(low=1, high=2, size=1)[0]
            self.items.append(Item(mean, variance-.5, string.ascii_lowercase[i], self.m, self.n))

    def _initItemsProbabilities(self):
        # Calc items probabilities to be in states
        self.item_loc_probabilities = np.zeros([self.k, self.grid_size])
        for i, item in enumerate(self.items):
            mv_gaussian = stats.multivariate_normal(mean=[item.mean_x, item.mean_y], cov=[[item.variance, 0],
                                                                                      [0, item.variance]])
            ''''
            grid coding as follows (example 4x4), state num is inside each grid cell:
            Y
            4 _________________
            3 _0_|_4_|_8__|_12_|
            2 _1_|_5_|_9__|_13_|
            1 _2_|_6_|_10_|_14_|
            0 _3_|_7_|_11_|_15_|
                 1   2    3    4  X

            cells on the edge of the grid have bigger probabilities then the cells in the middle, as they get all
            probabilities of the space beyond them.
            For instance, the probability of the item x being from -inf to 1 will be crammed into cells rightmost col
            (states 0,1,2,3). The probability of the item being from 3 to inf will be crammed into cell on the leftmost
            col (states 12,13,14,15).
            etc..
            For example, the probability of the item being in cell 2 is the probability of item's x being -inf<x<=1,
            and y being 1<y<=2
            '''
            for bottom_cell in range(self.m - 1, self.grid_size, self.m):
                sum_of_prev_col_cell_probs = 0
                for row in reversed(range(0, self.m)):
                    state = bottom_cell - (self.m - 1) + row
                    (row, col) = self._getRowColFromState(state)
                    cell_top_left_y, cell_top_left_x = (self.m - row, col+1)
                    if not state % self.m:
                        cell_top_left_y = math.inf
                    if col == (self.n - 1):
                        cell_top_left_x = math.inf
                    state_prob = mv_gaussian.cdf([cell_top_left_x, cell_top_left_y])
                    state_prob -= sum_of_prev_col_cell_probs
                    if col >= 1:
                        state_prob -= mv_gaussian.cdf([col, cell_top_left_y])
                    self.item_loc_probabilities[i][state] = state_prob
                    sum_of_prev_col_cell_probs += state_prob

    def _placeItems(self):
        self.item_locations = [[[] for i in range(0, self.n)] for j in range(0, self.m)]
        for item in self.items:
            if (self.no_stochastisity):
                (x,y) = self._getRowColFromState(item.mean)
            else:
                ((x, y),) = np.random.multivariate_normal([item.mean_x, item.mean_y], [[item.variance, 0], [0,
                                                                                                   item.variance]], 1)
            x = self.m - 1 if x > self.m - 1 else (0 if x < 0 else x)
            y = self.n - 1 if y > self.n - 1 else (0 if y < 0 else y)
            item.setStateFromRowAndCol(np.rint(x), np.rint(y))
            self.item_locations[int(np.rint(x))][int(np.rint(y))].append(item)

    def _getRowColFromState(self, state):
        return (state % self. m), int(state / self.m)

    def _getStateFromColRow(self, col, row):
        return int(col*self.m + row)

    def _build_trans_mat(self):
        trans_mat = np.zeros((self.grid_size, len(Actions), self.grid_size))
        r_mat = np.zeros((self.grid_size, len(Actions), self.grid_size))

        # Build transition matrix
        for state in range(0, self.grid_size):

            if state < self.m:
                trans_mat[state][Actions.west][state] = 1
                r_mat[state][Actions.west][state] = -10
            else:
                trans_mat[state][Actions.west][state - self.m] = 1
                r_mat[state][Actions.west][state - self.m] = -1

            if state in range(0, self.grid_size, self.m):
                trans_mat[state][Actions.north][state] = 1
                r_mat[state][Actions.north][state] = -10
            else:
                trans_mat[state][Actions.north][state - 1] = 1
                r_mat[state][Actions.north][state - 1] = -1

            if state >= (self.n-1) * self.m:
                trans_mat[state][Actions.east][state] = 1
                r_mat[state][Actions.east][state] = -10.
            else:
                trans_mat[state][Actions.east][state + self.m] = 1
                r_mat[state][Actions.east][state + self.m] = -1

            if state in range(self.m-1, self.grid_size, self.m):
                trans_mat[state][Actions.south][state] = 1
                r_mat[state][Actions.south][state] = -10.
            else:
                trans_mat[state][Actions.south][state + 1] = 1
                r_mat[state][Actions.south][state + 1] = -1

        '''for item in self.items:
            if item.state < (self.n-1) * self.m:
                r_mat[item.state+self.m][Actions.west][item.state] += 20
            if item.state >= self.m:
                r_mat[item.state-self.m][Actions.east][item.state] += 20
            if item.state not in range(0, self.grid_size, self.m):
                r_mat[item.state - 1][Actions.south][item.state] += 20
            if item.state not in range(self.m - 1, self.grid_size, self.m):
                r_mat[item.state+1][Actions.north][item.state] += 20'''

        return trans_mat, r_mat

    def _update_trans_mat(self):
        #reward_after_visit = -5
        reward_after_visit = -15
        if self._grid_cell < (self.n - 1) * self.m:
            self.r_mat[self._grid_cell + self.m][Actions.west][self._grid_cell] = reward_after_visit
        if self._grid_cell >= self.m:
            self.r_mat[self._grid_cell - self.m][Actions.east][self._grid_cell] = reward_after_visit
        if self._grid_cell not in range(0, self.grid_size, self.m):
            self.r_mat[self._grid_cell - 1][Actions.south][self._grid_cell] = reward_after_visit
        if self._grid_cell not in range(self.m - 1, self.grid_size, self.m):
            self.r_mat[self._grid_cell + 1][Actions.north][self._grid_cell] = reward_after_visit

    def _items_found(self, row, col):
        self.items_to_go -= len(self.item_locations[row][col])
        for item_in_loc in self.item_locations[row][col]:
            item_indx = ord(item_in_loc.name) - 97
            self.items_status[item_indx] = 1
        self.item_locations[row][col] = []

    def reset(self, start_cell=0, random_start_cell=False):
        # Random_state wins start_state. Don't use together.
        assert start_cell < self.grid_size, "start state provided to reset function is out of bounds."
        self._grid_cell = start_cell
        if random_start_cell:
            self._grid_cell = np.random.randint(low=0, high=self.grid_size, size=1)[0]

        self.items_to_go = self.k
        self.items_status = np.zeros(self.k)
        self._placeItems()
        if self.debug:
            util.logmsg("")
            util.logmsg("resetting environment...")
            util.logmsg("")
            util.logmsg("###################")
            util.logmsg("## Generated Items: ")
            for item in self.items:
                util.printobj(item)

        (row, col) = self._getRowColFromState(self._grid_cell)
        if len(self.item_locations[row][col]) > 0:
            self._items_found(row, col)

        self.trans_mat, self.r_mat = self._build_trans_mat()
        self._update_trans_mat()

        self.already_visited = self.grid_size * [0]
        self.already_visited[self._grid_cell] = 1

        # covert items_status to a number between 0 to (2^k-1)

        if self.items_to_go == 0:
            self._final_state = True
        else:
            self._final_state = False

        if self.debug:
            util.logmsg("")
            util.logmsg("")
            util.logmsg("reset complete.")
            util.logmsg("grid_cell {}",(self._grid_cell,))

        return self._getRowColFromState(self._grid_cell), self.items_status, self.already_visited, self._final_state

    def step(self, action):
        assert Actions.west <= action < self.spec.nA, "Invalid Action!"
        assert not self._final_state, "Episode has already finished. Please restart!"

        prev_grid_cell = self._grid_cell

        self._grid_cell = np.random.choice(self.grid_size, p=self.trans_mat[self._grid_cell, action])
        r = self.r_mat[prev_grid_cell, action, self._grid_cell]

        if self.debug:
            util.logmsg("taking step with a:{}...".format(action))
            util.logmsg("s:{} r:{}".format(self._grid_cell, r))

        (row, col) = self._getRowColFromState(self._grid_cell)
        if len(self.item_locations[row][col]) > 0:
            if self.debug:
                util.logmsg("items found!")
            self._items_found(row, col)

        self._update_trans_mat()
        self.already_visited[self._grid_cell] = 1

        if self.items_to_go == 0:
            self._final_state = True

        return self._getRowColFromState(self._grid_cell), self.items_status, self.already_visited, self._final_state, r

    def step_using_policy(self, pi):
        action = pi.action(self._grid_cell, self.items_status, self.already_visited, self._final_state)
        return (action,) + self.step(action)

    def heatMap(self):
        pass

    @property
    def features(self):
        return self._getRowColFromState(self._grid_cell), self.items_status, self.already_visited, self._final_state

    @property
    def grid_cell(self):
        return self._getRowColFromState(self._grid_cell)

    @property
    def final(self):
        return self._final_state

    @property
    def V(self):
        return self._V

    @property
    def Q(self):
        return self._Q

    def setQ(self, Q , pi):
        self._Q = Q
        for s in range(0, self.spec.nS):
            for a in range(0, self.spec.nA):
                self.V[s] += pi.action_prob(s, a) * Q[s][a]

    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat

    @property
    def graph(self) -> GridGraphWithItems:
        return self.graph_rep
