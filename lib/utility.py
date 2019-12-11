from enum import IntEnum
import sys
import numpy as np
from policy import PolicyType
from bisect import bisect_left, bisect_right
import math
import collections

########
### Module For Utility Functions
########

class Actions(IntEnum):
    west = 0
    north = 1
    east = 2
    south = 3

class PriorityQ:
    def __init__(self):
        self.sortedList = list()

    def add(self, intNum):
        bisect.insort(self.sortedList, intNum)

    def pop(self):
        if (not self.isEmpty()):
            return self.sortedList.pop()
        else:
            return -101

    def isEmpty(self):
        if len(self.sortedList) > 0:
            return False
        else:
            return True

class BasicQueue:
    def __init__(self):
        self.qlist = list()

    def add(self, item):
        self.qlist.append(item)

    def pop(self):
        if (not self.isEmpty()):
            return self.qlist.pop()
        else:
            raise IndexError("Queue is empty, cannot pop.")

    def isEmpty(self):
        if len(self.qList) > 0:
            return False
        else:
            return True

class SortedList(list):
    def __init__(self, data=[]):
        super().__init__(sorted(data))

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __contains__(self, value):
        index= bisect_left(self, value)
        if index == len(self) or self[index] != value:
            return False
        else:
            return True

    def append(self,value):
        raise NotImplementedError

    def add(self, value):
        """Add an item to this list."""
        index = bisect_right(self, value)
        self.insert(index, value)

    def index(self, value, start=0, stop=None):
        """Return first index of value.
        Raise ValueError if not found."""
        if stop is None:
            stop = len(self)
        index = bisect_left(self, value, start, stop)
        if index != len(self) and self[index] == value:
            return index
        else:
            return None

log_fh = None
def openlog(logfile=None):
    global log_fh
    if logfile is None:
        logfile = "log.txt"
    try:
        log_fh = open(logfile,'w')
    except IOError as e:
        sys.exit("ERROR: Could not open log file {}. Quitting. Err message: {}".format(logfile,e))


def logmsg(msg,vars=None,tab=0,log_only=False):
    if vars is not None:
        msg = tab * "    " + msg.format(*vars)
    if not log_only:
        print(msg)
    global log_fh
    try:
        log_fh.write(msg + "\n")
    except AttributeError as e:
        openlog("log.txt")
        try:
            log_fh.write(msg + "\n")
        except AttributeError as e:
            sys.exit("Could not write to the log file. Err message: {}".format(e))

def error(msg):
    global log_fh
    msg = str(msg)
    try:
        log_fh.write(msg + "\n")
    except Exception:
        pass
    sys.exit("ERROR: {}".format(msg))

def printobj(obj):
    members = [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj, a))]
    logmsg("------- Obj Starts ---------")
    for member in members:
        logmsg("    {} : {}".format(member,obj.__dict__[member]))

    logmsg("")

def visualizeGridTxt(env,V,items_status=0):
        upper_border = " " + env.n * "_______"
        print(upper_border)
        for i in range(0, env.m):
            upper_border += '_'
            middle_row = ''
            bottom_border = ""
            for j in range(i, env.grid_size, env.m):
                strV = "{: ^6.2f}".format(V[j+items_status*env.grid_size])
                middle_row += "|{}".format(strV)
                row = int(j % env.m)
                col = int(j/env.m)
                if len(env.item_locations[row][col]) > 0:
                    bottom_border += "|_{: ^4}_".format(''.join([item.name for item in env.item_locations[row][col]]))
                else:
                    bottom_border += "|______"
            middle_row += '|'
            bottom_border += '|'
            print(middle_row)
            print(bottom_border)

def visualizePolicyTxt(pi:np.array,m,n, item_status = 0 ,policy_type=PolicyType.greedy, eps = 0):

    upper_border = " " + ((n-1) * "____________") + "__________"
    print(upper_border)
    treshold = 0
    for i in range(0,m):
        tile_row = 5 * ['']
        for j in range(i,n*m,m):
            state = j+item_status*m*n
            num_optimal_actions = sum(math.ceil(prob) for prob in pi.P[state])
            if (policy_type == PolicyType.greedy):
                for prob in pi.P[state]:
                    assert prob == (1/num_optimal_actions) or prob == 0, "Policy that is given is not greedy," \
                                                                                      "can not visualize"
                treshold = 0
            elif (policy_type == PolicyType.e_soft):
                treshold = eps/len(Actions)


            #print("state {}".format(j))
            #print("    optAction#{} {} ".format(num_optimal_actions, pi.P[j]))
            if (pi.P[state][Actions.north] > treshold):
                tile_row[0] += '|     ^     '
                tile_row[1] += '|     |     '
            else:
                tile_row[0] += '|           '
                tile_row[1] += '|           '

            tile_row[2] += '| '
            if (pi.P[state][Actions.west] > treshold):
                tile_row[2] += '<- - '
            else:
                tile_row[2] += '     '
            if (pi.P[state][Actions.east] > treshold):
                tile_row[2] += '- -> '
            else:
                tile_row[2] += '     '

            if (pi.P[state][Actions.south] > treshold):
                tile_row[3] += '|     |     '
                tile_row[4] += '|_____v_____'
            else:
                tile_row[3] += '|           '
                tile_row[4] += '|___________'

        for curr_row in tile_row:
            curr_row += '|'
            print(curr_row)

class Simulator:
    def __init__(self,env,pi,random_start=False)->None:
        assert env.spec.nA == pi.nA and env.spec.nS == pi.nS, "policy and environment are not compatible. Not the " \
                                                              "the number of the states and actions differ."
        self._env = env
        self._pi = pi
        self._random_start = random_start

    @property
    def pi(self):
        return self._pi

    @property
    def env(self):
        return self._env

    def get_trajectory(self,N_truncation = None):
        init_state, done = self._env.reset(random_start_cell=self._random_start)
        states, actions, rewards, done = \
            [[init_state], [], [], done]
        step = 0
        while not done and (N_truncation is None or step < N_truncation):
            a = self._pi.action(states[-1])
            s, r, done,a = self._env.step_using_policy(self._pi)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            step += 1
        print("   traj steps:" + str(step))
        return list(zip(states[:-1],actions,rewards,states[1:]))

    def get_trajectories(self,episode_num, N_truncation = None):
        util.logmsg("generating trajectories...")
        trajs = []
        for _ in tqdm(range(episode_num)):
            trajs.append(self.get_trajectory(N_truncation))
        return trajs

