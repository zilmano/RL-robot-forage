import sys
import numpy as np
from policy import PolicyType
'''
 ########
 ### Module For Utility Functions
 ########
'''
'''class Printer:
    def __init__(self,logfile=None):
        if logfile is None:
            logfile = "log.txt"
        self._logfile = logfile
        try:
            self._fh = open(logfile,"w")
        except IOError:
            sys.exit("ERROR: Could not open log file {}. Exiting".format(logfile))

    def __del__(self):
        self._fh.close()

    def logmsg(self,msg):
        print(msg)
        self._fh.write(msg + "\n")

    def error(self,msg):
        sys.exit("ERROR: {}".format(msg))
'''

log_fh = None
def openlog(logfile=None):
    global log_fh
    if logfile is None:
        logfile = "log.txt"
    try:
        log_fh = open(logfile,'w')
    except IOError as e:
        sys.exit("ERROR: Could not open log file {}. Quitting. Err message: {}".format(logfile,e))


def logmsg(msg):
    msg = str(msg)
    print(msg)
    global log_fh
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

def visualizeGridTxt(env,V):
        upper_border = " " + env.n * "_______"
        print(upper_border)
        for i in range(0, env.m):
            upper_border += '_'
            middle_row = ''
            bottom_border = ""
            for j in range(i, env.spec.nS, env.m):
                strV = "{: ^6.2f}".format(V[j])
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

def visualizePolicyTxt(pi:np.array,m,n, policy_type=PolicyType.greedy, eps = 0):
    assert m*n == pi.nS, "grid column and row are in discrepancy with array size!"

    upper_border = " " + ((n-1) * "____________") + "__________"
    print(upper_border)
    treshold = 0
    for i in range(0,m):
        #print("row#{} ------------------------------------".format(i))
        tile_row = 5 * ['']
        for j in range(i,25,5):
            num_optimal_actions = sum(math.ceil(prob) for prob in pi.P[j])
            if (policy_type == PolicyType.greedy):
                for prob in pi.P[j]:
                    assert prob == (1/num_optimal_actions) or prob == 0, "Policy that is given is not greedy," \
                                                                                      "can not visualize"
                treshold = 0
            elif (policy_type == PolicyType.e_soft):
                treshold = eps/len(Actions)


            #print("state {}".format(j))
            #print("    optAction#{} {} ".format(num_optimal_actions, pi.P[j]))
            if (pi.P[j][Actions.north] > treshold):
                tile_row[0] += '|     ^     '
                tile_row[1] += '|     |     '
            else:
                tile_row[0] += '|           '
                tile_row[1] += '|           '

            tile_row[2] += '| '
            if (pi.P[j][Actions.west] > treshold):
                tile_row[2] += '<- - '
            else:
                tile_row[2] += '     '
            if (pi.P[j][Actions.east] > treshold):
                tile_row[2] += '- -> '
            else:
                tile_row[2] += '     '

            if (pi.P[j][Actions.south] > treshold):
                tile_row[3] += '|     |     '
                tile_row[4] += '|_____v_____'
            else:
                tile_row[3] += '|           '
                tile_row[4] += '|___________'

        for curr_row in tile_row:
            curr_row += '|'
            print(curr_row)
