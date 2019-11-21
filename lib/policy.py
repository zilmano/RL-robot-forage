import numpy as np
import enum

class PolicyType(enum.Enum):
    greedy = 1
    e_soft = 2

class Policy(object):
    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        raise NotImplementedError()

    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        raise NotImplementedError()

class NewPolicy(Policy):
    def __init__(self,nA,nS):
        self._nA = nA
        self._nS = nS
        self._p = np.ones([nS,nA])*(1/nA)

    def action_prob(self,state:int,action:int):
        return self._p[state][action]

    def action(self,state):
        return np.random.choice(self._nA, p=self._p[state])

    def set_greedy_action(self,state,new_greedy_actions):
        self._p[state] = np.zeros(self._nA)
        for action in new_greedy_actions:
           self._p[state][action] = 1/len(new_greedy_actions)

    def set_e_soft_action(self, state, new_action, e):
        self._p[state] = np.array(self._nA*[e/self.nA])
        self._p[state][new_action] = 1 - e + e/self.nA

    @property
    def P(self) -> np.array:
        return self._p

    @property
    def nA(self) -> int:
        return self._nA

    @nA.setter
    def nA(self,nA):
        self._nA = nA

    @property
    def nS(self) -> int:
        return self._nS

    @nS.setter
    def nS(self, nS):
        self._nS = nS
