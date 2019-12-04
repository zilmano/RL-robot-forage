import numpy as np
from tqdm import tqdm
from policy import PolicyType
class Simulation:
    def __init__(self,env,pi)->None:
        assert env.spec.nA == pi.nA and env.spec.nS == pi.nS, "policy and environment are not compatible. Not the " \
                                                              "the number of the states and actions differ."
        self._env = env
        self._pi = pi

    @property
    def pi(self):
        return self._pi

    @property
    def env(self):
        return self._env

    def get_trajectory(self,N_truncation = None):
        init_state, done = self._env.reset(random_start_cell=True)
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



def on_policy_mc_control(
    initQ:np.array,
    e_soft_epsilon,
    simulation,
    episode_num,
    N_truncation = None,
    alpha = None,
    everyVisit = False
    ) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """
    env_spec = simulation.env.spec
    V = np.zeros(env_spec.nS)
    Q = initQ
    num_of_S_A = np.ones([env_spec.nS,env_spec.nA])
    pi = simulation.pi
    ep_count = 0
    for _ in tqdm(range(episode_num)):
        print("doing episode num {}".format(ep_count))
        ep_count += 1
        traj = simulation.get_trajectory(N_truncation)
        occurance_num_in_traj = np.zeros([env_spec.nS, env_spec.nA])
        for (s_t,a_t,r,s_next) in traj:
            occurance_num_in_traj[s_t][a_t] += 1
        G = 0
        for (s_t,a_t,r,s_next) in reversed(traj):
            G = env_spec.gamma*G + r
            if occurance_num_in_traj[s_t][a_t] == 1:
                if alpha is None:
                    Q[s_t][a_t] += (1 / num_of_S_A[s_t][a_t]) * (G - Q[s_t][a_t])
                else:
                    Q[s_t][a_t] += alpha * (G - Q[s_t][a_t])
                num_of_S_A[s_t][a_t] += 1
                new_action = Q[s_t].argmax()
                pi.set_e_soft_action(s_t,new_action,e_soft_epsilon)
            elif occurance_num_in_traj[s_t][a_t] == 0:
                # OlegDbg: Sanity check, remove later.
                sys.exit("Internal Error")
            else:
                occurance_num_in_traj[s_t][a_t] -= 1

    for s in range(0,env_spec.nS):
        for a in range(0, env_spec.nA):
            V[s] += pi.action_prob(s,a)*Q[s][a]


    print(V)
    print(Q)
    return Q,V,pi
