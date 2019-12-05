import numpy as np
from tqdm import tqdm
from policy import PolicyType


def on_policy_mc_control(
    initQ:np.array,
    e_soft_epsilon,
    sim,
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
    env_spec = sim.env.spec
    V = np.zeros(env_spec.nS)
    Q = initQ
    num_of_S_A = np.ones([env_spec.nS,env_spec.nA])
    pi = sim.pi
    ep_count = 0
    for _ in tqdm(range(episode_num)):
        print("doing episode num {}".format(ep_count))
        ep_count += 1
        traj = sim.get_trajectory(N_truncation)
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

    return Q,V,pi
