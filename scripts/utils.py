import numpy as np
import random
import torch

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def StyblinskiTang_reward(X,bias):
    X = np.array(X)
    dims = len(X[0])
    val = 0.0
    for i in range(dims):
        val += (np.power(X[:,i], 4, dtype=np.longdouble) - 16.0 * np.power(X[:,i], 2, dtype=np.longdouble) + 5.0 * X[:,i])
    val = -val # switch to positive
    return val+bias

def rollout(agent,make_env,H,n_traj):
    state_acc, action_acc, reward_acc, prob_acc = [],[],[],[]
    for i in range(n_traj):
        done = False
        env = make_env()
        state = env.reset()
        states = []
        actions = []
        rewards = []
        probs = []
        for t in range(H):
            action, prob = agent(state)
            actions.append(action)
            probs.append(prob)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            rewards.append(reward)
            states.append(state)
            if done:
                break
        state_acc.append(states)
        action_acc.append(actions)
        reward_acc.append(rewards)
        prob_acc.append(probs)
    states = np.array(state_acc)
    rewards = np.array(reward_acc)
    actions = np.array(action_acc)
    probs = np.array(prob_acc)
    return states, rewards, actions, probs

def set_seed(seed,cuda):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_batch(X,Z,y,batch_size):
    for i in range(len(X)//batch_size):
        yield X[i*batch_size:(i+1)*batch_size], Z[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]

def rolling_window_helper(a, window_size):
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def sliding_window(a,w,step,dim=1):
    """
    a: n-dimensional tensor
    w: positive integer, window size
    step: positive integer < w, step size
    dim: dimension along which to perform sliding window
    """
    acc = [] 
    n_dim = min(min([len(aa) for aa in a]),w)
    w = n_dim
    for i in range(len(a)):
        frames = rolling_window_helper(np.array(a[i]),w)
        idx = ( np.arange(len(frames)) % step ) == 0
        frames = frames[idx] # pick every `step` windows
        acc.append(frames)
    arr = np.concatenate(acc)

    return arr

def session_reward_Y_T_m_Y_1(R):
    return R[-1]-R[0]

def session_reward_Y_T(R):
    return R[-1]

def session_reward_sum_Y_t(R):
    return np.sum(R)

def session_reward_mean_Y_t(R):
    return np.mean(R)