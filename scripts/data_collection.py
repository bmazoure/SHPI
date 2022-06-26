from .utils import rollout, sliding_window
import numpy as np

class BatchDataset():
    def __init__(self, agent, make_env, n_timesteps_offline, n_traj_offline, rho, tau, delta, eps_behavior, p_mu_threshold, gamma_behavior):
        self.agent = agent
        self.make_env = make_env
        self.H = n_timesteps_offline
        self.n_traj = n_traj_offline
        self.window_X = rho
        self.tau = tau
        self.step = delta
        self.eps = eps_behavior
        self.p_mu_threshold = p_mu_threshold
        self.gamma_behavior = gamma_behavior

        self.X_t = self.Z_t = self.Y_t = self.P_mu_t = None

    def construct(self):
        """
        Collect a logged dataset by some random agent mu

        X_t: N*rho x rho x n_features
        """
        states,rewards,actions,probs = rollout(lambda state:self.agent.act(state,eps=self.eps),self.make_env,H=self.H,n_traj=self.n_traj)
        
        long_idx = np.where(np.array([len(s) for s in states])>=self.window_X)[0]

        states = [traj for i,traj in enumerate(states) if i in long_idx]
        rewards =[traj for i,traj in enumerate(rewards) if i in long_idx ]
        actions = [traj for i,traj in enumerate(actions) if i in long_idx ]
        probs = [traj for i,traj in enumerate(probs) if i in long_idx ]
        
        self.X_t = sliding_window(states,w=self.window_X,step=self.step)
        self.Z_t = sliding_window(actions,w=self.window_X,step=self.step)
        self.Y_t = sliding_window(rewards,w=self.window_X,step=self.step)
        self.P_mu_t = sliding_window(probs,w=self.window_X,step=self.step)

        self.window_X = self.X_t.shape[1]

        non_nan_idx = np.unique(np.where(1- np.any(np.isnan(self.X_t),1))[0])

        self.X_t = self.X_t[non_nan_idx]
        self.Z_t = self.Z_t[non_nan_idx]
        self.Y_t = self.Y_t[non_nan_idx]
        self.P_mu_t = self.P_mu_t[non_nan_idx]



    def __len__(self):
        return len(self.X_t)

class RandomizedBatchDataset():
    def __init__(self,X,Z,r,w,dones,batch_size):
        self.X = X
        self.Z = Z
        self.r = r
        self.w = w
        self.dones = dones
        self.batch_size = batch_size

        self.idx = np.random.permutation(np.arange(0,len(X)*len(X[0])))
        self.unraveled_idx = np.unravel_index(self.idx,self.X.shape[:2])
        
    
    def batch(self,i):
        idx = self.unraveled_idx[0][i*self.batch_size:min((i+1)*self.batch_size,len(self.idx))]
        idx2 = self.unraveled_idx[1][i*self.batch_size:min((i+1)*self.batch_size,len(self.idx))]
        idx2_next = np.minimum(self.unraveled_idx[1][i*self.batch_size:min((i+1)*self.batch_size,len(self.idx))] + 1,len(self.X[0])-1)
        idx2_init = np.zeros_like(idx2)
        return self.X[idx,idx2_init], self.Z[idx,idx2_init], self.r[idx,idx2_init], self.X[idx,idx2], self.Z[idx,idx2], self.r[idx,idx2], self.w[idx,idx2], self.X[idx,idx2_next], self.Z[idx,idx2_next], self.r[idx,idx2_next], self.w[idx,idx2_next], self.dones[idx,idx2]

    def get_dataset(self):
        for i in range(len(self.X) // self.batch_size):
            yield self.batch(i)
        