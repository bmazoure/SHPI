import gym
import numpy as np
import logging
from copy import deepcopy
from scipy.integrate import odeint, ode
import recogym
from scipy.stats import binom,norm
from recogym import env_1_args, Configuration


class Toy_GaussianContextDiscreteAction():
    def __init__(self,tau,threshold,feature_weight_fn,mu,Ls,rho,n_timesteps,Y_bias,n_actions,n_features,reward_fn,mean_X,std_X,**kwargs):
        self.tau = tau
        self.threshold = threshold
        self.n_actions = n_actions
        self.n_features = n_features
        self.action_space = gym.spaces.Discrete( n_actions )
        self.state_space = gym.spaces.Box(low=float('-inf'),high=float('inf'),shape=(self.n_features,))
        self.feature_weight_fn = feature_weight_fn
        self.rho = rho
        self.n_timesteps = n_timesteps
        self.Y_bias = Y_bias

        self.reward_fn = reward_fn

        self.mean_X = mean_X
        self.std_X = std_X

        self.mu0 = mu
        self.L0s = Ls
        self.As = []
        self.reset()

    def threshold_continuous(self,Ns,mu,Sigma,threshold):
        W = np.zeros_like(Ns)
        U = np.zeros_like(Ns)
        for t in range(Ns.shape[0]):
            for j in range(Ns.shape[1]):
                U[t,j] = norm.cdf(Ns[t,j],loc=mu[t],scale=Sigma[t,t])
                W[t,j] = binom.ppf(U[t,j], 1, threshold[t,j]) # Use quantile + probability transforms
        return W, U

    def gaussian_conditional_sim(self,mu,cov,cond_idx,conditional_data):
        n_features = conditional_data.shape[1]
        c11 = cov[:cond_idx, :cond_idx] # Covariance matrix of the dependent variables
        c12 = cov[:cond_idx, cond_idx:] # Custom array only containing covariances, not variances
        c21 = cov[cond_idx:, :cond_idx] # Same as above
        c22 = cov[cond_idx:, cond_idx:] # Covariance matrix of independent variables

        m1 = mu[:cond_idx].T # Mu of dependent variables
        m2 = mu[cond_idx:].T # Mu of independent variables

        N_tp1 = np.zeros(shape=(cond_idx,n_features))
        for j in range(n_features):
            conditional_mu = m1 + c12.dot(np.linalg.inv(c22)).dot((conditional_data[:,j] - m2).T).T
            conditional_cov = np.linalg.inv(np.linalg.inv(cov)[:cond_idx, :cond_idx])
            N_tp1[:,j] = np.array([np.random.multivariate_normal([c_mu], conditional_cov, 1)[0] for c_mu in conditional_mu]).reshape(-1)
        return N_tp1

    def rolling_window_sum(self,a,w=28):
        acc = []
        for i in range(len(a)-w):
            acc.append( np.sum(a[i:i+w],axis=0) )
        return np.array(acc)

    def make_reward(self):
        if self.Y_bias == "non-stationary":
            self.bias = self._bias()
        Y_t = self.reward_fn(self.feature_weights,self.Xs[0].reshape(1,-1),self.bias)
        # Y_t = self.feature_weights @ (self.Xs[0]) + self.bias
        return Y_t

    def _bias(self):
        return ( np.random.normal(0,3,size=1)  ).item() 

    def reset(self):
        self.feature_weights = self.feature_weight_fn()

        self.mu = self.mu0.copy()

        if self.Y_bias == "none":
            self.bias = 0.
        elif self.Y_bias == "stationary":
            self.bias = self._bias()

        """
        N, W and U are matrices of size (window_X,n_features), e.g. (28,7)
        We first generate tau+1 random features, which will be used to condition the p(W_t|w_{t-tau+1:t})
        """

        self.Ns = np.zeros(shape=(self.n_timesteps,self.n_features))
        self.Ws = np.zeros(shape=(self.n_timesteps,self.n_features))
        self.Us = np.zeros(shape=(self.n_timesteps,self.n_features))
        self.As = np.zeros(shape=(self.n_timesteps,))

        self.Ns[self.Ns==0] = 'nan'
        self.Ws[self.Ws==0] = 'nan'
        self.Us[self.Us==0] = 'nan'
        
        self.As[:self.tau] = [self.action_space.sample() for _ in range(self.tau)]

        idx = 0
        for i in range(self.tau):
            idx += self.As[i]*(self.n_actions-1)**i
        idx = int(idx)

        L = self.L0s[idx]
        Sigma = L.T @ L

        self.Ns[:self.tau+1] = np.random.multivariate_normal(mean=self.mu,cov=Sigma,size=self.n_features).T
        self.Ws[:self.tau+1], self.Us[:self.tau+1] = self.threshold_continuous(self.Ns[:self.tau+1],self.mu,Sigma,self.threshold)

        self.Xs = self.rolling_window_sum(self.Ws,self.rho)

        return self.Xs[0]

    def step(self, a):
        self.As[1:] = self.As[:-1]
        self.As[0] = a

        idx = 0
        for i in range(self.tau):
            idx += self.As[i]*(self.n_actions-1)**i
        idx = int(idx)

        L = self.L0s[idx]
        Sigma = L.T @ L
        N_tp1 = self.gaussian_conditional_sim(self.mu,Sigma,1,self.Ns[:self.tau])
        self.Ns[1:], self.Ws[1:], self.Us[1:] = self.Ns[:-1], self.Ws[:-1], self.Us[:-1]
        self.Ns[0] = N_tp1 # drop the oldest observation
        self.Ws[0], self.Us[0] = self.threshold_continuous(self.Ns[0].reshape(1,-1),self.mu,Sigma,self.threshold)

        self.Xs = self.rolling_window_sum(self.Ws,self.rho)

        if self.mean_X and self.std_X:
            for j in range(self.Xs.shape[1]):
                self.Xs[:,j] = (self.Xs[:,j] - self.mean_X) / self.std_X
                
        Y_t = self.make_reward().item()
        return self.Xs[0], Y_t, False, None

class Toy_GaussianContextDiscreteAction_simple():
    def __init__(self,tau,mu,Sigma,rho,n_timesteps_offline,Y_bias,n_actions,n_features,reward_fn,a_max,eval,**kwargs):
        self.tau = tau
        self.n_actions = n_actions
        self.n_features = n_features
        self.action_space = gym.spaces.Discrete( n_actions )
        self.state_space = gym.spaces.Box(low=float('-inf'),high=float('inf'),shape=(self.n_features,))
        self.rho = rho
        self.n_timesteps = n_timesteps_offline
        self.Y_bias = Y_bias
        self.a_max = a_max
        self.eval = eval

        # constrained between -a_max and a_max
        n_non_orthonormal_actions = (n_actions - 2*n_features) // 2
        orthonormal_actions = np.concatenate([np.eye(n_features),-1*np.eye(n_features)])
        if n_non_orthonormal_actions > 0:
            random_actions = np.random.beta(a=0.5,b=0.5,size=(n_non_orthonormal_actions,n_features))
            self.true_action_vectors = np.concatenate([orthonormal_actions,random_actions,-random_actions])
        else:
            self.true_action_vectors = orthonormal_actions
        self.true_action_vectors = self.true_action_vectors / np.abs(self.true_action_vectors.sum(axis=1)[:, np.newaxis]) * a_max
        self.reward_fn = reward_fn

        self.mu = mu
        self.Sigma = Sigma
        self.As = []
        self.reset()

    def rolling_window_fn(self,a,w=28,fn=np.sum):
        acc = []
        for i in range(len(a)-w):
            acc.append( fn(a[i:i+w],axis=0) )
        return np.array(acc)

    def make_reward(self):
        if (self.step_count % 10 ) == 0 and self.Y_bias == "non-stationary": # Exo MRP eps_t updates 1/10th times of Y_t
            self.bias = self._bias()
        if self.eval:
            self.bias = 0.
        Y_t = self.reward_fn(self.Xs[0].reshape(1,-1),self.bias)
        return Y_t

    def _bias(self):
        # if self.curr_Y is None or self.prev_Y is None:
        std = 1e4
        # else:
        #     std = abs(self.prev_Y - self.curr_Y)/2
        return 1e5 + ( np.random.normal(0,std,size=1)  ).item() # for SY reward

    def reset(self):
        self.prev_Y, self.curr_Y = None, None
        self.step_count = 0
        if self.Y_bias == "none":
            self.bias = 0.
        elif self.Y_bias in ["stationary","non-stationary"]:
            self.bias = self._bias()

        """
        N, W and U are matrices of size (window_X,n_features), e.g. (28,7)
        We first generate tau+1 random features, which will be used to condition the p(W_t|w_{t-tau+1:t})
        """

        self.Ns = np.zeros(shape=(self.n_timesteps,self.n_features))
        self.Ws = np.zeros(shape=(self.n_timesteps,self.n_features))
        self.Us = np.zeros(shape=(self.n_timesteps,self.n_features))
        self.As = np.zeros(shape=(self.n_timesteps,))

        self.Ns[self.Ns==0] = 'nan'
        self.Ws[self.Ws==0] = 'nan'
        self.Us[self.Us==0] = 'nan'
        
        self.As[:self.tau] = [self.action_space.sample() for _ in range(self.tau)]

        # self.Ws[:self.tau+1] = np.random.multivariate_normal(mean=self.mu,cov=self.Sigma,size=self.tau+1)
        self.Ws = np.random.multivariate_normal(mean=self.mu,cov=self.Sigma,size=self.n_timesteps)
    
        self.Xs = self.rolling_window_fn(self.Ws,self.rho,fn=np.mean)

        self.curr_Y = self.reward_fn(self.Xs[0].reshape(1,-1),self.bias)

        return self.Xs[0]

    def step(self, a):
        self.step_count += 1
        self.As[1:] = self.As[:-1]
        self.As[0] = a

        # idx = 0
        action_vec = np.zeros(shape=(self.n_features))
        for i in range(self.tau):
            action_vec += self.true_action_vectors[int(self.As[i])] / self.tau

        W_tp1 = np.nanmean(self.Ws[:self.tau],0)

        #  = W_t_avg + action_vec # add the average action vector to the average context smoothed over tau last steps

        # L = self.L0s[idx]
        # Sigma = L.T @ L
        # N_tp1 = self.gaussian_conditional_sim(self.mu,Sigma,1,self.Ns[:self.tau])
        self.Ws[1:] = self.Ws[:-1]
        self.Ws[0] = W_tp1 + action_vec # drop the oldest observation

        self.Xs = self.rolling_window_fn(self.Ws,self.rho,fn=np.mean)

        # for j in range(self.Xs.shape[1]):
        #     self.Xs[:,j] = (self.Xs[:,j] - 2.7) / 20
        Y_t = self.make_reward().item()
        self.prev_Y = self.curr_Y
        self.curr_Y = Y_t
        
        return self.Xs[0], Y_t, False, None

class RecoGym(gym.Env):
    def __init__(self,n_features,n_actions,Y_bias,eval,seed):
        self.n_actions = n_actions
        self.n_features = n_features
        self.Y_bias = Y_bias
        self.eval = eval

        env_1_args['num_products'] = n_actions
        env_1_args['random_seed'] = seed
        self.env = gym.make('reco-gym-v1')
        self.env.init_gym(env_1_args)

        self.action_space = gym.spaces.Discrete( n_actions )
        self.state_space = gym.spaces.Box(low=0,high=1000,shape=(n_features,))

        self.true_action_vectors = np.eye(n_features)
        self.a_max = 1
        
        self.reset()

    def make_reward(self):
        if self.Y_bias == "non-stationary":
            self.bias = self._bias()
        if self.eval:
            self.bias = 0.
        Y_t = self.reward_fn(self.Zs,self.As,self.bias)
        return Y_t

    def reward_fn(self,actions,clicks,bias):
        filtered_actions = [a for i,a in enumerate(actions) if clicks[i]==1]
        if len(filtered_actions) == 0:
            reward = 0
            return reward
        filtered_actions = np.array(actions).max(0)
        affinity_score = filtered_actions @ self.env.omega
        return affinity_score.item()

    def _bias(self):
        return ( np.random.normal(0,1,size=1) ).item() # for sub-modular CTR reward
       
    def reset(self):
        self.Zs = []
        self.As = []
        if self.Y_bias == "none":
            self.bias = 0.
        elif self.Y_bias == "stationary":
            self.bias = self._bias()

        self.env.reset()
        obs,reward,done,info = self.env.step(None) # must be done
        organic_views = np.zeros(self.n_actions)
        if obs:
            for session in obs.sessions():
                organic_views[session['v']] += 1
        return organic_views

    def step(self,action):
        obs,reward,done,info = self.env.step(action)
        organic_views = np.zeros(self.n_actions)
        if obs:
            for session in obs.sessions():
                organic_views[session['v']] += 1
        self.Zs.append(self.env.Gamma[action])
        self.As.append(reward)
        new_reward = self.make_reward()
        # new_reward = reward
        return organic_views,new_reward,done,info



class HIVTreatment():
  
    state_names = ("T1", "T2", "T1*", "T2*", "V", "E")
    eps_values_for_actions = np.array([[0., 0.], [.7, 0.], [0., .3], [.7, .3]])

    def __init__(self, logspace=True, dt=5, model_derivatives=None, perturb_params=False,perturb_rate = 0.0, \
        p_T1=0, p_T2=0, p_T1s=0, p_T2s=0, p_V=0, p_E=0, **kw):
        """
        Initialize the environment.

        Keyword arguments:
        logspace --  return the state as log(state)
        dt -- change in time for each action (in days)
        model_derivatives -- option to pass specific model derivatives
        perturb_params -- boolean indicating whether to perturb the initial state
        p_T1 -- initial perturbation factor for specific state dimension
        p_T2 -- initial perturbation factor for specific state dimension
        p_T1s -- initial perturbation factor for specific state dimension
        p_T2s -- initial perturbation factor for specific state dimension
        p_V -- initial perturbation factor for specific state dimension
        p_E -- initial perturbation factor for specific state dimension
        """
        self.logspace = logspace
        if logspace:
            self.statespace_limits = np.array([[-5, 8]] * 6)
        else:
            self.statespace_limits = np.array([[0., 1e8]] * 6)
        if model_derivatives is None:
            model_derivatives = dsdt
        self.model_derivatives = model_derivatives
        self.dt = dt
        self.reward_bound = 1e300
        self.num_actions = 4
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.state_space = gym.spaces.Box(low=float('-inf'),high=float('inf'),shape=(6,))

        self.true_action_vectors = np.eye(self.num_actions)
        self.a_max = 1
        self.perturb_params = ('p_lambda1','p_lambda2','p_k1','p_k2','p_f', \
            'p_m1','p_m2','p_lambdaE','p_bE','p_Kb','p_d_E','p_Kd')
        self.perturb_rate = perturb_rate
        self.reset(perturb_params, p_T1, p_T2, p_T1s, p_T2s, p_V, p_E, **kw)


    def reset(self, perturb_params=False, p_T1=0, p_T2=0, p_T1s=0, p_T2s=0, p_V=0, p_E=0, **kw):
        """Reset the environment."""
        self.t = 0
        baseline_state = np.array([163573., 5., 11945., 46., 63919., 24.])
        # Slightly perturb initial state of patient
        # self.state = baseline_state + (baseline_state * np.array([p_T1, p_T2, p_T1s, p_T2s, p_V, p_E]))# could scale the random perturbations to reduce their effect by multiplyting by d < 1
        self.state = baseline_state + (baseline_state *
                                       np.random.uniform(low=-self.perturb_rate,high=self.perturb_rate, size=6))# could scale the random perturbations to reduce their effect by multiplyting by d < 1
        return self.observe()

    def observe(self):
        """Return current state."""
        if self.logspace:
            return np.log10(self.state)
        else:
            return self.state

    def is_done( self, episode_length=200, **kw ):
        """Check if we've finished the episode."""
        return True if self.t >= episode_length else False

    def calc_reward(self, action=0, state=None, **kw ):
        """Calculate the reward for the specified transition."""
        eps1, eps2 = self.eps_values_for_actions[action]
        if state is None:
            state = self.observe()
        if self.logspace:
            T1, T2, T1s, T2s, V, E = 10**state
        else:
            T1, T2, T1s, T2s, V, E = state
        # the reward function penalizes treatment because of side-effects
        reward = -0.1*V - 2e4*eps1**2 - 2e3*eps2**2 + 1e3*E
        # Constrain reward to be within specified range
        if np.isnan(reward):
            reward = -self.reward_bound
        elif reward > self.reward_bound:
            reward = self.reward_bound
        elif reward < -self.reward_bound:
            reward = -self.reward_bound
        return reward / 1e7


    def step(self, action, perturb_params=False, p_lambda1=0, p_lambda2=0, p_k1=0, \
        p_k2=0, p_f=0, p_m1=0, p_m2=0, p_lambdaE=0, p_bE=0, p_Kb=0, p_d_E=0, p_Kd=0, **kw):
        """Perform the specifed action and upate the environment.

        Arguments:
        action -- action to be taken

        Keyword Arguments:
        perturb_params -- boolean indicating whether to perturb dynamics (default: False)
        p_lambda1 -- hidden parameter (default: 0)
        p_lambda2 -- hidden parameter (default: 0)
        p_k1 -- hidden parameter (default: 0)
        p_k2 -- hidden parameter (default: 0)
        p_f -- hidden parameter (default: 0)
        p_m1 -- hidden parameter (default: 0)
        p_m2 -- hidden parameter (default: 0)
        p_lambdaE -- hidden parameter (default: 0)
        p_bE -- hidden parameter (default: 0)
        p_Kb -- hidden parameter (default: 0)
        p_d_E -- hidden parameter (default: 0)
        p_Kd -- hidden parameter (default: 0)
        """
        self.t += 1
        self.action = action
        eps1, eps2 = self.eps_values_for_actions[action]
        r = ode(self.model_derivatives).set_integrator('vode',nsteps=10000,method='bdf')
        t0 = 0
        deriv_args = (eps1, eps2, perturb_params, p_lambda1, p_lambda2, p_k1, p_k2, p_f, p_m1, p_m2, p_lambdaE, p_bE, p_Kb, p_d_E, p_Kd)
        r.set_initial_value(self.state, t0).set_f_params(deriv_args)
        self.state = r.integrate(self.dt)
        reward = self.calc_reward(action=action)
        return self.observe(), reward, self.is_done(), {}

def dsdt(t, s, params):
    """Wrapper for system derivative with respect to time"""
    derivs = np.empty_like(s)
    eps1,eps2,perturb_params,p_lambda1,p_lambda2,p_k1,p_k2,p_f,p_m1,p_m2,p_lambdaE,p_bE,p_Kb,p_d_E,p_Kd = params 
    dsdt_(derivs, s, t, eps1, eps2, perturb_params, p_lambda1, p_lambda2, p_k1, \
    p_k2, p_f, p_m1, p_m2, p_lambdaE, p_bE, p_Kb, p_d_E, p_Kd)
    return derivs


def dsdt_(out, s, t, eps1, eps2, perturb_params=False, p_lambda1=0, p_lambda2=0, p_k1=0, \
p_k2=0, p_f=0, p_m1=0, p_m2=0, p_lambdaE=0, p_bE=0, p_Kb=0, p_d_E=0, p_Kd=0):
    """System derivate with respect to time (days).

    Arguments:
    out -- output
    s -- state
    t -- time
    eps1 -- action effect
    eps2 -- action effect
    """
    # baseline model parameter constants
    lambda1 = 1e4     # Target cell, type 1, production rate *CAN BE VARIED*
    lambda2 = 31.98   # Target cell, type 2, production rate *CAN BE VARIED*
    d1 = 0.01         # Target cell, type 1, death rate
    d2 = 0.01         # Target cell, type 2, death rate
    f = .34           # Treatment efficacy, reduction in population 2 \in[0,1] *CAN BE VARIED*
    k1 = 8e-7         # Population 1, infection rate, *SENSITIVE TO REDUCTION, CAN BE VARIED*
    k2 = 1e-4         # Population 2, infection rate, *SENSITIVE TO REDUCTION, CAN BE VARIED*
    delta = .7        # Infected cell death rate
    m1 = 1e-5         # Immune-induced clearance rate, population 1 *CAN BE VARIED*
    m2 = 1e-5         # Immune-induced clearance rate, population 2 *CAN BE VARIED*
    NT = 100.         # Virions produced per infected cell
    c = 13.           # Virius natural death rate
    rho1 = 1.         # Average number of virions infecting type 1 cell
    rho2 = 1.         # Average number of virions infecting type 2 cell
    lambdaE = 1.      # Immune effector production rate *CAN BE VARIED*
    bE = 0.3          # Maximum birth rate for immune effectors *SENSITVE TO GROWTH, CAN BE VARIED*
    Kb = 100.         # Saturation constant for immune effector birth *CAN BE VARIED*
    d_E = 0.25        # Maximum death rate for immune effectors *CAN BE VARIED*
    Kd = 500.         # Saturation constant for immune effectors death *CAN BE VARIED*
    deltaE = 0.1      # Natural death rate for immune effectors

    if perturb_params:
        # Perturb empirically varied parameters...
        d = 1 # Scaling factor
        lambda1 += lambda1 * (p_lambda1 * d)
        lambda2 += lambda2 * (p_lambda2 * d)
        k1      += k1 * (p_k1 * d)
        k2      += k2 * (p_k2 * d)
        f       += f * (p_f * d)
        m1      += m1 * (p_m1 * d)
        m2      += m2 * (p_m2 * d)
        lambdaE += lambdaE * (p_lambdaE * d)
        bE      += bE * (p_bE * d)
        Kb      += Kb * (p_Kb * d)
        d_E     += d_E * (p_d_E * d)
        Kd      += Kd * (p_Kd * d)

    # decompose state
    T1, T2, T1s, T2s, V, E = s

    # compute derivatives
    tmp1 = (1. - eps1) * k1 * V * T1
    tmp2 = (1. - f * eps1) * k2 * V * T2
    out[0] = lambda1 - d1 * T1 - tmp1
    out[1] = lambda2 - d2 * T2 - tmp2
    out[2] = tmp1 - delta * T1s - m1 * E * T1s
    out[3] = tmp2 - delta * T2s - m2 * E * T2s
    out[4] = (1. - eps2) * NT * delta * (T1s + T2s) - c * V \
        - ((1. - eps1) * rho1 * k1 * T1 +
           (1. - f * eps1) * rho2 * k2 * T2) * V
    out[5] = lambdaE + bE * (T1s + T2s) / (T1s + T2s + Kb) * E \
        - d_E * (T1s + T2s) / (T1s + T2s + Kd) * E - deltaE * E

try:
    import numba
except ImportError as e:
    print("Numba acceleration unavailable, expect slow runtime.")
else:
    dsdt_ = numba.jit(
        numba.void(numba.float64[:], numba.float64[:], numba.float64, numba.float64, numba.float64, numba.bool_, \
         numba.float64, numba.float64, numba.float64, numba.float64, numba.float64, numba.float64, \
         numba.float64, numba.float64, numba.float64, numba.float64, numba.float64, numba.float64),
        nopython=True, nogil=True)(dsdt_)



if __name__ == "__main__":
    env = HIVTreatment()
    s = env.reset()
    print(s)
    import ipdb; ipdb.set_trace()