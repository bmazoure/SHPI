import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from vowpalwabbit import pyvw


class Regressor(torch.nn.Module):
    def __init__(self, dim_list):
        super(Regressor, self).__init__()
        self.layers = []
        for i in range(len(dim_list)-1):
            self.layers.append( torch.nn.Linear(dim_list[i], dim_list[i+1]) )
            if i < len(dim_list)-2:
                self.layers.append( nn.ReLU() )
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

def CBPolicy():
    def __init__(self, **args):
        pass
    def probs(self, X):
        pass
    def partial_fit(self, X, z, r, p):
        """
        Fits the specified policy learner to partially-labeled data collected from a different policy.
        
        Parameters
        ----------
        X : array (n_samples, n_features)
            Matrix of covariates for the available data.
        z : array (n_samples), int type
            Arms or actions that were chosen for each observations.
        r : array (n_samples), float type
            Rewards that were observed for the chosen actions.
        p : array (n_samples)
            Behavior policy weights.
        """
        pass

class NeuralPolicy():
    def __init__(self, dim_list):
        self.learner = Regressor(dim_list=dim_list)
        self.opt = torch.optim.Adam(self.learner.parameters(),lr=1e-3)

    def probs(self, x):
        logits = self.learner.forward(x)
        return F.softmax(logits)

    def partial_fit(self, X, Z, r, p):
        pass

class VWPolicy():
    def __init__(self, params, n_actions):
        self.learner = pyvw.vw(params)
        self.params = params
        self.n_actions = n_actions

    def handle_vw_(self, x, z=None,r=None,p=None):
        sharedfeat = ' '.join([ 'shared |x'] + [ f'{k}:{v}' for k, v in enumerate(x)])
        if r is None and p is None:
            # predict pmf
            exstr = '\n'.join([ sharedfeat ] + [ f' |a {k+1}:1' for k in range(self.n_actions) ])
            return exstr
        # learn on example
        labelexstr = '\n'.join([ sharedfeat ] + [ f' {l} |a {k+1}:1' 
                                                  for k in range(self.n_actions)
                                                  for l in (f'0:{r if k == z else 0}:{p}' if z == k else '',)
                                                ])
        return labelexstr

    def partial_fit_example_(self, x,z,r,p):
        # learn_example = str(z+1) + ":" + str(r) + ":" + str(p) + " | " + " ".join([str(feature) for feature in x])
        learn_example = self.handle_vw_(x,z,r,p)
        self.learner.learn(learn_example)

    def probs_example_(self, x, z):
        # test_example = " | " + " ".join([str(feature) for feature in x])
        test_example = self.handle_vw_(x,z,r=None,p=None)
        probs = self.learner.predict(test_example)
        if z is not None:
            probs = probs[z]
        return probs
    
    def act(self, X):
        # test_example = " | " + " ".join([str(feature) for feature in X])
        test_example = self.handle_vw_(X,z=None,r=None,p=None)
        probs = np.array(self.learner.predict(test_example))
        probs /= sum(probs)
        action = np.random.choice(np.arange(0,len(probs)),size=1,p=probs)
        return action

    def partial_fit(self, X, z, r, p):
        for i in range(len(X)):
            self.partial_fit_example_(X[i],z[i],r[i],p[i])

    def probs(self, X, z):
        acc = []
        for i in range(len(X)):
            prob = self.probs_example_(X[i],z[i])
            acc.append(prob)
        return np.array(acc)

def fit_policy(X,Z,r,p,policy,reward_fn,feature_weights):
    policy.fit(X=X, a=Z, r=r, p=p)
    
    Z_t = policy.predict(X)
    pred_Y_t = reward_fn(feature_weights,X,bias=0)

    return policy, pred_Y_t

class FC_Q(nn.Module):
    """
    BCQ Q-net
    """
    def __init__(self, state_dim, latent_dim, num_actions):
        super(FC_Q, self).__init__()
        self.q1 = nn.Linear(state_dim, latent_dim)
        self.q2 = nn.Linear(latent_dim, latent_dim)
        self.q3 = nn.Linear(latent_dim, num_actions)

        self.i1 = nn.Linear(state_dim, latent_dim)
        self.i2 = nn.Linear(latent_dim, latent_dim)
        self.i3 = nn.Linear(latent_dim, num_actions)


    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        return self.q3(q), F.log_softmax(i, dim=1), i