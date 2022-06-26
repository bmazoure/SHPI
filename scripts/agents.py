import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scripts.models import FC_Q
import copy
from scipy.special import softmax
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
tf.compat.v1.enable_eager_execution()

class Agent():
    def __init__(self,model, env, device='cpu'):
        self.action_space = env.action_space
        self.model = model
        self.device = device

    def act(self,state):
        pass

class RandomAgent(Agent):
    def act(self, state, eps=1):
        return self.action_space.sample(), 1/self.action_space.n

class RandomCosAgent(Agent):
    def __init__(self,model, env, device='cpu'):
        self.action_space = env.action_space
        self.model = model
        self.device = device

        self.counter = 0
        self.e1 = np.zeros(shape=(env.true_action_vectors.shape[1]))
        self.e1[0] = 1
        self.action_angles = env.true_action_vectors @ self.e1 / env.a_max

        self.U = np.random.uniform(0,2*np.pi,size=(1,)).item()

    def act(self, state, eps=1):
        cts_a = np.cos(self.counter + self.U)
        p = softmax((self.action_angles-cts_a)**2)
        discrete_a = np.random.choice([i for i in range(self.action_space.n)],p=p)
        self.counter += 1
        return discrete_a, p[discrete_a]
    
class SARSAAgent(Agent):
    def act(self, state, eps=0.1):
        if np.random.uniform(size=1).item() < eps:
            return self.action_space.sample(), 1/self.action_space.n
        state = torch.FloatTensor([state]).to(self.device)
        # if torch.cuda.is_available():
            # state = state.cuda()
        q_s_a = self.model(state)
        action = q_s_a.max(1)[1].item()
        prob = (q_s_a.detach().cpu().numpy() / q_s_a.detach().cpu().numpy().sum())[0][action]
        return action, prob

class QLAgent(Agent):
    def __init__(self,model, env, device='cpu'):
        from scripts.replay_buffer import ReplayBuffer
        self.action_space = env.action_space
        self.model = model
        self.device = device
        self.replay_buffer = ReplayBuffer(100000)

    def act(self, state, eps=0.1):
        if np.random.uniform(size=1).item() < eps:
            return self.action_space.sample(), 1/self.action_space.n
        state = torch.FloatTensor([state]).to(self.device)
        # if torch.cuda.is_available():
            # state = state.cuda()
        q_s_a = self.model(state)
        action = q_s_a.max(1)[1].item()
        prob = (q_s_a.detach().cpu().numpy() / q_s_a.detach().cpu().numpy().sum())[0][action]
        return action, prob

class CBAgent(Agent):
    def act(self, state, eps=0):
        if np.any(np.isnan(state)):
            return self.action_space.sample(), 1/self.action_space.n
        if np.random.uniform(size=1).item() < eps:
            return self.action_space.sample(), 1/self.action_space.n
        action = self.model.act(state).item()
        prob = self.model.probs_example_(state,action)
        return action, prob

class DiscreteBCQAgent(object):
    def __init__(
        self, 
        num_actions,
        state_dim,
        latent_dim,
        device,
        BCQ_threshold=0.3,
        discount=0.99,
        optimizer="Adam",
        optimizer_parameters={},
        polyak_target_update=False,
        target_update_frequency=8e3,
        tau=0.005,
        initial_eps = 1,
        end_eps = 0.001,
        eps_decay_period = 25e4,
        eval_eps=0.001,
    ):
    
        self.device = device

        # Determine network type
        self.Q = FC_Q(state_dim, latent_dim, num_actions).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Threshold for "unlikely" actions
        self.threshold = BCQ_threshold

        # Number of training iterations
        self.iterations = 0


    def act(self, state, eval=False):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > self.eval_eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
                q, imt, i = self.Q(state)
                imt = imt.exp()
                imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
                # Use large negative number to mask actions from argmax
                return int((imt * q + (1. - imt) * -1e8).argmax(1)), 1
        else:
            return np.random.randint(self.num_actions), 1/self.num_actions


    def partial_fit(self, state,action,next_state,reward,done):
        # Sample replay buffer
        # state, action, next_state, reward, done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            q, imt, i = self.Q(next_state)
            imt = imt.exp()
            imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

            q, imt, i = self.Q_target(next_state)
            target_Q = reward.to(self.device) + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate
        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action.to(self.device))

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q.reshape(-1))
        i_loss = F.nll_loss(imt, action.reshape(-1))

        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()


    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
           target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
             self.Q_target.load_state_dict(self.Q.state_dict())


class DiceAgent(CBAgent):
    def __init__(self,model, env, device='cpu'):
        self.action_space = env.action_space
        self.model = model
        self.device = device

    def distribution(self,x,z):
        probs = self.model.probs(x,z)
        return probs