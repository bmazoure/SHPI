import copy
import json
import os
import pickle

import gym
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import pandas as pd
import sklearn
import tensorflow.compat.v2 as tf
import torch
from sklearn.datasets import make_spd_matrix
from sklearn.neural_network import MLPClassifier
from torch.nn import functional as F
from tqdm import tqdm

tf.compat.v1.enable_v2_behavior()
tf.compat.v1.enable_eager_execution()

from tf_agents.environments import gym_wrapper, tf_py_environment
from tf_agents.environments.suite_gym import load as load_gym

import scripts.estimator_dice as estimator_lib
import scripts.utils_dice as common_utils
from scripts.agents import (CBAgent, DiscreteBCQAgent, QLAgent, RandomAgent,
                            RandomCosAgent, SARSAAgent)
from scripts.data_collection import BatchDataset
from scripts.dataset_dice import Dataset, EnvStep, StepType
from scripts.envs import (HIVTreatment, RecoGym,
                          Toy_GaussianContextDiscreteAction,
                          Toy_GaussianContextDiscreteAction_simple)
from scripts.models import Regressor, VWPolicy
from scripts.neural_dice import NeuralDice
from scripts.replay_buffer import ReplayBuffer
from scripts.training_routines import (fit_policy_offline_dataset,
                                       fit_policy_online)
from scripts.utils import (StyblinskiTang_reward, rollout,
                           session_reward_mean_Y_t, session_reward_sum_Y_t,
                           session_reward_Y_T, session_reward_Y_T_m_Y_1,
                           set_seed, sliding_window)
from scripts.utils_dice import reward_fn as reward_fn_dice
from scripts.value_network_dice import ValueNetwork
from scripts.wrappers import DiscreteActionWrapper1d

"""
Synthetic:
python run_experiments.py --env_name toy --n_features 2 --n_actions 10 --n_timesteps_offline 150 --n_timesteps_online 150 --n_timesteps_test 150 --n_traj_offline 10 --n_traj_online 1000 --delta 20 --rho 30 --k 5 --policy_improvement_steps=5 --Y_bias="none" --eps_behavior 0.3 --dice_RL 0 --seed 0

Synthetic - Large:
python run_experiments.py --env_name toy --n_features 100 --n_actions 300 --n_timesteps_offline 150 --n_timesteps_online 150 --n_timesteps_test 150 --n_traj_offline 10 --n_traj_online 1000 --delta 20 --rho 30 --k 5 --policy_improvement_steps=5 --Y_bias="none" --eps_behavior 0.3 --dice_RL 0 --seed 0

"""


def main(cmd_args):
    args = {
            # Env params
            'seed':                   np.random.randint(0,10000000,1).item(),
            'device':                 torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'env_name':               'toy',
            'n_features':             10,
            'n_actions':              cmd_args['n_actions'],

            'a_max':                  0.1,
            'tau':                    1,
            'reward_fn':              StyblinskiTang_reward,
            'p_mu_threshold':         0.01,

            'Y_bias':                 'none',
            # Dataset params
            'n_timesteps_offline':    150,
            'n_timesteps_online':     150,
            'n_timesteps_test':       150,
            'n_traj_offline':         10,
            'n_traj_online':          1000,
            'n_traj_test':            100,
            'delta':                  25,
            'rho':                    28,
            # RL params
            'eps_behavior':           -1, # [0-1]: e-greedy SARSA
            'cb_params':              "--coin --learning_rate 0.01 --cb_type mtr --cb_explore_adf ",
            'k':                      5,
            'offline_RL_agent':       'BCQ',
            'online_RL_agent':        'Online SARSA (MDP)',
            'gamma_RL':               0.99,
            'lr_RL':                  0.003,
            'batch_size_RL':          64,
            'eps_RL':                 0.1,
            'policy_improvement_steps':0,
            'gamma_behavior':         0,
            # From DICE repo
            'dice_RL':                0,
            'dice_primal_regularizer':0,
            'dice_dual_regularizer':  1,
            'dice_zero_reward':       0,
            'dice_norm_regularizer':  1,
            'dice_zeta_pos':          1,
            'dice_f_exponent':        2,
            'dice_primal_form':       False,
            'dice_nu_regularizer':    0,
            'dice_zeta_regularizer':  0,
            'dice_solve_for_state_action_ratio': False
    }

    args = dict(args, **cmd_args) # overwrite keys which exist in cmd_args

    primal_regularizer = args['dice_primal_regularizer']
    dual_regularizer = args['dice_dual_regularizer']
    zero_reward = args['dice_zero_reward']
    norm_regularizer = args['dice_norm_regularizer']
    zeta_pos = args['dice_zeta_pos']
    f_exponent = args['dice_f_exponent']
    primal_form = args['dice_primal_form']
    nu_regularizer = args['dice_nu_regularizer']
    zeta_regularizer = args['dice_zeta_regularizer']

    run_dir = os.path.join(args['output_dir'],'runs') 
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    args_cp = copy.deepcopy(args)
    args_cp['reward_fn'] = args_cp['reward_fn'].__name__
    args_cp['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    with open(os.path.join(run_dir,'params.json'),'w') as fp:
        json.dump(args_cp,fp)

    set_seed(args['seed'],cuda=torch.cuda.is_available())

    """
    Create and save env
    """

    if args['env_name'] == 'toy':
        Sigma = np.eye(args['n_features']) / 10 # n_actions^\tau x \tau+1 x \tau+1
        if args['n_features'] == 2:
            mu = np.array([0,-10])
        else:
            mu = np.zeros(args['n_features'])
            mu[np.random.randint(0,high=args['n_features'],size=(1,)).item()] = -10
            
        dummy_env = Toy_GaussianContextDiscreteAction_simple(**args,mu=mu,Sigma=Sigma,eval=False)
        eval_env = Toy_GaussianContextDiscreteAction_simple(**args,mu=mu,Sigma=Sigma,eval=True)
        session_reward = session_reward_Y_T_m_Y_1
    elif args['env_name'] == 'reco-gym-v1':
        dummy_env = RecoGym(n_features=args['n_features'],n_actions=args['n_actions'],seed=args['seed'],Y_bias=args['Y_bias'],eval=False)
        eval_env = RecoGym(n_features=args['n_features'],n_actions=args['n_actions'],seed=args['seed'],Y_bias=args['Y_bias'],eval=True)
        session_reward = session_reward_mean_Y_t
    elif args['env_name'] == 'hiv':
        dummy_env = HIVTreatment()
        eval_env = HIVTreatment()
        session_reward = session_reward_sum_Y_t
    else:
        dummy_env = gym.make(args['env_name'])
        eval_env = gym.make(args['env_name'])
        dummy_env.state_space = dummy_env.observation_space
        eval_env.state_space = eval_env.observation_space
        session_reward = session_reward_sum_Y_t
    
    if 'Pendulum' in args['env_name']:
        dummy_env = DiscreteActionWrapper1d(dummy_env,n_discrete_actions=args['n_actions'])
        eval_env = DiscreteActionWrapper1d(eval_env,n_discrete_actions=args['n_actions'])

    def make_env():
        return dummy_env

    def make_eval_env():
        return eval_env

    pickle.dump(dummy_env,open(os.path.join(run_dir,'env.pkl'),'wb'))

    """
    ONLINE
    ---------------
    Train deep RL / CB by online interaction with environment
    """

    random_agent = RandomAgent(model=None,env=dummy_env)
    loss_func = torch.nn.MSELoss()

    critic = Regressor(dim_list=[args['n_features'],128,128,args['n_actions']]).to(args['device'])
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args['lr_RL'], amsgrad=True)
    pi3_Y_t_raw_online = VWPolicy(args['cb_params'] + " --final_regressor " + os.path.join(run_dir,"pi3_online.weights"),args['n_actions'])
    cb3_online = CBAgent(pi3_Y_t_raw_online,env=dummy_env)
    
    model_type_RL = args['online_RL_agent']
    if model_type_RL == 'Online SARSA (MDP)':
        online_RL_agent = SARSAAgent(critic,dummy_env,device=args['device'])
    elif model_type_RL == 'Online DQN (MDP)':
        online_RL_agent = QLAgent(critic,dummy_env,device=args['device'])


    online_RL_agent, critic, r_acc_RL, v_acc_RL, l_acc_RL = fit_policy_online(make_env,online_RL_agent,random_agent,critic,opt_critic,args['n_traj_online'],args['n_timesteps_online'],args['eps_RL'],loss_func,args['gamma_RL'],args['batch_size_RL'],model_type_RL,args['device'])
    torch.save(critic.state_dict(),os.path.join(run_dir,args['online_RL_agent']+'.pth'))

    pi3_Y_t_raw_online, critic, r_acc_online_CB, v_acc_online_CB, l_acc_online_CB = fit_policy_online(make_env,cb3_online,random_agent,None,None,args['n_traj_online'],args['n_timesteps_online'],args['eps_RL'],loss_func,args['gamma_RL'],args['batch_size_RL'],'Online $r_t=Y_t$ (CB)',args['device'])

    """
    OFFLINE
    ---------------
    Collect a logged dataset by some RL agent mu destroyed with eps-greedy (eps=1 is random_agent exactly)

    All of (X_t,Y_t,Z_t,P_mu_t) should be of size N*rho
    """

    if args['eps_behavior'] == -1: # CosAgent
        behavior_agent = RandomCosAgent(model=None,env=dummy_env)
    elif 0 <= args['eps_behavior'] <= 1: # e-greedy + SARSA
        behavior_agent = online_RL_agent
        
    offline_dataset = BatchDataset(agent=behavior_agent,
                                   make_env=make_env,
                                   n_timesteps_offline=args['n_timesteps_offline'],
                                   n_traj_offline=args['n_traj_offline'],
                                   rho=args['rho'], tau=args['tau'],
                                   delta=args['delta'],
                                   eps_behavior=args['eps_behavior'],
                                   p_mu_threshold=args['p_mu_threshold'],
                                   gamma_behavior=args['gamma_behavior'])
    offline_dataset.construct() # deploy the agent and collect a dataset
    
    """
    OFFLINE
    ---------------
    Define all offline CB and RL models, with optimizers and function approximation
    """

    """
    DualDice estimator for d^pi/d^mu
    """
    full_spec = common_utils._create_spec(dummy_env,_episode_step_limit=args['n_timesteps_offline'])
    if args['dice_solve_for_state_action_ratio']:
        input_spec = (full_spec.observation,
                full_spec.action)
    else:
        input_spec = (full_spec.observation,)
        
    activation_fn = tf.nn.relu
    kernel_initializer = tf.keras.initializers.GlorotUniform()
    hidden_dims = (128, 128)

    nu_network_pi1 = ValueNetwork(
    input_spec,
    fc_layer_params=hidden_dims,
    activation_fn=activation_fn,
    kernel_initializer=kernel_initializer,
    last_kernel_initializer=kernel_initializer)
    
    output_activation_fn = tf.math.square if zeta_pos else tf.identity
    zeta_network_pi1 = ValueNetwork(
        input_spec,
        fc_layer_params=hidden_dims,
        activation_fn=activation_fn,
        output_activation_fn=output_activation_fn,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=kernel_initializer)

    nu_optimizer_pi1 = tf.keras.optimizers.Adam(0.0001, clipvalue=1.0)
    zeta_optimizer_pi1 = tf.keras.optimizers.Adam(0.0001, clipvalue=1.0)
    lam_optimizer_pi1 = tf.keras.optimizers.Adam(0.0001, clipvalue=1.0)


    estimator_pi1 = NeuralDice(
        full_spec,
        nu_network_pi1,
        zeta_network_pi1,
        nu_optimizer_pi1,
        zeta_optimizer_pi1,
        lam_optimizer_pi1,
        args['gamma_RL'],
        solve_for_state_action_ratio=args['dice_solve_for_state_action_ratio'],
        zero_reward=zero_reward,
        f_exponent=f_exponent,
        primal_form=primal_form,
        reward_fn=reward_fn_dice,
        primal_regularizer=primal_regularizer,
        dual_regularizer=dual_regularizer,
        norm_regularizer=norm_regularizer,
        nu_regularizer=nu_regularizer,
        zeta_regularizer=zeta_regularizer)

    nu_network_pi2 = ValueNetwork(
    input_spec,
    fc_layer_params=hidden_dims,
    activation_fn=activation_fn,
    kernel_initializer=kernel_initializer,
    last_kernel_initializer=kernel_initializer)

    zeta_network_pi2 = ValueNetwork(
        input_spec,
        fc_layer_params=hidden_dims,
        activation_fn=activation_fn,
        output_activation_fn=output_activation_fn,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=kernel_initializer)

    nu_optimizer_pi2 = tf.keras.optimizers.Adam(0.0001, clipvalue=1.0)
    zeta_optimizer_pi2 = tf.keras.optimizers.Adam(0.0001, clipvalue=1.0)
    lam_optimizer_pi2 = tf.keras.optimizers.Adam(0.0001, clipvalue=1.0)


    estimator_pi2 = NeuralDice(
        full_spec,
        nu_network_pi2,
        zeta_network_pi2,
        nu_optimizer_pi2,
        zeta_optimizer_pi2,
        lam_optimizer_pi2,
        args['gamma_RL'],
        solve_for_state_action_ratio=args['dice_solve_for_state_action_ratio'],
        zero_reward=zero_reward,
        f_exponent=f_exponent,
        primal_form=primal_form,
        reward_fn=reward_fn_dice,
        primal_regularizer=primal_regularizer,
        dual_regularizer=dual_regularizer,
        norm_regularizer=norm_regularizer,
        nu_regularizer=nu_regularizer,
        zeta_regularizer=zeta_regularizer)

    pi1_Y_t = VWPolicy(args['cb_params'] + " --final_regressor " + os.path.join(run_dir,"pi1.weights"),args['n_actions'])
    pi2_Y_t_minus_Y_tm1 = VWPolicy(args['cb_params'] + " --final_regressor " + os.path.join(run_dir,"pi2.weights"),args['n_actions'])
    pi3_Y_t_raw = VWPolicy(args['cb_params'] + " --final_regressor " + os.path.join(run_dir,"pi3.weights"),args['n_actions'])
    pi4_Y_t_minus_Y_tm1_raw = VWPolicy(args['cb_params'] + " --final_regressor " + os.path.join(run_dir,"pi4.weights"),args['n_actions'])
    V_cb_1 = Regressor(dim_list=[args['n_features'],128,128,1]).to(args['device'])
    V_cb_2 = Regressor(dim_list=[args['n_features'],128,128,1]).to(args['device'])
    Q_cb_1 = Regressor(dim_list=[args['n_features'],128,128,args['n_actions']]).to(args['device'])
    Q_cb_2 = Regressor(dim_list=[args['n_features'],128,128,args['n_actions']]).to(args['device'])
    opt_V_1 = torch.optim.Adam(V_cb_1.parameters(), lr=args['lr_RL'], amsgrad=False)
    opt_V_2 = torch.optim.Adam(V_cb_2.parameters(), lr=args['lr_RL'], amsgrad=False)
    opt_Q_1 = torch.optim.Adam(Q_cb_1.parameters(), lr=args['lr_RL'], amsgrad=True)
    opt_Q_2 = torch.optim.Adam(Q_cb_2.parameters(), lr=args['lr_RL'], amsgrad=True)

    pi3_Y_t_raw_online = VWPolicy(args['cb_params'] + " --final_regressor " + os.path.join(run_dir,"pi3_online.weights"),args['n_actions'])
    cb3_online = CBAgent(pi3_Y_t_raw_online,env=dummy_env)
    # loss_func = F.smooth_l1_loss
    if args['offline_RL_agent'] == 'BCQ':
        offline_RL = DiscreteBCQAgent(args['n_actions'],args['n_features'],128,args['device'],discount=args['gamma_RL'],optimizer_parameters={'lr':args['lr_RL']})

    """
    Fit all CB algorithms and offline RL on the MDP data cut into chunks of weakly correlated episodes
    Initial phase under mu=random
    """

    train_mask = (np.random.uniform(0,1,size=offline_dataset.X_t.shape[:2]) < 1-args['gamma_behavior'])

    if args['policy_improvement_steps'] > 0:
        for pi_step in range(args['policy_improvement_steps']):
            _,                       (V_cb_1,Q_cb_1),       _,          v_acc_1, l_acc_1 = fit_policy_offline_dataset(offline_dataset,make_env,None,None,V_cb_1,opt_V_1,Q_cb_1,opt_Q_1,loss_func,args['gamma_RL'],train_mask,args['k'],'$r_t=Y_t$ (Value)',session_reward,args['device'],args['dice_RL'],args['dice_solve_for_state_action_ratio'],args['batch_size_RL'])
            _,                       (V_cb_2,Q_cb_2),       _,          v_acc_2, l_acc_2 = fit_policy_offline_dataset(offline_dataset,make_env,None,None,V_cb_2,opt_V_2,Q_cb_2,opt_Q_2,loss_func,args['gamma_RL'],train_mask,args['k'],'$r_t=Y_t-Y_{t-1}$ (Value)',session_reward,args['device'],args['dice_RL'],args['dice_solve_for_state_action_ratio'],args['batch_size_RL'])
    
    pi1_Y_t,                 _,      r_acc_1,          a_acc_1, z_acc_1 = fit_policy_offline_dataset(offline_dataset,make_env,pi1_Y_t,estimator_pi1,V_cb_1,opt_V_1,Q_cb_1,opt_Q_1,loss_func,args['gamma_RL'],train_mask,args['k'],'$r_t=Y_t$ (CB++)',session_reward,args['device'],args['dice_RL'],args['dice_solve_for_state_action_ratio'],args['batch_size_RL'])
    pi2_Y_t_minus_Y_tm1,     _,      r_acc_2,          a_acc_2, z_acc_2 = fit_policy_offline_dataset(offline_dataset,make_env,pi2_Y_t_minus_Y_tm1,estimator_pi2,V_cb_2,opt_V_2,Q_cb_1,opt_Q_1,loss_func,args['gamma_RL'],train_mask,args['k'],'$r_t=Y_t-Y_{t-1}$ (CB++)',session_reward,args['device'],args['dice_RL'],args['dice_solve_for_state_action_ratio'],args['batch_size_RL'])
    pi3_Y_t_raw,             _,      r_acc_3,          _,       _       = fit_policy_offline_dataset(offline_dataset,make_env,pi3_Y_t_raw,None,None,None,None,None,loss_func,args['gamma_RL'],train_mask,args['k'],'$r_t=Y_t$ (CB)',session_reward,args['device'],args['dice_RL'],args['dice_solve_for_state_action_ratio'],args['batch_size_RL'])
    pi4_Y_t_minus_Y_tm1_raw, _,      r_acc_4,          _,       _       = fit_policy_offline_dataset(offline_dataset,make_env,pi4_Y_t_minus_Y_tm1_raw,None,None,None,None,None,loss_func,args['gamma_RL'],train_mask,args['k'],'$r_t=Y_t-Y_{t-1}$ (CB)',session_reward,args['device'],args['dice_RL'],args['dice_solve_for_state_action_ratio'],args['batch_size_RL'])
    offline_RL,              _,      r_acc_offline_RL, _,       _       = fit_policy_offline_dataset(offline_dataset,make_env,offline_RL,None,None,None,None,None,loss_func,args['gamma_RL'],train_mask,args['k'],'Offline BCQ (MDP)',session_reward,args['device'],args['dice_RL'],args['dice_solve_for_state_action_ratio'],args['batch_size_RL'])
    torch.save(offline_RL.Q.state_dict(),os.path.join(run_dir,args['offline_RL_agent']+'.pth'))

    offline_dataset_extra = offline_dataset # or generate new one via pi rollouts

    a_acc_1['iteration'] = 0
    a_acc_2['iteration'] = 0

    adv_acc = pd.concat([a_acc_1,a_acc_2])

    if args['policy_improvement_steps'] > 0:
        for pi_step in range(args['policy_improvement_steps']):
            print('Step %d/%d for advantage estimation' % (pi_step+1,args['policy_improvement_steps']))
            pi1_Y_t,             _, r_acc_1, a_acc_1, _ = fit_policy_offline_dataset(offline_dataset_extra,make_env,pi1_Y_t,estimator_pi1,V_cb_2,opt_V_1,Q_cb_1,opt_Q_1,loss_func,args['gamma_RL'],train_mask,args['k'],'$r_t=Y_t$ (CB++)',session_reward,args['device'],args['dice_RL'],args['dice_solve_for_state_action_ratio'],args['batch_size_RL'])
        
            pi2_Y_t_minus_Y_tm1, _, r_acc_2, a_acc_2, _ = fit_policy_offline_dataset(offline_dataset_extra,make_env,pi2_Y_t_minus_Y_tm1,estimator_pi2,V_cb_2,opt_V_2,Q_cb_2,opt_Q_2,loss_func,args['gamma_RL'],train_mask,args['k'],'$r_t=Y_t-Y_{t-1}$ (CB++)',session_reward,args['device'],args['dice_RL'],args['dice_solve_for_state_action_ratio'],args['batch_size_RL'])

            a_acc_1['iteration'] = pi_step+1
            a_acc_2['iteration'] = pi_step+1
            adv_acc = pd.concat([adv_acc,a_acc_1,a_acc_2])
    
    adv_acc['step'] = adv_acc.index
    adv_acc.loc[adv_acc['policy']=='$r_t=Y_t$ (CB++)','step'] = np.arange(0,len(adv_acc[adv_acc['policy']=='$r_t=Y_t$ (CB++)']))
    adv_acc.loc[adv_acc['policy']=='$r_t=Y_t-Y_{t-1}$ (CB++)','step'] = np.arange(0,len(adv_acc[adv_acc['policy']=='$r_t=Y_t-Y_{t-1}$ (CB++)']))

    """
    Test-time rollouts of CB policies, offline and online RL in the env simulator
    """

    cb1 = CBAgent(pi1_Y_t,env=dummy_env)
    cb2 = CBAgent(pi2_Y_t_minus_Y_tm1,env=dummy_env)
    cb3 = CBAgent(pi3_Y_t_raw,env=dummy_env)
    cb4 = CBAgent(pi4_Y_t_minus_Y_tm1_raw,env=dummy_env)

    _,rewards_0,         _, _ = rollout(lambda state:behavior_agent.act(state,args['eps_behavior']),make_eval_env,H=args['n_timesteps_test'],n_traj=args['n_traj_test'])
    _,rewards_1,         _, _ = rollout(lambda state:cb1.act(state),make_eval_env,H=args['n_timesteps_test'],n_traj=args['n_traj_test'])
    _,rewards_2,         _, _ = rollout(lambda state:cb2.act(state),make_eval_env,H=args['n_timesteps_test'],n_traj=args['n_traj_test'])
    _,rewards_3,         _, _ = rollout(lambda state:cb3.act(state),make_eval_env,H=args['n_timesteps_test'],n_traj=args['n_traj_test'])
    _,rewards_4,         _, _ = rollout(lambda state:cb4.act(state),make_eval_env,H=args['n_timesteps_test'],n_traj=args['n_traj_test'])
    _,rewards_3_online,  _, _ = rollout(lambda state:cb3_online.act(state),make_eval_env,H=args['n_timesteps_test'],n_traj=args['n_traj_test'])
    _,rewards_offline_RL,_, _ = rollout(lambda state:offline_RL.act(state,eval=True),make_eval_env,H=args['n_timesteps_test'],n_traj=args['n_traj_test'])
    _,rewards_online_RL, _, _ = rollout(lambda state:online_RL_agent.act(state,eps=0.01),make_eval_env,H=args['n_timesteps_test'],n_traj=args['n_traj_test'])

    learning_curves = pd.concat([r_acc_1,r_acc_2,r_acc_3,r_acc_4,r_acc_offline_RL,r_acc_RL,r_acc_online_CB])
    learning_curves.to_csv(os.path.join(run_dir,'learning_curves.csv'),index=False)

    value_curves = pd.concat([v_acc_1,v_acc_2,v_acc_online_CB,v_acc_RL]) 
    value_curves.to_csv(os.path.join(run_dir,'value_curves.csv'),index=False)

    loss_curves = pd.concat([l_acc_1,l_acc_2,l_acc_online_CB,l_acc_RL]) 
    loss_curves.to_csv(os.path.join(run_dir,'loss_curves.csv'),index=False)

    adv_acc.to_csv(os.path.join(run_dir,'advantage_curves.csv'),index=False)

    if args['dice_RL']:
        zeta_loss_curves = pd.concat([z_acc_1,z_acc_2]) 
        zeta_loss_curves.to_csv(os.path.join(run_dir,'zeta_loss_curves.csv'),index=False)

    online_results = pd.DataFrame({
                                'returns':np.concatenate([[session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards_0],
                                                          [session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards_3],
                                                          [session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards_4],
                                                          [session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards_1],
                                                          [session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards_2],
                                                          [session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards_3_online],
                                                          [session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards_offline_RL],
                                                          [session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards_online_RL]]),
                                'policy':np.repeat(['$\mu$ (CB)',
                                                    '$r_t=Y_t$ (CB)',
                                                    '$r_t=Y_t-Y_{t-1}$ (CB)',
                                                    '$r_t=Y_t$ (CB++)',
                                                    '$r_t=Y_t-Y_{t-1}$ (CB++)',
                                                    'Online $r_t=Y_t$ (CB)',
                                                    'Offline BCQ (MDP)',
                                                    'Online SARSA (MDP)'],args['n_traj_test'])
                                })  

    online_results.to_csv(os.path.join(run_dir,'test_returns.csv'),index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launcher for ML jobs')
    parser.add_argument('--seed', help='Random seed',type=int,default=1234)
    parser.add_argument('--output_dir', type=str, default="./")
    # MDP parameters
    parser.add_argument('--env_name', help='Env name (either toy, recogym or hiv)',type=str,default='toy')
    parser.add_argument('--n_features', help='Context size',type=int,default=2)
    parser.add_argument('--n_actions', help='Number of possible actions',type=int,default=4)
    parser.add_argument('--delta', help='Sliding window overlap',type=int,default=7)
    parser.add_argument('--tau', help='Order of Markov chain',type=int,default=1) # Obsolete parameter, regulates p(X_tau+1|X_1:tau,A_1:tau)
    parser.add_argument('--rho', help='Size of dependence window',type=int,default=28)
    parser.add_argument('--Y_bias', help='Additive reward bias',type=str,default="none",choices=["none","stationary","non-stationary"])
    # Algorithms parameters
    parser.add_argument('--k', help='Number of timesteps in TD false horizon',type=int,default=1)
    parser.add_argument('--policy_improvement_steps', help='Number of policy improvement steps (1 step = 1 dataset pass)',type=int,default=0)
    parser.add_argument('--n_traj_offline', help='Number of training episodes for offline learner',type=int,default=10)
    parser.add_argument('--n_traj_online', help='Number of training episodes for online learner',type=int,default=1000)
    parser.add_argument('--n_timesteps_offline', help='Number of training steps/ episode for offline learner',type=int,default=150)
    parser.add_argument('--n_timesteps_online', help='Number of training steps/ episode for online learner',type=int,default=150)
    parser.add_argument('--n_timesteps_test', help='Number of test steps/ episode',type=int,default=150)
    # Misc
    parser.add_argument('--eps_behavior', help='Epsilon for initial e-greedy behavior agent (between 0 and 1 inclusively)',type=float,default=1)
    parser.add_argument('--gamma_behavior', help='Rejection probability to simulate draws from d^mu',type=float,default=0.0) # In case  we need samples from d^mu, add samples to buffer w.p. 1-gamma_behavior
    parser.add_argument('--dice_RL', help='Train DualDice to approximate d^pi/d^mu?',type=int,default=0)
    
    args = parser.parse_args()
    main(vars(args))
