import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .agents import CBAgent, DiceAgent
from .utils import rollout
from .data_collection import RandomizedBatchDataset
from .dataset_dice import Dataset, EnvStep, StepType
from tf_agents.trajectories import time_step


def fit_policy(X,Z,r,p,policy,reward_fn,feature_weights):
    policy.fit(X=X, a=Z, r=r, p=p)
    
    Z_t = policy.predict(X)
    pred_Y_t = reward_fn(feature_weights,X,bias=0)

    return policy, pred_Y_t

def fit_policy_offline_dataset(dataset,make_env,model,estimator,critic,opt_V,Q_critic,opt_Q,loss_func,gamma,train_mask,k,model_type,session_reward,device,dice_rl,dice_solve_for_state_action_ratio,batch_size):
    """
    model_type in ['behaviour','CB','CB++']
    """
    print('###########################')
    print('Training %s offline'%model_type)
    print('###########################')
    validation_period = max(int(0.1 * len(dataset)), 1)
    v_acc = []
    r_acc = []
    l_acc = []

    X_t = dataset.X_t
    Z_t = dataset.Z_t
    Y_t = dataset.Y_t
    P_mu_t = dataset.P_mu_t

    if '(CB++)' in model_type:
        target_policy_dice = DiceAgent(model,env=make_env())

    r = np.zeros(shape=(len(dataset),dataset.window_X,))
    w = np.ones(shape=(len(dataset),dataset.window_X,))
    w_stat = np.ones(shape=(len(dataset),dataset.window_X,))
    done = np.zeros(shape=(len(dataset),dataset.window_X,))
    done[:,-1] = 1

    print('Computing reward estimates')
    for n in tqdm(range(len(dataset))):
        for t in range(0,dataset.window_X,1):
            if '$r_t=Y_t$' in model_type or 'Offline BCQ' in model_type:
                r[n,t] = Y_t[n,t]
            elif '$r_t=Y_t-Y_{t-1}$' in model_type:
                if t == 0:
                    r[n,t] = 0 #Y_t[n,t]
                else:
                    r[n,t] = Y_t[n,t] - Y_t[n,t-1]
            else:
                raise Exception('No such reward estimator.')
            
                    
    """
    Batched estimators (BCQ, V^mu, DualDice)
    """
    if 'Value' in model_type or model_type == 'Offline BCQ (MDP)' or ('(CB++)' in model_type and dice_rl):
        print('Training batched estimator: %s'%model_type)
        batch_dataset = RandomizedBatchDataset(X_t,Z_t,r,w,done,batch_size)

        for n,(X_0_batch,Z_0_batch,r_0_batch,X_batch,Z_batch,r_batch,p_mu_batch,X_next_batch,Z_next_batch,r_next_batch,p_mu_next_batch,done_batch) in enumerate(batch_dataset.get_dataset()):
            if 'Value' in model_type:
                """
                Train V^mu function approximator
                """
                state = torch.FloatTensor(X_batch).to(device)
                action = torch.LongTensor(Z_batch).to(device)
                reward = torch.FloatTensor(r_batch).reshape(-1,1).to(device)
                dones = torch.FloatTensor(done_batch).reshape(-1,1).to(device)
                next_state = torch.FloatTensor(X_next_batch).to(device)
                
                with torch.no_grad():
                    target = reward + (1-dones) * gamma * critic(next_state)

                value_t = critic(state)
                loss = loss_func(value_t.to(device), target.to(device))
                opt_V.zero_grad()
                loss.backward()
                opt_V.step()
                
                v = value_t.mean().cpu().detach().item()
                l = loss.cpu().detach().item()
                
                """
                Train Q^mu function approximator
                """
                next_action = torch.LongTensor(Z_next_batch).to(device)

                target = Q_critic(state)
                
                with torch.no_grad():
                    target = reward + (1-dones) * gamma * Q_critic(next_state).gather(1,next_action.unsqueeze(1))

                
                q_value_t = Q_critic(state).gather(1,action.unsqueeze(1))
                loss = loss_func(q_value_t, target)
                opt_Q.zero_grad()
                loss.backward()
                opt_Q.step()

            if '(CB++)' in model_type and dice_rl:
                    initial_steps_batch = EnvStep(step_type=time_step.StepType.FIRST,
                                step_num=0,
                                observation=X_batch.astype(np.float32),
                                action=Z_batch.astype(np.int64),
                                reward=r_batch,
                                discount=np.repeat(gamma,repeats=len(X_batch)),
                                policy_info=None,
                                env_info=None,
                                other_info=None
                                )
                    batch_t = EnvStep(step_type=time_step.StepType.MID,
                                step_num=1,
                                observation=X_batch.astype(np.float32),
                                action=Z_next_batch.astype(np.int64),
                                reward=r_next_batch,
                                discount=np.repeat(gamma,repeats=len(X_batch)),
                                policy_info={'log_probability':np.log(p_mu_batch)},
                                env_info=None,
                                other_info=None
                                )
                    batch_tp1 = EnvStep(step_type=time_step.StepType.MID,
                                step_num=2,
                                observation=X_next_batch.astype(np.float32),
                                action=Z_next_batch.astype(np.int64),
                                reward=r_next_batch,
                                discount=np.repeat(gamma,repeats=len(X_batch)),
                                policy_info={'log_probability':np.log(p_mu_next_batch)},
                                env_info=None,
                                other_info=None
                                )
                    
                    nu_loss,zeta_loss,lam_loss = estimator.train_step(initial_steps_batch, batch_t, batch_tp1,
                                    target_policy_dice)
        
                    v = 0
                    l = zeta_loss.numpy().item()
    
            if model_type == 'Offline BCQ (MDP)':
                state = torch.FloatTensor(X_batch).to(device)
                next_state = torch.FloatTensor(X_next_batch).to(device)
                action = torch.LongTensor(Z_batch).reshape(-1,1).to(device)
                reward = torch.FloatTensor(r_batch).to(device)
                dones = torch.FloatTensor(done_batch).reshape(-1,1).to(device)

                model.partial_fit(state,action,next_state,reward,dones)

                v = 0
                l = 0
                
            v_acc.append([v,0.,model_type,n])
            l_acc.append([l,0.,model_type,n])

    """
    Update cost-to-go for CB++
    """
    if '(CB++)' in model_type: 
        print('Estimating advantages')
        A_k = np.zeros(shape=(len(dataset),dataset.window_X))
        for n in tqdm(range(len(dataset))):
            """
            Advantage estimation using k-step false horizon target + PDIS weights + (optionally) DualDice
            """
            V_mu_table = critic(torch.FloatTensor(X_t[n]).to(device)).cpu().detach().numpy().reshape(-1)
            Q_mu_table = Q_critic(torch.FloatTensor(X_t[n]).to(device)).cpu().detach().numpy()
            if dice_rl:
                if dice_solve_for_state_action_ratio:
                    w_stat[n] = np.clip( estimator._zeta_network( (X_t[n].astype(np.float32),Z_t[n].astype(np.int64)))[0].numpy(), 0.5,2.0)
                else:
                    w_stat[n] = np.clip( estimator._zeta_network( (X_t[n].astype(np.float32),) )[0].numpy(), 0.5,2.0)
                D_KL = np.log(w_stat[n])
                
            lam = 0.5
            for t in range(0,dataset.window_X):
                k_step = 0
                k_step_mu = 0
                m = 0
                w = 1

                lam_step = 0
                conservative_estimate = False #dice_rl and (1/w_stat[n,t] < dice_rl)
                while m < k and t+m < dataset.window_X-1:
                    k_step += w * gamma**m * r[n,t+m] # w_t=1 for m=0
                    k_step_mu += gamma**m * r[n,t+m]
                    
                    m += 1
                    # w *= w_stat[n,t+m]
                    if conservative_estimate:
                        w = 1
                    else:
                        w *= np.clip(model.probs(np.array([X_t[n,t+m]]),np.array([Z_t[n,t+m]])).item() / P_mu_t[n,t+m], 0.5,2.0)

                    lam_step += lam**(m-1) * (k_step + w * gamma**m * V_mu_table[t+m])
                V_mu_xtpm = V_mu_table[t+m]
                V_mu_xt = V_mu_table[t]
                Q_pi_xt = k_step + w * gamma**m * V_mu_xtpm
                Q_pi_xt_lam = (1-lam) * lam_step + lam * w * gamma**m * V_mu_xtpm
                # A_k[n,t] = Q_pi_xt_lam - V_mu_xt 
                A_k[n,t] = Q_pi_xt - V_mu_xt
                if dice_rl:
                    A_k[n,t] += dice_rl * D_KL[t]

                
                
            v_acc.append([A_k[n].mean(),model_type,0,n])
        

    if '(CB++)' in model_type or '(CB)' in model_type:
        print('Training %s via CB oracle'%model_type)
        for n in tqdm(range(len(dataset))):
            if '(CB++)' in model_type:
                cost = -A_k[n]
                p_mu = np.ones(shape=(dataset.window_X))
            if '(CB)' in model_type:
                cost = -Y_t[n]
                p_mu = dataset.P_mu_t[n]
                
            model.partial_fit(X_t[n],Z_t[n],cost,p_mu)
    

    if 'Value' not in model_type:
        if model_type == 'Offline BCQ (MDP)':
            agent = model
        elif '(CB++)' in model_type or '(CB)' in model_type:
            agent = CBAgent(model,env=make_env())
        states,rewards,actions, probs = rollout(lambda state:agent.act(state),make_env,H=dataset.H,n_traj=10)
        avg_test_rewards = np.mean([session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards])
        std_test_rewards = np.std([session_reward(np.array(sess)[~np.isnan(sess)]) for sess in rewards])

        r_acc.append([avg_test_rewards,std_test_rewards,model_type,n])            


    """
    Save metrics
    """

    if 'Value' in model_type:
        r_acc = None

        v_acc = pd.DataFrame(v_acc)
        v_acc.columns = ["value_mean","value_std","policy","step"]

        l_acc = pd.DataFrame(l_acc)
        l_acc.columns = ["loss_mean","loss_std","policy","step"]
    elif dice_rl and '(CB++)' in model_type:
        v_acc = pd.DataFrame(v_acc)
        v_acc.columns = ["adv","policy","t","n"]

        l_acc = pd.DataFrame(l_acc)
        l_acc.columns = ["loss_mean","loss_std","policy","step"]

        r_acc = pd.DataFrame(r_acc)
        r_acc.columns = ["reward_mean","reward_std","policy","step"]
    elif not dice_rl and '(CB++)' in model_type:
        l_acc = None

        v_acc = pd.DataFrame(v_acc)
        v_acc.columns = ["adv","policy","t","n"]

        r_acc = pd.DataFrame(r_acc)
        r_acc.columns = ["reward_mean","reward_std","policy","step"]
    else:
        v_acc = l_acc = None

        r_acc = pd.DataFrame(r_acc)
        r_acc.columns = ["reward_mean","reward_std","policy","step"]

    return model, (critic,Q_critic), r_acc, v_acc, l_acc

def fit_policy_online(make_env,model,random_agent,critic,opt_V,n_traj,n_timesteps,eps,loss_func,gamma,batch_size,model_type,device):
    print('###########################')
    print('Training %s online'%model_type)
    print('###########################')
    r_acc = []
    v_acc = []
    l_acc = []
    for i in tqdm(range(n_traj)):
        done = False
        env = make_env()
        state = env.reset()
        ep_reward = 0
        ep_loss = 0
        ep_v = 0
        nb_steps = 0
        for t in range(n_timesteps):
            if np.any(np.isnan(state)):
                action, _ = random_agent.act(state)
                state, reward, done, _ = env.step(action)
                continue
            action, prob = model.act(state,eps=eps)
            next_state, reward, done, _ = env.step(action)

            if model_type == 'Online DQN (MDP)':
                model.replay_buffer.push(state, action, reward, next_state, done)

                if len(model.replay_buffer) >= batch_size:
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch, _ = model.replay_buffer.sample(batch_size)

                    state_batch = torch.FloatTensor(state_batch).type(torch.float32).to(device)
                    action_batch = torch.LongTensor(action_batch).to(device)
                    reward_batch = torch.FloatTensor(reward_batch).to(device)
                    next_state_batch = torch.FloatTensor(next_state_batch).type(torch.float32).to(device)
                    done_batch = torch.FloatTensor(done_batch).type(torch.float).to(device)

                    q_values = critic(state_batch)
                    next_q_values = critic(next_state_batch).detach()

                    q_value          = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
                    next_q_value     = next_q_values.max(1)[0]
                    expected_q_value = reward + gamma * next_q_value * (1 - done_batch)
                    expected_q_value.requires_grad = False
                    
                    loss = loss_func(q_value, expected_q_value)
                    
                    opt_V.zero_grad()
                    loss.backward()
                    opt_V.step()
                ep_loss += loss.detach().item()
                ep_v += q_value.max().detach().item()
            elif model_type == 'Online SARSA (MDP)':
                next_action, _ = model.act(next_state)
                reward = torch.FloatTensor([reward])

                target = critic(torch.FloatTensor([state]).to(device))[0]
                if done:
                    target[action] = reward
                else:
                    target[action] = (reward + gamma * critic(torch.FloatTensor([next_state]).to(device))[0][next_action].item())

                target = target.view(1,-1)
                q_value = critic(torch.FloatTensor([state]).to(device))[0]
                loss = loss_func(q_value, target)
                opt_V.zero_grad()
                loss.backward()
                opt_V.step()
                ep_loss += loss.detach().item()
                ep_v += q_value.max().detach().item()
            elif '$r_t=Y_t$ (CB)' in model_type:
                model.model.partial_fit_example_(state,action,-reward,prob)
            
            state = next_state

            try:
                ep_reward += reward.item()
            except:
                ep_reward += reward
            nb_steps += 1

            if done:
                break
        ep_loss /= nb_steps
        ep_v /= nb_steps
        r_acc.append([ep_reward,0.,model_type,i])
        v_acc.append([ep_v,0.,model_type,i])
        l_acc.append([ep_loss,0.,model_type,i])
        
    r_acc = pd.DataFrame(r_acc)
    r_acc.columns = ["reward_mean","reward_std","policy","step"]

    if len(v_acc) > 0:
        v_acc = pd.DataFrame(v_acc)
        v_acc.columns = ["value_mean","value_std","policy","step"]

        l_acc = pd.DataFrame(l_acc)
        l_acc.columns = ["loss_mean","loss_std","policy","step"]
    return model, critic, r_acc, v_acc, l_acc
