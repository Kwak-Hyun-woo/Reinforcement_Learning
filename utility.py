from mdp import MovielensUserMDP
from mdp import LOLUserMDP
import numpy as np
import random
from matrix_factorization import construct_user_latent_state, run_matrix_factorization
from replay_buffer import Transition
from tqdm import tqdm
from agent import RandomAgent
import torch
import pickle

def run_test_rand(data, agent, T):
    available_mdps = {user_id: MovielensUserMDP(user_id, data[data['user_id'] == user_id])
                      for user_id in list(data['user_id'].unique())}

    rewards = []
    for user_id in available_mdps.keys():
        user_mdp = available_mdps[user_id]
        with torch.no_grad():
            for t in range(T):
                a_t = agent.action(user_mdp, None)
                r_t = user_mdp.reward(action=a_t)
                rewards.append(r_t)


    return np.mean(rewards), np.std(rewards)

def run_test(data, V, agent, T):

    with open("./data/lol/data_frame_match_merged.pickle", "rb") as fr:
        champs_numbering_file  = pickle.load(fr)
    """
    Runs a test on the given data.
    :param data: test data
    :param V: the item representation vectors
    :param agent: the agent
    :param T: the amount of time steps per user
    :return: mean and standard deviation of the observed rewards
    """
    available_mdps = {user_id: LOLUserMDP(user_id, data[data['user_id'] == user_id], V.shape[1], agent.device)
                      for user_id in list(data['user_id'].unique())}
    # specify second dimension as 1 due to matrix operations
    user_vectors = {user_id: np.zeros((64, 1)) for user_id in available_mdps.keys()}
    rewards = []
    agent.enter_test_mode()
    for user_id in available_mdps.keys():
        user_mdp = available_mdps[user_id]
        match_data = champs_numbering_file
        try:
          s_t = np.concatenate([user_vectors[user_id], np.expand_dims(np.array(match_data[user_id][0]), axis=1)])
        except:
          print(f"user_id: {user_id}")
          continue
          
          
        with torch.no_grad():
            for t in range(T-1):
                a_t = agent.action(user_mdp, s_t)
                r_t = user_mdp.reward(action=a_t, state = s_t)
                s_tp = construct_user_latent_state(s_t[:len(user_vectors[user_id])], V, a_t, r_t)
                s_tp1 = np.concatenate([s_tp, np.expand_dims(np.array(match_data[user_id][t+1]), axis=1)])
                rewards.append(r_t)
                s_t = s_tp1

    return np.mean(rewards), np.std(rewards)

def run_final_test(data, V, agent, T):
    with open("./data/lol/data_frame_match_test.pickle", "rb") as fr:
        data_frame_match_test  = pickle.load(fr)
    
    with open("./data/lol/champs_numbering.pickle", "rb") as fr:
        champs_numbering  = pickle.load(fr)
    reverse_champs_numbering = {v:k for k,v in champs_numbering.items()}
    """
    Runs a test on the given data.
    :param data: test data
    :param V: the item representation vectors
    :param agent: the agent
    :param T: the amount of time steps per user
    :return: mean and standard deviation of the observed rewards
    """
    available_mdps = {user_id: LOLUserMDP(user_id, data[data['user_id'] == user_id], V.shape[1], agent.device)
                      for user_id in list(data['user_id'].unique())}
    # specify second dimension as 1 due to matrix operations
    user_vectors = {user_id: np.zeros((64, 1)) for user_id in available_mdps.keys()}
    rewards = []
    logged_data = {}
    agent.enter_test_mode()
    count = 0
    for user_id in available_mdps.keys():
        user_log = []
        user_mdp = available_mdps[user_id]
        match_data = data_frame_match_test
        # breakpoint()
        try:
          s_t = np.concatenate([user_vectors[user_id], np.expand_dims(np.array(match_data[user_id][0][1:]), axis=1)])
          s_t_seleted_champ = match_data[user_id][0][0]
        except:
          print(f"user_id: {user_id}")
          continue
        with torch.no_grad():
            for t in range(T-1):
                cur_match = s_t.squeeze()[-10:]
                # 본인 챔피언 선택 정보 추가
                if t == 0:
                  cur_match = np.append(cur_match, s_t_seleted_champ)
                else:
                  cur_match = np.append(cur_match, match_data[user_id][t+1][0])

                # 변환
                cur_match_list = cur_match.tolist()
                
                cur_match_processed = []
                for idx, number in enumerate(cur_match_list):
                  if idx == 9:
                    cur_match_processed.append(int(number))
                    continue
                  cur_match_processed.append(reverse_champs_numbering[number])

                a_t = agent.action(user_mdp, s_t)
                recommended_champ = reverse_champs_numbering[a_t]
                r_t = user_mdp.reward(action=a_t, state = s_t)
                cur_reward = r_t
                rating = user_mdp.rewards_map.get(a_t)
                if (rating==None):
                    rating = 0
                s_tp = construct_user_latent_state(s_t[:len(user_vectors[user_id])], V, a_t, rating)
                s_tp1 = np.concatenate([s_tp, np.expand_dims(np.array(match_data[user_id][t+1][1:]), axis=1)])
                rewards.append(r_t)
                user_log.append((cur_match_processed, recommended_champ, cur_reward))
                s_t = s_tp1
        if (count <= 5):
            logged_data[user_id] = user_log
            count += 1

    return np.mean(rewards), np.std(rewards), logged_data

def train(data, agent, V, iterations, T, nth_fold):

    with open("./data/lol/data_frame_match_merged.pickle", "rb") as fr:
        champs_numbering_file  = pickle.load(fr)
    """
    Trains the agent on the given data.
    :param data: the training data
    :param agent: agent to be trained
    :param V: the item representations
    :param iterations: number of training iterations
    :param T: length of a training iteration
    """
    avg_reward_log = []

    if type(agent) == RandomAgent:
        return  # no training required

    available_mdps = {user_id: LOLUserMDP(user_id, data[data['user_id'] == user_id], V.shape[1], agent.device)
                      for user_id in list(data['user_id'].unique())}
    #specify second dimension as 1 due to matrix operations
    user_vectors = {user_id: np.zeros((64, 1)) for user_id in available_mdps.keys()}
    agent.enter_train_mode()

    pbar = tqdm(range(iterations))
    rolling_reward = 0
    for _ in pbar:  # outer episode loop
        rewards = []
        user_mdp = available_mdps[random.choice(list(available_mdps.keys()))]
        # user list 
        user = random.choice(list(available_mdps.keys()))

        user_vector = user_vectors[user_mdp.user_id]
        match_data = champs_numbering_file
        s_t = np.concatenate([user_vector, np.expand_dims(np.array(match_data[user][0]), axis=1)])  # Algorithm 1 returns zero vector on t=0
        
        for t in range(0, T-1):
            a_t = agent.action(user_mdp, s_t)
            r_t = user_mdp.reward(action=a_t, state = s_t)
            s_tp = construct_user_latent_state(s_t[:len(user_vector)], V, a_t, r_t)
            #breakpoint()
            s_tp1 = np.concatenate([s_tp, np.expand_dims(np.array(match_data[user][t+1]), axis=1)])
            rewards.append(r_t)
            agent.store_transition(
                Transition(s_t, torch.tensor([a_t], device=agent.device), s_tp1, torch.tensor([r_t], device=agent.device)))
            s_t = s_tp1

            agent.optimize()
        rolling_reward = 0.99 * rolling_reward + 0.01 * np.mean(rewards)
        avg_reward_log.append(rolling_reward)
        pbar.set_description(f"Avg.Reward: {rolling_reward}")
        agent.update_target()

    # train avg reward log save
    with open(f"./data/lol/result/{nth_fold}th_fold_reward_mean_log.pickle", "wb") as fw:
      pickle.dump(avg_reward_log, fw)


def run_cross_validation(dat, train_iter, T, agent_cls, agent_params, mf_params, device):
    """
    Runs cross validation on the provided data.
    :param dat: dataset
    :param train_iter: number of training iterations per fold
    :param T: length of one training iteration
    :param agent_cls: the agent class
    :param agent_params: the parameters to instantiate an agent
    :return: mean and standard deviation of the runs
    """
    means = []
    for i, d in enumerate(dat):  # d:tuple(train_data, test_data)
        agent = agent_cls(*agent_params)
        if agent_cls != RandomAgent:
            U, V, ids = run_matrix_factorization(d[0], agent.out_size, device, *mf_params)
            train(d[0], agent, V, train_iter, T, i+1)
            agent.save_model()
            mean, std = run_test(d[1], V, agent, T)
            # mean, std, logged_data = run_final_test(d[1], V, agent, T)
        else:
            mean, std = run_test_rand(d[0], agent, T)
        means.append(mean)
        with open(f"./data/lol/result/{i+1}th_fold_val_mean_std_log.pickle", "wb") as fw:
          pickle.dump([mean, std], fw)
        print(f'In fold {i+1}: Mean {mean}')
        # print(f'logged_data: \n')
        # print(logged_data[:10])
    print(f'Avg. Mean: {np.mean(means)}, Std.: {np.std(means)}')
    return np.mean(means)
