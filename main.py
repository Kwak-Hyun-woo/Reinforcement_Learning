from preprocessing import generate_train_test_splits
from pathlib import Path
import torch
from utility import run_cross_validation, run_final_test, run_test
import argparse
from agent import DQNAgent, RandomAgent
from matrix_factorization import run_matrix_factorization
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', type=int, help='number of iterations', default=700) # default=2000
    parser.add_argument('--test', type=bool, help='Test Mode', default=False)
    parser.add_argument('--path', type=str, default='data/lol/data_frame_new_merged.pickle', help='file location')
    parser.add_argument('--data_type', type=str, default='lol', help='Data passed. Currently supports [ml1m].')
    parser.add_argument('--agent_type', type=str, default='dqn',
                        help='Agent to be used. Currently supports [rand, dqn].')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size for the deep learning agents.')
    parser.add_argument('--lr_dl', default=0.001, type=float,
                        help='Learning rate for the deep neural network based agents.')
    parser.add_argument('--lr_mf', default=0.001, type=float, help='Learning rate for matrix factorization.')
    parser.add_argument('--num_iterations_mf', default=4000, type=int,
                        help='Number of training iterations for matrix factorization.')
    parser.add_argument('--lbd_mf', default=0.01, type=float,
                        help='The regularization factor lambda for matrix factorization')
    parser.add_argument('--cuda', type=bool, default=False, help='Set to True for running on GPU, false for CPU.')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of cross-validation folds.')
    parser.add_argument('--min_interactions_user', type=int, default=100,
                        help='The minimum number of interactions a user should have to be included '
                             '(users present in the data with less interactions will not be considered).')
    parser.add_argument('--train_pct', type=float, default=0.90,
                        help='The amount of data to be used for training in each cross-validation split.')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='The hidden dimension of the deep neural network based agents.')
    parser.add_argument('--emb_dim', type=int, default=74,
                        help='The dimension of the item embeddings generated by the matrix factorization algorithm.'
                             ' Also, the input dimension to the deep learning agents.')
    parser.add_argument('--replay_buffer_size', type=int, default=10000, help='The size of the replay buffer.')
    parser.add_argument('--gamma', type=float, default=0.2, help='The discount factor for Q-learning.')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='The exploration/exploitation factor (agent will not explore during test time).')
    parser.add_argument('--T', type=int, default=20,
                        help='The length of an episode (number of ongoing interaction steps per user)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = Path(args.path)
    dat, id_mapping_dict, total_items, test_dat = generate_train_test_splits(
        data_path=path,
        data_type=args.data_type,
        number_splits=args.num_splits,
        min_interactions_user=args.min_interactions_user,
        train_pct=args.train_pct)
    device = torch.device('cuda') if args.cuda else torch.device('cpu')
    agent_cls = None
    agent_params = None
    if args.agent_type.lower() == 'dqn':
        agent_cls = DQNAgent
        agent_params = (args.emb_dim, args.hidden_dim, total_items[0], device, args.batch_size, args.gamma, args.epsilon,
                        args.replay_buffer_size, args.lr_dl)
    elif args.agent_type.lower() == 'rand' or args.agent_type.lower() == 'random':
        agent_cls = RandomAgent
        agent_params = ()  # no parameters required for random agent
    else:
        raise ValueError(f'Agent type {args.agent_type.lower()} was not recognized.')

    mf_params = (args.num_iterations_mf, args.lbd_mf, args.lr_mf)
    print(f"Param: num_iteration:{args.num_iterations}, \
            agent_param:{agent_params}")

    # run cross validation
    if args.test == False:
      run_cross_validation(dat, args.num_iterations, args.T, agent_cls, agent_params, mf_params, device)
    else:
      # model load and test
      agent = agent_cls(*agent_params)
      check_point = torch.load("./data/lol/checkpoints/checkpoint" + "all.tar") 
      agent.policy.load_state_dict(check_point['policy'])
      agent.target.load_state_dict(check_point['target'])

      with open(f"./data/lol/data_frame_new_test.pickle", "rb") as fr:
        test_dat  = pickle.load(fr)

      test_dat.columns = ['user_id', 'item_id', 'rating']
      U, V, ids = run_matrix_factorization(test_dat, agent.out_size, device, *mf_params)
      test_reward_mean, test_reward_std, logged_data = run_final_test(test_dat, V, agent, args.T)
      print(f"test_reward_mean: {test_reward_mean}")
      print(f"test_reward_std: {test_reward_std}")
      print(f"logged_data: {logged_data}")

      with open(f"./data/lol/result/test_result.pickle", "wb") as fw:
        pickle.dump([test_reward_mean, test_reward_std, logged_data], fw)

