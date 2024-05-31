import abc
import random
from random import choice
from DQN import DQNet
from replay_buffer import ReplayBuffer, Transition
import torch
from copy import deepcopy
import numpy as np
import time
from datetime import datetime

class Agent(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def action(self, mdp, s_t):
        ...

    @abc.abstractmethod
    def optimize(self):
        ...

    @abc.abstractmethod
    def store_transition(self, transition):
        ...


class RandomAgent(Agent):
    def action(self, mdp, s_t):
        """chooses to execute a random action"""
        return choice(mdp.action_space)

    def optimize(self):
        pass  # random predictions, no training required

    def store_transition(self, transition):
        pass  # random predictions, no storing required


class DQNAgent(Agent):

    def __init__(self, emb_size, hidden_size, out_size, device, batch_size=32, gamma=0.2,
                 epsilon=0.1, replay_buffer_size=1000, lr=0.01):  # param mapping_dict removed 
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.test_mode = None
        # self.mapping_dict = mapping_dict

        self.policy = DQNet(emb_size, hidden_size, out_size).to(device=self.device)
        self.target = DQNet(emb_size, hidden_size, out_size).to(device=self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def action(self, mdp, x):
        """selects the best action (according to q-value) which is present in the user's mdp"""
        # actions are chosen according to the policy
        game_state = x[-10:]
        position = x[-1]
        if (position == 1):
            action_mask = np.array([20, 24, 115, 82, 37, 76, 1, 136, 39, 106, 86, 143, 117, 153, 129, 138, 66, 45, 41, 152, 108, 97, 46, 137, 158, 100, 133, 4, 119, 51, 36, 151, 38, 109, 120, 107, 157, 162, 96, 54, 110, 63, 150, 140, 43])
        if (position == 2):
            action_mask = np.array([39, 93, 50, 41, 152, 97, 151, 162, 103, 127, 140, 148, 79, 69, 116, 14, 156, 16, 49, 64, 130, 47, 65, 102, 89, 91, 62, 28, 71, 58, 92, 42, 40, 6, 118, 131, 78])
        if (position == 3):
            action_mask = np.array([37, 76, 106, 138, 46, 21, 4, 51, 36, 157, 96, 110, 127, 150, 123, 25, 130, 28, 131, 11, 59, 144, 7, 60, 44, 35, 8, 146, 34, 128, 135, 155, 126, 68, 84, 72, 94, 77, 145, 57, 12, 23])
        if (position == 4):
            action_mask = np.array([143, 123, 135, 139, 52, 67, 90, 73, 111, 55, 31, 27, 121, 81, 154, 142, 9, 56, 113])
        if (position == 5):
            action_mask = np.array([122, 20, 129, 161, 116, 16, 32, 78, 75, 35, 155, 126, 57, 10, 113, 134, 85, 15, 74, 17, 70, 80, 125, 101, 132, 104, 165, 5, 105, 13, 83, 114])
        new_action_vector = [x - 1 for x in action_mask]
        if self.device is not None:
            # create numpy vector of interacted with items
            action_vector = torch.zeros(167).bool().to(self.device)
            action_vector[(new_action_vector)] = 1

        if not self.test_mode and random.uniform(0, 1) < self.epsilon:
            a_t = random.choice(new_action_vector)
            #a_t = random.choice(mdp.action_space)
            return a_t  # return a random possible action from the user's mdp to explore
        else:
            q_vals = self.policy(torch.tensor(x, device=self.device).float())
            # masks illegal (i.e., not in user mdp) actions with -np.inf and selects the max legal action
            #return torch.argmax(q_vals.masked_fill(~mdp.action_vector, -np.inf)).item()
            return (torch.argmax(q_vals.masked_fill(~(action_vector), -np.inf)).item())

    def store_transition(self, transition):
        self.replay_buffer.add(transition)

    def optimize(self):
        """samples a batch from the replay memory and optimizes the agent"""
        if len(self.replay_buffer) < self.batch_size:
            return

        assert not self.test_mode

        transitions = self.replay_buffer.sample(self.batch_size)

        # NOTE: due to the nature of the interactive MDP, the next state is defined for every state (no terminal states)
        batch = Transition(*zip(*transitions))
        nextstate_batch = torch.tensor(batch.next_state, device=self.device).float()
        state_batch = torch.tensor(batch.state, device=self.device).float()
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy(state_batch).gather(1, action_batch.unsqueeze(0))
        assert not torch.isnan(state_action_values).any()

        next_state_values = self.target(nextstate_batch).max(1)[0].detach()
        assert not torch.isnan(next_state_values).any()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.loss_fn(state_action_values.squeeze(), expected_state_action_values)
        assert not torch.isnan(loss).any()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def enter_test_mode(self):
        self.test_mode = True
        self.policy.eval()
        self.target.eval()

    def enter_train_mode(self):
        self.test_mode = False
        self.policy.train()
        self.target.train()

    def update_target(self):
        self.target = deepcopy(self.policy)

    def save_model(self, data_path = "./data/lol/checkpoints/"):
        torch.save({
          'policy': self.policy.state_dict(),
          'target': self.target.state_dict(),
          'optimizer': self.optimizer.state_dict()
        },data_path + "checkpoint" +'all.tar')

class BDQNAgent(DQNAgent):
    def __init__(self):
        self.biases = {}
