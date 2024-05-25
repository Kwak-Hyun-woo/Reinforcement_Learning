import torch


class MovielensUserMDP:
    """
    Contains the MDP-model of a single user.
    (Assumes the rewards for each action are fixed, irrespectively of time-step (e.g., as would be gathered from logs))
    """

    def __init__(self, user_id, transition_history, total_items = None, device = None):
        self.user_id = user_id
        self.transition_history = transition_history

        # check items only interacted with once
        assert len(list(self.transition_history['item_id'].unique())) == \
               len(self.transition_history['item_id'].to_list())

        self.action_space = self.transition_history['item_id'].to_list()
        self.rewards_map = {k: v for (k, v) in zip(self.transition_history['item_id'].to_list(),
                                                   self.transition_history['rating'].to_list())}

        if total_items is not None and device is not None:
            # create numpy vector of interacted with items
            self.action_vector = torch.zeros(total_items).bool().to(device)
            self.action_vector[self.action_space] = 1

    def reward(self, action):
        return self.rewards_map[action]

    def generate_rewards_map(self):
        pass
