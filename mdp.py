import torch
import pandas as pd
import numpy as np


data_path = './data/lol'
pairing_bot_sup = pd.read_excel(data_path + '/pairing_bot_sup.xlsx')
pairing_jung_sup = pd.read_excel(data_path + '/pairing_jung_sup.xlsx')
pairing_mid_jung = pd.read_excel(data_path + '/pairing_mid_jung.xlsx')
pairing_top_jung = pd.read_excel(data_path + '/pairing_top_jung.xlsx')
meta_info = pd.read_excel(data_path + '/meta_info.xlsx')
adc_counters = pd.read_excel(data_path + '/adc_champion_counters.xlsx')
jungle_counters = pd.read_excel(data_path + '/jungle_champion_counters.xlsx')
mid_counters = pd.read_excel(data_path + '/mid_champion_counters.xlsx')
support_counters = pd.read_excel(data_path + '/support_champion_counters.xlsx')
top_counters = pd.read_excel(data_path + '/top_champion_counters.xlsx')



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


class LOLUserMDP:
    """
    Contains the MDP-model of a single user.
    (Assumes the rewards for each action are fixed, irrespectively of time-step (e.g., as would be gathered from logs))
    """
    # game_history : user's rating_matrix


    def __init__(self, user_id, game_history, total_items = None, device = None):
        self.user_id = user_id
        self.game_history = game_history
        
        # check items only interacted with once
        assert len(list(self.game_history['item_id'].unique())) == \
               len(self.game_history['item_id'].to_list())

        self.total_action_space = np.arange(0, 167)
        self.action_space = self.game_history['item_id'].to_list()
        self.rewards_map = {k: v for (k, v) in zip(self.game_history['item_id'].to_list(),
                                                   self.game_history['rating'].to_list())}
        #print(self.rewards_map)
        if total_items is not None and device is not None:
            # create numpy vector of interacted with items
            self.action_vector = torch.zeros(total_items).bool().to(device)
            self.action_vector[self.action_space] = 1
        

    def reward(self, action, state):
        state = state.squeeze()
        game_state = state[-10:]
        position = state[-1]
        return self.calculate_reward(position = position, game_state = game_state, action = action)

    def calculate_reward(self, position, game_state, action):
        # weight parameter
        weight_game = 0.5
        weight_preference = 0.5
        game_state = [int(x + 1) for x in game_state]
        action = int(action + 1)

        # pairing score calculation
        if position == 1:
            # top-jungle pairing score
            pairing = pairing_top_jung[((pairing_top_jung['champ1'] == action) & (pairing_top_jung['champ2'] == game_state[0]))|((pairing_top_jung['champ2'] == action) & (pairing_top_jung['champ1'] == game_state[0]))]['score'].copy()
            if not pairing.empty:
                pairing_score = pairing.iloc[0]
            else:
                pairing_score = 0

            counter = top_counters[(top_counters['champion'] == action) & (top_counters['counter'] == game_state[4])]['winrate'].copy()
            if not counter.empty:
                counter_score = counter.iloc[0]
            else:
                counter_score = 0
        
        if position == 2: # jungle
            pairing1 = pairing_top_jung[((pairing_top_jung['champ1'] == game_state[0]) & (pairing_top_jung['champ2'] == action))|((pairing_top_jung['champ2'] == game_state[0]) & (pairing_top_jung['champ1'] == action))]['score'].copy()
            pairing2 = pairing_jung_sup[((pairing_jung_sup['champ1'] == action) & (pairing_jung_sup['champ2'] == game_state[3]))|((pairing_jung_sup['champ2'] == action) & (pairing_jung_sup['champ1'] == game_state[3]))]['score'].copy()
            if not pairing1.empty:
                if not pairing2.empty:
                    pairing_score = (pairing1.iloc[0] + pairing2.iloc[0])/2
                else:
                    pairing_score = pairing1.iloc[0]
            else:
                if not pairing2.empty:
                    pairing_score = pairing2.iloc[0]
                else:
                    pairing_score = 0
            
            counter = jungle_counters[(jungle_counters['champion'] == action) & (jungle_counters['counter'] == game_state[5])]['winrate'].copy()
            if not counter.empty:
                counter_score = counter.iloc[0]
            else:
                counter_score = 0

        if position == 3: # mid
            pairing = pairing_mid_jung[((pairing_mid_jung['champ1'] == action) & (pairing_mid_jung['champ2'] == game_state[1]))|((pairing_mid_jung['champ2'] == action) & (pairing_mid_jung['champ1'] == game_state[1]))]['score'].copy()
            if not pairing.empty:
                pairing_score = pairing.iloc[0]
            else:
                pairing_score = 0
            
            counter = mid_counters[(mid_counters['champion'] == action) & (mid_counters['counter'] == game_state[6])]['winrate'].copy()
            if not counter.empty:
                counter_score = counter.iloc[0]
            else:
                counter_score = 0

        if position == 4: # adc
            pairing = pairing_bot_sup[((pairing_bot_sup['champ1'] == action) & (pairing_bot_sup['champ2'] == game_state[3]))|((pairing_bot_sup['champ2'] == action) & (pairing_bot_sup['champ1'] == game_state[3]))]['score'].copy()
            if not pairing.empty:
                pairing_score = pairing.iloc[0]
            else:
                pairing_score = 0
            
            counter = adc_counters[(adc_counters['champion'] == action) & (adc_counters['counter'] == game_state[7])]['winrate'].copy()
            if not counter.empty:
                counter_score = counter.iloc[0]
            else:
                counter_score = 0

        if position == 5: # support
            pairing1 = pairing_jung_sup[((pairing_jung_sup['champ1'] == game_state[1]) & (pairing_jung_sup['champ2'] == action))|((pairing_jung_sup['champ2'] == game_state[1]) & (pairing_jung_sup['champ1'] == action))]['score'].copy()
            pairing2 = pairing_bot_sup[((pairing_bot_sup['champ1'] == game_state[3]) & (pairing_bot_sup['champ2'] == action))|((pairing_bot_sup['champ2'] == game_state[3]) & (pairing_bot_sup['champ1'] == action))]['score'].copy()
            if not pairing1.empty:
                if not pairing2.empty:
                    pairing_score = (pairing1.iloc[0] + pairing2.iloc[0])/2
                else:
                    pairing_score = pairing1.iloc[0]
            else:
                if not pairing2.empty:
                    pairing_score = pairing2.iloc[0]
                else:
                    pairing_score = 0
    
        counter = support_counters[(support_counters['champion'] == action) & (support_counters['counter'] == game_state[8])]['winrate'].copy()
        if not counter.empty:
            counter_score = counter.iloc[0]
        else:
            counter_score = 0
        

        str_position = str(int(position))
        condition = (meta_info['Unnamed: 0'] == action)

        try:
            meta_score = meta_info.loc[condition, str_position].values[0]
        except KeyError:
            print("KeyError: The specified key does not exist.")
            meta_score = 0 
        #print(pairing_score)
        game_score = (0.45*meta_score + 0.1*pairing_score + 0.45*counter_score)
        # user's preference score
        max_value = max(self.game_history['rating'])
        min_value = 0

        # zero division prevention
        cur_reward = self.rewards_map.get(str(action), 0)
        if (min_value == max_value):
            preference_score = min_value
        else: 
            preference_score = (cur_reward - min_value) / (max_value - min_value)
        preference_score = (preference_score) * 100
        #print(preference_score)
        total_reward = weight_game * game_score + weight_preference * preference_score
        #print('game_score : %f and preference_score %f', game_score, preference_score)
        return total_reward.astype(np.float32)