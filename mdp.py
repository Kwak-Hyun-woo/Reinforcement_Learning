import torch
import pandas as pd

data_path = '/content/drive/MyDrive/Colab_Notebooks/Reinforcement_Learning/data/lol'
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

        self.action_space = self.game_history['item_id'].to_list()
        self.rewards_map = {k: v for (k, v) in zip(self.game_history['item_id'].to_list(),
                                                   self.game_history['rating'].to_list())}

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

        # pairing score calculation
        if position == 1:
            # top-jungle pairing score
            pairing = pairing_top_jung[((pairing_top_jung['champ1'] == action) & (pairing_top_jung['champ2'] == game_state[1]))|((pairing_top_jung['champ2'] == action) & (pairing_top_jung['champ1'] == game_state[1]))]['score']
            if not pairing.empty:
                pairing_score = pairing.iloc[0]
            else:
                pairing_score = 0

            counter = top_counters[(top_counters['champion'] == action) & (top_counters['counter'] == game_state[5])]['winrate']
            if not counter.empty:
                counter_score = counter.iloc[0]
            else:
                counter_score = 0
        
        if position == 2: # jungle
            pairing1 = pairing_top_jung[((pairing_top_jung['champ1'] == game_state[0]) & (pairing_top_jung['champ2'] == action))|((pairing_top_jung['champ2'] == game_state[0]) & (pairing_top_jung['champ1'] == action))]['score']
            pairing2 = pairing_jung_sup[((pairing_jung_sup['champ1'] == action) & (pairing_jung_sup['champ2'] == game_state[4]))|((pairing_jung_sup['champ2'] == action) & (pairing_jung_sup['champ1'] == game_state[4]))]['score']
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
            
            counter = jungle_counters[(jungle_counters['champion'] == action) & (jungle_counters['counter'] == game_state[6])]['winrate']
            if not counter.empty:
                counter_score = counter.iloc[0]
            else:
                counter_score = 0

        if position == 3: # mid
            pairing = pairing_mid_jung[((pairing_mid_jung['champ1'] == action) & (pairing_mid_jung['champ2'] == game_state[1]))|((pairing_mid_jung['champ2'] == action) & (pairing_mid_jung['champ1'] == game_state[1]))]['score']
            if not pairing.empty:
                pairing_score = pairing.iloc[0]
            else:
                pairing_score = 0
            
            counter = mid_counters[(mid_counters['champion'] == action) & (mid_counters['counter'] == game_state[7])]['winrate']
            if not counter.empty:
                counter_score = counter.iloc[0]
            else:
                counter_score = 0

        if position == 4: # adc
            pairing = pairing_bot_sup[((pairing_bot_sup['champ1'] == action) & (pairing_bot_sup['champ2'] == game_state[4]))|((pairing_bot_sup['champ2'] == action) & (pairing_bot_sup['champ1'] == game_state[4]))]['score']
            if not pairing.empty:
                pairing_score = pairing.iloc[0]
            else:
                pairing_score = 0
            
            counter = adc_counters[(adc_counters['champion'] == action) & (adc_counters['counter'] == game_state[8])]['winrate']
            if not counter.empty:
                counter_score = counter.iloc[0]
            else:
                counter_score = 0

        if position == 5: # support
            pairing1 = pairing_jung_sup[((pairing_jung_sup['champ1'] == game_state[1]) & (pairing_jung_sup['champ2'] == action))|((pairing_jung_sup['champ2'] == game_state[1]) & (pairing_jung_sup['champ1'] == action))]['score']
            pairing2 = pairing_bot_sup[((pairing_bot_sup['champ1'] == game_state[3]) & (pairing_bot_sup['champ2'] == action))|((pairing_bot_sup['champ2'] == game_state[3]) & (pairing_bot_sup['champ1'] == action))]['score']
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
    
        counter = support_counters[(support_counters['champion'] == action) & (support_counters['counter'] == game_state[9])]['winrate']
        if not counter.empty:
            counter_score = counter.iloc[0]
        else:
            counter_score = 0
        
        try:
            meta_score = meta_info.loc[action, position]
        except KeyError:
            #print("KeyError: The specified key does not exist.")
            meta_score = 0 
        #print(pairing_score)
        #print(counter_score)
        #print(meta_score)
        game_score = (meta_score + pairing_score + counter_score)/3


        # user's preference score
        max_value = max(self.game_history['rating'])
        min_value = min(self.game_history['rating'])

        # zero division prevention
        if (min_value == max_value):
            preference_score = min_value
        else: 
            preference_score = (self.rewards_map[action] - min_value) / (max_value - min_value)
        preference_score = preference_score * 100
        #print(preference_score)
        total_reward = weight_game * game_score + weight_preference * preference_score

        return total_reward 