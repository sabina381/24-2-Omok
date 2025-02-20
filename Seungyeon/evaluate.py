# import
import torch
import pandas as pd
import pickle
import copy

from file_save_load import *
from environment import *
from enemy_agents import *
from mcts import *

from config import *

# class EvalAgent:
#     __slots__ = ('model', 'player')

#     def __init__(self, model):
#         self.model = model
#         self.player = False
#         self.state_type = np.arange(8)

#     def get_policy(self, state):
#         state = copy.copy(state)
#         policy, value = predict(state, self.model)
#         return policy, value

#     def make_state(self, state):
#         random.choice(self.state_type)

#         pass


class Evaluate:
    __slots__ = ('env', 'file_name', 'result_list', 'model_type')

    def __init__(self, file_name, model_type):
        self.env = Environment(STATE_SIZE, WIN_CONDITION)
        self.file_name = file_name
        # [label, point, win, draw, lose]
        self.result_list = []
        self.model_type = model_type


    def _save_game_result(self, label, point, win, draw):
        self.result_list.append([label, point, win, draw, EVAL_NUM_GAME-win-draw])


    def save_result_data(self):
        try:
            with open(f'{self.file_name}_result_data.pkl', 'rb') as f:
                load_data = pickle.load(f)

            df = pd.DataFrame(self.result_list)
            df = pd.concat([load_data, df], axis=0)

        except FileNotFoundError:
            df = pd.DataFrame(self.result_list)

        with open(f'{self.file_name}_result_data.pkl', 'wb') as f:
            pickle.dump(df, f)


    def evaluate_network(self, episode):
        model_latest = load_model(f'{self.file_name}_model_latest.pkl', self.model_type).to(DEVICE)
        model_best = load_model(f'{self.file_name}_model_best.pkl', self.model_type).to(DEVICE)

        mcts_latest = Mcts(model_latest, TEMPERATURE)
        mcts_best = Mcts(model_best, TEMPERATURE)

        player_list = [mcts_latest, mcts_best]
        # player_list = [EvalAgent(model_latest), EvalAgent(model_best)]

        print(" << Start Evaluation w/ best >>")
        # 대전
        average_point, cnt_win, cnt_draw = self.env.evaluate_first_algorithm("Evaluate network", player_list, EVAL_NUM_GAME, select_max=True)
        self._save_game_result('latest vs. best', average_point, cnt_win, cnt_draw)

        # game image
        _, data = self.env.play_one_game(player_list, is_save=True, select_max=True)
        self.env.save_game_image(f"{self.file_name}_{episode}_best_black", data)

        inverse_player_list = [None, None]
        inverse_player_list[0], inverse_player_list[1] = player_list[1], player_list[0]
        _, data = self.env.play_one_game(inverse_player_list, is_save=True, select_max=True)
        self.env.save_game_image(f"{self.file_name}_{episode}_best_white", data)

        # best player 교체
        if average_point > CRITERIA:
            save_model(f'{self.file_name}_model_best.pkl', model_latest.to('cpu'))
            return True
        else:
            return False


    def evaluate_best_player(self, episode):
        model = load_model(f'{self.file_name}_model_best.pkl', self.model_type).to(DEVICE)

        mcts_best = Mcts(model, TEMPERATURE)
        agents_dict = {'random': RandomAgent(False),
                        # 'alpha-beta': AlphaBetaAgent(False),
                        'mcs': McsAgent(False),
                        'mcts': MctsAgent(False)}

        count = 0
        for key in agents_dict.keys():
            agent = agents_dict[key]
            player_list = [mcts_best, agent]
            count += 1
            average_point, cnt_win, cnt_draw = self.env.evaluate_first_algorithm(f"[{count}/{len(agents_dict)}] vs. {key}", player_list, TEST_NUM_GAME, selct_max=True)
            self._save_game_result(key, average_point, cnt_win, cnt_draw)

            # game image
            _, data = self.env.play_one_game(player_list, is_save=True, select_max=True)
            self.env.save_game_image(f"{self.file_name}_{episode}_{key}_black", data)

            inverse_player_list = [None, None]
            inverse_player_list[0], inverse_player_list[1] = player_list[1], player_list[0]
            _, data = self.env.play_one_game(inverse_player_list, is_save=True, select_max=True)
            self.env.save_game_image(f"{self.file_name}_{episode}_{key}_white", data)