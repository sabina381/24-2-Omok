# import
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd

from file_save_load import *
from mcts import *
from environment import *
from net import *

from config import *


class TrainNetwork:
    __slots__ = ('env', 'file_name', 'model_type', 'model', 'device', 'temp', 'learn_rate', 'learn_decay', 'learn_epoch', 'L2', 'optimizer', 'scheduler', 'cross_entropy', 'memory', 'policy_loss_list', 'value_loss_list', 'loss_list', 'step_list')

    def __init__(self, file_name, model_type):
        self.env = Environment(STATE_SIZE, WIN_CONDITION)
        self.file_name = file_name
        self.device = DEVICE
        self.temp = TEMPERATURE

        self.model_type = model_type
        self.model = load_model(f'{file_name}_model_latest.pkl', model_type)

        self.learn_rate = LEARN_RATE
        self.learn_decay = LEARN_DECAY
        self.learn_epoch = LEARN_EPOCH
        self.L2 = L2

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate, weight_decay=self.L2)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.learn_epoch, gamma=self.learn_decay)
        self.cross_entropy = CROSS_ENTROPY

        self.memory = load_history(f'{file_name}_history.pkl')

        self.policy_loss_list = []
        self.value_loss_list = []
        self.loss_list = []
        self.step_list = []


    def _self_play_one_game(self):
        '''
        model을 통한 MCTS 알고리즘에 따라
        1번의 게임을 진행하는 함수
        '''
        history = []

        state = State()
        is_done = False

        n_steps = 0

        while not is_done:
            n_steps += 1

            if n_steps > EXPLORE_REGULATION:
                self.temp *= self.temp * TEMPERATURE_DECAY
                self.temp = max(self.temp, 0.1)
            
            mcts = Mcts(self.model, temperature = self.temp)

            policies = mcts.get_policy(state) # 가능한 행동들의 확률 분포 얻기
            legal_actions = state.get_legal_actions() # 가능한 행동 (index)

            # 전체 행동에 대한 policy
            policy = [0] * self.env.num_actions
            for action, p in zip(legal_actions, policies):
                policy[action] = p

            if PLAYER_INFO:
                player_arr = np.full(self.env.state_size, state.check_first_player()).reshape(1, self.env.n, self.env.n)
                state_arr = np.concatenate([state.history, player_arr], axis=0)
            else:
                state_arr = np.array(state.history)
            
            history.append([state_arr, policy, None])

            # 가능한 행동 중 랜덤으로 선택해서 게임 진행
            action = mcts.get_action(state)
            state, is_done, _ = self.env.step(state, action)

        # first player 기준의 reward로 바꾸기
        reward = self.env.get_first_reward(state)

        for i in range(len(history)):
            history[i][-1] = reward if i % 2 == 0 else -reward

        print(f" << n_steps: {n_steps} >>")

        return history, n_steps


    def _self_play(self):
        '''
        self-play를 진행해 data를 만드는 함수
        '''
        data = []

        for _ in range(SP_NUM_TRAIN):
            history, n_steps = self._self_play_one_game()
            data.extend(history)
            self.step_list.append(n_steps)

        return data


    def save_self_play_history(self):
        '''
        self_play를 진행하고, file 경로에 history를 저장하는 함수
        '''
        data = self._self_play()

        if DATA_AGUMENTATION:
            new_data = []
            for _, temp in enumerate(data):
                board, policy, value = temp
                policy_arr = np.array(policy).reshape(self.env.n, self.env.n)

                new_data.append((board, policy, value))
                new_data.append((np.transpose(board.copy(), (0, 2, 1)), policy_arr.copy().T.flatten(), value))
                new_data.append((np.fliplr(board.copy()), np.fliplr(policy_arr.copy()).flatten(), value))
                new_data.append((np.transpose(np.fliplr(board.copy()), (0, 2, 1)), np.fliplr(policy_arr.copy()).T.flatten(), value))
                new_data.append((np.flip(board.copy(), axis=2), np.flip(policy_arr.copy()).flatten(), value))
                new_data.append((np.transpose(np.flip(board.copy(), axis=2), (0, 2, 1)), np.flip(policy_arr.copy()).T.flatten(), value))
                new_data.append((np.fliplr(np.flip(board.copy(), axis=2)), np.fliplr(np.flip(policy_arr.copy())).flatten(), value))
                new_data.append((np.transpose(np.fliplr(np.flip(board.copy(), axis=2)), (0, 2, 1)), np.fliplr(np.flip(policy_arr.copy())).T.flatten(), value))

            self.memory.extend(new_data)

        else:
            self.memory.extend(data)

        save_data(f"{self.file_name}_history.pkl", self.memory)


    def _loss_function(self, pred_policy, pred_value, y):
        '''
        Define loss function
        '''
        y_policy = y[:, :-1]
        y_value = y[:, -1:]

        mse = F.mse_loss(pred_policy, y_policy)
        cross_entropy = self.cross_entropy(pred_value, y_value)
        return mse, cross_entropy, mse + cross_entropy


    def _make_dataset(self):
        '''
        history에서 랜덤 추출하여 학습 가능한 dataset 형태로 만드는 함수
        '''
        batch_size = min(BATCHSIZE, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        states, policies, results = zip(*mini_batch)

        policies = np.array(policies)
        results = np.array(results).reshape(-1, 1)
        Y_array = np.concatenate([policies, results], axis=1)

        X = torch.tensor(states, dtype=torch.float32).to(self.device) # (batch_size, STATE_DIM, env.n, env.n)
        Y = torch.tensor(Y_array, dtype=torch.float32).to(self.device) # (batch_size, 82)

        return X, Y


    def train_network(self):
        '''
        최근 모델을 불러와서 학습하는 함수
        '''
        self.model = self.model.to(self.device)
        policy_loss_list = []
        value_loss_list = []
        loss_list = []
        for i in range(TRAIN_EPOCHS):
            X, Y = self._make_dataset()
            pred_policy, pred_value = self.model.forward(X)
            policy_loss, value_loss, loss = self._loss_function(pred_policy, pred_value, Y)
            # 역전파
            self.optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()

            policy_loss_list.append(policy_loss.item())
            value_loss_list.append(value_loss.item())
            loss_list.append(loss.item())

            if (i+1) % PRINT_LOSS_FREQENCY == 0:
                print(f">> train step {i+1}/{TRAIN_EPOCHS} p_loss:{round(np.mean(policy_loss_list), 5)}, v_loss:{round(np.mean(value_loss_list), 5)}")

        self.scheduler.step()
        # 최근 모델 저장
        save_model(f'{self.file_name}_model_latest.pkl', self.model.to('cpu'))

        return policy_loss_list, value_loss_list, loss_list


    def train_cycle(self, eval):
        for i in range(EPISODES):
            print(f"- - - - Episode {i+1}/{EPISODES} - - - -")
            self.model = load_model(f'{self.file_name}_model_latest.pkl', self.model_type)
            self.save_self_play_history()
            policy_loss_list, value_loss_list, loss_list = self.train_network()
            self.policy_loss_list.extend(policy_loss_list)
            self.value_loss_list.extend(value_loss_list)
            self.loss_list.extend(loss_list)
            print(f" << policy loss: {round(np.mean(self.policy_loss_list[-10:]), 5)} / value_loss: {round(np.mean(self.value_loss_list[-10:]), 5)} >>")

            df = pd.DataFrame([self.policy_loss_list, self.value_loss_list, self.loss_list])
            save_data(f"{self.file_name}_loss_data.pkl", df)
            save_data(f"{self.file_name}_step.pkl", self.step_list)

            if (i+1) % EVAL_FREQUENCY == 0:
                is_update_model = eval.evaluate_network(i)

                if is_update_model:
                    eval.evaluate_best_player(i)

                eval.save_result_data()

        print("Finish train")