# import
import numpy as np
import copy
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os

# parameter
from config import *
#class
class Environment:
    __slots__ = ('state_size', 'win_condition', 'n', 'num_actions', 'action_space', 'reward_dict')

    def __init__(self, state_size:Tuple, win_condition:int):
        self.state_size = state_size
        self.win_condition = win_condition
        self.n = self.state_size[0]
        self.num_actions = self.n ** 2
        self.action_space = np.arange(self.num_actions)

        self.reward_dict = {'win': REWARD_WIN, 'lose': REWARD_LOSE, 'draw': REWARD_DRAW}


    def step(self, present_state, action_idx :int):
        '''
        present_state에 대해 action_idx의 행동에 따라 게임을 한 턴 진행시키고
        next_state, is_done, is_lose를 반환한다.
        '''
        next_state = present_state.next(action_idx)
        is_done, is_lose = next_state.check_done()

        return next_state, is_done, is_lose


    def get_reward(self, final_state):
        '''
        게임이 종료된 state에 대해 last player의 reward를 반환한다.
        final_state: 게임이 종료된 state
        '''
        _, is_lose = final_state.check_done()
        reward = self.reward_dict['lose'] if is_lose else self.reward_dict['draw']

        return reward


    def get_first_reward(self, final_state):
        '''
        게임이 종료된 state에 대해 first player의 reward를 반환한다.
        final_state: 게임이 종료된 state
        note: final_state가 is_lose라면, 해당 state에서 행동할 차례였던 플레이어가 패배한 것.
        '''
        is_first_player = final_state.check_first_player()
        reward = self.get_reward(final_state) if is_first_player else -self.get_reward(final_state)

        return reward


    def render(self, state):
        '''
        입력받은 state를 문자열로 출력한다.
        X: first_player, O: second_player
        '''
        is_first_player = state.check_first_player()
        board = state.my_state - state.enemy_state if is_first_player else state.enemy_state - state.my_state
        board = board.reshape(-1)
        board_list = list(map(lambda x: '○' if board[x] == 1 else '●' if board[x] == -1 else '.', self.action_space))

        board_string = ' '.join(board_list)
        formatted_string = '\n'.join([board_string[i:i+(self.n * 2)] for i in range(0, len(board_string), self.n * 2)])

        print(formatted_string)

    ######## play game method ######
    def play_one_game(self, player_list:list, is_show=False, is_save=False, select_max=False):
        is_done = False
        state = State()

        data = []
        n_steps = 0

        while not is_done:
            n_steps += 1
            player = player_list[0] if state.check_first_player() else player_list[1]

            if 'model' in player.__slots__:
                if n_steps == EXPLORE_REGULATION:
                    player.temperature = 0

                policy = player.get_policy(state)

                # 전체 행동에 대한 정책으로 변경
                legal_actions = state.get_legal_actions()
                policies = [0] * self.num_actions
                for act, p in zip(legal_actions, policy):
                    policies[act] = p
                    
                if is_save:
                    # data.append((state, policies, value))
                    data.append((state, policies))
                
                if select_max:
                    max_actions = np.where(policies == np.max(policies))[0]
                    action = np.random.choice(max_actions)
                else:
                    action = np.random.choice(legal_actions, p=policy)
                
                state, is_done, _ = self.step(state, action)

            else:
                action = player.get_action(state)
                state, is_done, _ = self.step(state, action)

            if is_show:
                self.render(state)
                print("-"*30)

        # 게임 종료 후 first player 기준 reward, point 계산
        reward = self.get_first_reward(state)
        point = POINT_WIN if reward == REWARD_WIN else POINT_DRAW if reward == REWARD_DRAW else POINT_LOSE

        if is_show:
            print(f"first reward: {reward}")

        if is_save:
            return point, data
        else:
            return point, None # first player point


    def evaluate_first_algorithm(self, label:str, player_list:list, num_game:int, select_max):
        inverse_player_list = [None, None]
        inverse_player_list[0], inverse_player_list[1] = player_list[1], player_list[0]
        total_point = 0
        cnt_win = 0
        cnt_draw = 0

        print_frequency = num_game / 5

        for i in range(num_game):
            if 'model' in player_list[0].__slots__:
                 player_list[0].temperature = TEMPERATURE
            if 'model' in player_list[1].__slots__:
                 player_list[1].temperature = TEMPERATURE

            if i % 2 == 0:
                point, _ = self.play_one_game(player_list, select_max=select_max)
                total_point += point
                cnt_win += 1 if point == POINT_WIN else 0

            else:
                point, _ = self.play_one_game(inverse_player_list, select_max=select_max)
                total_point += 1 - point
                cnt_win += 1 if point == POINT_LOSE else 0

            cnt_draw += 1 if point == POINT_DRAW else 0

            # change player
            player_list[0].player = not player_list[0].player
            player_list[1].player = not player_list[1].player

            if (i+1) % print_frequency == 0:
                print(f"Evaluate {label}: {num_game}/{i+1}")

        average_point = total_point / num_game
        print(f"{label}: {round(average_point, 5)} / win:{cnt_win} / draw:{cnt_draw} / lose:{num_game-cnt_win-cnt_draw}")

        return average_point, cnt_win, cnt_draw


    def save_game_image(self, file_name, data):
        fig, axes = plt.subplots(len(data), 2, figsize=(12, 6*len(data)))
        for num in range(len(data)):
            
            # 히트맵 그리기
            policy = np.array(data[num][1]).reshape(self.n, self.n)
            sns.heatmap(policy, annot=True, fmt='.2f', cmap='YlGnBu', cbar=True, ax=axes[num, 1])
            # axes[num, 1].set_title(f'Mcts Policy / value: {data[num][2]}')
            axes[num, 1].set_title(f'Mcts Policy')

            # 격자 그리기
            first_player = data[num][0].check_first_player()
            board = data[num][0].get_total_state() if first_player else -data[num][0].get_total_state()

            axes[num, 0].set_facecolor('#CD853F')
            axes[num, 0].set_title(f"Mcts is first: {first_player}")
            for x in range(self.n):
                axes[num, 0].plot([x, x], [0, self.n-1], color="black", linewidth=1, zorder = 1)  # 세로선
                axes[num, 0].plot([0, self.n-1], [x, x], color="black", linewidth=1, zorder = 1)  # 가로선

            for i in range(self.n):
                for j in range(self.n):
                    if board[i, j] == 1:  # 흑돌
                        circle = plt.Circle((j, self.n-1 - i), 0.4, color="black", zorder = 2)
                        axes[num, 0].add_artist(circle)
                    elif board[i, j] == -1:  # 백돌
                        circle = plt.Circle((j, self.n-1 - i), 0.4, facecolor='white', edgecolor="black", linewidth=1.5, zorder = 2)
                        axes[num, 0].add_artist(circle)

            axes[num, 0].set_xlim(-0.5, self.n - 0.5)
            axes[num, 0].set_ylim(-0.5, self.n - 0.5)
            axes[num, 0].set_aspect('equal')
            
        # 레이아웃 조정 및 저장
        # 저장할 디렉토리 경로 생성
        output_dir = os.path.join(F_PATH, "game_images")
        os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성

        # 이미지 파일 경로 설정
        output_file = os.path.join(output_dir, f"{file_name}.png")

        # 그래프 저장
        plt.tight_layout()
        plt.savefig(output_file)  # 이미지 파일로 저장
        plt.close()               # 메모리 해제

        print(f"<< Game image succefully saved in \'game_images/{file_name}.png\'. >>")


########################################################################
# define State class
class State():
    __slots__ = ('env', 'my_state', 'enemy_state', 'previous_action', 'previous_state', 'history')

    def __init__(self, my_state=None, enemy_state=None, previous_action=None, previous_state=None):
        self.env = Environment(STATE_SIZE, WIN_CONDITION)

        self.my_state = my_state if my_state != None else [0] * (self.env.n ** 2)
        self.enemy_state = enemy_state if enemy_state != None else [0] * (self.env.n ** 2)
        self.previous_action = previous_action
        self.previous_state = previous_state

        self.my_state = np.array(self.my_state).reshape(self.env.state_size)
        self.enemy_state = np.array(self.enemy_state).reshape(self.env.state_size)
        self.history = self.make_history(NUM_HISTORY)


    def make_history(self, num_history):
        history = [self.my_state, self.enemy_state]
        temp = self.previous_state
        
        if num_history > 0:
            for _ in range(num_history):
                if temp is not None:
                    history.extend([temp.my_state, temp.enemy_state])
                    temp = temp.previous_state
                else:
                    history.extend([np.zeros(self.env.state_size), np.zeros(self.env.state_size)])

        return history

    def total_pieces_count(self):
        '''
        이 state의 전체 돌의 개수를 반환한다.
        '''
        total_state = self.my_state + self.enemy_state
        return np.sum(total_state)


    def get_legal_actions(self):
        '''
        이 state에서 가능한 action idx의 리스트를 반환한다.
        '''
        total_state = (self.my_state + self.enemy_state).reshape(-1)
        legal_actions = np.where(total_state == 0)[0]
        return legal_actions


    def get_onehot_legal_actions(self):
        '''
        이 state에서 가능한 action을
        one-hot encoding 형식의 array로 반환한다.
        '''
        legal_actions = self.get_legal_actions()
        onehot_legal_actions = np.array([legal_actions[x] == 0 for x in self.env.action_space], dtype = int)
        return onehot_legal_actions


    def check_line_done(self, line):
        k = self.env.win_condition
        line = np.pad(line, (0, 1), 'constant', constant_values=0)

        window = np.sum(line[:k-1])
        for i in range(k-1, len(line)):
            window += line[i] - line[i - k]
            if window == k:
                # 육목 확인 조건
                window += line[i+1] - line[i+1 - k]
                if window == k:
                    break
                else:
                    return True

        return False


    def check_lose(self):
        '''
        이 state의 lose 여부를 반환한다.
        note: 상대가 행동한 후, 자신의 행동을 하기 전 이 state를 확인한다.
        따라서 이전 state에서 상대의 행동으로 상대가 이긴 경우는 이 state의 플레이어가 진 경우이다.
        '''
        if self.total_pieces_count() < self.env.win_condition * 2 - 1:
            return False

        is_lose = False
        x, y = np.divmod(self.previous_action, self.env.n)

        # check row
        row = self.enemy_state[x, :]
        if not np.sum(row) < self.env.win_condition:
            is_lose = self.check_line_done(row)
            # print(f"row: {is_lose}")
            if is_lose:
                return is_lose

        # check col
        col = self.enemy_state[:, y]
        if not np.sum(col) < self.env.win_condition:
            is_lose = self.check_line_done(col)
            # print(f"col: {is_lose}")
            if is_lose:
                return is_lose

        # check diag
        diag = np.diag(self.enemy_state, k = y - x)
        if not np.sum(diag) < self.env.win_condition:
            is_lose = self.check_line_done(diag)
            # print(f"diag: {is_lose}")
            if is_lose:
                return is_lose

        # check anti diag
        anti_diag = np.diag(np.fliplr(self.enemy_state), k = self.env.n - 1 - y - x)
        if not np.sum(anti_diag) < self.env.win_condition:
            is_lose = self.check_line_done(anti_diag)
            # print(f"anti diag: {is_lose}")
            if is_lose:
                return is_lose

        return is_lose


    def check_draw(self):
        if self.total_pieces_count() == self.env.num_actions:
            return True

        return False


    def check_done(self):
        is_lose = self.check_lose()
        is_done = is_lose or self.check_draw()

        return is_done, is_lose


    def next(self, action_idx):
        '''
        주어진 action에 따라 다음 state를 생성한다.
        note: 다음 state는 상대의 차례이므로 state 순서를 바꾼다.
        '''
        x, y =np.divmod(action_idx, self.env.n)
        state = self.my_state.copy()
        state[x, y] = 1

        state = list(state.reshape(-1))
        enemy_state = list(copy.copy(self.enemy_state).reshape(-1))

        return State(enemy_state, state, action_idx, self)


    def check_first_player(self):
        '''
        이 state의 플레이어가 first player인지 확인한다.
        '''
        return (self.total_pieces_count() % 2) == 0


    def get_random_action(self):
        '''
        이 state에서 가능한 action 중 랜덤으로 action을 반환한다.
        '''
        legal_actions = self.get_legal_actions()
        action = np.random.choice(legal_actions)
        return action


    def get_total_state(self):
        '''
        history에 넣을 전체 게임보드 state
        자신의 수: 1 / 상대의 수: -1 / 빈칸: 0
        '''
        # return (self.state, self.enemy_state)
        return self.my_state - self.enemy_state


    def render(self):
        '''
        이 state를 렌더링한다.
        '''
        self.env.render(self)