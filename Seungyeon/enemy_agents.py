# import
import numpy as np
import random
from math import log

from environment import Environment

# parameters
from config import *

# random agent ################################
class RandomAgent:
    __slots__ = ('player')

    def __init__(self, player:bool):
        self.player = player

    def get_action(self, state):
        return state.get_random_action()


# alpha-beta(minimax) agent ################################
# parameter
AB_DEPTH = 100

# class
class AlphaBetaAgent:
    __slots__ = ('player', 'best_action', 'root_node')

    def __init__(self, player:bool):
        self.env = Environment(STATE_SIZE, WIN_CONDITION)
        self.player = player
        self.best_action = None
        self.root_node = None


    def get_action(self, state):
        if self.root_node == None:
            self.root_node = state

        self.minimax(state, AB_DEPTH, -np.Inf, np.Inf)
        return self.best_action


    def minimax(self, present_state, depth, alpha, beta):
        is_done, _ = present_state.check_done()
        reward = 0
        state = present_state
        legal_actions = present_state.get_legal_actions()

        if is_done or (depth == 0):
            reward = self.env.get_reward(present_state)
            return reward

        if present_state.check_first_player() == self.player: # max player
            max_eval = -np.Inf

            for action in legal_actions:
                # state = copy.deepcopy(present_state)
                next_state, _, _ = self.env.step(state, action)

                eval = self.minimax(next_state, depth-1, alpha, beta)

                if eval > max_eval:
                    best_action = action
                    max_eval = eval

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

                if depth == AB_DEPTH: # 최상위 호출에서만 best action 저장
                    self.best_action = best_action

            return max_eval

        else: # min player
            min_eval = np.Inf
            for action in legal_actions:
                # state = copy.deepcopy(present_state)
                next_state, _, _ = self.env.step(state, action)

                eval = self.minimax(next_state, depth-1, alpha, beta)
                min_eval = min(min_eval, eval)

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval


# MCS agent ################################
# parameter
MCS_PO_NUM = 30

# class
class McsAgent():
    __slots__ = ('env', 'player')

    def __init__(self, player:bool):
        self.env = Environment(STATE_SIZE, WIN_CONDITION)
        self.player = player

    def get_action(self, state):
        legal_actions = state.get_legal_actions()
        value_list = np.zeros(len(legal_actions))

        for i, action in enumerate(legal_actions):
            next_state, _, _ = self.env.step(state, action)

            for _ in range(MCS_PO_NUM):
                value_list[i] += - self.playout(next_state)

        max_idx = self.get_max_idx(value_list)
        return legal_actions[max_idx]


    def playout(self, state):
        is_done, _ = state.check_done()

        if is_done:
            return self.env.get_reward(state)

        action = state.get_random_action()
        next_state, _, _ = self.env.step(state, action)

        return - self.playout(next_state)


    def get_max_idx(self, value_list):
        value_list = list(value_list)
        return value_list.index(max(value_list))


# MCTS Agent ################################
# define Node class ##################
class PureNode:
    __slots__ = ('env', 'state', 'n', 'w', 'child_nodes')

    def __init__(self, state):
        self.env = Environment(STATE_SIZE, WIN_CONDITION)
        self.state = state
        self.n = 0 # visit count
        self.w = 0 # cumulative sum of values
        self.child_nodes = None

    def evaluate(self):
        is_done, _ = self.state.check_done()

        # 게임 종료 시 승패 여부에 따라 value 업데이트
        if is_done:
            value = self.env.get_reward(self.state)
            self.w += value
            self.n += 1
            return value

        # child node가 없는 경우 확장
        if not self.child_nodes:
            value = self.playout(self.state)

            self.w += value
            self.n += 1

            # expand child node
            if self.n == 10:
                self.expand()

            return value

        # end node가 아니고, child node가 있는 경우 -> 전개
        else:
            next_child_node = self.get_next_child_node()
            value = - next_child_node.evaluate()

            self.w += value
            self.n += 1

            return value


    def expand(self):
        '''
        Expand child node
        '''
        legal_actions = self.state.get_legal_actions()
        self.child_nodes = []

        for action in legal_actions:
            next_state, _, _ = self.env.step(self.state, action)
            self.child_nodes.append(PureNode(next_state))


    def get_next_child_node(self):
        '''
        UCB1이 가장 큰 child node를 선택
        '''
        node_scores = np.array(list(map(lambda c: c.n, self.child_nodes)))

        # 방문 횟수가 0인 child node 반환
        if np.any(node_scores == 0):
            zero_idx = random.choice(np.where(node_scores == 0)[0])
            return self.child_nodes[zero_idx]

        total_scores = np.sum(node_scores)

        # UCB1 계산 함수
        def ucb1(c):
            return -c.w/c.n + (2*log(total_scores)/c.n)**0.5

        # ucb1 값에 따라 정렬한 child nodes list (마지막이 최댓값을 갖는 child node)
        ucb1_sorted = sorted(self.child_nodes, key = lambda c: ucb1(c))

        return ucb1_sorted[-1]


    def playout(self, state):
        is_done, _ = state.check_done()

        if is_done:
            return self.env.get_reward(state)

        action = state.get_random_action()
        next_state, _, _ = self.env.step(state, action)

        return - self.playout(next_state)

# class
class MctsAgent():
    __slots__ = ('player')

    def __init__(self, player:bool):
        self.player = player


    def get_action(self, state):
        root_node = PureNode(state)
        root_node.expand()

        for _ in range(MCTS_EV_NUM):
            root_node.evaluate()

        legal_actions = state.get_legal_actions()

        child_node_scores = list(map(lambda c: c.n, root_node.child_nodes))
        max_idx = self.get_max_idx(child_node_scores)
        return legal_actions[max_idx]


    def get_max_idx(self, value_list):
        return value_list.index(max(value_list))