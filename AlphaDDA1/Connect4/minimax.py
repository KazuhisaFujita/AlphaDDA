#---------------------------------------
#Since : 2019/04/10
#Update: 2021/11/14
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from copy import deepcopy
import random
import math
from connect4 import Connect4
import time
from parameters import Parameters

class Node():
    def __init__(self, state, player, move = None, terminal = False, winner = 0, parent = None):
        self.p        = player
        self.move     = move
        self.state    = state
        self.children = []
        self.parent   = parent
        self.terminal = terminal
        self.winner   = winner
        self.value    = None

    def Get_state(self):
        return deepcopy(self.state)

    def Get_player(self):
        return deepcopy(self.p)

    def Add_child(self, state, player, move, terminal, winner):
        child = Node(state, player, move, terminal, winner, self)
        self.children.append(child)

class Minimax:
    def __init__(self, game):
        self.g = game
        self.p = self.g.current_player
        self.params = Parameters()

        self.root = Node(state = self.g.Get_board(), player = self.g.current_player)

        self.depth = 3 # the depth of a game tree

        self.base_reward = 100

    def Expand_node(self, node):
        temp_g = Connect4()
        temp_g.board = node.Get_state()
        temp_g.current_player = node.Get_player()
        valid_moves = temp_g.Get_valid_moves()
        for m in valid_moves:
            temp_g.Ini_board()
            temp_g.board = node.Get_state()
            temp_g.current_player = node.Get_player()
            temp_g.Play_action(m)
            player = temp_g.current_player
            terminal = temp_g.Check_game_end()
            winner = temp_g.Get_winner()
            state = temp_g.Get_board()
            node.Add_child(state, player, m, terminal, winner)

    def Make_tree(self, node, depth):
        depth -= 1
        if node.terminal != True and depth >= 0:
            self.Expand_node(node)
            for i in node.children:
                self.Make_tree(i, depth)

    def Run(self):
        node = self.root

        self.Make_tree(node, self.depth)

        return self.Search(self.root).move

    def Check_row(self, b, i, j, n):
        for t in range(n):
            if i + t >= self.params.board_x:
                return 0
            if b[i + t][j] == 0:
                return 0
            if t == n-1 and i + t + 1 < self.params.board_x:
                if b[i + t + 1][j] == 1:
                    # If the number of discs in connection is more then n
                    return 0
            if t == 0 and i - 1 >= 0:
                if b[i - 1][j] == 1:
                    # If the number of discs in connection is more then n
                    return 0
        return 1

    def Check_column(self, b, i, j, n):
        for t in range(n):
            if j + t >= self.params.board_y:
                return 0
            if b[i][j + t] == 0:
                return 0
            if t == n-1 and j + t + 1 < self.params.board_y:
                if  b[i][j + t + 1] == 1:
                    # If the number of discs in connection is more then n
                    return 0
            if t == 0 and j - 1 >= 0:
                if b[i][j - 1] == 1:
                    # If the number of discs in connection is more then n
                    return 0
        return 1

    def Check_diagonal1(self, b, i, j, n):
        for t in range(n):
            if i + t >= self.params.board_x or j + t >= self.params.board_y:
                return 0
            if b[i + t][j + t] == 0:
                return 0
            if t == n-1 and i + t + 1 < self.params.board_x and j + t + 1 < self.params.board_y:
                if b[i + t + 1][j + t + 1] == 1:
                    # If the number of discs in connection is more then n
                    return 0
            if t == 0 and i - 1 >= 0 and j - 1 >= 0:
                if  b[i - 1][j - 1] == 1:
                    # If the number of discs in connection is more then n
                    return 0
        return 1

    def Check_diagonal2(self, b, i, j, n):
        for t in range(n):
            if i - t < 0 or j + t >= self.params.board_y:
                return 0
            if b[i - t][j + t] == 0:
                return 0
            if t == n-1 and i - t - 1 >= 0 and j + t + 1 < self.params.board_y:
                if b[i - t - 1][j + t + 1] == 1:
                    # If the number of discs in connection is more then n
                    return 0
            if t == 0 and i + 1 < self.params.board_x and j - 1 >= 0:
                if b[i + 1][j - 1] == 1:
                    # If the number of discs in connection is more then n
                    return 0
        return 1

    def Check_connection(self, b):
        result = 0
        for i in range(self.params.board_x):
            for j in range(self.params.board_y):
                for n in [3, 2]:
                    s = self.Check_row(b, i, j, n) * (self.base_reward**(n - 1))
                    result += s
                for n in [3, 2]:
                    s = self.Check_column(b, i, j, n) * (self.base_reward**(n - 1))
                    result += s
                for n in [3, 2]:
                    s = self.Check_diagonal1(b, i, j, n) * (self.base_reward**(n - 1))
                    result += s
                for n in [3, 2]:
                    s = self.Check_diagonal2(b, i, j, n) * (self.base_reward**(n - 1))
                    result += s

        return result

    def Evaluate(self, s, p):
        # Extract the board position of the root node.
        b = np.where(s == p, 1, 0)
        # Caclurate the score of the board position of minimax player.
        v = self.Check_connection(b)
        # Extract the board position of the opponent.
        b = np.where(s == -p, 1, 0)
        # Caclurate the score of the board position fo the opponent.
        # Subtract the opponent's score from the minimax player's score.'
        v -= self.Check_connection(b)
        return v

    def Search(self, root):
        temp_g = Connect4()

        stack = [root] # the list to store the nodes

        while len(stack) != 0:
            n = stack.pop()

            if len(n.children) != 0:
                stack += n.children
            else:
                v = 0
                if n.terminal:
                    v = root.p * n.winner * 1000000 # the value when the node is a terminal node.
                else:
                    if len(n.children) == 0:
                        v = self.Evaluate(n.state, root.p)
                    else:
                        v = n.value

                # When the terminal node is in second nodes, v is None and Errot occur. This code avoid this problem.
                n.value = v

                while n.parent != None:
                    # Set the node to the parent node.
                    n = n.parent

                    if n.value == None:
                        n.value = v
                    elif n.p == root.p and n.value < v:
                        n.value = v
                    elif n.p != root.p and n.value > v:
                        n.value = v

        return root.children[np.argmax(np.array([i.value for i in root.children]))]
