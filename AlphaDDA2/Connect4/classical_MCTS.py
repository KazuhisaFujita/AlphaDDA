#---------------------------------------
#Since : 2019/04/10
#Update: 2021/11/12
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from copy import deepcopy
import random
import math
from connect4 import Connect4
from player import Random_player

class Node():
    def __init__(self, state, player, move = None, terminal = False, winner = 0, parent = None):
        self.n        = 0 # the visit count
        self.q        = 0
        self.p        = player
        self.move     = move
        self.state    = state
        self.children = []
        self.parent   = parent
        self.terminal = terminal
        self.winner   = winner

    def Get_state(self):
        return deepcopy(self.state)

    def Get_player(self):
        return deepcopy(self.p)

    def Add_child(self, state, player, move, terminal, winner):
        child = Node(state, player, move, terminal, winner, self)
        self.children.append(child)

class MCTS:
    def __init__(self, game):
        self.g = game
        self.p = self.g.current_player

        self.root = Node(state = self.g.Get_board(), player = self.g.current_player)

        self.num_sim = 400 # the number of simulations
        self.th_open_leaf = 5 # A leaf nod is expanded if its visit count is more than this number.

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

        node.children[0].n = 1


    def Run(self):
        for _ in range(self.num_sim):
            node = self.root
            while node.terminal == False:
                if len(node.children) == 0 and (node == self.root or node.n >= self.th_open_leaf):
                    # If the node is a leaf node and the visit count is more than N_opene, add child nodes to the node.
                    self.Expand_node(node)
                else:
                    if len(node.children) == 0:
                        # Playout step
                        node.winner = self.random_play(node)
                        break
                    else:
                        # Selection step
                        node = self.Search(node)

            reward = deepcopy(node.winner)

            self.BACKUP(node, reward)

        return self.Decide_move()

    def random_play(self, node):
        temp_g = Connect4()
        temp_g.board = node.Get_state()
        temp_g.current_player = node.Get_player()

        while(1):
            rand = Random_player(temp_g)
            move = rand.Move()
            temp_g.Play_action(move)
            if temp_g.Check_game_end():
                winner = temp_g.Get_winner()
                break
        return winner


    def l(self, q, n, N):
        # UCT score
        return float(q)/(n + 1e-7) + 0.5 * math.sqrt(2 * math.log(N + 1) / (n + 1e-7))

    def Search(self, node):
        N = np.sum(np.array([i.n for i in node.children])) # the visit count of the parent node
        best_child = node.children[np.argmax(np.array([self.l(i.q, i.n, N) for i in node.children]))]
        return best_child

    def Decide_move(self):
        # The move corresponding to the child node with the maximum visit count is selected.
        return self.root.children[np.argmax(np.array([i.n for i in self.root.children]))].move

    def BACKUP(self, node, reward):
        while node != None:
            node.n += 1
            node.q += reward * (-node.p)
            node = node.parent
