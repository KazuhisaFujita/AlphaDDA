#---------------------------------------
#Since : 2019/04/10
#Update: 2020/02/26
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from copy import deepcopy
import random
import math
from Othello import Othello
from player import Random_player

class Node():
    def __init__(self, state, player, move = None, terminal = False, winner = 0, parent = None):
        self.n        = 0 # the number of times the node has been visited
        self.q        = 0 # np.random.randint(0,2)    #
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
        # The player of the node give the last move. Then, The player is not the current player using MCTS.
        self.num_sim = 400
        self.th_open_leaf = 5

    def Expand_node(self, node):
        temp_g = Othello()
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
                    self.Expand_node(node)
                else:
                    if len(node.children) == 0:
                        node.winner = self.random_play(node)
                        break
                    else:
                        node = self.Search(node)

            reward = deepcopy(node.winner)

            self.BACKUP(node, reward)

        # for i in self.root.children:
        #     print(i.move, i.q, i.n)
        # print("")


        #return self.Search(self.root).move
        return self.Decide_move()

    def random_play(self, node):
        temp_g = Othello()
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
        return float(q)/(n + 1e-7) + 0.5 * math.sqrt(2 * math.log(N + 1) / (n + 1e-7))

    def Search(self, node):
        N = np.sum(np.array([i.n for i in node.children])) #node.n
        best_child = node.children[np.argmax(np.array([self.l(i.q, i.n, N) for i in node.children]))]
        return best_child

    def Decide_move(self):
        return self.root.children[np.argmax(np.array([i.n for i in self.root.children]))].move

    def BACKUP(self, node, reward):
        while node != None:
            node.n += 1
            node.q += reward * (-node.p)
            node = node.parent
