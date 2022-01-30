#---------------------------------------
#Since : 2019/04/10
#Update: 2022/01/13
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from copy import deepcopy
import random
import math
from nn import NNetWrapper as nnet
from parameters import Parameters
from connect4 import Connect4
from statistics import mean

class Node():
    def __init__(self, board, states, player, move = None, psa = 0, terminal = False, winner = 0, parent = None, depth = 0):
        self.nsa      = 0 # the number of times the node has been visited
        self.wsa      = 0
        self.qsa      = 0
        self.psa      = psa
        self.player   = player
        self.move     = move
        self.board    = board
        self.states    = states
        self.children = []
        self.parent   = parent
        self.terminal = terminal
        self.winner = winner

    def Get_states(self):
        return deepcopy(self.states)

    def Get_board(self):
        return deepcopy(self.board)

    def Get_player(self):
        return deepcopy(self.player)

    def Get_winner(self):
        return  deepcopy(self.winner)

    def Add_child(self, board, states, player, move, psa, terminal, winner, parent):
        child = Node(board = board, states = states, player = player, move = move, psa = psa, terminal = terminal, winner = winner, parent = parent)
        self.children.append(child)

class A_MCTS:
    def __init__(self, game, net = None, params = Parameters(), num_mean = 1, X0 = 0.1, A = 1000, N_MAX =300, states = None):
        self.num_moves = None

        self.max_num_values = num_mean
        self.estimated_outcome = []

        g = game
        self.player = g.current_player
        if net == None:
            self.nn = nnet()
        else:
            self.nn = net

        self.root = Node(board = g.Get_board(), states = g.Get_states(), player = g.current_player)
        self.params = params

        self.A = A
        self.X0 = X0
        self.N_MAX = N_MAX

        self.states = states

    def softmax(self, x):
        x = np.exp(x / self.params.Temp)
        return x/np.sum(x)

    def Expand_node(self, node, psa_vector):
        temp_g = Connect4()
        temp_g.board = node.Get_board()
        temp_g.current_player = node.Get_player()
        valid_moves = temp_g.Get_valid_moves()
        for m in valid_moves:
            temp_g.Ini_board()
            temp_g.board = node.Get_board()
            temp_g.state = node.Get_states()
            temp_g.current_player = node.Get_player()
            temp_g.Play_action(m)
            psa = psa_vector[m]
            board = temp_g.Get_board()
            player = temp_g.current_player
            states = temp_g.Get_states()
            terminal = temp_g.Check_game_end()
            winner = temp_g.Get_winner()
            node.Add_child(board = board, states = states , player = player, move = m, psa = psa, terminal = terminal, winner = winner, parent = node)

    def Store_outcome(self, state):
        # Store the value of the board state.

        # Estimate the value of the board state.
        _, estimated_outcome =  self.nn.predict(state)

        # Add the value to the queue.
        self.estimated_outcome.append(np.asscalar(estimated_outcome))

        # Pop the value if size of the queue is more than the max.
        if len(self.estimated_outcome) > self.max_num_values:
            self.estimated_outcome.pop(0)

    def Run(self):
        temp_g = Connect4()

        for i in self.states:
            self.Store_outcome(i)

        # Change the number of simulations.
        self.params.num_mcts_sims = math.ceil(10**(- self.A * (mean(self.estimated_outcome) * self.root.player + self.X0)))
        if self.params.num_mcts_sims < 1:
            self.params.num_mcts_sims = 1
        if self.params.num_mcts_sims > self.N_MAX:
            self.params.num_mcts_sims = self.N_MAX


        for _ in range(self.params.num_mcts_sims):
            # Travel the game tree from the root node to the leaf node.
            node = self.root

            # Travel toa leaf node.
            while len(node.children) != 0:
                node = self.Search(node)
            # Here, the node is a leaf node.

            # Calculate the value.
            v = 0
            if node.terminal:
                # The value is the color of the winner when the node is a terminal node.
                v = node.Get_winner()
            else:
                # Obtain the outputs of the deep neural network.
                psa_vector, v =  self.nn.predict(node.Get_states())

                # Calculate the move probability.
                temp_g.Ini_board()
                temp_g.board = node.Get_board()
                temp_g.current_player = node.Get_player()
                # Get the valid moves.
                valid_moves = temp_g.Get_valid_moves()
                # Normalize the output of the policy head. Only the outputs of the units representing the valid actions are used.
                psa_vector /= np.sum(np.array([psa_vector[i] for i in valid_moves])) + 1e-7

                # Expan node
                self.Expand_node(node, psa_vector)

            # Backpropagation
            self.Back_prop(node, v)

        # Return the move.
        return self.Decide_move()

    def Decide_move(self):
        if self.num_moves > self.params.opening:
            # The action with the maximum visits is chosen.
            return self.root.children[np.argmax(np.array([i.nsa for i in self.root.children]))].move
        else:
            # The action is chosen at random with softmax function at openings.
            pi = self.softmax(np.array([i.nsa for i in self.root.children]))
            best_child = self.root.children[np.random.choice(len(self.root.children), p = pi.tolist())]
            return best_child.move

    def Search(self, node):
        if node.parent != None:
            # The action with the maximum UCT score is chosen.
            N = np.sum(np.array([i.nsa for i in node.children]))
            best_child = node.children[np.argmax(np.array([self.l(i.qsa, i.nsa, i.psa, N) for i in node.children]))]
        else:
            # The action is chosen based on epsilon gready algorithm when the node is the root node.
            if np.random.rand() > self.params.rnd_rate:
                N = np.sum(np.array([i.nsa for i in node.children]))
                best_child = node.children[np.argmax(np.array([self.l(i.qsa, i.nsa, i.psa, N) for i in node.children]))]
            else:
                best_child = random.choice(node.children)

        return best_child

    def l(self, qsa, nsa, psa, N):
        # UCT score
        return qsa + self.params.cpuct * psa * math.sqrt(N) / (nsa + 1)

    def Back_prop(self, node, v):
        # backpropagation
        while node != self.root:
            node.nsa += 1
            # wsa is the cumulative value for the player corresponding with the parent node.
            # The stone color of the player of parent node is -1 times the stone color of the player of this node.
            node.wsa += v * ( - node.player)
            node.qsa = node.wsa / node.nsa
            node = node.parent

    def Get_prob(self):
        # Calculate the probabilities of selected actions in MCTS.
        prob = np.zeros(self.params.action_size)
        for i in self.root.children:
            prob[i.move] += i.nsa

        prob /= np.sum(prob)
        return(prob)
