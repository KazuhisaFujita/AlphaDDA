#---------------------------------------
#Since : 2019/04/25
#Update: 2021/11/16
# -*- coding: utf-8 -*-
#---------------------------------------

class Parameters:
    def __init__(self):
        #Game setting
        self.board_x     = 8                               # boad size
        self.board_y     = self.board_x                    # boad size
        self.action_size = self.board_x * self.board_y + 1 # the maximum number of actions
        self.black       = 1                               # stone color of the first player
        self.white       = -1                              # stone color of the second player

        #------------------------
        # AlphaZero

        # parameters for parallel processing
        self.num_processes_training = 10 # the number of parallelized processes of training of AlphaZero
        self.num_processes_test = 10 # the number of parallelized processes of test games
        self.devices       = ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1'] # used devices

        self.num_iterations      = 800 # the number of iterations
        self.num_games           = 10   # the number of games played in self-play

        # AlphaZero plays games with the other AI agent every checkpoint_interval to test the strength of AlphaZero.
        self.checkpoint_interval = 5
        self.num_test            = 10 # the number of games in test play

        #MCTS
        self.num_mcts_sims = 400                # the number of MCTS simulations
        self.cpuct         = 1.25               # the exploration rate
        self.opening_train = 6                  # the opening of a game for training
        self.opening_test  = 0                  # the opening of a game for test
        self.opening       = self.opening_train # the opening of a game
        self.Temp          = 40                 # the temperature parameter of softmax function for calculating the move probability
        self.rnd_rate      = 0.2                # the probability to select a move at random

        #Neural Network
        self.input_size     = 50000                   # the number of inputs
        self.k_boards       = 1                       # the number of board states in one input
        self.input_channels = (self.k_boards * 2) + 1 # the number of channels of an input
        self.num_filters    = 256                     # the number of filters in the body
        self.num_filters_p  = 2                       # the number of filters in the policy head
        self.num_filters_v  = 1                       # the number of filters in the value head
        self.num_res        = 5                       # the number of redidual blocks in the body
        self.epochs         = 1                       # the number of epochs every iteration
        self.batch_size     = 2048                    # the size of the mini batch
        self.lam            = 2e-1                    # learning rate
        self.weight_decay   = 1e-4                    # weight decay
        self.momentum       = 0.9                     # momentum
