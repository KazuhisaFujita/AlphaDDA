#---------------------------------------
#Since : 2019/04/25
#Update: 2021/11/15
# -*- coding: utf-8 -*-
#---------------------------------------

class Parameters:
    def __init__(self):
        # Game settings
        self.board_x     = 7            # boad size
        self.board_y     = 6            # boad size
        self.connect     = 4            # the number of stones in a connection to win
        self.action_size = self.board_x # the maximum number of actions
        self.red         = 1            # stone color of the first player
        self.yellow      = -1           # stone color of the second player

        #------------------------
        # AlphaZero
        #MCTS
        self.num_mcts_sims = 200  # the number of MCTS simulations
        self.cpuct         = 1.25 # the exploration rate
        self.opening       = 0    # the opening of a game
        self.Temp          = 50   # the temperature parameter of softmax function for calculating the move probability
        self.rnd_rate      = 0.2  # the probability to select a move at random

        #Neural Network
        self.k_boards       = 1                       # the number of board states in one input
        self.input_channels = (self.k_boards * 2) + 1 # the number of channels of an input
        self.num_filters    = 256                     # the number of filters in the body
        self.num_filters_p  = 2                       # the number of filters in the policy head
        self.num_filters_v  = 1                       # the number of filters in the value head
        self.num_res        = 3                       # the number of redidual blocks in the body
