#---------------------------------------
#Since : 2019/04/25
#Update: 2022/01/13
# -*- coding: utf-8 -*-
#---------------------------------------

class Parameters:
    def __init__(self):
        #Game
        self.board_x     = 6
        self.board_y     = self.board_x
        self.action_size = self.board_x * self.board_y + 1
        self.black       = 1
        self.white       = -1

        #AlphaZero
        #MCTS
        self.num_mcts_sims = 200 # Number of MCTS simulations per game.
        self.cpuct         = 1.25 # The level of exploration used in MCTS.
        self.opening       = 0#4
        self.Temp          = 20
        self.rnd_rate      = 0.2
        self.eps           = 0.25
        self.alpha         = 0.15

        #Neural Network
        self.k_boards       = 1
        self.input_channels = (self.k_boards * 2) + 1
        self.num_filters    = 256
        self.num_filters_p  = 2
        self.num_filters_v  = 1
        self.num_res        = 3
