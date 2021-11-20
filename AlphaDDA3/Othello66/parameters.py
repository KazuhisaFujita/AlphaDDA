#---------------------------------------
#Since : 2019/04/25
#Update: 2021/11/21
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
        self.num_mcts_sims = 200
        self.cpuct         = 1.25
        self.opening       = 0
        self.Temp          = 20
        self.rnd_rate      = 0.2

        #Neural Network
        self.k_boards       = 1
        self.input_channels = (self.k_boards * 2) + 1
        self.num_filters    = 256
        self.num_filters_p  = 2
        self.num_filters_v  = 1
        self.num_res        = 2
