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
        self.num_iterations      = 600
        self.num_games           = 10
        self.checkpoint_interval = 5
        self.num_test            = 10

        self.num_processes_training = 10
        self.num_processes_test = 10
        self.devices       = ["cuda:0", "cuda:0", "cuda:0", "cuda:0", "cuda:0", "cuda:1", "cuda:2", "cuda:0", "cuda:1", "cuda:2"]

        #MCTS
        self.num_mcts_sims = 200
        self.cpuct         = 1.25
        self.opening_train = 4
        self.opening_test  = 0
        self.opening       = self.opening_train
        self.Temp          = 20
        self.rnd_rate      = 0.2

        #Neural Network
        self.input_size     = 20000
        self.k_boards       = 1
        self.input_channels = (self.k_boards * 2) + 1
        self.num_filters    = 256
        self.num_filters_p  = 2
        self.num_filters_v  = 1
        self.num_res        = 2
        self.epochs         = 1
        self.batch_size     = 2048
        self.lam            = 2e-1
        self.weight_decay   = 1e-4
        self.momentum       = 0.9
