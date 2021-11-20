#---------------------------------------
#Since : 2019/02/26
#Update: 2019/05/07
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np

class Random_player():
    def __init__(self, game):
        self.g = game

    def Move(self):
        moves = np.array(self.g.Get_valid_moves())
        if moves.shape[0] != 0:
            return moves[np.random.randint(moves.shape[0])]
        else:
            return [None, None]

# class HumanPlay():
#     def __init__(self):
