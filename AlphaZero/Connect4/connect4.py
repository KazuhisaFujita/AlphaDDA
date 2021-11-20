#---------------------------------------
#Since : 2018/09/16
#Update: 2020/02/26
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from ringbuffer import RingBuffer
from copy import deepcopy
from parameters import Parameters

class Connect4():
    def __init__(self):
        self.params = Parameters()

        self.xnum = self.params.board_x
        self.ynum = self.params.board_y
        self.winner = 0
        self.put_location = [0,0]

        self.Ini_board()

    def Ini_board(self):
        self.board = self.Create_board()
        self.current_player = self.params.red
        self.seq_boards = RingBuffer(self.params.k_boards)
        for i in range(self.params.k_boards):
            self.seq_boards.add(np.zeros((self.xnum, self.ynum)))
        self.seq_boards.add(deepcopy(self.board))

    def Create_board(self):
        return np.zeros((self.xnum, self.ynum))

    def Print_board(self):
        print("  ", end='')
        for j in range(self.xnum):
            print(str(j)+"|", end='')
        print("")

        for i in range(self.ynum):
            print(str(self.ynum - 1 - i)+"|", end='')
            for j in range(self.xnum):
                if self.board[j][self.ynum - 1 - i] == self.params.red:
                    print("o|", end='')
                elif self.board[j][self.ynum - 1 - i] == self.params.yellow:
                    print("x|", end='')
                else:
                    print(" |", end='')
            print("")


    def Can_put(self, b):
        return b == 0

    def Put(self, b, x, y, player):
        b[x][y] = player
        return b

    def Get_player(self):
        return deepcopy(self.player)

    def Get_board(self):
        return deepcopy(self.board)

    def Get_board_size(self):
        return (self.xnum, self.ynum)

    def Get_action_size(self):
        return self.xnum * self.ynum

    def Check_row(self, b, i, j):
        for t in range(self.params.connect):
            if b[i + t][j] == 0:
                return 0
        return 1

    def Check_column(self, b, i, j):
        for t in range(self.params.connect):
            if b[i][j + t] == 0:
                return 0
        return 1

    def Check_diagonal1(self, b, i, j):
        for t in range(self.params.connect):
            if b[i + t][j + t] == 0:
                return 0
        return 1

    def Check_diagonal2(self, b, i, j):
        for t in range(self.params.connect):
            if b[i - t][j + t] == 0:
                return 0
        return 1

    def Check_connection(self, b):
        result = 0
        for i in range(self.xnum - self.params.connect + 1):
            for j in range(self.ynum):
                result = self.Check_row(b, i, j)
                if result != 0:
                    return result
        for i in range(self.xnum):
            for j in range(self.ynum - self.params.connect + 1):
                result = self.Check_column(b, i, j)
                if result != 0:
                    return result
        for i in range(self.xnum - self.params.connect + 1):
            for j in range(self.ynum - self.params.connect + 1):
                result = self.Check_diagonal1(b, i, j)
                if result != 0:
                    return result
        for i in range(self.params.connect - 1, self.xnum):
            for j in range(self.ynum - self.params.connect + 1):
                result = self.Check_diagonal2(b, i, j)
                if result != 0:
                    return result
        return result

    def Check_winner(self):
        b_red = np.where(self.board == self.params.red, 1, 0)
        if self.Check_connection(b_red) == 1:
            return self.params.red
        b_yellow = np.where(self.board == self.params.yellow, 1, 0)
        if self.Check_connection(b_yellow) == 1:
            return self.params.yellow
        return 0

    def Get_winner(self):
        return self.winner

    def Get_valid_moves(self):
        moves = np.argwhere(self.board[:, self.ynum - 1] == 0)
        return(moves)

    def Check_game_end(self):
        self.winner = self.Check_winner()
        if self.winner != 0:
            return(True)
        else:
            if np.size(self.Get_valid_moves()) == 0:
                return(True)
            else:
                return(False)

    def Get_states(self):
        temp_states = self.seq_boards.Get_buffer()
        states = []
        for i in range(self.params.k_boards):
            states.append(np.where(temp_states[i] == 1, 1, 0))
            states.append(np.where(temp_states[i] == -1, 1, 0))

        if self.current_player == 1:
            states.append(np.ones((self.xnum, self.ynum)))
        else:
            states.append(np.zeros((self.xnum, self.ynum)))

        return np.array(states)

    def Put_stone(self, action):
        for i in range(self.ynum):
            if self.board[action][i] == 0:
                self.board[action][i] = self.current_player
                self.put_location = [action, i]
                break

    def Play_action(self, action):
        self.Put_stone(action[0])
        self.current_player *= -1
        self.seq_boards.add(deepcopy(self.board))

    def Get_current_player(self):
        return self.current_player

if __name__ == '__main__':
    tc = Tic_tac_toe()

    tc.Print_board()

    while(1):
        print(tc.Get_states())
        print(tc.Get_valid_moves())
        action = input()
        tc.Play_action(action)
#        tc.Print_board()

        if tc.Check_game_end():
            winner = tc.Get_winner()
            print(winner)
            exit()
