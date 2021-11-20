#---------------------------------------
#Since : 2018/09/16
#Update: 2019/07/25
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from ringbuffer import RingBuffer
from copy import deepcopy
from parameters import Parameters

class Othello():
    def __init__(self):
        self.params = Parameters()

        self.num = self.params.board_x
        self.directions = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])

        self.Ini_board()

    def Ini_board(self):
        self.board          = self.Create_board()
        self.current_player = self.params.black
        self.seq_boards     = RingBuffer(self.params.k_boards)
        for i in range(self.params.k_boards):
            self.seq_boards.add(np.zeros((self.num, self.num)))
        self.seq_boards.add(deepcopy(self.board))

    def Create_board(self):
        b = np.zeros((self.num, self.num))
        b[self.num // 2 - 1][self.num // 2 - 1] = self.params.white
        b[self.num // 2    ][self.num // 2 -1]  = self.params.black
        b[self.num // 2 - 1][self.num // 2]     = self.params.black
        b[self.num // 2    ][self.num // 2]     = self.params.white
        return b

    def Print_board(self):
        print("  ", end='')
        for j in range(self.num):
            print(str(j)+"|", end='')
        print("")

        for i in range(self.num):
            print(str(i)+"|", end='')
            for j in range(self.num):
                if self.board[i][j] == self.params.white:
                    print("o|", end='')
                elif self.board[i][j] == self.params.black:
                    print("x|", end='')
                else:
                    print(" |", end='')
            print("")


    def Put(self, b, x, y, player):
        b[x][y] = player
        return b

    def Get_board(self):
        return deepcopy(self.board)

    def Get_board_size(self):
        return (self.num, self.num)

    def Get_action_size(self):
        return self.params.action_size

    def Get_winner(self):
        b = self.board
        winner = None

        if self.Check_game_end():
            if np.sum(b) * self.params.white > 0:
                winner = self.params.white
            elif np.sum(b) * self.params.white < 0:
                winner = self.params.black
            else:
                winner = 0

        return winner

    def Get_valid_moves(self, p = None):
        if p == None:
            player = self.current_player
        else:
            player = p

        valid_moves = []
        moves = np.argwhere(self.board == 0)

        if np.size(moves) != 0:
            valid_flag = False
            for x, y in moves:
                valid = False

                for i, j in self.directions:
                    row = x + i
                    col = y + j
                    if row >= 0 and col >= 0 and row < self.num and col < self.num:
                        if self.board[row][col] == player * (-1):
                            while True:
                                row += i
                                col += j
                                if row >= 0 and col >= 0 and row < self.num and col < self.num:
                                    if self.board[row][col] == 0:
                                        break
                                    elif self.board[row][col] == player:
                                        valid = True
                                        break
                                else:
                                    break

                if valid:
                    valid_moves.append([x, y])
                    valid_flag = True

            if valid_flag == False:
                valid_moves = [[self.num - 1, self.num]]

        else:
            valid_moves = [[self.num - 1, self.num]]

        return(valid_moves)

    def Check_game_end(self):
        if np.size(np.argwhere(self.board == 0)) == 0:
            return(True)
        elif np.size(np.argwhere(self.board == self.params.white)) == 0:
            return(True)
        elif np.size(np.argwhere(self.board == self.params.black)) == 0:
            return(True)
        elif self.Get_valid_moves(p = self.params.white)[0][1] == self.num and self.Get_valid_moves(p = self.params.black)[0][1] == self.num:
            return(True)
        else:
            return(False)


    def Put_stone(self, action):
        if action[1] != self.num:
            player = self.current_player
            x = action[0]
            y = action[1]
            self.board[x][y] = self.current_player
            for i, j in self.directions:
                reverse = False
                row = x + i
                col = y + j
                if row >= 0 and col >= 0 and row < self.num and col < self.num:
                    if self.board[row][col] == player * (-1):
                        while True:
                            row += i
                            col += j
                            if row >= 0 and col >= 0 and row < self.num and col < self.num:
                                if self.board[row][col] == 0:
                                    break
                                elif self.board[row][col] == player:
                                    reverse = True
                                    break
                            else:
                                break

                if reverse:
                    row = x + i
                    col = y + j
                    self.board[row][col] = player
                    while True:
                        row += i
                        col += j
                        if row >= 0 and col >= 0 and row < self.num and col < self.num:
                            if self.board[row][col] == player * (-1):
                                self.board[row][col] = player
                            else:
                                break


    def Get_states(self):
        temp_states = self.seq_boards.Get_buffer()
        states = []
        for i in range(self.params.k_boards):
            states.append(np.where(temp_states[i] == self.params.white, 1, 0))
            states.append(np.where(temp_states[i] == self.params.black, 1, 0))

        if self.current_player == 1:
            states.append(np.ones((self.num, self.num)))
        else:
            states.append(np.zeros((self.num, self.num)))

        return np.array(states)

    def Play_action(self, action):
        self.Put_stone(action)
        self.current_player *= -1
        self.seq_boards.add(deepcopy(self.board))

    def Get_current_player(self):
        return self.current_player

if __name__ == '__main__':
    ot = Othello()

    ot.board = np.array([
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1,  0,  0],
        [-1, -1, -1, -1, -1,  0],
        [-1, -1, -1, -1, -1,  1],
        [-1, -1, -1, -1, -1,  0],
        [-1,  0, -1, -1, -1, -1]])
    ot.Print_board()
    print(ot.Get_valid_moves())
    print(ot.Get_winner())
    print(ot.Check_game_end())
#     while(1):
# #        print(ot.Get_states())
#         print(ot.Get_valid_moves())
#         action = list(map(int, input().split(",")))
#         ot.Play_action(action)
#         ot.Print_board()

#         if ot.Check_game_end():
#             winner = ot.Get_winner()
#             print(winner)
#             exit()
