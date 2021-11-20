#---------------------------------------
#Since : 2018/09/16
#Update: 2021/11/16
# -*- coding: utf-8 -*-
# I used the sorce code in https://github.com/taihsuanho/KyleOthello/blob/master/KyleOthello/OthelloLogic.py for the bitboard engene.
#---------------------------------------
import numpy as np
from ringbuffer import RingBuffer
from copy import deepcopy
from parameters import Parameters

kBitBoardShiftRight = ((1, 0xfefefefefefefefe), (7, 0x7f7f7f7f7f7f7f00), (8, 0xffffffffffffff00), (9, 0xfefefefefefefe00))
kBitBoardShiftLeft = ((1, 0x7f7f7f7f7f7f7f7f), (7, 0x00fefefefefefefe), (8, 0x00ffffffffffffff), (9, 0x007f7f7f7f7f7f7f))

class Othello():
    def __init__(self):
        self.params = Parameters()

        self.num = self.params.board_x
        self.directions = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])

        self.Ini_board()

    def Ini_board(self):
        self.board = self.Create_board()
        self.bitboard = self.To_bitboard(self.board)
        self.current_player = self.params.black
        self.seq_boards = RingBuffer(self.params.k_boards)
        for i in range(self.params.k_boards):
            self.seq_boards.add(np.zeros((self.num, self.num)))
        self.seq_boards.add(deepcopy(self.board))

    def Create_board(self):
        b = np.zeros((self.num, self.num))
        b[self.num // 2 - 1][self.num // 2 - 1] = self.params.white
        b[self.num // 2][self.num // 2 -1]      = self.params.black
        b[self.num // 2 - 1][self.num // 2]     = self.params.black
        b[self.num // 2][self.num // 2]         = self.params.white
        return b


    def To_bitboard(self, board):
        bb0 = bb1 = 0
        for i in range(self.num):
            for j in range(self.num):
                bb0 <<= 1
                bb1 <<= 1
                if board[i][j] ==  self.params.white:
                    bb0 |= 1
                if board[i][j] == self.params.black:
                    bb1 |= 1
        return (bb0, bb1)

    def BitBoardToIndex(self, bb):
        pos = []
        num_cells = self.num * self.num
        for i in range(num_cells):
            if bb & 1:
                pos.append([(num_cells -1 - i) // self.num, ((num_cells - 1 - i)) % self.num])
            bb >>= 1

        print(pos)

    def To_board(self, bitboard):
        bb0, bb1 = bitboard
        board = []
        for i in range(self.num*self.num):
            if bb0 & 1:
                board.append(self.params.white)
            elif bb1 & 1:
                board.append(self.params.black)
            else:
                board.append(0)
            bb0 >>= 1
            bb1 >>= 1

        board.reverse()
        return(np.array(board).reshape((self.num, self.num)))


    def Get_valid_moves(self, p = None):
        if p == None:
            player = self.current_player
        else:
            player = p

        bitboard = self.To_bitboard(self.board)

        if player == self.params.white:
            bbOwn, bbOpp = bitboard
        else:
            bbOpp, bbOwn = bitboard
        empty = ~(bbOwn | bbOpp)

        moves = 0;
        for (d, mask) in kBitBoardShiftRight:
            candidates = bbOpp & ((bbOwn & mask) >> d)
            while candidates:
                moves |= empty & ((candidates & mask) >> d)
                candidates = bbOpp  & ((candidates & mask) >> d)

        for (d, mask) in kBitBoardShiftLeft:
            candidates = bbOpp & ((bbOwn & mask) << d)
            while candidates:
                moves |= empty & ((candidates & mask) << d)
                candidates = bbOpp  & ((candidates & mask) << d)

        board = []
        for i in range(self.num*self.num):
            if moves & 1:
                board.append(1)
            else:
                board.append(0)
            moves >>= 1

        board.reverse()
        moves = np.argwhere(np.array(board).reshape((self.num, self.num)))
        if moves.size != 0:
            return moves
        else:
            return [[self.params.board_y - 1, self.params.board_x]]

    def Put_stone(self, action):
        bitboard = self.bitboard

        if action[1] != self.params.board_x:
            if self.current_player == self.params.white:
                bbOwn, bbOpp = bitboard
            else:
                bbOpp, bbOwn = bitboard

            flips = 0;
            x = int(action[0])
            y = int(action[1])
            bb = 0x0000000000000001 << (self.num * self.num - 1 - x * self.num - y);
            for (d, mask) in kBitBoardShiftRight:
                flip = 0
                cc = (bb & mask) >> d
                while cc & bbOpp:
                    flip |= cc
                    cc = (cc & mask) >> d
                if cc & bbOwn:
                    flips |= flip
            for (d, mask) in kBitBoardShiftLeft:
                flip = 0
                cc = (bb & mask) << d
                while cc & bbOpp:
                    flip |= cc
                    cc = (cc & mask) << d
                if cc & bbOwn:
                    flips |= flip

            bbOwn = bbOwn | flips | bb
            bbOpp = bbOpp & ~flips
            if self.current_player == self.params.white:
                self.bitboard = (bbOwn, bbOpp)
            else:
                self.bitboard = (bbOpp, bbOwn)


    def Print_board(self):
        print("  ", end='')
        for j in range(self.params.board_y):
            print(str(j)+"|", end='')
        print("")

        for i in range(self.params.board_x):
            print(str(i)+"|", end='')
            for j in range(self.params.board_y):
                if self.board[i][j] == self.params.white:
                    print("o|", end='')
                elif self.board[i][j] == self.params.black:
                    print("x|", end='')
                else:
                    print(" |", end='')
            print("")

    def Can_put(self, b):
        return b == 0

    def Put(self, b, x, y, player):
        b[x][y] = player
        return b

    def Get_board(self):
        return deepcopy(self.board)

    def Get_board_size(self):
        return (self.num, self.num)

    def Get_action_size(self):
        return self.num * self.num

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

    def Check_game_end(self):
        if np.size(np.argwhere(self.board == 0)) == 0:
            return(True)
        elif self.bitboard[0] == 0:
            return(True)
        elif self.bitboard[1] == 0:
            return(True)
        elif self.Get_valid_moves(p = 1)[0][1] == self.params.board_x and self.Get_valid_moves(p = -1)[0][1] == self.params.board_x:
            return(True)
        else:
            return(False)


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
        self.bitboard = self.To_bitboard(self.board)
        self.Put_stone(action)
        self.board = self.To_board(self.bitboard)
        self.current_player *= -1
        self.seq_boards.add(deepcopy(self.board))


    def Get_current_player(self):
        return self.current_player

if __name__ == '__main__':
    ot = Othello()

    ot.Print_board()

    while(1):
#        print(ot.Get_states())
        print(ot.Get_valid_moves())
        action = list(map(int, input().split(",")))
        ot.Play_action(action)
        ot.Print_board()

        if ot.Check_game_end():
            winner = ot.Get_winner()
            print(winner)
            exit()
