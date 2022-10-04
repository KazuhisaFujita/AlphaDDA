#---------------------------------------
#Since : 2019/04/23
#Update: 2022/10/04
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from nn import NNetWrapper
from Othello import Othello
from AlphaZero_mcts import A_MCTS
from collections import deque
from parameters import Parameters
from minimax import Minimax
from copy import deepcopy
from player import Random_player
from ringbuffer import RingBuffer
import time
import multiprocessing as mp
import math

class Train():
    def __init__(self):
        self.params = Parameters()
        self.comp_time = 0
        self.net    = NNetWrapper(params = self.params)

    def Make_schedule(self, num, players):
        num_players = len(players)
        schedule = []
        if players[0] == players[1] and len(players) == 2:
            for n in range(num):
                schedule.append([players[0], players[1]])
        else:
            for n in range(num//2):
                for i in range(num_players):
                    for j in range(num_players):
                        if i != j:
                            schedule.append([players[i], players[j]])

        return schedule

    def AlphaZero(self, g, count):
        amcts = A_MCTS(game = g, net = self.net, params = self.params)
        amcts.num_moves = count
        action = amcts.Run()
        prob = amcts.Get_prob()
        return action, prob

    def Minimax(self, g):
        mm = Minimax(g)
        action = mm.Run()
        return action

    def Action(self, g, count = 0, player = "alphazero"):
        if player == "alphazero":
            action, prob = self.AlphaZero(g, count)
            return action, prob
        elif player == "minimax":
            action = self.Minimax(g)
            return action


    def Run(self):
        # Start training.
        mp.set_start_method('spawn')

        # Make buffers to store the training data.
        buf_board = RingBuffer(self.params.input_size)
        buf_prob = RingBuffer(self.params.input_size)
        buf_v = RingBuffer(self.params.input_size)


        # Make the schedule for training.
        schedule = self.Make_schedule(self.params.num_games, ["alphazero", "alphazero"])
        schedule_list = [schedule[i::self.params.num_processes_training] for i in range(self.params.num_processes_training)]

        for i in range(1, self.params.num_iterations+1):
            # Set the openings for the training.
            self.params.opening = self.params.opening_train

            start = time.time()

            # Initialize the lists to store boards, probabilities, and values obtained in self-play.
            training_board = []
            training_prob  = []
            training_v     = np.empty(0)

            # Set the devices.
            devices = self.params.devices

            # Start self-play.
            pool    = mp.Pool(self.params.num_processes_training)
            results = [pool.apply_async(self.self_play, args=(devices[i],schedule_list[i],)) for i in range(self.params.num_processes_training)]
            output  = [p.get() for p in results]
            pool.close()
            pool.join()

            # Add the data obtained in self-play to the lists.
            for j in range(self.params.num_processes_training):
                training_board += output[j][0]
                training_prob += output[j][1]
                training_v = np.append(training_v, output[j][2])

            # Augment the data obtained in self-play.
            training_board, training_prob, training_v = self.Augment_data(training_board, training_prob, training_v)

            # Add the augmented data to the input buffers.
            for j in range(len(training_board)):
                buf_board.add(training_board[j])
                buf_prob.add(training_prob[j])
                buf_v.add(training_v[j])

            # Train the DNN.
            self.Learning(buf_board.Get_buffer_start_end(), buf_prob.Get_buffer_start_end(), buf_v.Get_buffer_start_end(), i)

            #Calculate computation time.
            self.comp_time = time.time() - start

            # Test the trained AlphaZero.
            if i%self.params.checkpoint_interval == 0:
                self.test(i)


    def self_play(self, device, schedule):
        self.net.device = device
        g = Othello()

        board_data = []
        prob_actions = []
        v_data = np.empty(0)

        for i in range(len(schedule)):
            p = schedule[i]

            temp_v_data = np.empty(0)

            winner = 0
            g.Ini_board()

            count = 0
            while True:
                count += 1

                board_data.append(g.Get_states())
                temp_v_data = np.append(temp_v_data, 1)
                action, prob = self.Action(g = g, count = count, player = p[(count - 1)%2])
                g.Play_action(action)
                prob_actions.append(prob)

                if g.Check_game_end():
                    winner = g.Get_winner()
                    break

            temp_v_data *= winner
            v_data = np.append(v_data, temp_v_data)

        return(board_data, prob_actions, v_data)

    def Learning(self, training_board, training_prob, training_v, i):
        self.net.device = 'cuda'
        self.net.train(np.array(training_board), np.array(training_prob), np.array(training_v))
        self.net.save_checkpoint(i)

    def Augment_data(self, training_board, training_prob, training_v):
        aug_training_board = deepcopy(training_board)
        aug_training_prob = deepcopy(training_prob)
        aug_training_v = deepcopy(training_v)

        for i in range(len(training_board)):
            board = training_board[i]
            prob = training_prob[i]

            # flip
            flip_board = np.zeros((self.params.k_boards * 2 + 1, self.params.board_x, self.params.board_y))
            for j in range(board.shape[0]):
                flip_board[j] = np.flip(board[j], axis=0)
            aug_training_board.append(flip_board)

            flip_prob = prob
            prob_none = flip_prob[-1]
            flip_prob = np.delete(flip_prob, -1)
            flip_prob = np.flip(flip_prob.reshape((self.params.board_x, self.params.board_y)), axis = 0).reshape(self.params.board_x*self.params.board_y)
            flip_prob = np.insert(flip_prob, flip_prob.size, prob_none)
            aug_training_prob.append(flip_prob)
            aug_training_v = np.append(aug_training_v, training_v[i])

            for _ in range(3):
                # rot90
                rot_board = np.zeros((self.params.k_boards * 2 + 1, self.params.board_x, self.params.board_y))
                for j in range(board.shape[0]):
                    rot_board[j] = np.rot90(board[j])
                board = rot_board
                aug_training_board.append(rot_board)

                rot_prob = prob
                prob_none = rot_prob[-1]
                rot_prob = np.delete(rot_prob, -1)
                rot_prob = np.rot90(rot_prob.reshape((self.params.board_x, self.params.board_y))).reshape(self.params.board_x*self.params.board_y)
                rot_prob = np.insert(rot_prob, rot_prob.size, prob_none)
                prob = rot_prob
                aug_training_prob.append(rot_prob)
                aug_training_v = np.append(aug_training_v, training_v[i])

                # flip
                flip_board = np.zeros((self.params.k_boards * 2 + 1, self.params.board_x, self.params.board_y))
                for j in range(board.shape[0]):
                    flip_board[j] = np.flip(board[j], axis=0)
                aug_training_board.append(flip_board)

                flip_prob = prob
                prob_none = flip_prob[-1]
                flip_prob = np.delete(flip_prob, -1)
                flip_prob = np.flip(flip_prob.reshape((self.params.board_x, self.params.board_y)), axis = 0).reshape(self.params.board_x*self.params.board_y)
                flip_prob = np.insert(flip_prob, flip_prob.size, prob_none)
                aug_training_prob.append(flip_prob)
                aug_training_v = np.append(aug_training_v, training_v[i])

        return aug_training_board, aug_training_prob, aug_training_v

    def test(self, i):
        self.params.opening = self.params.opening_test

        schedule = self.Make_schedule(self.params.num_test, ["alphazero", "minimax"])
        schedule_list = [schedule[i::self.params.num_processes_test] for i in range(self.params.num_processes_test)]

        devices = self.params.devices
        pool = mp.Pool(self.params.num_processes_test)
        results = [pool.apply_async(self.arena_test, args=(devices[i],schedule_list[i],)) for i in range(self.params.num_processes_test)]
        output = [p.get() for p in results]
        pool.close()
        pool.join()

        num_win_alpha = 0
        num_win_mm = 0

        for j in range(self.params.num_processes_test):
            num_win_alpha += output[j][0]
            num_win_mm += output[j][1]
        print(i, num_win_alpha/self.params.num_test,  num_win_mm/self.params.num_test, (self.params.num_test - num_win_alpha - num_win_mm)/self.params.num_test, self.comp_time)

    def arena_test(self, device, schedule):
        self.net.device = device
        g = Othello()

        num_win_alpha = 0
        num_win_mm = 0

        for i in range(len(schedule)):
            p = schedule[i]

            g.Ini_board()

            count = 0
            while True:
                count += 1

                if p[(count - 1)%2] == "alphazero":
                    action, _ = self.Action(g = g, count = count, player = p[(count - 1)%2])
                else:
                    action = self.Action(g = g, count = count, player = p[(count - 1)%2])

                g.Play_action(action)

                if g.Check_game_end():
                    winner = g.Get_winner()
                    if winner == self.params.black:
                        if p[0] == "alphazero":
                            num_win_alpha += 1
                        else:
                            num_win_mm += 1
                    if winner == self.params.white:
                        if p[1] == "alphazero":
                            num_win_alpha += 1
                        else:
                            num_win_mm += 1
                    break

        return(num_win_alpha, num_win_mm)

if __name__ == '__main__':
    tr = Train()
    tr.Run()
