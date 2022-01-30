#---------------------------------------
#Since : 2019/06/12
#Update: 2022/01/22
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from connect4 import Connect4
from player import Random_player
from classical_MCTS import MCTS
from minimax import Minimax
from minimax2 import Minimax as Minimax2
from AlphaZero_mcts import A_MCTS
from AlphaDDA1 import A_MCTS as DDA
from nn import NNetWrapper
import time
import multiprocessing as mp
from parameters import Parameters
import random
import sys

class Cal:
    def __init__(self, num_mean = 1, A = 1000, X0 = 0.0, N_MAX = 300):
        self.num_games = 50
        self.dda_player  = "alphazero_dda"
        self.players  = ["alphazero", "mcts1", "mcts2", "mcts3", "mcts4", "minimax1", "minimax2", "random"]

        self.params = Parameters()
        self.score_win = 1
        self.net  = NNetWrapper()
        self.net.load_checkpoint()
        mp.set_start_method('spawn')

        self.num_mean = num_mean
        self.A  = A
        self.X0 = X0
        self.N_MAX = N_MAX

    def Init_win(self, players):
        results = {}
        for p in players:
            results[p] = np.array([0, 0])
        return results

    def Make_schedule_first(self, players):
        num_players = len(players)
        schedule = []

        for i in range(num_players):
            for j in range(self.num_games):
                schedule.append([self.dda_player, players[i]])

        return schedule

    def Make_schedule_second(self, players):
        num_players = len(players)
        schedule = []

        for i in range(num_players):
            for j in range(self.num_games):
                schedule.append([players[i], self.dda_player])

        return schedule

    def Action(self, g, count = 0, player = "random", states = None):
        if player == "random":
            self.Random(g)
        elif player == "alphazero":
            self.AlphaZero(g, count)
        elif player == "alphazero_dda":
            self.dda(g, count, states)
        elif player == "minimax":
            self.Minimax(g)
        elif player == "minimax2":
            self.Minimax2(g)
        elif player == "mcts1":
            self.MCTS(g, 300)
        elif player == "mcts2":
            self.MCTS(g, 100)
        elif player == "mcts3":
            self.MCTS(g, 200)
        elif player == "mcts4":
            self.MCTS(g, 50)

    def dda(self, g, count, states):
        amcts = DDA(game = g, net = self.net, num_mean = self.num_mean, X0 = self.X0, A = self.A, N_MAX = self.N_MAX, states = states)
        amcts.num_moves = count
        action = amcts.Run()
        g.Play_action(action)

    def AlphaZero(self, g, count):
        amcts = A_MCTS(game = g, net = self.net)
        amcts.num_moves = count
        action = amcts.Run()
        g.Play_action(action)

    def Random(self, g):
        rand = Random_player(g)
        move = rand.Move()
        g.Play_action(move)

    def Minimax(self, g):
        mm = Minimax(g)
        action = mm.Run()
        g.Play_action(action)

    def Minimax2(self, g):
        mm = Minimax2(g)
        action = mm.Run()
        g.Play_action(action)

    def MCTS(self, g, sim):
        mcts = MCTS(g)
        mcts.num_sim = sim
        action = mcts.Run()
        g.Play_action(action)

    def Cal(self):
        win_lose = self.Init_win(self.players)

        schedule = self.Make_schedule_first(self.players)
        win_lose_first = self.parallel_play(schedule)
        for i in self.players:
            print("first:", self.num_mean, self.A, self.X0, self.N_MAX, i, win_lose_first[i][0], win_lose_first[i][1], self.num_games - win_lose_first[i][0] - win_lose_first[i][1])

        for i in self.players:
            win_lose[i] += win_lose_first[i]

        schedule = self.Make_schedule_second(self.players)
        win_lose_second = self.parallel_play(schedule)
        for i in self.players:
            print("second:", self.num_mean, self.A, self.X0, self.N_MAX, i, win_lose_second[i][0], win_lose_second[i][1], self.num_games - win_lose_second[i][0] - win_lose_second[i][1])

        for i in self.players:
            win_lose[i] += win_lose_second[i]

        for i in self.players:
            print("total:", self.num_mean, self.A, self.X0, self.N_MAX, i, win_lose[i][0], win_lose[i][1], self.num_games * 2- win_lose[i][0] - win_lose[i][1])

    def parallel_play(self, schedule):
        win_lose = self.Init_win(self.players)

        num_processes = 7
        devices       = ['cuda:0', 'cuda:0', 'cuda:0', 'cuda:0', 'cuda:0', 'cuda:0', 'cuda:0', 'cuda:0', 'cuda:0', 'cuda:0']

        random.shuffle(schedule)
        schedule_list = [schedule[i::num_processes] for i in range(num_processes)]

        pool = mp.Pool(num_processes)
        results = [pool.apply_async(self.play, args=(devices[i], schedule_list[i],)) for i in range(num_processes)]
        output = [p.get() for p in results]
        pool.close()
        pool.join()
        processes = []

        for j in range(num_processes):
            for l in output[j]:
                pl = l[0]
                score = l[1]
                if pl[0] == self.dda_player:
                    win_lose[pl[1]][0] += score[0]
                    win_lose[pl[1]][1] += score[1]
                else:
                    win_lose[pl[0]][0] += score[1]
                    win_lose[pl[0]][1] += score[0]

        return win_lose

    def play(self, device, schedule):
        self.net.device = device
        g = Connect4()
        results = []
        states = []

        for i in range(len(schedule)):
            p = schedule[i]
            score = np.zeros(2)
            g.Ini_board()

            count = 0
            while(1):
                count += 1

                states.append(g.Get_states())
                if len(states) > 2:
                    states.pop(0)

                self.Action(g = g, count = count, player = p[(count - 1)%2], states = states)
                if g.Check_game_end():
                    winner = g.Get_winner()
                    if winner == self.params.red:
                        score[0] += self.score_win
                    elif winner == self.params.yellow:
                        score[1] += self.score_win
                    break

            results.append([p, score])

        return results


if __name__ == '__main__':
    num_mean = int(sys.argv[1])
    A = float(sys.argv[2])
    X0 = float(sys.argv[3])
    N_MAX = int(sys.argv[4])
    cal = Cal(num_mean = num_mean, A = A, X0 = X0, N_MAX = N_MAX)
    cal.Cal()
