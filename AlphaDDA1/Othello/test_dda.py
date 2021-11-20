#---------------------------------------
#Since : 2019/06/12
#Update: 2021/11/20
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from Othello_bitboard import Othello
from player import Random_player
from classical_MCTS import MCTS
from minimax import Minimax
from AlphaZero_mcts import A_MCTS
from AlphaDDA1 import A_MCTS as DDA
from nn import NNetWrapper
import time
import multiprocessing as mp
from parameters import Parameters
import random
import sys

class Cal_Elo_rating:
    def __init__(self, num_mean = 1, A = 1000, X0 = 0.0, N_MAX = 600):
        self.num_games = 20
        self.dda_player  = "alphazero_dda"
        self.players  = ["alphazero", "minimax", "mcts1", "mcts2", "random"]

        self.params = Parameters()
        self.schedule = self.Make_schedule(self.players)
        self.win_lose = self.Init_win(self.players)
        self.score_win = 1
        self.net  = NNetWrapper()
        self.net.load_checkpoint()
        mp.set_start_method('spawn')
        self.estimated_outcome = []

        self.num_mean = num_mean

        self.A  = A
        self.X0 = X0
        self.N_MAX = N_MAX

    def Init_win(self, players):
        elo = {}
        for p in players:
            elo[p] = [0, 0]
        return elo

    def Make_schedule(self, players):
        num_players = len(players)
        schedule = []

        for i in range(num_players):
            for j in range(self.num_games):
                schedule.append([self.dda_player, players[i]])
                schedule.append([players[i], self.dda_player])

        return schedule

    def Action(self, g, count = 0, player = "random"):
        if player == "random":
            self.Random(g)
        elif player == "alphazero":
            self.AlphaZero(g, count)
        elif player == "alphazero_dda":
            self.dda(g, count)
        elif player == "minimax":
            self.Minimax(g)
        elif player == "mcts2":
            self.MCTS(g, 100)
        elif player == "mcts3":
            self.MCTS(g, 300)

    def dda(self, g, count):
        amcts = DDA(game = g, net = self.net, estimated_outcome = self.estimated_outcome, num_mean = self.num_mean, X0 = self.X0, A = self.A, N_MAX = self.N_MAX)
        amcts.Store_outcome()
        amcts.num_moves = count
        action = amcts.Run()
        g.Play_action(action)
        amcts.Store_outcome()
        self.estimated_outcome = amcts.estimated_outcome

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

    def MCTS(self, g, sim):
        mcts = MCTS(g)
        mcts.num_sim = sim
        action = mcts.Run()
        g.Play_action(action)

    def Cal_elo(self):

        win_opponent = 0

        num_processes = 14
        devices       = ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1', 'cuda:0', 'cuda:1']

        random.shuffle(self.schedule)
        schedule_list = [self.schedule[i::num_processes] for i in range(num_processes)]

        pool = mp.Pool(num_processes)
        results = [pool.apply_async(self.elo, args=(devices[i], schedule_list[i],)) for i in range(num_processes)]
        output = [p.get() for p in results]
        pool.close()
        pool.join()
        processes = []

        for j in range(num_processes):
            for l in output[j]:
                pl = l[0]
                score = l[1]
                if pl[0] != self.dda_player:
                    self.win_lose[pl[0]][0] += score[0]
                    self.win_lose[pl[0]][1] += score[1] #AlphaZero_dda wins
                else:
                    self.win_lose[pl[1]][0] += score[1]
                    self.win_lose[pl[1]][1] += score[0] #AlphaZero_dda wins

        for i in self.players:
            print(self.num_mean, self.A, self.X0, self.N_MAX, i, self.win_lose[i][0], self.win_lose[i][1], self.num_games * 2 - self.win_lose[i][0] - self.win_lose[i][1])

    def elo(self, device, schedule):
        self.net.device = device
        g = Othello()
        results = []

        for i in range(len(schedule)):
            p = schedule[i]
            score = np.zeros(2)
            g.Ini_board()

            count = 0
            while(1):
                count += 1
                self.Action(g = g, count = count, player = p[(count - 1)%2])
                if g.Check_game_end():
                    winner = g.Get_winner()
                    if winner == self.params.black:
                        score[0] += self.score_win
                    elif winner == self.params.white:
                        score[1] += self.score_win
                    break

            results.append([p, score])

        return results

if __name__ == '__main__':
    num_mean = int(sys.argv[1])
    A = float(sys.argv[2])
    X0 = float(sys.argv[3])
    N_MAX = int(sys.argv[4])
    cal_elo = Cal_Elo_rating(num_mean = num_mean, A = A, X0 = X0, N_MAX = N_MAX)
    cal_elo.Cal_elo()
