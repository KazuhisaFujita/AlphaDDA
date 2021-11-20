#---------------------------------------
#Since : 2019/04/10
#Update: 2020/09/03
# -*- coding: utf-8 -*-
#---------------------------------------
import numpy as np
from Othello import Othello
from player import Random_player
from classical_MCTS import MCTS
from minimax import Minimax
from AlphaZero_mcts_dda_simple_v import A_MCTS
from nn import NNetWrapper
import time

if __name__ == '__main__':
    win_alpha = 0
    win_mcts = 0
    win_rand = 0
    win_mm = 0

    net  = NNetWrapper()
    net.device = "cuda"
    net.load_checkpoint()

    for i in range(1):
        g = Othello()

        count = 0

        while(1):

            count += 1

            g.Print_board()
            print(g.Get_valid_moves())
            action = list(map(int, input().split(",")))
            print(action)
            g.Play_action(action)

            if g.Check_game_end():
                winner = g.Get_winner()
                print(winner)
                exit()

            count += 1
            print("Alpha")
            start = time.time()
            g.Print_board()
            psa_vector, v =  net.predict(g.Get_states())
            print(v)
            print(g.Get_valid_moves())
            amcts = A_MCTS(game = g, net = net)
            amcts.num_moves = count
            amcts.params.opening = 1
            amcts.params.Temp = 10
            action = amcts.Run()
            print(action)
            g.Play_action(action)
            print("time:", time.time() - start)
            if g.Check_game_end():
                winner = g.Get_winner()
                print(winner)
                if winner == -1:
                    win_alpha += 1
                if winner == 1:
                    win_rand += 1
                #exit()
                break

            # count += 1
            # print("Minimax")
            # start = time.time()
            # g.Print_board()
            # mm = Minimax(g)
            # action = mm.Run()
            # g.Play_action(action)
            # print(g.Get_valid_moves())
            # print(action)
            # print("time:", time.time() - start)
            # if g.Check_game_end():
            #     winner = g.Get_winner()
            #     print(winner)
            #     if winner == 1:
            #         win_mm += 1
            #     #exit()
            #     break

            # count += 1
            # print("MCTS")
            # start = time.time()
            # mcts = MCTS(g)
            # mcts.num_sim = 400
            # action = mcts.Run()
            # g.Play_action(action)
            # g.Print_board()
            # print(action)
            # print("time:", time.time() - start)
            # if g.Check_game_end():
            #     winner = g.Get_winner()
            #     print(winner)
            #     if winner == 1:
            #         win_mcts += 1
            #     #exit()
            #     break

            # print("Random")
            # rand = Random_player(g)
            # move = rand.Move()
            # g.Play_action(move)
            # g.Print_board()
            # if g.Check_game_end():
            #     winner = g.Get_winner()
            #     print(winner)
            #     if winner == 1:
            #         win_rand += 1
            #     if winner == -1:
            #         win_alpha += 1
            #     #exit()
            #     break

        g = Othello()

        count = 0

        print("\n\n")
        while(1):
            count += 1

            # print("MCTS")
            # start = time.time()
            # mcts = MCTS(g)
            # action = mcts.Run()
            # g.Play_action(action)
            # #g.Print_board()
            # print("time:", time.time() - start)
            # if g.Check_game_end():
            #     winner = g.Get_winner()
            #     if winner == -1:
            #         win_mcts += 1
            #     #exit()
            #     break


            # print("Random")
            # rand = Random_player(g)
            # move = rand.Move()
            # g.Play_action(move)
            # if g.Check_game_end():
            #     winner = g.Get_winner()
            #     print(winner)
            #     if winner == -1:
            #         win_rand += 1
            #     if winner == 1:
            #         win_alpha += 1
            #     #exit()
            #     break


            count += 1
            print("Minimax")
            start = time.time()
            g.Print_board()
            mm = Minimax(g)
            action = mm.Run()
            g.Play_action(action)
            print(g.Get_valid_moves())
            print(action)
            print("time:", time.time() - start)
            if g.Check_game_end():
                winner = g.Get_winner()
                print(winner)
                if winner == 1:
                    win_mm += 1
                #exit()
                break

            count += 1
            print("Alpha")
            start = time.time()
            g.Print_board()
            psa_vector, v =  net.predict(g.Get_states())
            print(v)
            print(g.Get_valid_moves())
            amcts = A_MCTS(game = g, net = net)
            amcts.num_moves = count
            amcts.params.opening = 1
            amcts.params.Temp = 10
            action = amcts.Run()
            print(action)
            g.Play_action(action)
            print("time:", time.time() - start)
            if g.Check_game_end():
                winner = g.Get_winner()
                print(winner)
                if winner == -1:
                    win_alpha += 1
                if winner == 1:
                    win_rand += 1
                #exit()
                break

        print("Alpha:", win_alpha / i / 2, "MCTS:", win_mcts / i / 2,  "draw:", (i * 2 - win_alpha - win_mcts) / i / 2)
        print("Alpha:", win_alpha / i / 2, "Rand:", win_rand / i / 2,  "draw:", (i * 2 - win_alpha - win_rand) / i / 2)
        print("Alpha:", win_alpha / i / 2, "Minimax:", win_mm / i / 2, "draw:", (i * 2 - win_alpha - win_mm) / i / 2)
        print("MCTS:",  win_mcts / i / 2,  "Rand:", win_rand / i / 2,  "draw:", (i * 2 - win_mcts - win_rand) / i / 2)
        print("MCTS:", win_mcts / i / 2,   "Minimax:", win_mm / i / 2, "draw:", (i * 2 - win_mcts - win_mm) / i / 2)
        print("Minimax:", win_mm / i / 2,  "Rand:", win_rand / i / 2,  "draw:", (i * 2 - win_mm - win_rand) / i / 2)
