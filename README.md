# AlphaDDA

AlphaDDA is an AlphaZero-based game AI with dynamic difficulty adjustment.
It consists of MCTS and a deep neural network (DNN) like AlphaZero.
It changes its skill according to the state's value estimated by the DNN.'a
I propose three types of AlphaDDA.
- AlphaDDA1: It changes the number of simulations in MCTS according to the value.
- AlphaDDA2: It changes the dropout probability according to the value. In AlphaDDA2, the DNN used by MCTS is damaged by the dropout and outputs the inaccurate value.
- AlphaDDA3: It applies a new UCT score. The UCT score is made based on the two assumptions: Its opponent continues to make a board state with the same value as the current board state. It makes a board state with the inverse value of the current board state.
In this study, AlphaDDAs play Connect4, 6x6 Othello, and Othello with the AI players.
6x6 Othello is Othello using a 6x6 board.
The weights of AlphaDDAs are the same as trained AlphaZero.
The codes of AlphaZero used in this study are opened in the "AlphaZero directory".
The details of AlphaDDA are denoted in Fujita 2021.

## References

- Kazuhisa Fujita (2021) AlphaDDA: game artificial intelligence with dynamic difficulty adjustment using AlphaZero, arXiv:2111.06266.
