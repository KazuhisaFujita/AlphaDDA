ulimit -n 1000000
rm -rf *dat
python3 test_dda.py 3 1.6 -1.5 200 >>  alphadda1.dat
