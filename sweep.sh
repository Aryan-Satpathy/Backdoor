#!/bin/bash

bash claasrun.sh --method simclr  
bash claasrun.sh --method byol 
bash claasrun.sh --method simsiam  


bash claasrun.sh --method simclr --defense blur 
bash claasrun.sh --method byol --defense blur
bash claasrun.sh --method simsiam --defense blur 


python robust.py --method simclr  
python robust.py --method simclr --blur 

python robust.py --method byol 
python robust.py --method byol --blur

python robust.py --method simsiam  
python robust.py --method simsiam --blur 