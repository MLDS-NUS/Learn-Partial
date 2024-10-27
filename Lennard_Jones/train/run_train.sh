#!/bin/bash

python3 G_lx.py --model_name='onsagernet'
python3 G_lx.py --model_name='MLP'
python3 G_lx.py --model_name='GFINNs'

python3 G_lz.py --model_name='onsagernet'
python3 G_lz.py --model_name='MLP'
python3 G_lz.py --model_name='GFINNs'

