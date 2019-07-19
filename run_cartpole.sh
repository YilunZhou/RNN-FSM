#!/bin/bash

python main_control.py --env CartPole-v1 --gru_size 32 --bhx_size 64 --ox_size 100 --generate_bn_data --generate_max_steps 500 --no_render
