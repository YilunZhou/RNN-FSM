# This is an example script for sequence of commands to run
# Usage : ./run_atari.sh PongDeterministic-v4
#!/usr/bin/env bash

ENV=$1
GRU_SIZE=32
FILE=main_atari.py

# Assuming Pre-trained models exist
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --generate_bn_data --generate_max_steps 100 --no_render
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --bhx_train --no_render
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --bhx_test --no_render
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --ox_train --no_render
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --ox_test --no_render
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --bgru_train --no_render
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --bgru_test --no_render
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --generate_fsm --no_render
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --evaluate_fsm --no_render

exit
