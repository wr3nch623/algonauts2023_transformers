#!/bin/bash
nohup (time ~/miniconda3/bin/python3 main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'streams_inc' --device 'cuda:1') |& tee output.txt &

