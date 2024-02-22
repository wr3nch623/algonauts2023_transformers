#!/bin/bash
(nohup time ~/miniconda3/bin/python3 main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'streams_inc' --device 'cuda:1') |& tee output.txt
(nohup time ~/miniconda3/bin/python3 main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'visuals' --device 'cuda:1') |& tee output.txt
(nohup time ~/miniconda3/bin/python3 main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'words' --device 'cuda:1') |& tee output.txt
(nohup time ~/miniconda3/bin/python3 main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'faces' --device 'cuda:1') |& tee output.txt
(nohup time ~/miniconda3/bin/python3 main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'bodies' --device 'cuda:1') |& tee output.txt
(nohup time ~/miniconda3/bin/python3 main.py --run 1  --subj 1 --enc_output_layer 1 --readout_res 'places' --device 'cuda:1') |& tee output.txt
