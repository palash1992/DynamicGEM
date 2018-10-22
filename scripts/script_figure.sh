#!/bin/bash
for emb in $(seq 64 64 256) 
do
	python -W ignore utils/fig_util.py -t sbm_cd -emb $emb -fig 0 -fs 0 -mn 1
done




