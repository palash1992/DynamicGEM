#!/bin/bash
for emb in $(seq 64 64 128) 
do
for data in academic hep AS 
	do
		python -W ignore utils/fig_util.py -t all -ts $data -emb $emb -sm 2000 -fig 1 -fs 0 
		python -W ignore utils/fig_util.py -t all -ts $data -emb $emb -sm 2000 -fig 0 -fs 0 
	done
done	




