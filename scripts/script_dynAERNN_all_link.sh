#!/bin/bash
for emb in $(seq 64 64 256) 
	do
	for nm in $(seq 10 10 20) 
		do
			python -W ignore ./embedding/dynAERNN.py -emb $emb -l 10 -nm $nm 
		done
done


