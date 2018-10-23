#!/bin/bash
for emb in $(seq 64 64 256) 
	do
		for iter in $(seq 10 20 300) 
			do
				python  ./embedding/dynRNN.py -iter $iter -emb $emb
			done
done


