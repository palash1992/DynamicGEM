#!/bin/bash
for emb in $(seq 64 64 256) 
	do
			python -W ignore ./embedding/ae_static.py -iter 250 -l 20 -emb $emb -t enron 
	done
	

