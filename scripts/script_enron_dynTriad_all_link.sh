#!/bin/bash
for emb in $(seq 64 64 256) 
	do
		python -W ignore ./embedding/dynamicTriad.py -iter 10 -n 20 -K $emb  -t enron
			
	done

