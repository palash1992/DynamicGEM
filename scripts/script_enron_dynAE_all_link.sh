#!/bin/bash
for emb in $(seq 128 128 256)

do
	for lb in $(seq 4 1 15)
		do
				python -W ignore ./embedding/dynAE.py -iter 200 -l 30 -lb $lb -emb $emb -t enron
		done
done	

