#!/bin/bash
for sm in $(seq 2000 3000 5000) 
	do
		for emb in $(seq 64 64 256) 
			do
					python -W ignore ./embedding/dynAE.py -iter 300 -l 20 -emb $emb -t academic -sm $sm -rd ./resultsdynAE
					python -W ignore ./embedding/dynAE.py -iter 300 -l 50 -emb $emb -t hep -sm $sm -rd ./resultsdynAE 
					python -W ignore ./embedding/dynAE.py -iter 300 -l 50 -emb $emb -t AS -sm $sm -rd ./resultsdynAE
			done
	done	

python -W ignore ./embedding/dynAE.py -iter 300 -l 100 -emb 128 -t hep -sm 5000  -rd ./resultsdynAE 
python -W ignore ./embedding/dynAE.py -iter 300 -l 100 -emb 128 -t AS -sm 5000 -rd ./resultsdynAE