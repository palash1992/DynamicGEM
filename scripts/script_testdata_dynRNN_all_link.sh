#!/bin/bash
for sm in $(seq 2000 3000 5000) 
	do
		for emb in $(seq 64 64 256) 
			do
					python -W ignore ./embedding/dynRNN.py -iter 300 -l 20 -emb $emb -t academic -sm $sm 
					python -W ignore ./embedding/dynRNN.py -iter 300 -l 50 -emb $emb -t hep -sm $sm  
					python -W ignore ./embedding/dynRNN.py -iter 300 -l 50 -emb $emb -t AS -sm $sm
			done
	done	

python -W ignore ./embedding/dynRNN.py -iter 300 -l 100 -emb 128 -t hep -sm 5000  
python -W ignore ./embedding/dynRNN.py -iter 300 -l 100 -emb 128 -t AS -sm 5000