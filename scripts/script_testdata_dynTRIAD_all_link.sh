#!/bin/bash
for sm in $(seq 2000 3000 5000) 
	do
		for emb in $(seq 192 64 256) 
			do
					python -W ignore ./embedding/dynamicTriad.py -iter 30 -n 20 -K $emb -sm $sm -t academic  
					python -W ignore ./embedding/dynamicTriad.py -iter 30 -n 50 -K $emb -sm $sm -t hep   
					python -W ignore ./embedding/dynamicTriad.py -iter 30 -n 50 -K $emb -sm $sm -t AS 
			done
	done	

python -W ignore ./embedding/dynamicTriad.py -iter 30 -n 100 -K 128 -sm 5000 -t hep   
python -W ignore ./embedding/dynamicTriad.py -iter 30 -n 100 -K 128 -sm 5000 -t AS 