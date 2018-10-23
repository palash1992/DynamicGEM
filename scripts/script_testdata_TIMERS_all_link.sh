#!/bin/bash
for sm in $(seq 2000 3000 5000) 
	do
		for emb in $(seq 64 64 256) 
			do
					python  -W ignore ./embedding/TIMERS.py -l 20 -emb $emb -sm $sm -t academic  -rd ./resultsTIMERS
					python  -W ignore ./embedding/TIMERS.py -l 50 -emb $emb -sm $sm -t hep -rd ./resultsTIMERS  
					python  -W ignore ./embedding/TIMERS.py -l 50 -emb $emb -sm $sm -t AS  -rd ./resultsTIMERS
			done
	done	

python  -W ignore ./embedding/TIMERS.py -l 100 -emb 128 -sm 5000 -t hep  -rd ./resultsTIMERS 
python -W ignore  ./embedding/TIMERS.py -l 100 -emb 128 -sm 5000 -t AS   -rd ./resultsTIMERS