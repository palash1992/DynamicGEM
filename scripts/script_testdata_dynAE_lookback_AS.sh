#!/bin/bash
for lb in $(seq 2 1 10) 
	do
			# python -W ignore ./embedding/dynAE.py -iter 250 -l 20 -lb $lb -emb 128 -sm $sm 2000 -rd ./resultsdynAE_lookback -t academic
			# python -W ignore ./embedding/dynAE.py -iter 250 -l 50 -lb $lb -emb 128 -sm $sm 2000 -rd ./resultsdynAE_lookback -t hep
			python -W ignore ./embedding/dynAE.py -iter 250 -l 50 -lb $lb -emb 128 -sm $sm 2000 -rd ./resultsdynAE_lookback_AS -t AS
			# python -W ignore ./embedding/dynAE.py -iter 100 -l 20 -lb $lb -emb 128 -sm $sm 2000 -rd ./resultsdynAE_lookback -t enron
	done

