#!/bin/bash
for bs in $(seq 100 100 500) 
	do 
		for iter in $(seq 100 50 500) 
			do
				for eta in 1e-3 1e-4 1e-5 1e-6 1e-7
					do
						for lb in $(seq 1 1 5) 
							do
								for emb in $(seq 64 64 320) 
									do
									python -W ignore ./embedding/dynAE.py -iter $iter -eta $eta -bs $bs -emb $emb -lb $lb -l 10 -nm 10 -rd ./results_hyper -ht 1
									python -W ignore ./embedding/dynRNN.py -iter $iter -eta $eta -bs $bs -emb $emb -lb $lb -l 10 -nm 10 -rd ./results_hyper -ht 1
									python -W ignore ./embedding/dynAERNN.py -iter $iter -eta $eta -bs $bs -emb $emb -lb $lb -l 10 -nm 10 -rd ./results_hyper -ht 1
									done
							done		
					done
			done
	done		

