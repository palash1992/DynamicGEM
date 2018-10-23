#!/bin/bash

criterias='degree eigenvector katz closeness betweenness load harmonic'

for criteria in $criterias
do
	for j in $(seq 5 5 30) 
	do
	 sudo python -W ignore ./embedding/dynAE.py -c $criteria  -l $j
	done

done


