#!/bin/bash

criterias='degree eigenvector katz closeness betweenness load harmonic'

for criteria in $criterias
do
	sudo python -W ignore ./embedding/dynAE.py -c $criteria -rc True
	sudo python -W ignore ./embedding/dynRNN.py -c $criteria -rc True
	sudo python -W ignore ./embedding/dynAE.py -c $criteria -rc False
	sudo python -W ignore ./embedding/dynRNN.py -c $criteria -rc False
done


