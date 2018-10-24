# DynamicGEM: Dynamic graph to vector embedding


## Implemented Methods
dynamicGEM implements the following graph embedding techniques:
* [Incremental SVD]
* [Rerun SVD]
* [Optimal SVD]
* [Dynamic TRIAD]
* [Static AE]
* [Dynamic AE]
* [Dynamic RNN]
* [Dynamic AERNN]

## Graph Format

## Repository Structure
* **dyngraph2vec/embedding**: 
* **dyngraph2vec/evaluation**: 
* **dyngraph2vec/utils**: 
* **dyngraph2vec/graph_generation**: 
* **dyngraph2vec/visualization**:
* **dyngraph2vec/matlab**: 
* **dyngraph2vec/graphs**:
* **dyngraph2vec/experiments**:
* **dyngraph2vec/TIMERS**:
* **dyngraph2vec/dynamicTriad**:

## Dependencies
dynamicgem is tested to work on python 3.5. The module with working dependencies are listed as follows:

* h5py                   2.8.0
* joblib                 0.12.5
* Keras                  2.0.2
* Keras-Applications     1.0.6
* Keras-Preprocessing    1.0.5
* matlabruntimeforpython R2017a
* matplotlib             3.0.0
* networkx               1.11
* numpy                  1.15.3
* pandas                 0.23.4
* scikit-learn           0.20.0
* scipy                  1.1.0
* seaborn                0.9.0
* setuptools             39.1.0
* six                    1.11.0
* sklearn                0.0
* tensorflow-gpu         1.11.0
* Theano                 1.0.3
* wheel                  0.32.2
## Install
Before setting up DynamicGEM, it is suggested that the dynamic triad and TIMERS are properly set up.

* To setup up the dynamicTriad please follow the repository "https://github.com/luckiezhou/DynamicTriad"
* The TIMERS is originally writtien in matlab, in dynamicgem we have created python modules for Timers using Matlab Library Compiler. We have used Matlab R2017a to generate modules that work with python 3.5. To run the matlab runtime please configure the Matlab runtime by downloading it from "https://www.mathworks.com/products/compiler/matlab-runtime.html" and following steps mentioned in "https://www.mathworks.com/help/compiler/install-the-matlab-runtime.html"


The package uses setuptools, which is a common way of installing python modules. To install in your home directory, use:
```bash
    python setup.py install --user
```

To install for all users on Unix/Linux:
```bash 
    sudo python setup.py install
```
## Usage
### Example 1


## Cite
   
