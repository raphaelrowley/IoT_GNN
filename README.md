# IoT_GNN

## Environment Setup
Development was done on Linux machines. Python 3.11 is necessary due to the underlying packages used.  
Anaconda is used to create a conda environment and install the necessary packages inside that environment.  

The commands to create the conda environment with the required packages are below. We used the following Anaconda version (Anaconda3-2025.06-0-Linux-x86_64).

Commands to create the environment and install the minimal packages:
```bash
conda create -n iot_env python=3.11
conda activate iot_env
conda install pytorch
conda install pandas
conda install scikit-learn
conda install matplotlib
conda install pytorch::torchdata
conda install conda-forge::dgl
```  
Then, the conda environment has to be exported in order to use it for a Jupyter Notebook.  
```bash
conda install ipykernel
python -m ipykernel install --user --name iot_env --display-name "Python (iot_env)"
```  
To run a notebook using iot_env, select the iot_env kernel to run the ipynb.

## Datasets

## Important Scripts
### iot.ipynb
### main.py
### hyperparameters.py
### getResults-hyperparameter.py
### data_info.py
### final_tests.py

## Results
### @Moritz (how to use the results thing)
(We have NF-BoT, NF-ToN and UNSW all randomized. Plus, we have NF-BoT non randomized.)
### @Joao (hyperparam directory)

## Notes on Infrastructure
For training and testing, we used a PNY GeForce RTX 2080 Ti 11GB Blower GPU graphics card.