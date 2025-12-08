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
In this script, we run the code to execute the hyperparameter study for EGS.
The relevant parameters are set in the main function, and they are:
{
    dataset: str, specifying the name of the dataset (for this study, we only use 'NF-BoT-IoT')
    version: int, specifying the version of the dataset (for this study, we only use version 1)
    randomized_list: list of bool, specifying whether to randomize source IPs or not
    model_type: str, specifying the type of GNN model to use (for this study, we only use EGS, i.e., 'E_GraphSAGE')
    multiclass: bool, specifying whether the task is multiclass classification (for this study, we use True)
    numEpochs: int, specifying the number of training epochs
    numRealizations: int, specifying the number of runs for experiments, where the random seed is different for each run
    numK: int, specifying the number of neighbors to consider in EGS
    dimH: int, specifying the hidden dimension size in the EGS
}
The results, namely class-weighted recall and F1-score, are saved in the 'hyperparam' directory. These results are obtained by averaging the metrics over all runs for each hyperparameter combination. For each run, they correspond to the metrics obtained on the validation set for the epoch with the lowest validation loss.

### getResults-hyperparameter.py
### data_info.py
### final_tests.py

## Results
### @Moritz (how to use the results thing)
(We have NF-BoT, NF-ToN and UNSW all randomized. Plus, we have NF-BoT non randomized.)
### @Joao (hyperparam directory)

## Notes on Infrastructure
For training and testing, we used a PNY GeForce RTX 2080 Ti 11GB Blower GPU graphics card.