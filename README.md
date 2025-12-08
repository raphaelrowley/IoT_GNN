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
    model_type: str, specifying the type of the model to use (for this study, we only use EGS, i.e., 'E_GraphSAGE')
    multiclass: bool, specifying whether the task is multiclass classification (for this study, we use True)
    numEpochs: int, specifying the number of training epochs
    numRealizations: int, specifying the number of runs for experiments, where the random seed is different for each run
    numK: int, specifying the number of layers to consider in EGS
    dimH: int, specifying the hidden dimension size in EGS
}
The results, namely class-weighted recall and F1-score, are saved in the 'hyperparam' directory. For each run, they correspond to the metrics obtained on the validation set for the epoch with the lowest validation loss. Observe that even for the non-randomized case, we still have multiple runs with different random seeds, as they affect the initialization of the model weights.

### getResults-hyperparameter.py
This script is used to read the results obtained from the hyperparameter study and print the results of EGS for parameter combinations. For a given set of parameters, the printed results consider the mean, standard deviation, minimum, and maximum of the class-weighted recall and F1-score over all runs. The set of relevant parameters to read the results are similar to the hyperparameters.py script.

### data_info.py
This script is used to study the dataset of interest, that is, NF-BoT-IoT version 1. The relevant parameters for this script are:
{
    dataset: str, specifying the name of the dataset (for this study, we only use 'NF-BoT-IoT')
    version: int, specifying the version of the dataset (for this study, we only use version 1)
    randomized_source_ip: bool, specifying whether to randomize source IPs or not
    print_df_info: bool, specifying whether to print the dataset information
    print_graph_info: bool, specifying whether to print graph information associated with the dataset
    random_seed: int, specifying the random seed used for randomization of source IPs (if applicable)
}
In this script, we print important information on the dataset, such as the number of samples, number of features, data type of each feature and so on. Regarding the associated graph, we print information such as the number of nodes, number of edges, degree distribution. More importantly, we check whether or not the graph has cycles. If yes, we print the longest path found in the graph. This is important to understand whether multi-hop message passing can be effective in this graph.

### final_tests.py

## Results
### @Moritz (how to use the results thing)
(We have NF-BoT, NF-ToN and UNSW all randomized. Plus, we have NF-BoT non randomized.)
### @Joao (hyperparam directory)
The 'hyperparam' directory contains the results obtained from the hyperparameter study for EGS on NF-BoT-IoT version 1. Each file in this directory corresponds to a specific combination of hyperparameters (numK and dimH) and whether source IPs were randomized or not. The files can be read using the getResults-hyperparameter.py script to obtain the summarized results (mean, standard deviation, minimum, and maximum of class-weighted recall and F1-score) over multiple runs (in this study, five) for each parameter combination. Each run corresponds to a different random seed.

## Notes on Infrastructure
For training and testing, we used a PNY GeForce RTX 2080 Ti 11GB Blower GPU graphics card.