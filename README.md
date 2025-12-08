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

Alternatively, the conda environment can be created using the provided environment.yaml file. We still need to export the environment to use with a Jupyter Notebook.
```bash
conda env create --name iot_env -f environment.yaml
conda activate iot_env
python -m ipykernel install --user --name iot_env --display-name "Python (iot_env)"
```  

## Datasets

## Important Scripts

### iot.ipynb
This notebook serves as an example of how to use the different classes and modules we have created. It shows how to load in datasets, train models and test them. Specifically, the notebook allows to train and test the E-GraphSAGE baseline model, our 2 hidden layer FNN model and our enhanced E-GraphSAGE model that aggregates both edge and node features. This can be done whilst randomizing the source IPs of the dataset or not. This is done for the NF-BoT-IoT-v1 dataset.

### main.py

Similar to ```iot.ipynb```, this script can be used to preprocess and load different datasets, initializing models, training and evaluating them.
At the top of the main function, the dictionary ```dataset_config``` contains all necessary parameters to configure the dataset. Please refer to the docstrings of class ```IoTDataset``` for a detailed description of the parameters.
The dictionary ```training_config``` sets all hyperparameters for the training loop. A description can be found in the docstrings of class ```ModelTrainer```.

The script initializes all models used in our presentation (the E-GraphSAGE baseline model, our 2 hidden layer FNN model, our enhanced E-GraphSAGE model and a DIDS-inspired FNN). 
The hyperparameters of these models, such as number of layers or the dimension of the node embeddings, can be changed as desired.

###### Checkpoints 
The boolean ```USE_CHECKPOINTS``` controls whether the ```ModelTrainer``` uses existing checkpoints which we provided in the directory ```checkpoints```. 
If ```ModelTrainer.train_model(model, use_checkpoint=True)``` is called, and valid checkpoints for the model and the dataset are available, the training loop automatically loads and continues from them.

If the number of epochs is kept at 300, the training loop automatically terminates after loading the weights into the model and generates learning curves ```…_risk.png``` and ```…_classification.png``` in ```checkpoints```, showing how the empirical risk and the classification metrics (precision, recall, F1), respectively, developed during training.

The ```ModelTester``` finally generates a ```…_test_report.txt``` and the ```confusion_matrix.png``` in ```checkpoints```.

All file names are prefixed with ```…```, which contains the model identifier and the dataset ID (e.g., ```NF-BoT-IoT-v1-multiclassE_GraphSAGE_K2_H128```).

Our provided checkpoints can be overwritten when calling the main script with ```USE_CHECKPOINTS = False```

We have provided checkpoints for the datasets  
- ````'NF-BoT-IoT'```` (version 1, randomized and non-randomized IPs)
- ```'NF-ToN-IoT'``` (version 1, randomized IPs)
- ```'NF_UNSW-NB15-v1'``` (version 1, randomized IPs).

For each of these datasets, checkpoints are provided for multiclass classification with the following models:
- Baseline EGS with 2 hidden EGS layers and a node embedding size of 128
- All variants of enhanced EGS with 2 hidden enhanced EGS layers and a node embedding size of 128.
- Baseline fully connected FNN with 2 hidden layers and hidden layer dimensions of 128 and 192.
- DIDS-inspired FNN with 5 hidden layers and hidden layer widths of 18, 36, 72, 144, and 256.

###### Executing the script
To execute the script, call the following:

```bash
(iot_env)$ python main.py
```

### hyperparameters.py
In this script, we run the code to execute the hyperparameter study for EGS.
The relevant parameters are set in the main function, and they are:

<pre>{
    dataset: str, specifying the name of the dataset (for this study, we only use 'NF-BoT-IoT')
    version: int, specifying the version of the dataset (for this study, we only use version 1)
    randomized_list: list of bool, specifying whether to randomize source IPs or not
    model_type: str, specifying the type of the model to use (for this study, we only use EGS, i.e., 'E_GraphSAGE')
    multiclass: bool, specifying whether the task is multiclass classification (for this study, we use True)
    numEpochs: int, specifying the number of training epochs
    numRealizations: int, specifying the number of runs for experiments, where the random seed is different for each run
    numK: int, specifying the number of hidden layers to consider in EGS
    dimH: int, specifying the hidden dimension size in EGS
}</pre>
The results, namely class-weighted recall and F1-score, are saved in the 'hyperparam' directory. For each run, they correspond to the metrics obtained on the validation set for the epoch with the lowest validation loss. Observe that even for the non-randomized case, we still have multiple runs with different random seeds, as they affect the initialization of the model weights.
After setting the relevant parameters, the script can be run as follows:
```bash
(iot_env)$ python hyperparameters.py
```

### getResults-hyperparameter.py
This script is used to read the results obtained from the hyperparameter study for EGS and print the results for all parameter combinations. For a given set of parameters, the printed results consider the mean, standard deviation, minimum, and maximum of the class-weighted recall and F1-score over all runs. The set of relevant parameters to read the results are similar to the hyperparameters.py script.
After setting the relevant parameters, the script can be run as follows:
```bash
(iot_env)$ python getResults-hyperparameter.py
```

### data_info.py
This script is used to study the dataset of interest, that is, NF-BoT-IoT version 1. The relevant parameters for this script are:
<pre>{
    dataset: str, specifying the name of the dataset (for this study, we only use 'NF-BoT-IoT')
    version: int, specifying the version of the dataset (for this study, we only use version 1)
    randomized_source_ip: bool, specifying whether to randomize source IPs or not
    print_df_info: bool, specifying whether to print the dataset information
    print_graph_info: bool, specifying whether to print graph information associated with the dataset
    random_seed: int, specifying the random seed used for randomization of source IPs (if applicable)
}</pre>
In this script, we print important information on the dataset, such as the number of samples, number of features, data type of each feature and so on. Regarding the associated graph, we print information such as the number of nodes, number of edges, degree distribution. More importantly, we check whether or not the graph has cycles. If yes, we print the longest path found in the graph. This is important to understand whether multi-hop message passing can be effective in this graph.

After setting the relevant parameters, the script can be run as follows:
```bash
(iot_env)$ python data_info.py
```

### final_tests.py
This script is used to train and test all of our different GNNs in one script. It was used to obtain our final results on the test set for NF-BoT-IoT-v1, as well as the other datasets. By default, this script executes training and testing on a GPU. This is set in the script at line 109 for training and at line 166 for testing.

The script first sets the dataset_config dict and loads the training, validation and testing datasets. Next, the various models are instantiated and a breakdown of their parameters is printed. The models are then trained and tested; all of the models are first trained and then all of them are tested.  

The recommended way of running:
```bash
(iot_env)$ python final_tests.py > final_test.log
```  
**Note:** as stated in the script, the conda environment has to be updated to use this script (if the environment was created with the manual steps).  
```bash
(iot_env)$ conda install conda-forge::torchinfo
```  

## Results

### hyperparam directory
The 'hyperparam' directory contains the results obtained from the hyperparameter study for EGS on NF-BoT-IoT version 1. Each file in this directory is associated with results for each run index (given by 'g') and whether source IPs are randomized or not. Each run corresponds to a different random seed. Each file corresponds to the results for the set of attempted hyperparameters (numK and dimH). The files can be read using the getResults-hyperparameter.py script to obtain the summarized results (mean, standard deviation, minimum, and maximum of class-weighted recall and F1-score) over multiple runs (in this study, five) for each parameter combination.

## Notes on Infrastructure
For training and testing, we used a PNY GeForce RTX 2080 Ti 11GB Blower GPU graphics card.