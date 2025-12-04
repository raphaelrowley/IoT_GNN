from configuration import *

from models import e_graphsage, fnn_model, e_graphsage_hembed
from data import IoTDataset
from train import ModelTrainer


def validation(model, trainer):
    """
    Function to validate the model and return the results of the epoch wit the best F1-score.
    --------------
    Parameters:
    model : Model
        The model to be validated.
    trainer : ModelTrainer
        The trainer object used for training and validation.
    --------------
    Returns:
    best_f1 : float
        The best F1-score achieved during validation.
    best_recall : float
        The recall corresponding to the best F1-score.
    --------------
    """
    prec, recall, f1, val_risk = trainer.train_model(model, False)
    best_epoch = np.argmin(val_risk)
    print(f'Precision: {prec[best_epoch]:.2f},  \
          Recall: {recall[best_epoch]:.2f}, F1-Score: {f1[best_epoch]:.2f} at epoch {best_epoch+1}')
    return f1[best_epoch], recall[best_epoch]

def main(dataset, version, randomized, model_type, multiclass, numEpochs, numK, dimH, g):
    """
    Function to run hyperparameter tuning for different models on the specified dataset.
    --------------
    Parameters:
    dataset : str
        Name of the dataset to be used.
    version : int
        Version of the dataset to be used.
    randomized : bool
        Whether to randomize source IPs in the dataset.
    model_type : str
        Type of model to be used.
    multiclass : bool
        Whether the classification task is multiclass (true) or binary (false).
    numEpochs : int
        Number of training epochs.
    numK : list of int
        List of values for the number of layers (K) to be validated.
    dimH : list of int
        List of values for the hidden dimension (H) to be validated.
    g : int
        Random seed for reproducibility.
    --------------
    Returns:
    f1_score : np.ndarray of dtype float and dimensions (len(numK), len(dimH), numRealizations)
        Array of F1-scores for each combination of K and H across realizations.
    re_score : np.ndarray of dtype float and dimensions (len(numK), len(dimH), numRealizations)
        Array of recall scores for each combination of K and H across realizations.
    --------------
    """
    dataset_config = {'multiclass': multiclass, 'dataset': dataset, 'randomize_source_ip': randomized, 'version': version}
    

    training_config = {
    'num_epochs': numEpochs,             
    'lr': 1e-3,
    'gpu': True,
    'lr_sched_factor': np.sqrt(10),    
    'lr_sched_patience': 100,        
    }

    f1_score = np.zeros((len(numK), len(dimH)))
    re_score = np.zeros((len(numK), len(dimH)))

    np.random.seed(g) #random seed for the randomization of ip addresses
    torch.manual_seed(g) #random seed for pytorch
    train_data = IoTDataset(**dataset_config,g=g)
    val_data = IoTDataset(**dataset_config, split='val', g=g)
    trainer = ModelTrainer(training_config, train_data, val_data, g=g)
    for idx_k, k in enumerate(numK):
        for idx_h, h in enumerate(dimH):
            print(f'Validating {model_type} with g={g}, K={k}, H={h}')
            if model_type == 'E_GraphSAGE':
                model = e_graphsage.E_GraphSAGE(numLayers=k,
                                            dim_node_embed=h,
                                            num_edge_attr=train_data.num_features,
                                            num_classes=len(train_data.classes),
                                            dropout=0.2,
                                            normalization=False
                                            )
            elif model_type == 'E_GraphSAGE_hEmbed':
                model = e_graphsage_hembed.E_GraphSAGE_hEmbed(numLayers=k,
                                            dim_node_embed=h,
                                            num_edge_attr=train_data.num_features,
                                            num_classes=len(train_data.classes)
                                            )
            else:
                print('Model type not recognized.')
            f1_score[idx_k,idx_h], re_score[idx_k,idx_h] = validation(model, trainer)

    data_path = os.path.join(os.getcwd(), 'data', dataset + f'-v{version}')
    os.remove(f'{data_path}-g{g}-train{("-randomized" if randomized else "")}.pkl')
    os.remove(f'{data_path}-g{g}-val{("-randomized" if randomized else "")}.pkl')
    os.remove(f'{data_path}-g{g}-test{("-randomized" if randomized else "")}.pkl')

    data_path = os.path.join(os.getcwd(), 'hyperparam') 
    file_name_f1 = f'f1_{dataset}_v{version}_{"randomized" if randomized else ""}_{model_type}_{"multiclass" if multiclass else "binary"}_g{g}'
    file_name_re = f're_{dataset}_v{version}_{"randomized" if randomized else ""}_{model_type}_{"multiclass" if multiclass else "binary"}_g{g}'
    np.save(os.path.join(data_path, file_name_f1), f1_score)
    np.save(os.path.join(data_path, file_name_re), re_score)

    for idx_k, k in enumerate(numK):
        for idx_h, h in enumerate(dimH):
            print ("################################")
            print(f'g = {g}, K={k}, H={h} =>')
            print(f'F1-score: {f1_score[idx_k,idx_h]:.2f}, Recall: {re_score[idx_k,idx_h]:.2f}')
            print ("################################")
    return

if __name__ == "__main__":
    dataset = 'NF-BoT-IoT' #e.g., 'NF-BoT-IoT'
    version = 1
    randomized = True
    model_type = 'E_GraphSAGE' #e.g., 'E_GraphSAGE', E_GraphSAGE_hEmbed
    multiclass = True
    numEpochs = 2000
    numRealizations = 5
    numK = [2]
    dimH = [64,128]

    data_path = os.path.join(os.getcwd(), 'hyperparam') 

    if os.path.exists(data_path) == False:
        os.mkdir(data_path)
    for g in range(numRealizations):
        main(dataset, version, randomized, model_type, multiclass, numEpochs, numK, dimH, g)
    #pool = multiprocessing.Pool(processes=5)
    #pool.starmap(main, [(dataset, version, randomized, model_type, multiclass, numEpochs, numK, dimH, g) for g in range(numRealizations)])
