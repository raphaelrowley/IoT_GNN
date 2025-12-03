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
    prec, recall, f1, _ = trainer.train_model(model, False)
    best_epoch = np.argmax(f1)
    print(f'Precision: {prec[best_epoch]:.2f},  \
          Recall: {recall[best_epoch]:.2f}, F1-Score: {f1[best_epoch]:.2f} at epoch {best_epoch+1}')
    return f1[best_epoch], recall[best_epoch]

def main(dataset, version, randomized, model_type, multiclass, numEpochs, numRealizations, numK, dimH):
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
    numRealizations : int
        Number of realizations for averaging results.
    numK : list of int
        List of values for the number of layers (K) to be validated.
    dimH : list of int
        List of values for the hidden dimension (H) to be validated.
    --------------
    Returns:
    f1_score : np.ndarray of dtype float and dimensions (len(numK), len(dimH), numRealizations)
        Array of F1-scores for each combination of K and H across realizations.
    re_score : np.ndarray of dtype float and dimensions (len(numK), len(dimH), numRealizations)
        Array of recall scores for each combination of K and H across realizations.
    --------------
    """
    dataset_config = {'multiclass': multiclass, 'dataset': dataset, 'randomize_source_ip': randomized, 'version': version}
    
    data_path = os.path.join(os.getcwd(), 'data', dataset + f'-v{version}')

    training_config = {
    'num_epochs': numEpochs,             
    'lr': 1e-3,
    'gpu': False,
    'lr_sched_factor': np.sqrt(10),    
    'lr_sched_patience': 100,        
    }

    f1_score = np.zeros((len(numK), len(dimH), numRealizations))
    re_score = np.zeros((len(numK), len(dimH), numRealizations))

    for g in range(numRealizations):
        print(f'Realization {g+1}/{numRealizations}')
        np.random.seed(g) #random seed for the randomization of ip addresses
        train_data = IoTDataset(**dataset_config)
        val_data = IoTDataset(**dataset_config, split='val')
        trainer = ModelTrainer(training_config, train_data, val_data)
        for idx_k, k in enumerate(numK):
            for idx_h, h in enumerate(dimH):
                print(f'Validating {model_type} with K={k}, H={h}')
                if model_type == 'E_GraphSAGE':
                    model = e_graphsage.E_GraphSAGE(numLayers=k,
                                                dim_node_embed=h,
                                                num_edge_attr=train_data.num_features,
                                                num_classes=len(train_data.classes),
                                                dropout=0.2,
                                                normalization=True
                                                )
                elif model_type == 'E_GraphSAGE_hEmbed':
                    model = e_graphsage_hembed.E_GraphSAGE_hEmbed(numLayers=k,
                                                dim_node_embed=h,
                                                num_edge_attr=train_data.num_features,
                                                num_classes=len(train_data.classes)
                                                )
                else:
                    print('Model type not recognized.')
                f1_score[idx_k,idx_h,g], re_score[idx_k,idx_h,g] = validation(model, trainer)

        os.remove(f'{data_path}-train{("-randomized" if randomized else "")}.pkl')
        os.remove(f'{data_path}-val{("-randomized" if randomized else "")}.pkl')
        os.remove(f'{data_path}-test{("-randomized" if randomized else "")}.pkl')
    for k_idx, k in enumerate(numK):
        for h_idx, h in enumerate(dimH):
            print(f'K={k}, H={h} =>')
            print(f'F1-Score: {np.mean(f1_score[idx_k,idx_h,:]):.2f} ± {np.std(f1_score[idx_k,idx_h,:]):.2f}, {np.min(f1_score[idx_k,idx_h,:]):.2f}-{np.max(f1_score[idx_k,idx_h,:]):.2f}')
            print(f'Recall:   {np.mean(re_score[idx_k,idx_h,:]):.2f} ± {np.std(re_score[idx_k,idx_h,:]):.2f}, {np.min(re_score[idx_k,idx_h,:]):.2f}-{np.max(re_score[idx_k,idx_h,:]):.2f}')
    return f1_score, re_score

if __name__ == "__main__":
    dataset = 'NF-BoT-IoT' #e.g., 'NF-BoT-IoT'
    version = 1
    randomized = True
    model_type = 'E_GraphSAGE' #e.g., 'E_GraphSAGE', E_GraphSAGE_hEmbed
    multiclass = True
    numEpochs = 200
    numRealizations = 1
    numK = [2]
    dimH = [128]
    f1_score, re_score = main(dataset, version, randomized, model_type, multiclass, numEpochs, numRealizations, numK, dimH)

    data_path = os.path.join(os.getcwd(), 'hyperparam') 
    file_name_f1 = f'f1_{dataset}_v{version}_{"randomized" if randomized else ""}_{model_type}_{"multiclass" if multiclass else "binary"}'
    file_name_re = f're_{dataset}_v{version}_{"randomized" if randomized else ""}_{model_type}_{"multiclass" if multiclass else "binary"}'
    np.save(os.path.join(data_path, file_name_f1), f1_score)
    np.save(os.path.join(data_path, file_name_re), re_score)