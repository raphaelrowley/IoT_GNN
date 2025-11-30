from configuration import *

from models import e_graphsage, fnn_model
from data import IoTDataset
from train import ModelTrainer


def validation(model, trainer):
    prec, recall, f1, val_risk = trainer.train_model(model, True)
    best_epoch = np.argmin(val_risk)
    print(f'Precision: {prec[best_epoch]:.2f},  \
          Recall: {recall[best_epoch]:.2f}, F1-Score: {f1[best_epoch]:.2f} at epoch {best_epoch+1}')
    #print(f'Precision: {prec}, Recall: {recall}, F1-Score: {f1}')

def main():
    dataset_config = {'multiclass': True, 'dataset': 'NF-BoT-IoT', 'randomize_source_ip': True}

    train_data = IoTDataset(**dataset_config)
    val_data = IoTDataset(**dataset_config, split='val')
    
    ## Validation of E_GraphSAGE
    training_config = {
    'num_epochs': 200,             
    'lr': 1e-3,
    'gpu': False,
    'lr_sched_factor': np.sqrt(10),    
    'lr_sched_patience': 100,        
    }

    trainer = ModelTrainer(training_config, train_data, val_data)

    numK = [2,3,4]
    dimH = [8]
    for (k,h) in [(k,h) for k in numK for h in dimH]:
        print(f'Validating E_GraphSAGE with K={k}, H={h}')
        model_EGS = e_graphsage.E_GraphSAGE(numLayers=k,
                                    dim_node_embed=h,
                                    num_edge_attr=train_data.num_features,
                                    num_classes=len(train_data.classes)
                                    )
    
        validation(model_EGS, trainer)
if __name__ == "__main__":
    main()