from configuration import *

from models import e_graphsage
from data import IoTDataset
from train import ModelTrainer


def main():

    multiclass = False
    print('\rLoading and preprocessing data…', end='')
    train_data = IoTDataset(version=1, multiclass=multiclass)
    val_data = IoTDataset(version=1, multiclass=multiclass, split='val')

    print('\rInitializing model…', end='')
    model = e_graphsage.E_GraphSAGE(numLayers=2,
                                    dim_node_embed=64,
                                    num_edge_attr=train_data.num_features,
                                    num_classes=len(train_data.classes)
                                    )

    training_config = {
        'num_epochs': 100,
        'lr': 1e-3,
        'gpu': False,
        'lr_sched_factor': np.sqrt(10),
        'lr_sched_patience': 10,
    }

    print('\rStarting training…', end='')
    trainer = ModelTrainer(training_config, train_data, val_data)

    trainer.train_model(model)


if __name__ == "__main__":
    main()