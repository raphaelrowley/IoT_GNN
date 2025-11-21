from configuration import *

from models import e_graphsage, fnn_model
from data import IoTDataset
from train import ModelTrainer
from tester import ModelTester


def main():
    multiclass = True
    print('\rLoading and preprocessing data…', end='')
    train_data = IoTDataset(version=1, multiclass=multiclass)
    val_data = IoTDataset(version=1, multiclass=multiclass, split='val')
    test_data = IoTDataset(version=1, multiclass=multiclass, split='test')

    print('\rInitializing model…', end='')
    model = e_graphsage.E_GraphSAGE(numLayers=2,
                                    dim_node_embed=64,
                                    num_edge_attr=train_data.num_features,
                                    num_classes=len(train_data.classes)
                                    )

    model2 = fnn_model.TestFNN(num_hidden_layers=3,
                               hidden_layer_widths=[64, 64, 128],  # Should be approximately comparable to EGS
                               num_edge_attr=train_data.num_features,
                               num_classes=len(train_data.classes),
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

    trainer.train_model(model, False)
    trainer.train_model(model2, False)

    tester = ModelTester(test_data, False)
    tester.test_model(model)


if __name__ == "__main__":
    main()