from configuration import *

from models import e_graphsage, fnn_model
from data import IoTDataset
from train import ModelTrainer
from tester import ModelTester


def main():
    dataset_config = {
        'multiclass': True,
        'dataset': 'NF-ToN-IoT',
        'version': 1,
        'randomize_source_ip': True,
        'relabel_nodes': False,
    }

    print('\rLoading and preprocessing data…', end='')
    train_data = IoTDataset(**dataset_config)
    val_data = IoTDataset(**dataset_config, split='val')
    test_data = IoTDataset(**dataset_config, split='test')

    print('\rInitializing model…', end='')
    model = e_graphsage.E_GraphSAGE(numLayers=2,
                                    dim_node_embed=128,
                                    num_edge_attr=train_data.num_features,
                                    num_classes=len(train_data.classes)
                                    )

    model2 = fnn_model.TestFNN(num_hidden_layers=3,
                               hidden_layer_widths=[128, 128, 256],  # Should be approximately comparable to EGS
                               num_edge_attr=train_data.num_features,
                               num_classes=len(train_data.classes),
                               )

    training_config = {
        'num_epochs': 5000,             # TODO Increase?
        'lr': 1e-3,
        'gpu': False,
        'lr_sched_factor': np.sqrt(10),     # TODO or set to 1 first?
        'lr_sched_patience': 100,        # TODO maybe scale appropriately
    }

    print('\rStarting training…', end='')
    trainer = ModelTrainer(training_config, train_data, val_data)

    trainer.train_model(model, True)
    trainer.train_model(model2, True)

    tester = ModelTester(test_data, False)
    tester.test_model(model)


if __name__ == "__main__":
    main()