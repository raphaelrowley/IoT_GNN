from configuration import *

from models import e_graphsage, fnn_model, enhanced_e_graphsage
from data import IoTDataset
from train import ModelTrainer
from tester import ModelTester


def main():

    # ------------------------------------------------------------------------
    dataset_config = {
        'multiclass': True,
        'dataset': 'NF-BoT-IoT',
        'version': 1,
        'randomize_source_ip': True,
        'relabel_nodes': False,
    }

    training_config = {
        'num_epochs': 300,
        'lr': 1e-3,
        'gpu': True,
        'lr_sched_factor': np.sqrt(10),
        'lr_sched_patience': 300,  # Decrease to enable learning rate scheduling
    }

    USE_CHECKPOINTS = False
    # ------------------------------------------------------------------------

    print('\rLoading and preprocessing data…', end='')
    train_data = IoTDataset(**dataset_config)
    print("\rDone processing train.", end='')
    val_data = IoTDataset(**dataset_config, split='val')
    print("\rDone processing val.", end='')
    test_data = IoTDataset(**dataset_config, split='test')
    print("\rDone processing test.", end='')

    print('\rInitializing models…', end='')
    egs_baseline = e_graphsage.E_GraphSAGE(numLayers=2,
                                           dim_node_embed=128,
                                           num_edge_attr=train_data.num_features,
                                           num_classes=len(train_data.classes)
                                          )

    fnn_baseline = fnn_model.TestFNN(num_hidden_layers=2,
                                     hidden_layer_widths=[128, 192],  # Should be approximately comparable to EGS
                                     num_edge_attr=train_data.num_features,
                                     num_classes=len(train_data.classes),
                                    )

    egs_enhanced = enhanced_e_graphsage.Enhanced_E_GraphSAGE(numLayers=2,
                                                             dim_node_embed=128,
                                                             num_edge_attr=train_data.num_features,
                                                             num_classes=len(train_data.classes),
                                                             attention=False, gating=False, residual=False
                                                            )

    dids_model = fnn_model.TestFNN(num_hidden_layers=5,                 # DIDS inspired model
                                   hidden_layer_widths=[18, 36, 72, 144, 256],
                                   num_edge_attr=train_data.num_features,
                                   num_classes=len(train_data.classes),
                                   )

    print('\rStarting training…', end='')
    trainer = ModelTrainer(training_config, train_data, val_data)

    for model in [egs_baseline, fnn_baseline, egs_enhanced, dids_model]:
        trainer.train_model(model, use_checkpoint=USE_CHECKPOINTS)

    tester = ModelTester(test_data, False)
    for model in [egs_baseline, fnn_baseline, egs_enhanced, dids_model]:
        tester.test_model(model)


if __name__ == "__main__":
    main()