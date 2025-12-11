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
        'lr_sched_patience': 100,  # Decrease to scale learning rate scheduling more aggressively
    }

    USE_CHECKPOINTS = False
    # ------------------------------------------------------------------------

    print('Loading and preprocessing data…')
    train_data = IoTDataset(**dataset_config)
    print("Done processing train.")
    val_data = IoTDataset(**dataset_config, split='val')
    print("Done processing val.")
    test_data = IoTDataset(**dataset_config, split='test')
    print("Done processing test.")

    print('Initializing models…')
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

    print('Starting training…')
    trainer = ModelTrainer(training_config, train_data, val_data)

    for model in [egs_baseline, fnn_baseline, egs_enhanced, dids_model]:
        trainer.train_model(model, use_checkpoint=USE_CHECKPOINTS)

    tester = ModelTester(test_data, False)
    for model in [egs_baseline, fnn_baseline, egs_enhanced, dids_model]:
        tester.test_model(model)


if __name__ == "__main__":
    main()