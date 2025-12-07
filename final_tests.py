from configuration import *

from models import e_graphsage, fnn_model, e_graphsage_hembed, enhanced_e_graphsage
from data import IoTDataset
from train import ModelTrainer
from tester import ModelTester

# Conda env (iot_env) has to be updated in the following manner to run this script:
# conda install conda-forge::torchinfo
from torchinfo import summary


def main():
    # Appropriate dataset csv should be in the data directory.
    dataset_config = {
        'multiclass': True,
        'dataset': 'NF-BoT-IoT',
        'version': 1,
        'randomize_source_ip': True,
        'relabel_nodes': False,
    }

    print('\rLoading and preprocessing data…', end='')
    train_data = IoTDataset(**dataset_config)
    print("\rDone processing train.", end='')
    val_data = IoTDataset(**dataset_config, split='val')
    print("\rDone processing val.", end='')
    test_data = IoTDataset(**dataset_config, split='test')
    print("\rDone processing test.", end='')

    print('\rInitializing models…', end='')

    print(f"Number of edge attributes: {train_data.num_features}")

    model = e_graphsage.E_GraphSAGE(numLayers=2,
                                    dim_node_embed=128,
                                    num_edge_attr=train_data.num_features,
                                    num_classes=len(train_data.classes)
                                    )

    model2 = fnn_model.TestFNN(num_hidden_layers=2,
                               hidden_layer_widths=[128, 192],  # Should be approximately comparable to EGS
                               num_edge_attr=train_data.num_features,
                               num_classes=len(train_data.classes),
                               )

    # DIDS inspired model.
    model11 = fnn_model.TestFNN(num_hidden_layers=5,
                               hidden_layer_widths=[18, 36, 72, 144, 256],
                               num_edge_attr=train_data.num_features,
                               num_classes=len(train_data.classes),
                               )

    model3 = enhanced_e_graphsage.Enhanced_E_GraphSAGE(numLayers=2,
                                                       dim_node_embed=128,
                                                       num_edge_attr=train_data.num_features,
                                                       num_classes=len(train_data.classes),
                                                       )
    model4 = enhanced_e_graphsage.Enhanced_E_GraphSAGE(numLayers=2,
                                                       dim_node_embed=128,
                                                       num_edge_attr=train_data.num_features,
                                                       num_classes=len(train_data.classes),
                                                       attention=False, gating=True, residual=True
                                                       )
    model5 = enhanced_e_graphsage.Enhanced_E_GraphSAGE(numLayers=2,
                                                       dim_node_embed=128,
                                                       num_edge_attr=train_data.num_features,
                                                       num_classes=len(train_data.classes),
                                                       attention=True, gating=False, residual=True
                                                       )
    model6 = enhanced_e_graphsage.Enhanced_E_GraphSAGE(numLayers=2,
                                                       dim_node_embed=128,
                                                       num_edge_attr=train_data.num_features,
                                                       num_classes=len(train_data.classes),
                                                       attention=True, gating=True, residual=False
                                                       )
    model7 = enhanced_e_graphsage.Enhanced_E_GraphSAGE(numLayers=2,
                                                       dim_node_embed=128,
                                                       num_edge_attr=train_data.num_features,
                                                       num_classes=len(train_data.classes),
                                                       attention=True, gating=False, residual=False
                                                       )
    model8 = enhanced_e_graphsage.Enhanced_E_GraphSAGE(numLayers=2,
                                                       dim_node_embed=128,
                                                       num_edge_attr=train_data.num_features,
                                                       num_classes=len(train_data.classes),
                                                       attention=False, gating=True, residual=False
                                                       )
    model9 = enhanced_e_graphsage.Enhanced_E_GraphSAGE(numLayers=2,
                                                       dim_node_embed=128,
                                                       num_edge_attr=train_data.num_features,
                                                       num_classes=len(train_data.classes),
                                                       attention=False, gating=False, residual=True
                                                       )
    model10 = enhanced_e_graphsage.Enhanced_E_GraphSAGE(numLayers=2,
                                                       dim_node_embed=128,
                                                       num_edge_attr=train_data.num_features,
                                                       num_classes=len(train_data.classes),
                                                       attention=False, gating=False, residual=False
                                                       )

    print('\r' + ' ' * 50 + '\r', end='')
    for model_t in [model, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11]:
        print(f'Number of learnable parameters in {model_t.id}: {sum(p.numel() for p in model_t.parameters() if p.requires_grad)}')
        summary(model_t)

    training_config = {
        'num_epochs': 300,
        'lr': 1e-3,
        'gpu': True,
        'lr_sched_factor': np.sqrt(10),
        'lr_sched_patience': 100,
    }

    print('\rStarting training…', end='')
    trainer = ModelTrainer(training_config, train_data, val_data)
    print("Model 01")
    trainer.train_model(model, False)
    print("Model 02")
    trainer.train_model(model2, False)
    print("Model 03")
    trainer.train_model(model3, False)
    print("Model 04")
    trainer.train_model(model4, False)
    print("Model 05")
    trainer.train_model(model5, False)
    print("Model 06")
    trainer.train_model(model6, False)
    print("Model 07")
    trainer.train_model(model7, False)
    print("Model 08")
    trainer.train_model(model8, False)
    print("Model 09")    
    trainer.train_model(model9, False)
    print("Model 10")
    trainer.train_model(model10, False)
    print("Model 11")
    trainer.train_model(model11, False)

    tester = ModelTester(test_data, True)
    print('\rStarting testing…', end='')
    print("Model 01")
    tester.test_model(model)
    print("Model 02")
    tester.test_model(model2)
    print("Model 03")
    tester.test_model(model3)
    print("Model 04")
    tester.test_model(model4)
    print("Model 05")
    tester.test_model(model5)
    print("Model 06")
    tester.test_model(model6)
    print("Model 07")
    tester.test_model(model7)
    print("Model 08")
    tester.test_model(model8)
    print("Model 09")
    tester.test_model(model9)
    print("Model 10")
    tester.test_model(model10)
    print("Model 11")
    tester.test_model(model11)


if __name__ == "__main__":
    main()