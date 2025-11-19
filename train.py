import copy

from configuration import *
from tqdm import trange


class ModelTrainer:

    def __init__(self, training_config, train_data, test_data):

        self.num_epochs = training_config['num_epochs']
        self.lr = training_config['lr']

        self.lr_sched_factor = training_config['lr_sched_factor']
        self.lr_sched_patience = training_config['lr_sched_patience']

        self.train_data = train_data
        self.test_data = test_data

        if training_config['gpu']:
            self.use_gpu = True
            self.device = torch.device('cuda')

            self.train_data.graph = self.train_data.graph.to(self.device)
            self.test_data.graph = self.test_data.graph.to(self.device)
        else:
            self.use_gpu = False
            self.device = torch.device('cpu')

        if len(train_data.classes) > 2:
            self.loss_fn = nn.CrossEntropyLoss(
                weight = torch.tensor(train_data.class_weights, dtype=torch.float32)
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(train_data.class_weights[1] / train_data.class_weights[1], dtype=torch.float32)
            )

        self.optimizer = None
        self.lr_scheduler = None

        self.checkpoint_base_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(self.checkpoint_base_path, exist_ok=True)
        self.checkpoint_base_path = os.path.join(self.checkpoint_base_path, train_data.id)

        self.checkpoint_path = None


    def train_model(self, model):
        if self.use_gpu:
            model = model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=1/self.lr_sched_factor,
            patience=self.lr_sched_patience
        )

        self.checkpoint_path = self.checkpoint_base_path + model.id + '.pt'
        if os.path.isfile(self.checkpoint_path):
            epoch, train_risk, test_risk, test_accuracy = self.load_checkpoint(model)
        else:
            epoch = 0

            train_risk = []
            test_risk = []
            test_accuracy = []

        with trange(epoch, self.num_epochs, initial=epoch, total=self.num_epochs,
                    desc=f'Training {model.id}', unit='epoch') as pbar:
            for epoch in pbar:
                model.train()

                # We are doing full-batch training, no need for iterators or dataloaders :)
                train_graph = copy.deepcopy(self.train_data.__getitem__(0))
                del train_graph.edata['edge_label']   # prevent target leakage

                model.forward(train_graph)
                logits = train_graph.edata['edge_pred']

                target = self.train_data.__getitem__(0).edata['edge_label']
                loss = self.loss_fn(logits, target)
                loss.backward()

                del train_graph     # Maybe unnecessary, but free up memory ASAP for large datasets

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)  # Recommended in Torch Performance Tuning Guide

                model.eval()
                with torch.no_grad():
                    test_graph = copy.deepcopy(self.test_data.__getitem__(0))
                    del test_graph.edata['edge_label']  # prevent target leakage
                    model.forward(test_graph)
                    logits = test_graph.edata['edge_pred']

                    del test_graph      # Maybe unnecessary, but free up memory ASAP for large datasets

                    # Compute test loss (using the same weights as in training data set;
                    # negligible difference due to stratified split)
                    target = self.test_data.__getitem__(0).edata['edge_label']
                    test_loss = self.loss_fn(logits, target)
                    test_risk.append(test_loss)

                    self.lr_scheduler.step(test_loss)

                    # TODO Define a good accuracy score to use in testing, maybe a class-weighted F1?

                train_risk.append(loss.item())
                pbar.set_postfix({
                    "train loss": f'{train_risk[-1]:.4f}', #f"{loss:.4f}",
                    "test loss": f'{test_risk[-1]:.4f}',
                    "learning rate": f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                })
                self.set_checkpoint(epoch, model, train_risk, test_risk, test_accuracy)

        plt.plot([i + 1 for i in range(self.num_epochs)], train_risk, label='train')
        plt.plot([i + 1 for i in range(self.num_epochs)], test_risk, label='test')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Risk')
        plt.show()


    def load_checkpoint(self, model):
        (epoch, train_risk, test_risk, test_accuracy,
         rng_state, _ ,optim_sd, lr_sched_sd) = torch.load(self.checkpoint_path, weights_only=False)

        torch.set_rng_state(rng_state)

        self.lr_scheduler.load_state_dict(lr_sched_sd)
        self.optimizer.load_state_dict(optim_sd)

        # load the model state dict separately and map it to the correct device
        _, _, _, _, _, model_sd, _, _ = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
        model.load_state_dict(model_sd)

        epoch += 1

        return epoch, train_risk, test_risk, test_accuracy


    def set_checkpoint(self, epoch, model, train_risk, test_risk, test_accuracy):
        torch.save([epoch, train_risk, test_risk, test_accuracy,
                    torch.get_rng_state(), model.state_dict(), self.optimizer.state_dict(),
                    self.lr_scheduler.state_dict()],
                   self.checkpoint_path)
        return


def test():
    from models import e_graphsage
    from data import IoTDataset

    multiclass = True
    print('\rLoading and preprocessing data…', end='')
    train_data = IoTDataset(version=1, multiclass=multiclass)
    test_data = IoTDataset(version=1, multiclass=multiclass, split='test')

    print('\rInitializing model…', end='')
    model = e_graphsage.E_GraphSAGE(numLayers=2,
                                    dim_node_embed=64,
                                    num_edge_attr=train_data.num_features,
                                    num_classes=len(train_data.classes)
                                    )

    training_config = {
        'num_epochs': 200,
        'lr': 1e-3,
        'gpu': False,
        'lr_sched_factor': np.sqrt(10),
        'lr_sched_patience': 10,
    }

    print('\rStarting training…', end='')
    trainer = ModelTrainer(training_config, train_data, test_data)

    trainer.train_model(model)


if __name__ == "__main__":
    test()


