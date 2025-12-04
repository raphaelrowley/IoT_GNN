from configuration import *


class ModelTrainer:
    """
    Train neural network models on IoT network-flow data.

    The ``ModelTrainer`` encapsulates the full training loop, including optimizer
    and scheduler setup, checkpointing, metric tracking, and plotting of training
    and validation behavior. It operates on full-batch DGL graphs provided by
    the dataset.

    Attributes
    ----------
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate used by the optimizer.
    lr_sched_factor : float
        Factor used in the ReduceLROnPlateau scheduler (via ``1 / lr_sched_factor``).
    lr_sched_patience : int
        Number of epochs without improvement before the learning rate is reduced.
    train_data : torch.utils.data.Dataset
        Training dataset instance.
    val_data : torch.utils.data.Dataset
        Validation dataset instance.
    use_gpu : bool
        Whether training is performed on a CUDA device.
    device : torch.device
        Device on which the model and graphs are placed.
    loss_fn : torch.nn.Module
        Loss function used for optimization (``CrossEntropyLoss`` or
        ``BCEWithLogitsLoss``).
    optimizer : torch.optim.Optimizer or None
        Optimizer used during training, initialized in :meth:`train_model`.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler, initialized in :meth:`train_model`.
    checkpoint_base_path : str
        Base directory used for storing checkpoint files.
    checkpoint_path : str or None
        Full path to the current checkpoint file.

    Notes
    -----
    This docstring was created with assistance from ChatGPT.
    """

    def __init__(self, training_config, train_data, val_data, g=None):
        """
        Initializes the model trainer with configuration parameters, datasets,
        loss function, and device settings.

        Parameters
        ----------
        training_config : dict
            Dictionary containing training hyperparameters. Expected keys:

            - ``num_epochs`` : int
              Number of training epochs.
            - ``lr`` : float
              Learning rate used for the Adam optimizer.
            - ``lr_sched_factor`` : float
              Multiplicative factor used as 1/lr_sched_factor in ReduceLROnPlateau.
            - ``lr_sched_patience`` : int
              Number of epochs with no improvement before LR is reduced.
            - ``gpu`` : bool
              If True, training is executed on a CUDA device.
        train_data : torch.utils.data.Dataset
            Training dataset object. Expected attributes:

            - ``graph`` : DGLGraph containing edge features.
            - ``classes`` : array-like of class indices.
            - ``class_weights`` : array-like of class weights.
            - ``id`` : string identifier for checkpoint naming.

        val_data : torch.utils.data.Dataset
            Validation dataset object with the same structure and attributes
            as ``train_data``.
        g : int or None, optional
            Random seed for reproducibility.
        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """

        self.num_epochs = training_config['num_epochs']
        self.lr = training_config['lr']

        self.lr_sched_factor = training_config['lr_sched_factor']
        self.lr_sched_patience = training_config['lr_sched_patience']

        self.train_data = train_data
        self.val_data = val_data

        if training_config['gpu']:
            self.use_gpu = True
            self.device = torch.device('cuda')

            self.train_data.graph = self.train_data.graph.to(self.device)
            self.val_data.graph = self.val_data.graph.to(self.device)
        else:
            self.use_gpu = False
            self.device = torch.device('cpu')

            self.train_data.graph = self.train_data.graph.to(self.device)
            self.val_data.graph = self.val_data.graph.to(self.device)

        if len(train_data.classes) > 2:
            self.loss_fn = nn.CrossEntropyLoss(
                weight = torch.tensor(train_data.class_weights, dtype=torch.float32, device=self.device)
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(train_data.class_weights[1] / train_data.class_weights[1], dtype=torch.float32, device=self.device)
            )

        self.optimizer = None
        self.lr_scheduler = None

        self.checkpoint_base_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(self.checkpoint_base_path, exist_ok=True)
        self.checkpoint_base_path = os.path.join(self.checkpoint_base_path, train_data.id + ("" if g is None else f"-g{g}-"))

        self.checkpoint_path = None

    def train_model(self, model, use_checkpoint):
        """
        Train the model for a fixed number of epochs, optionally resuming from a
        saved checkpoint.

        Parameters
        ----------
        model : nn.Module
            Model instance to be trained.
        use_checkpoint : bool
            If True and a checkpoint exists, resume training using the stored epoch,
            optimizer state, and recorded metrics.

        Returns
        -------
        prec : list of float
            Precision scores per epoch. For multiclass: weighted average. For binary:
            positive-class precision.
        recall : list of float
            Recall scores per epoch. Same class-handling as ``prec``.
        f1 : list of float
            F1 scores per epoch. Same class-handling as ``prec``.
        val_risk : list of float
            Validation risk values per epoch.

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """

        if self.use_gpu:
            model = model.to(self.device)
        else:
            model = model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=1/self.lr_sched_factor,
            patience=self.lr_sched_patience
        )

        self.checkpoint_path = self.checkpoint_base_path + model.id + '.pt'
        if os.path.isfile(self.checkpoint_path) and use_checkpoint:
            epoch, train_risk, val_risk, progress_reports = self.load_checkpoint(model)
        else:
            epoch = 0

            train_risk = []
            val_risk = []
            if len(self.val_data.classes) > 2:
                # multiclass_classification
                progress_reports = {}
                for cls in self.val_data.encoder.inverse_transform(self.val_data.classes):
                    progress_reports[cls] = {}
                    progress_reports[cls]['precision'] = []
                    progress_reports[cls]['recall'] = []
                    progress_reports[cls]['f1-score'] = []
                progress_reports['weighted avg'] = {}
                progress_reports['weighted avg']['precision'] = []
                progress_reports['weighted avg']['recall'] = []
                progress_reports['weighted avg']['f1-score'] = []
            else:
                progress_reports = {}
                progress_reports['precision'] = []
                progress_reports['recall'] = []
                progress_reports['f1-score'] = []

        with trange(epoch, self.num_epochs, initial=epoch, total=self.num_epochs,
                    desc=f'Training {model.id}', unit='epoch') as pbar:
            for epoch in pbar:
                model.train()

                # We are doing full-batch training, no need for iterators or dataloaders :)
                train_graph = copy.deepcopy(self.train_data.__getitem__(0))
                del train_graph.edata['edge_label']   # prevent target leakage
                train_graph = train_graph.to(self.device)

                model.forward(train_graph)
                logits = train_graph.edata['edge_pred']

                target = self.train_data.__getitem__(0).edata['edge_label'].to(self.device)
                loss = self.loss_fn(logits, target)
                # Zero gradients before the backward step too
                self.optimizer.zero_grad(set_to_none=True)  # Recommended in Torch Performance Tuning Guide
                loss.backward()

                del train_graph     # Maybe unnecessary, but free up memory ASAP for large datasets

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)  # Recommended in Torch Performance Tuning Guide

                model.eval()
                with torch.no_grad():
                    val_graph = copy.deepcopy(self.val_data.__getitem__(0))
                    del val_graph.edata['edge_label']  # prevent target leakage
                    val_graph = val_graph.to(self.device)
                    model.forward(val_graph)
                    logits = val_graph.edata['edge_pred']
                    if ((epoch % 25) == 0):
                        print(f"[Epoch {epoch}] logit mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")

                    del val_graph      # Maybe unnecessary, but free up memory ASAP for large datasets

                    # Compute validation loss (using the same weights as in training data set;
                    # negligible difference due to stratified split)
                    target = self.val_data.__getitem__(0).edata['edge_label'].to(self.device)
                    val_loss = self.loss_fn(logits, target)
                    val_risk.append(val_loss.item())

                    self.lr_scheduler.step(val_loss)

                    if len(self.val_data.classes) > 2:
                        # Need to compute cls_report on the CPU.
                        y_pred_cpu = torch.argmax(logits, dim=-1)
                        target_cpu = target
                        if self.use_gpu:
                            y_pred_cpu = torch.argmax(logits, dim=-1).detach().cpu()
                            target_cpu = target.detach().cpu()
                        cls_report = sk.metrics.classification_report(y_true=target_cpu, y_pred=y_pred_cpu,
                                                                      labels=self.val_data.classes,
                                                                      target_names=self.val_data.encoder.inverse_transform(self.val_data.classes),
                                                                      output_dict=True, zero_division=0.0)
                        for key in progress_reports.keys():
                            progress_reports[key]['precision'].append(cls_report[key]['precision'])
                            progress_reports[key]['recall'].append(cls_report[key]['recall'])
                            progress_reports[key]['f1-score'].append(cls_report[key]['f1-score'])
                    else:
                        # Need to compute cls_report on the CPU.
                        y_pred_cpu = 0.5 * (1+torch.sgn(logits))
                        target_cpu = target
                        if self.use_gpu:
                            y_pred_cpu = 0.5 * (1+torch.sgn(logits)).detach().cpu()
                            target_cpu = target.detach().cpu()
                        cls_report = sk.metrics.classification_report(y_true=target_cpu, y_pred=y_pred_cpu, output_dict=True, zero_division=0.0)
                        progress_reports['precision'].append(cls_report['1.0']['precision'])
                        progress_reports['recall'].append(cls_report['1.0']['recall'])
                        progress_reports['f1-score'].append(cls_report['1.0']['f1-score'])


                train_risk.append(loss.item())
                f1 = progress_reports['weighted avg']['f1-score'][-1] if len(self.val_data.classes) > 2 else progress_reports['f1-score'][-1]
                pbar.set_postfix({
                    "train loss": f'{train_risk[-1]:.4f}', #f"{loss:.4f}",
                    "validation loss": f'{val_risk[-1]:.4f}',
                    "learning rate": f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    "F1 score": f'{f1:.5f}'
                })
                self.set_checkpoint(epoch, model, train_risk, val_risk, progress_reports)

        # TODO: do a small print of the checkpoint path.
        plt.figure('Risk', figsize=(7, 5))
        plt.plot([i + 1 for i in range(self.num_epochs)], train_risk, label='train')
        plt.plot([i + 1 for i in range(self.num_epochs)], val_risk, label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Risk')
        plt.yscale('log')
        plt.savefig(self.checkpoint_path.replace('.pt', '_risk.png'), dpi=300)
        plt.close()

        plt.figure('Classification Results')
        prec = progress_reports['weighted avg']['precision'] if len(self.val_data.classes) > 2 else progress_reports['precision']
        recall = progress_reports['weighted avg']['recall'] if len(self.val_data.classes) > 2 else progress_reports['recall']
        f1 = progress_reports['weighted avg']['f1-score'] if len(self.val_data.classes) > 2 else progress_reports['f1-score']
        plt.plot([i + 1 for i in range(self.num_epochs)], prec, label='Precision')
        plt.plot([i + 1 for i in range(self.num_epochs)], recall, label='Recall')
        plt.plot([i + 1 for i in range(self.num_epochs)], f1, label='F1 Score')
        plt.legend()
        plt.xlabel('Epoch')
        plt.savefig(self.checkpoint_path.replace('.pt', '_classification.png'), dpi=300)
        plt.close()

        return prec, recall, f1, val_risk

    def load_checkpoint(self, model) -> tuple[int, list[float], list[float], dict]:
        """
        Loads training progress and model weights from a given checkpoint.

        Parameters
        ----------
        model : nn.Module
            Model instance into which the checkpoint weights are loaded.

        Returns
        -------
        epoch : int
            Epoch saved in the checkpoint.
        train_risk : list of float
            List of training risk scores.
        val_risk : list of float
            List of validation risk scores.
        progress_reports : dict

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """

        (epoch, train_risk, val_risk, progress_reports,
         rng_state, _ ,optim_sd, lr_sched_sd) = torch.load(self.checkpoint_path, weights_only=False)

        torch.set_rng_state(rng_state)

        self.lr_scheduler.load_state_dict(lr_sched_sd)
        self.optimizer.load_state_dict(optim_sd)

        # load the model state dict separately and map it to the correct device
        _, _, _, _, _, model_sd, _, _ = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
        model.load_state_dict(model_sd)

        epoch += 1

        return epoch, train_risk, val_risk, progress_reports


    def set_checkpoint(self, epoch, model, train_risk, val_risk, progress_reports):
        """
        Save a training checkpoint containing model state, optimizer state,
        scheduler state, and recorded training progress.

        Parameters
        ----------
        epoch : int
            Epoch index to save in the checkpoint.
        model : nn.Module
            Model whose parameters are stored via ``state_dict()``.
        train_risk : list of float
            Training loss values recorded so far.
        val_risk : list of float
            Validation loss values recorded so far.
        progress_reports : dict
            Per-epoch classification metrics such as precision, recall, and F1 score.

        Returns
        -------
        None
            This function returns ``None``. The checkpoint is written to
            ``self.checkpoint_path``.

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """
        torch.save([epoch, train_risk, val_risk, progress_reports,
                    torch.get_rng_state(), model.state_dict(), self.optimizer.state_dict(),
                    self.lr_scheduler.state_dict()],
                   self.checkpoint_path)
        return

