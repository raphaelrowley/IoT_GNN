from configuration import *
import stats_util

class ModelTester:
    """
    Test neural network models on IoT network-flow data.

    The `ModelTester` encapsulates the end-to-end evaluation of a trained
    intrusion-detection model. It loads the test graph, runs a
    forward pass, computes multiclass or binary metrics, generates
    confusion matrices, and writes detailed reports to a file and in the console.

    Attributes
    ----------
    test_data : IoTDataset
        The dataset object containing the DGL test graph, class labels,
        encoders, and class weights.
    
    use_gpu : bool
        Whether inference is performed on a CUDA-enabled GPU.
    
    device : torch.device
        The PyTorch device (`'cuda'` or `'cpu'`) on which the model and
        DGL graph are placed during testing.
    
    loss_fn : torch.nn.Module
        The loss function used for evaluation. This is a
        `CrossEntropyLoss` for multiclass classification or
        `BCEWithLogitsLoss` for binary classification.
    
    checkpoint_base_path : str
        Directory where all test reports and confusion matrices will be
        written.
    
    checkpoint_path : str or None
        Full path prefix for the currently tested model (including the
        model ID). Used to name the output report and PNG files. Set
        inside `test_model()`.

    Notes
        ------
        This docstring was created with assistance from ChatGPT.
    """
    def __init__(self, test_data, use_gpu):
        """
        Initialize the ModelTester.

        This sets up the evaluation environment by configuring the device
        (CPU or GPU), transferring the DGL test graph to that device,
        constructing the appropriate loss function (binary or multiclass),
        and preparing the directory structure for saving reports and visual
        artifacts.

        Parameters
        ----------
        test_data : IoTDataset
            Dataset object containing the DGL test graph, class list,
            encoder, and class weights used for loss balancing.

        use_gpu : bool
            If True, computations are performed on the first available CUDA
            device. Otherwise, computation runs on the CPU.

        Notes
        -----
            The test graph is immediately moved to the selected device so that
            the model can operate on it without further transfers.

            This docstring was created with assistance from ChatGPT.
        """
        self.test_data = test_data

        if use_gpu:
            self.use_gpu = True
            self.device = torch.device('cuda')

            self.test_data.graph = self.test_data.graph.to(self.device)
        else:
            self.use_gpu = False
            self.device = torch.device('cpu')

            self.test_data.graph = self.test_data.graph.to(self.device)

        # Multiclass or traditional binary classification
        if len(test_data.classes) > 2:
            self.loss_fn = nn.CrossEntropyLoss(
                weight = torch.tensor(test_data.class_weights, dtype=torch.float32, device=self.device)
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(test_data.class_weights[1] / test_data.class_weights[1], dtype=torch.float32, device=self.device)
            )

        # To print out results
        self.checkpoint_base_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(self.checkpoint_base_path, exist_ok=True)
        self.checkpoint_base_path = os.path.join(self.checkpoint_base_path, self.test_data.id)

        self.checkpoint_path = None

    def test_model(self, model):
        """
        Evaluate a trained model on the test dataset.

        This method performs a full evaluation pass on the test DGL graph.
        It executes the model forward pass, computes the loss, obtains
        predictions, and evaluates the classifier using a wide range of
        metrics. These include:

        - Precision, recall, and F1-score (per class and weighted).
        - Overall accuracy and balanced accuracy.
        - Confusion matrix (raw and normalized).
        - Derived binary metrics even for multiclass classification
            (treating "Benign" as the negative class and all attack classes as a combined positive class).

        A detailed test report is printed to the console and written to a
        text file. A normalized confusion matrix is also saved as a PNG
        image.

        Parameters
        ----------
        model : torch.nn.Module
            The trained model to evaluate. Must implement a forward pass
            that populates `edge_pred` in the test graph after execution.

        Returns
        -------
        None
            The method performs evaluation, prints results, and writes output
            files, but does not return a value.

        Outputs
        -------
        - A text report containing:
            * Class-wise precision, recall, F1-score.
            * Global accuracy and balanced accuracy.
            * Binary metrics derived from the multiclass confusion matrix.
            * Confusion matrix (printed), unnormalized.

        - A PNG image saved at:
            ``{checkpoint_path}_confusion_matrix.png``

        Side Effects
        ------------
        - Moves the model to the appropriate device.
        - Writes a text report to disk.
        - Writes a confusion matrix image to disk.

        Notes
        -----
        - For GPU execution, predictions and labels are detached and moved
        to CPU before computing scikit-learn metrics.
        - For multiclass evaluation, the first class in the dataset is
        treated as the negative ("Benign") class when deriving binary
        metrics.
        - This method expects the model to modify the graph in-place by
        setting `test_graph.edata['edge_pred']` during the forward pass.

            This docstring was created with assistance from ChatGPT.
        """
        if self.use_gpu:
            model = model.to(self.device)
        else:
            model = model.to(self.device)

        self.checkpoint_path = self.checkpoint_base_path + model.id
        
        # To keep consistency with train.py, we use same progress_reports dict as structure.
        test_risk = []
        progress_reports = {}
        if len(self.test_data.classes) > 2:
            # multiclass_classification
            for cls in self.test_data.encoder.inverse_transform(self.test_data.classes):
                progress_reports[cls] = {}
                progress_reports[cls]['precision'] = []
                progress_reports[cls]['recall'] = []
                progress_reports[cls]['f1-score'] = []
            progress_reports['weighted avg'] = {}
            progress_reports['weighted avg']['precision'] = []
            progress_reports['weighted avg']['recall'] = []
            progress_reports['weighted avg']['f1-score'] = []
        else:
            for cls in self.test_data.classes:
                progress_reports[cls] = {}
                progress_reports[cls]['precision'] = []
                progress_reports[cls]['recall'] = []
                progress_reports[cls]['f1-score'] = []
            progress_reports['accuracy'] = []
            progress_reports['far'] = []
            progress_reports['weighted avg'] = {}
            progress_reports['weighted avg']['precision'] = []
            progress_reports['weighted avg']['recall'] = []
            progress_reports['weighted avg']['f1-score'] = []

        
        # Load the test graph
        test_graph = copy.deepcopy(self.test_data.__getitem__(0))
        target = test_graph.edata['edge_label']
        del test_graph.edata['edge_label']   # prevent target leakage

        model.eval()

        with torch.no_grad():
            model.forward(test_graph)
            logits = test_graph.edata['edge_pred']
            
            del test_graph      # Maybe unnecessary, but free up memory ASAP for large datasets

            test_loss = self.loss_fn(logits, target)
            test_risk.append(test_loss)

            # Compute results
            if len(self.test_data.classes) > 2:
                # Need to compute cls_report on the CPU.
                y_pred_cpu = torch.argmax(logits, dim=-1)
                target_cpu = target
                if self.use_gpu:
                    y_pred_cpu = torch.argmax(logits, dim=-1).detach().cpu()
                    target_cpu = target.detach().cpu()
                cls_report = sk.metrics.classification_report(y_true=target_cpu, y_pred=y_pred_cpu,
                                                              labels=self.test_data.classes,
                                                              target_names=self.test_data.encoder.inverse_transform(self.test_data.classes),
                                                              output_dict=True, zero_division=0.0)
                # Accuracy does not make much sense for imbalanced datasets like ours.
                acc_report = sk.metrics.accuracy_score(y_true=target_cpu, y_pred=y_pred_cpu,
                                                       normalize=True)
                bal_acc_report = sk.metrics.balanced_accuracy_score(y_true=target_cpu, y_pred=y_pred_cpu)
                # FAR for multiclass does not make sense so we will not compute it.
                
                # Confusion Matrix.
                conf_matrix = sklearn.metrics.confusion_matrix(y_true=target_cpu, y_pred=y_pred_cpu, labels=self.test_data.classes)

                # #########################
                # Extract TP, FP, FN, TN
                # This allows us to compute the equivalent binary classification metrics from our multiclass classifier.
                # #########################
                # True benign and predicted benign
                num_classes = self.test_data.classes
                tn = 0
                fn = 0
                fp = 0
                tp = 0
                # Iterate over rows
                for i in range(len(conf_matrix)):
                    # True Label = Benign
                    if i == 0:
                        tn = conf_matrix[i][i]
                        # Calculate FP (sum of the other elements of row 1)
                        for j in range(len(conf_matrix[i])-1):
                            fp += conf_matrix[i][j+1]
                    # True Label = An Attack Class
                    else:
                        # Rest of first column is false negative
                        fn += conf_matrix[i][0]
                        # Rest of the row is part of TP
                        for j in range(len(conf_matrix[i])-1):
                            tp += conf_matrix[i][j+1]

                total_samples = tn + tp + fn + fp

                # Binary classification metrics.
                bin_acc = stats_util.get_accuracy(tn, tp, fn, fp)
                bin_bal_acc = stats_util.get_balanced_accuracy(tn, tp, fn, fp)
                bin_far = stats_util.get_far(tn, tp, fn, fp)
                bin_recall = stats_util.get_recall(tn, tp, fn, fp)
                bin_precision = stats_util.get_precision(tn, tp, fn, fp)
                bin_f1 = stats_util.get_f1_score(tn, tp, fn, fp)

                # Update progress report
                for key in progress_reports.keys():
                    progress_reports[key]['precision'].append(cls_report[key]['precision'])
                    progress_reports[key]['recall'].append(cls_report[key]['recall'])
                    progress_reports[key]['f1-score'].append(cls_report[key]['f1-score'])

                # Reporting / Display Section
                # Create a string with all of the results that can be written to file and console.
                report_text = []
                report_text.append("===== Multiclass Test Report =====\n")
                for key in progress_reports.keys():
                    report_text.append(f"== {key} ==\n")
                    report_text.append(f"Precision: {progress_reports[key]['precision'][0]:.6f}\n")
                    report_text.append(f"Recall:    {progress_reports[key]['recall'][0]:.6f}\n")
                    report_text.append(f"F1-score:  {progress_reports[key]['f1-score'][0]:.6f}\n\n")

                report_text.append(f"= Global Metrics =\n")
                report_text.append(f"Accuracy:           {acc_report:.6f}\n")
                report_text.append(f"Balanced Accuracy:  {bal_acc_report:.6f}\n\n")

                # Confusion matrix
                report_text.append("Confusion Matrix:\n")
                report_text.append(str(conf_matrix) + "\n\n")

                report_text.append(f"= Binary Metrics =\n")
                report_text.append(f"TN, TP, FN, FP are derived from the confusion matrix.\n")
                report_text.append(f"Benign is the negative class. The attack classes are positive.\n")
                print("Total Samples:", total_samples)
                report_text.append(f"Total Samples: {total_samples}\n")
                report_text.append(f"TN: {tn}\n")
                report_text.append(f"TP: {tp}\n")
                report_text.append(f"FN: {fn}\n")
                report_text.append(f"FP: {fp}\n\n")

                report_text.append(f"Accuracy: {bin_acc:.6f}\n")
                report_text.append(f"Balanced Accuracy: {bin_bal_acc:.6f}\n")
                report_text.append(f"FAR: {bin_far:.6f}\n")
                report_text.append(f"Recall: {bin_recall:.6f}\n")
                report_text.append(f"Precision: {bin_precision:.6f}\n")
                report_text.append(f"F1-score: {bin_f1:.6f}\n")

                # Convest report to text, print it and write to file.
                report_text = "".join(report_text)
                
                print(report_text)

                report_path = self.checkpoint_path + "_test_report.txt"
                with open(report_path, "w") as f:
                    f.write(report_text)
                
                sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
                    y_true=target_cpu, 
                    y_pred=y_pred_cpu, 
                    labels=self.test_data.classes, 
                    display_labels=self.test_data.encoder.inverse_transform(self.test_data.classes), 
                    normalize='true'
                )

                disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
                    y_true=target_cpu,
                    y_pred=y_pred_cpu,
                    labels=self.test_data.classes,
                    display_labels=self.test_data.encoder.inverse_transform(self.test_data.classes),
                    normalize='true',
                )
                
                # Save Confusion Matrix at same place as the checkpoints
                plt.savefig(self.checkpoint_path + "_confusion_matrix.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print("Report and confusion matrix recorded in: ", self.checkpoint_path)

            else:
                # Need to compute cls_report on the CPU.
                y_pred_cpu = 0.5 * (1+torch.sgn(logits))
                target_cpu = target
                if self.use_gpu:
                    y_pred_cpu = (0.5 * (1+torch.sgn(logits))).detach().cpu()
                    target_cpu = target.detach().cpu()
                cls_report = sk.metrics.classification_report(y_true=target_cpu, y_pred=y_pred_cpu, output_dict=True, zero_division=0.0)
                # Accuracy is still misleading given our imbalanced dataset.
                acc_report = sk.metrics.accuracy_score(y_true=target_cpu, y_pred=y_pred_cpu,
                                                       normalize=True)
                bal_acc_report = sk.metrics.balanced_accuracy_score(y_true=target_cpu, y_pred=y_pred_cpu)
                
                # Computation of FAR for the binary case.
                conf_matrix = sklearn.metrics.confusion_matrix(y_true=target_cpu, y_pred=y_pred_cpu, labels=self.test_data.classes)
                tn, fp, fn, tp = conf_matrix.ravel()

                far = fp / (fp + tn)

                # Update progress report
                for key in cls_report.keys():
                    pr_key = key
                    if key == 'accuracy':
                        continue
                    elif key == 'macro avg':
                        continue
                    if key == '0.0':
                        pr_key = 0
                    elif key == '1.0':
                        pr_key = 1
                    
                    progress_reports[pr_key]['precision'].append(cls_report[key]['precision'])
                    progress_reports[pr_key]['recall'].append(cls_report[key]['recall'])
                    progress_reports[pr_key]['f1-score'].append(cls_report[key]['f1-score'])
                progress_reports['accuracy'].append(cls_report['accuracy'])
                progress_reports['far'].append(far)

                # Reporting / Display Section
                report_text = []
                report_text.append("===== Binary Test Report =====\n")
                for key in progress_reports.keys():
                    if (key == 'accuracy') or (key == 'far'):
                        continue
                    if (key == 0):
                        report_text.append(f"== Benign ==\n")
                        report_text.append(f"Precision: {progress_reports[key]['precision'][0]:.6f}\n")
                        report_text.append(f"Recall:    {progress_reports[key]['recall'][0]:.6f}\n")
                        report_text.append(f"F1-score:  {progress_reports[key]['f1-score'][0]:.6f}\n\n") 
                    elif (key == 1):
                        report_text.append(f"== Attack ==\n")
                        report_text.append(f"Precision: {progress_reports[key]['precision'][0]:.6f}\n")
                        report_text.append(f"Recall:    {progress_reports[key]['recall'][0]:.6f}\n")
                        report_text.append(f"F1-score:  {progress_reports[key]['f1-score'][0]:.6f}\n\n") 
                    else:
                        report_text.append(f"== {key} ==\n")
                        report_text.append(f"Precision: {progress_reports[key]['precision'][0]:.6f}\n")
                        report_text.append(f"Recall:    {progress_reports[key]['recall'][0]:.6f}\n")
                        report_text.append(f"F1-score:  {progress_reports[key]['f1-score'][0]:.6f}\n\n")
                    
                report_text.append(f"= Global Metrics =\n")
                report_text.append(f"Accuracy:           {progress_reports['accuracy'][0]:.6f}\n")
                report_text.append(f"False Alarm Rate:  {progress_reports['far'][0]:.6f}\n\n")

                # Confusion matrix
                report_text.append("Confusion Matrix:\n")
                report_text.append(str(conf_matrix) + "\n\n")


                # Convest report to text, print it and write to file.
                report_text = "".join(report_text)
                
                print(report_text)

                report_path = self.checkpoint_path + "_test_report.txt"
                with open(report_path, "w") as f:
                    f.write(report_text)

                sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
                    y_true=target_cpu, 
                    y_pred=y_pred_cpu, 
                    labels=self.test_data.classes, 
                    display_labels=["Benign", "Attack"], 
                    normalize='true'
                )

                disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
                    y_true=target_cpu, 
                    y_pred=y_pred_cpu, 
                    labels=self.test_data.classes, 
                    display_labels=["Benign", "Attack"], 
                    normalize='true'
                )

                # Save Confusion Matrix at same place as the checkpoints
                plt.savefig(self.checkpoint_path + "_confusion_matrix.png", dpi=300, bbox_inches='tight')
                plt.close()

                print("Report and confusion matrix recorded in: ", self.checkpoint_path)
