from configuration import *


class ModelTester:
    def __init__(self, test_data, use_gpu):
        self.test_data = test_data

        if use_gpu:
            self.use_gpu = True
            self.device = torch.device('cuda')

            self.test_data.graph = self.test_data.graph.to(self.device)
            
        else:
            self.use_gpu = False
            self.device = torch.device('cpu')

        if len(test_data.classes) > 2:
            self.loss_fn = nn.CrossEntropyLoss(
                weight = torch.tensor(test_data.class_weights, dtype=torch.float32)
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(test_data.class_weights[1] / test_data.class_weights[1], dtype=torch.float32)
            )

        self.checkpoint_base_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(self.checkpoint_base_path, exist_ok=True)
        self.checkpoint_base_path = os.path.join(self.checkpoint_base_path, self.test_data.id)

        self.checkpoint_path = None

    def test_model(self, model):
        if self.use_gpu:
            model = model.to(self.device)

        self.checkpoint_path = self.checkpoint_base_path + model.id
        
        # TODO: Do we even need test_risk and progress_reports?
        test_risk = []
        if len(self.test_data.classes) > 2:
            # multiclass_classification
            progress_reports = {}
            for cls in self.test_data.encoder.inverse_transform(self.test_data.classes):
                progress_reports[cls] = {}
                progress_reports[cls]['precision'] = []
                progress_reports[cls]['recall'] = []
                progress_reports[cls]['f1-score'] = []
            progress_reports['weighted avg'] = {}
            progress_reports['weighted avg']['precision'] = []
            progress_reports['weighted avg']['recall'] = []
            progress_reports['weighted avg']['f1-score'] = []
            progress_reports['weighted avg']['accuracy'] = []
            progress_reports['weighted avg']['far'] = []
        else:
            progress_reports = {}
            progress_reports['precision'] = []
            progress_reports['recall'] = []
            progress_reports['f1-score'] = []
            progress_reports['accuracy'] = []
            progress_reports['far'] = []
        
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
                y_pred = torch.argmax(logits, dim=-1)
                cls_report = sk.metrics.classification_report(y_true=target, y_pred=y_pred,
                                                              labels=self.test_data.classes,
                                                              target_names=self.test_data.encoder.inverse_transform(self.test_data.classes),
                                                              output_dict=True, zero_division=0.0)
                # Accuracy does not make much sense for imbalanced datasets like ours.
                acc_report = sk.metrics.accuracy_score(y_true=target, y_pred=y_pred,
                                                       normalize=True)
                bal_acc_report = sk.metrics.balanced_accuracy_score(y_true=target, y_pred=y_pred)
                # FAR for multiclass does not make sense so we will not compute it.
                
                # Confusion Matrix.
                conf_matrix = sklearn.metrics.confusion_matrix(y_true=target, y_pred=y_pred, labels=self.test_data.classes)

                

                # Update progress report
                for key in progress_reports.keys():
                    progress_reports[key]['precision'].append(cls_report[key]['precision'])
                    progress_reports[key]['recall'].append(cls_report[key]['recall'])
                    progress_reports[key]['f1-score'].append(cls_report[key]['f1-score'])

                # Reporting / Display Section
                report_text = []
                report_text.append("===== Test Report =====\n")
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

                # Convest report to text, print it and write to file.
                report_text = "".join(report_text)
                
                print(report_text)

                report_path = self.checkpoint_path + "_test_report.txt"
                with open(report_path, "w") as f:
                    f.write(report_text)
                
                sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
                    y_true=target, 
                    y_pred=y_pred, 
                    labels=self.test_data.classes, 
                    display_labels=self.test_data.encoder.inverse_transform(self.test_data.classes), 
                    normalize='true'
                )

                disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
                    y_true=target,
                    y_pred=y_pred,
                    labels=self.test_data.classes,
                    display_labels=self.test_data.encoder.inverse_transform(self.test_data.classes),
                    normalize='true',
                )
                
                # Save Confusion Matrix at same place as the checkpoints
                plt.savefig(self.checkpoint_path + "_confusion_matrix.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print("Report and confusion matrix recorded in: ", self.checkpoint_path)

            else:
                y_pred = 0.5 * (1+torch.sgn(logits))
                cls_report = sk.metrics.classification_report(y_true=target, y_pred=y_pred, output_dict=True, zero_division=0.0)
                acc_report = sk.metrics.accuracy_score(y_true=target, y_pred=y_pred,
                                                       normalize=True)
                bal_acc_report = sk.metrics.balanced_accuracy_score(y_true=target, y_pred=y_pred)
                conf_matrix = sklearn.metrics.confusion_matrix(y_true=target, y_pred=y_pred, labels=self.test_data.classes)
                tn, fp, fn, tp = conf_matrix.ravel()

                far = fp / (fp + tn)

                print(self.test_data.classes)
                sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true=target, y_pred=y_pred, labels=self.test_data.classes , display_labels=self.test_data.encoder.inverse_transform(self.test_data.classes), normalize='true')
                print(cls_report)
                print("acc", acc_report)
                print("bal_acc", bal_acc_report)
                print("far", far)
                print(conf_matrix)





    # TODO: Load a pre-trained model?
    # TODO: Test many models at once.    
    # TODO: Add the methods for pre-trained model.