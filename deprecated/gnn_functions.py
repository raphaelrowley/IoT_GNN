from configuration import *



#Item 3
def test(model, test_loader, loss_function, device='cpu'):
    """
    Tests a GNN model on a test set.
    ----------------
    Input:
    - model: The GNN model to be tested.
    - test_loader: DataLoader for the test dataset.
    - loss_function: The loss function to be used during testing.
    - device: The device to run the testing on (e.g., 'cpu', 'cuda').
    ----------------
    Output:
    - test_risk: The average loss on the test dataset.
    - test_accuracy: The accuracy on the test dataset.
    """

    model = model.to(device = device)
    model.eval()

    with torch.no_grad():
        risk = 0
        accuracy = 0

        for data in test_loader:

            data = data.to(device=device)

            # forward pass
            outputs = model(data)
            loss = loss_function(outputs, data.y) #make sure outputs and data.y have compatible shapes and types

            predicted = torch.max(outputs.data, 1)[1].type(torch.float32) #Assuming outputs is a tensor of size [batch_size, num_classes]

            # compute the fraction of correctly predicted labels
            correct_predict = torch.sum(predicted == data.y)/(data.y.size(0))

            risk += loss.item()
            accuracy += correct_predict.item()

        test_risk = risk/len(test_loader)
        test_accuracy = accuracy/len(test_loader)

    return test_risk, test_accuracy

def train(model, num_epochs, loss_function, train_loader, test_loader, optimizer, device='cpu'):
    """
    Trains a GNN model and evaluates it on a test set after each epoch.
    ---------------
    Input:
    - model: The GNN model to be trained.
    - num_epochs: Number of epochs to train the model.
    - loss_function: The loss function to be used during training.
    - train_loader: DataLoader for the training dataset.
    - test_loader: DataLoader for the test (validation) dataset.
    - optimizer: The optimizer to update the model parameters.
    - device: The device to run the training on (e.g., 'cpu', 'cuda').
    ---------------
    Output:
    - train_risk: List of training losses for each epoch.
    - test_risk: List of test losses for each epoch.
    - test_accuracy: List of test accuracies for each epoch.
    ---------------
    Note:
    - Each sample in train_loader can be of type Data containing the edge indices, node features, edge features, and labels.
    - the node features can be a tensor x of shape [num_nodes, num_node_features]
    - the edge indices can be a tensor edge_idx of shape [2, num_edges]
    - the edge features can be a tensor edge_attr of shape [num_edges, num_edge_features]
    - the labels can be a tensor y of shape [num_edges] for edge classification tasks.
    """

    #Move our model to the configured device
    model = model.to(device = device)

    train_risk = []
    test_risk = []
    test_accuracy = []


    for epoch in range(num_epochs):
        # training risk in one epoch
        risk = 0

        model.train()

        # loop over training data
        for data in train_loader: #each data can be of type Data containing the edge indices, node features, edge features and labels
            data = data.to(device=device)

            # forward pass
            outputs = model(data)
            loss = loss_function(outputs, data.y) #make sure outputs and data.y have compatible shapes and types

            risk += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        risk_epoch, accuracy_epoch = test(model, test_loader, loss_function, device='cpu')

        train_risk.append(risk/len(train_loader))
        test_risk.append(risk_epoch)
        test_accuracy.append(accuracy_epoch)

        print(f"epoch {epoch} - test risk/accur {risk_epoch:.2f},{accuracy_epoch:.2f}")


    # plot the training and test losses
    plt.plot([i+1 for i in range(num_epochs)], train_risk, label='train')
    plt.plot([i+1 for i in range(num_epochs)], test_risk, label='test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Risk')
    plt.show()

    # plot the test accuracy
    plt.plot([i+1 for i in range(num_epochs)], test_accuracy)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    return train_risk, test_risk, test_accuracy