from configuration import *

class TestFNN(nn.Module):
    def __init__(self, num_inputs, num_hidden_layers, is_multiclass, num_classes):
        super().__init__()

        self.num_hidden_layers = num_hidden_layers

        # Input layer
        self.layer1 = nn.Linear(num_inputs, 2*num_inputs)
        self.active1 = nn.ReLU()

        # Hidden layers
        # TODO: make the width of the layers parameterized.
        self.hidden_layers = []
        self.activations = []
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(2*num_inputs, 2*num_inputs))
            self.activations.append(nn.ReLU())
        
        # Output layer
        self.output_layer = None
        self.output_activation = None
        if is_multiclass:
            self.output_layer = nn.Linear(2*num_inputs, num_classes)
            self.output_activation = nn.Softmax(num_classes)
        else:
            self.output_layer = nn.Linear(2*num_inputs, 1)
            self.output_activation = nn.Sigmoid()
    
    def forward(self, x):
        # Input layer
        x = self.active1(self.layer1(x))
        # Hidden layers
        for i in range(self.num_hidden_layers):
            x = self.activations[i](self.hidden_layers[i](x))
        # Output layer
        x = self.output_activation(self.output_layer(x))
        return x