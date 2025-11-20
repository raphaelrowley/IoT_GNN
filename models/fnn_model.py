from configuration import *

class TestFNN(nn.Module):
    def __init__(self, num_edge_attr, num_hidden_layers, hidden_layer_widths, num_classes, dropout=0.2):
        super().__init__()

        self.num_hidden_layers = num_hidden_layers

        assert num_hidden_layers == len(hidden_layer_widths)

        self.id = f'FCNN_K{num_hidden_layers}'

        self.layers = nn.Sequential()
        prev_width = num_edge_attr
        for width in hidden_layer_widths:
            self.layers.append(nn.Linear(in_features=prev_width, out_features=width, bias=True))
            self.layers.append(nn.ReLU())
            if dropout is not None:
                self.layers.append(nn.Dropout1d(p=dropout))     # TODO Check if this is too aggressive
            prev_width = width
        
        # Output layer
        self.output_layer = None
        dim_output = 1 if (num_classes == 2) else num_classes
        self.output_layer = nn.Linear(prev_width, dim_output)

    
    def forward(self, graph):
        x = graph.edata['edge_attr']
        logit = self.output_layer(self.layers(x))
        graph.edata['edge_pred'] = logit
        return graph