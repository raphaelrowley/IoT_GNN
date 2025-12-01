from configuration import *

class TestFNN(nn.Module):
    """
    Fully-connected feedforward neural network for edge classification.

    The model operates directly on edge attributes of a DGL graph and produces
    per-edge logits stored in ``graph.edata['edge_pred']``. Hidden layers,
    dropout usage, and output dimensionality are configured at initialization.

    Attributes
    ----------
    id : str
        String identifier for the model based on the number of hidden layers.
    num_hidden_layers : int
        Number of hidden layers in the network.
    layers : nn.Sequential
        Sequential container holding all hidden layers, including linear layers,
        activations, and optional dropout.
    output_layer : nn.Linear
        Final linear layer mapping hidden representations to output logits.

    Notes
    -----
    This docstring was created with assistance from ChatGPT.
    """

    def __init__(self, num_edge_attr, num_hidden_layers, hidden_layer_widths, num_classes, dropout=0.2):
        """
        Implements a simple fully-connected feedforward neural network for edge classification.

        Parameters
        ----------
        num_edge_attr : int
            Number of edge attributes used as input features.
        num_hidden_layers : int
            Number of hidden layers
        hidden_layer_widths : list of int
            Widths of hidden layers
        num_classes : int
            Number of output classes.
        dropout : float
            Dropout probability between layers. If ``None``,
            dropout is omitted.

        Notes
        -----
        This docstring was created with assistance from ChatGPT.
        """
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
        """
        Forward pass of the fully-connected neural network.

        Parameters
        ----------
        graph : dgl.DGLGraph
            DGL graph containing edge attributes in ``graph.edata['edge_attr']``.

        Returns
        -------
        graph : dgl.DGLGraph
            Graph with predicted logits stored in ``graph.edata['edge_pred']``.
        """
        x = graph.edata['edge_attr']
        logit = self.output_layer(self.layers(x))
        graph.edata['edge_pred'] = logit
        return graph