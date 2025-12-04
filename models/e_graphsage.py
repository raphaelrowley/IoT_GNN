from configuration import *

class E_GraphSAGE(nn.Module):
    """
    Implementation of the E-GraphSAGE model.

    Attributes
    ----------
    graphsage : nn.Sequential
        Sequence of ``E_GraphSAGE_Layer`` modules performing node updates.
    mlp : nn.Sequential
        Final classifier applied to concatenated node embeddings.
    id : str
        String identifier of the model based on number of layers and embedding size.

    References
    ----------
    W. W. Lo, S. Layeghy, M. Sarhan, M. Gallagher and M. Portmann,
    "E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT,"
    NOMS 2022-2022 IEEE/IFIP Network Operations and Management Symposium, Budapest, Hungary, 2022,
    pp. 1-9, doi: 10.1109/NOMS54207.2022.9789878.

    Notes
    -----
    This docstring was created with assistance from ChatGPT.
    """

    def __init__(self, numLayers, dim_node_embed, num_edge_attr, num_classes, dropout=0.2, normalization=False):
        """
        Initializes an E-GraphSAGE model.

        Parameters
        ----------
        numLayers : int
            Number of E-GraphSAGE layers.
        dim_node_embed : int
            Dimensionality of node embeddings.
        num_edge_attr : int
            Number of edge attribute features in the input graph.
        num_classes : int
            Number of output classes. If ``num_classes == 2``, the model outputs
            a single logit per edge.
        dropout : float, optional
            Dropout probability after each hidden layer except the last.
            If ``None``, dropout is omitted.
        normalization : bool, optional
            If True, add a normalization layer after activation in each hidden layer.

        Notes
        -----
        This docstring was created with assistance from ChatGPT.
        """

        super(E_GraphSAGE, self).__init__()

        # Use the given number of E_GraphSAGE_Layers, dropout between each layer
        self.graphsage = nn.Sequential(
            E_GraphSAGE_Layer(num_edge_attr=num_edge_attr,
                              dim_node_embed=num_edge_attr,
                              dim_new_node_embed=dim_node_embed,
                              dropout=(dropout if numLayers > 1 else None),
                              normalization = normalization if numLayers > 1 else False
                              ),
        )
        for k in range(numLayers-2):
            self.graphsage.append(
                E_GraphSAGE_Layer(num_edge_attr=num_edge_attr, dim_node_embed=dim_node_embed,
                                  dim_new_node_embed=dim_node_embed, dropout=dropout, normalization=normalization),
            )
        if numLayers > 1:
            self.graphsage.append(
                E_GraphSAGE_Layer(num_edge_attr=num_edge_attr, dim_node_embed=dim_node_embed,
                                  dim_new_node_embed=dim_node_embed, dropout=None, normalization=False),
            )

        dim_output = 1 if (num_classes == 2) else num_classes
        self.mlp = nn.Sequential(
            nn.Linear(2*dim_node_embed, dim_output),
        )

        self.id = f'E_GraphSAGE_K{numLayers}_H{dim_node_embed}'

    def forward(self, graph):
        """
        Forward pass of the E-GraphSAGE model.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph containing ``edge_attr`` and initial ``node_attr`` features.

        Returns
        -------
        graph : dgl.DGLGraph
            Output graph whose ``edata['edge_pred']`` contains per-edge logits.

        Notes
        -----
        This docstring was created with assistance from ChatGPT.
        """
        graph = self.graphsage(graph)
        graph.apply_edges(lambda edges:
                          {'edge_pred' : torch.cat((edges.src['node_attr'], edges.dst['node_attr']), dim=-1)}
                          )
        graph.edata['edge_pred'] = self.mlp(graph.edata['edge_pred'])

        return graph


class E_GraphSAGE_Layer(nn.Module):
    """
    Single E-GraphSAGE layer performing edge-based message aggregation
    and node embedding updates.

    Attributes
    ----------
    linear : nn.Linear
        Linear transformation applied to concatenated node- and message-features.
    relu : nn.ReLU
        Activation function applied after the linear transformation.
    dropout : nn.Dropout or None
        Optional dropout applied to updated node embeddings.
    normalization : nn.BatchNorm1d or None
        Optional normalization layer applied after activation.

    Notes
    -----
    This docstring was created with assistance from ChatGPT.
    """

    def __init__(self, num_edge_attr, dim_node_embed, dim_new_node_embed, dropout=0.2, normalization=False):
        """
        Initializes an E-GraphSAGE layer.

        Parameters
        ----------
        num_edge_attr : int
            Number of edge attribute features.
        dim_node_embed : int
            Dimensionality of node embeddings of the previous layer.
        dim_new_node_embed : int
            Dimensionality of new node embeddings after the update.
        dropout : float or None, optional
            Dropout rate applied to updated node embeddings. If ``None``,
            dropout is omitted.
        normalization : bool, optional
            If True, add a normalization layer after activation.

        Notes
        -----
        This docstring was created with assistance from ChatGPT.
        """
        super(E_GraphSAGE_Layer, self).__init__()

        self.linear = nn.Linear(in_features=(dim_node_embed+num_edge_attr), out_features=dim_new_node_embed, bias=True)
        self.relu = nn.ReLU()
        if normalization:
            self.normalization = nn.LayerNorm(dim_new_node_embed)
        else:
            self.normalization = None
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        """
        Forward pass of the E-GraphSAGE layer.

        Performs the following:
        1. Aggregates incoming edge attributes via mean aggregation.
        2. Concatenates old node embeddings with aggregated edge attributes.
        3. Applies linear transformation, ReLU, and optional normalization and dropout.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input graph containing ``edata['edge_attr']`` and ``ndata['node_attr']``.

        Returns
        -------
        graph : dgl.DGLGraph
            Graph with updated node embeddings stored in ``ndata['node_attr']``.

        Notes
        -----
        This docstring was created with assistance from ChatGPT.
        """

        # Line (4): Aggregate Edge Features
        graph.update_all(
            dgl.function.copy_e(e='edge_attr', out='m'),
            dgl.function.mean(msg='m', out='h_Nv')
        )

        # Line (5)
        graph.ndata['node_attr'] = self.linear(torch.cat((graph.ndata['node_attr'], graph.ndata['h_Nv']), dim=-1))
        if self.normalization is not None:
            graph.ndata['node_attr'] = self.normalization(graph.ndata['node_attr'])
        graph.ndata['node_attr'] = self.relu(graph.ndata['node_attr'])
        if self.dropout is not None:
            graph.ndata['node_attr'] = self.dropout(graph.ndata['node_attr'])
        del graph.ndata['h_Nv']

        return graph
