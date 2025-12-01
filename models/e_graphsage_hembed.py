from configuration import *

class E_GraphSAGE_hEmbed(nn.Module):
    """
    Implementation of the E-GraphSAGE model with modified aggregation.

    Attributes
    ----------
    graphsage : nn.Sequential
        Sequence of modified ``E_GraphSAGE_hEmbed_Layer`` modules performing node updates.
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
    def __init__(self, numLayers, dim_node_embed, num_edge_attr, num_classes, dropout=0.2):
        """
        Initializes a modified E-GraphSAGE model.

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

        Notes
        -----
        This docstring was created with assistance from ChatGPT.
        """
        super(E_GraphSAGE_hEmbed, self).__init__()

        # Use the given number of E_GraphSAGE_Layers, dropout between each layer
        self.graphsage = nn.Sequential(
            E_GraphSAGE_hEmbed_Layer(num_edge_attr=num_edge_attr,
                                     dim_node_embed=num_edge_attr,
                                     dim_new_node_embed=dim_node_embed,
                                     dropout=(dropout if numLayers > 1 else None),
                                    ),
        )
        for k in range(numLayers-2):
            self.graphsage.append(
                E_GraphSAGE_hEmbed_Layer(num_edge_attr=num_edge_attr, dim_node_embed=dim_node_embed,
                                         dim_new_node_embed=dim_node_embed, dropout=dropout),
            )
        if numLayers > 1:
            self.graphsage.append(
                E_GraphSAGE_hEmbed_Layer(num_edge_attr=num_edge_attr, dim_node_embed=dim_node_embed,
                                         dim_new_node_embed=dim_node_embed, dropout=None),
            )

        dim_output = 1 if (num_classes == 2) else num_classes
        self.mlp = nn.Sequential(
            nn.Linear(2*dim_node_embed, dim_output),
        )

        self.id = f'E_GraphSAGE_hEmbed_K{numLayers}_H{dim_node_embed}'

    def forward(self, graph):
        """
        Forward pass of the modified E-GraphSAGE model.

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


class E_GraphSAGE_hEmbed_Layer(nn.Module):
    """
    Single modified E-GraphSAGE layer performing both node- and
    edge-based message aggregation and node embedding updates.

    Attributes
    ----------
    linear : nn.Linear
        Linear transformation applied to concatenated node- and message-features.
    relu : nn.ReLU
        Activation function applied after the linear transformation.
    dropout : nn.Dropout or None
        Optional dropout applied to updated node embeddings.

    Notes
    -----
    This docstring was created with assistance from ChatGPT.
    """

    def __init__(self, num_edge_attr, dim_node_embed, dim_new_node_embed, dropout=0.2):
        """
        Initializes a modified E-GraphSAGE layer.

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

        Notes
        -----
        This docstring was created with assistance from ChatGPT.
        """
        super(E_GraphSAGE_hEmbed_Layer, self).__init__()

        self.linear = nn.Linear(in_features=(2*dim_node_embed+num_edge_attr), out_features=dim_new_node_embed, bias=True)
        self.relu = nn.ReLU()

        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        """
        Forward pass of the modified E-GraphSAGE layer.

        Performs the following:
        1. Aggregates incoming edge and node attributes via mean aggregation and concatenates them.
        2. Concatenates old node embeddings with aggregated edge attributes.
        3. Applies linear transformation, ReLU, and optional dropout.

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

        # Also aggregate node features
        graph.update_all(
            dgl.function.copy_u(u='node_attr', out='mh'),
            dgl.function.mean(msg='mh', out='h_Nvh')
        )

        # concat aggregations
        graph.ndata['h_Nv'] = torch.cat((graph.ndata['h_Nv'], graph.ndata['h_Nvh']), dim=-1)
        del graph.ndata['h_Nvh']

        # Line (5)
        graph.ndata['node_attr'] = self.linear(torch.cat((graph.ndata['node_attr'], graph.ndata['h_Nv']), dim=-1))
        graph.ndata['node_attr'] = self.relu(graph.ndata['node_attr'])
        if self.dropout is not None:
            graph.ndata['node_attr'] = self.dropout(graph.ndata['node_attr'])
        del graph.ndata['h_Nv']

        return graph
