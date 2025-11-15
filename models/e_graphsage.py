from configuration import *
from data import IoTDataset

class E_GraphSAGE(nn.Module):

    def __init__(self, numLayers, dim_node_embed, num_edge_attr, num_classes, dropout=0.2):
        super(E_GraphSAGE, self).__init__()

        # Use the given number of E_GraphSAGE_Layers, dropout between each layer
        self.graphsage = nn.Sequential(
            E_GraphSAGE_Layer(num_edge_attr=num_edge_attr,
                              dim_node_embed=num_edge_attr,
                              dim_new_node_embed=dim_node_embed,
                              dropout=(dropout if numLayers > 1 else None),
                              ),
        )
        for k in range(numLayers-2):
            self.graphsage.append(
                E_GraphSAGE_Layer(num_edge_attr=num_edge_attr, dim_node_embed=dim_node_embed,
                                  dim_new_node_embed=dim_node_embed, dropout=dropout),
            )
        if numLayers > 1:
            self.graphsage.append(
                E_GraphSAGE_Layer(num_edge_attr=num_edge_attr, dim_node_embed=dim_node_embed,
                                  dim_new_node_embed=dim_node_embed, dropout=None),
            )

        dim_output = 1 if (num_classes == 2) else num_classes
        self.mlp = nn.Sequential(
            nn.Linear(2*dim_node_embed, dim_output),
        )
        if dim_output > 1:
            self.mlp.append(nn.Softmax(dim=-1))

    def forward(self, graph):
        graph = self.graphsage(graph)
        graph.apply_edges(lambda edges:
                          {'edge_pred' : torch.cat((edges.src['node_attr'], edges.dst['node_attr']), dim=-1)}
                          )
        graph.edata['edge_pred'] = self.mlp(graph.edata['edge_pred'])

        return graph


class E_GraphSAGE_Layer(nn.Module):

    def __init__(self, num_edge_attr, dim_node_embed, dim_new_node_embed, dropout=0.2):
        super(E_GraphSAGE_Layer, self).__init__()

        self.linear = nn.Linear(in_features=(dim_node_embed+num_edge_attr), out_features=dim_new_node_embed, bias=False)
        self.relu = nn.ReLU()

        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, graph):

        # Line (4): Aggregate Edge Features
        graph.update_all(
            dgl.function.copy_e(e='edge_attr', out='m'),
            dgl.function.mean(msg='m', out='h_Nv')
        )

        # Line (5)
        graph.ndata['node_attr'] = self.linear(torch.cat((graph.ndata['node_attr'], graph.ndata['h_Nv']), dim=-1))
        graph.ndata['node_attr'] = self.relu(graph.ndata['node_attr'])
        if self.dropout is not None:
            graph.ndata['node_attr'] = self.dropout(graph.ndata['node_attr'])
        del graph.ndata['h_Nv']

        return graph


def test():
    train_data = IoTDataset(version=1)
    pyg_graph = train_data[0]
    nx_graph = torch_geometric.utils.to_networkx(
        pyg_graph,
        to_undirected=False,
        node_attrs=["node_attr"],
        edge_attrs=["edge_attr", "edge_label"]
    )
    dgl_graph = dgl.from_networkx(
        nx_graph,
        node_attrs=["node_attr"],
        edge_attrs=["edge_attr", "edge_label"]
    )
    print(dgl_graph)

    temp = E_GraphSAGE(numLayers=2, num_edge_attr=8, dim_node_embed=128, num_classes=5)
    temp.forward(dgl_graph)

test()


# TODO
#   – Maybe do copy.deepcopy() first in forward or ensure this is done in training?
#   – Otherwise, we modify everything etc.
#   – Prevent target leakage, do not provide labels!
#   – Why does v1 of NF-BoT-IoT allow no stratified split?