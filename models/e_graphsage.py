from configuration import *
from data import IoTDataset

import dgl

# class E_GraphSage_Layer(nn.Module):
#
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(E_GraphSage_Layer, self).__init__()
#
#         self.neighborhood_sampler = geo_nn.MessagePassing(
#             aggr='mean',
#             flow='source_to_target',
#
#         )


class E_GraphSage_Layer(geo_nn.MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(E_GraphSage_Layer, self).__init__(
            aggr='mean'
        )

    def forward(self, graph):
        edge_index = graph.edge_index
        edge_attr = torch.cat((graph.edge_attr, graph.edge_attr), dim=-1)

        print('EDGE INDEX', edge_index.shape)
        print('EDGE ATTR', edge_attr.shape)
        node_attr = graph.node_attr
        print('NODE ATTR', node_attr.shape)
        # print(edge_attr)

        temp = self.propagate(edge_index, edge_attr=edge_attr, x=node_attr)
        print(self.edge_updater(edge_index, edge_attr=edge_attr))
        print(temp.shape)

    def message(self, x_j):
        print("TEST")
        print(x_j.shape)
        return super(E_GraphSage_Layer, self).message(x_j)


def test():
    train_data = IoTDataset()
    nx_graph = pyg.utils.to_networkx(train_data)
    dgl_graph = dgl.from_networkx(nx_graph)
    print(dgl_graph)
    layer = E_GraphSage_Layer(train_data.num_features, 64)
    layer.forward(train_data.__getitem__(0))

test()