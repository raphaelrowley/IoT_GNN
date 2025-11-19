from configuration import *


#See https://pytorch-geometric.readthedocs.io/en/2.5.1/generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing
#This layer is based on the EGraphSAGE Layer from https://arxiv.org/abs/2103.16329
class ESageLayer(geo_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean') #Mean aggregation of the edge attributes
        #the input of the linear layer is the concatenation of the node embedding plus the aggregated embedding (see Algorithm 1 line 5)
        #that goes into the linear layer without bias
        self.lin = nn.Linear(in_channels*2, out_channels, bias = False)
        self.activation = nn.Sigmoid()




    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        return x