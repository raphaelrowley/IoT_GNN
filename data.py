import sklearn.utils.class_weight

from configuration import *

class IoTDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset='NF-BoT-IoT',
                 version=2,         # TODO change to v1 later
                 split='train',
                 multiclass=False,
                 randomize_source_ip=True,
                 test_size=0.2,     # TODO do we want to keep test size and val size fixed?
                 val_size=0.1,
                 data_parent_dir = None):

        assert split in ['train', 'val', 'test'], 'Invalid split argument'

        self.ordinal_encoder = None             # initialize to save when transforming labels for inverse_transform later


        # Assume "data" dir is in the same dir as data.py
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        # Can also specify another location.
        if data_parent_dir is not None:
            data_path = os.path.join(data_parent_dir, 'data')
        base_path = os.path.join(data_path, dataset + f'-v{version}')

        graph_path = f'{base_path}-{split}-{('randomized' if randomize_source_ip else '')}.pt'
        graph = None

        # Check if preprocessed graph exists, else initialize
        if os.path.exists(graph_path):
            graph = torch.load(graph_path, weights_only=False)
        else:
            csv_path = base_path + '.csv'
            graphs = self.init_graph(csv_path, randomize_source_ip, test_size, val_size)
            self.save_graphs(base_path, randomize_source_ip, graphs)
            graph = graphs[split]

        # Set the required labels
        self.set_labels(graph, multiclass=multiclass)

        self.graph = graph
        self.num_features = graph.edge_attr.shape[-1]
        self.classes = np.unique(graph.edge_label)
        self.class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                             classes=self.classes,
                                                                             y=graph.edge_label.numpy())

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph

    @staticmethod
    def init_graph(csv_path, randomize_source_ip, test_size, val_size):
        df = pd.read_csv(csv_path)

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Random mapping of IP addresses
        if randomize_source_ip:
            lower = int(ipaddress.IPv4Address('172.16.0.1'))
            upper = int(ipaddress.IPv4Address('172.31.0.1'))
            ip_ints = np.random.randint(low=lower, high=upper + 1, size=len(df), dtype=np.uint32)
            # This loop might be slow, might need to rewrite this for larger datasets using numpy string operations or so
            ips = [str(ipaddress.IPv4Address(int(i))) for i in ip_ints]
            df["IPV4_SRC_ADDR"] = ips

        # Merge information about SRC and DST
        df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].astype(str) + ':' + df["L4_SRC_PORT"].astype(str)
        df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].astype(str) + ':' + df["L4_DST_PORT"].astype(str)
        df.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT'], inplace=True)

        # Select numeric columns / categorical columns which are already transformed to numbers
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_cols = numeric_cols.drop('Label')  # Drop label, we want to keep those entries {0, 1}

        # Normalize the previously extracted numerical data
        numeric_data = df[numeric_cols]
        scaler = sk.preprocessing.StandardScaler().fit(numeric_data)        # Could be easily exchanged for another sklearn scaler
        numeric_data_normalized = scaler.transform(numeric_data)
        df[numeric_cols] = numeric_data_normalized

        # Split the data into train, test and validation datasets.
        try:
            # Use a stratified split to ensure attack types and benign traffic are represented equally
            train_df, test_df = sk.model_selection.train_test_split(df, stratify=df['Attack'],
                                                                    test_size=test_size + val_size, shuffle=True)
            test_df, val_df = sk.model_selection.train_test_split(test_df, stratify=test_df['Attack'],
                                                                  test_size=val_size / (test_size + val_size))
        except ValueError:
            print('No stratified split feasible')
            train_df, test_df = sk.model_selection.train_test_split(df, test_size=test_size + val_size, shuffle=True)
            test_df, val_df = sk.model_selection.train_test_split(test_df, test_size=val_size / (test_size + val_size))

        graphs = {}
        splits = ['train', 'val', 'test']
        for i, df in enumerate([train_df, val_df, test_df]):
            # Create Networkx graphs
            g = nx.from_pandas_edgelist(df, source='IPV4_SRC_ADDR', target='IPV4_DST_ADDR', edge_attr=True,
                                        create_using=nx.DiGraph())

            # Convert to PyG, group numerical features as edge attributes.
            # These attributes are the ones that we can later pass to the GNN for learning.
            pyg_graph = torch_geometric.utils.convert.from_networkx(g, group_edge_attrs=numeric_cols.tolist())

            # Add all-one-vectors as "dummy" node attributes
            # E-Graphsage-paper: "[…] dimension of all one constant vector is the same as the number of edge features."
            pyg_graph.node_attr = torch.ones(pyg_graph.num_nodes, pyg_graph.edge_attr.shape[-1])

            graphs[splits[i]] = pyg_graph

        return graphs

    @staticmethod
    def save_graphs(base_path, randomize_source_ip, graphs):
        splits = ['train', 'val', 'test']
        for split in splits:
            graph_path = f'{base_path}-{split}-{('randomized' if randomize_source_ip else '')}.pt'
            torch.save(graphs[split], graph_path)

    def set_labels(self, graph, multiclass):
        # Set the edge labels depending on binary or multiclass classification

        if multiclass:
            # Use sklearn for one-hot encoding of the strings to calculate CE-Loss later.
            labels = np.array(graph.Attack).reshape(-1, 1)
            self.ordinal_encoder = sk.preprocessing.OrdinalEncoder()
            ordinal_encoded = self.ordinal_encoder.fit_transform(labels)

            graph.edge_label = torch.tensor(ordinal_encoded.squeeze(-1), dtype=torch.long)

            # The code below performs one-hot encoding. However, the PyG-train-test-split does not
            # work with one-hot encodings
            # labels = np.array(pyg_graph.Attack).reshape(-1, 1)
            # one_hot_encoder = sk.preprocessing.OneHotEncoder()
            # one_hot_encoded = one_hot_encoder.fit_transform(labels).toarray()
            #
            # pyg_graph.edge_label = one_hot_encoded
        else:
            graph.edge_label = graph.Label

        del graph.Label, graph.Attack
        return graph


def test():
    test_data = IoTDataset()
    test_data2 = IoTDataset(multiclass=True)


if __name__ == "__main__":
    test()