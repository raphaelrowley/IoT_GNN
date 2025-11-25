import copy

import sklearn.utils.class_weight
import torch

from configuration import *


class IoTDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset='NF-BoT-IoT',
                 version=1,
                 split='train',
                 multiclass=False,
                 randomize_source_ip=True,
                 test_size=0.2,  # TODO do we want to keep test size and val size fixed?
                 val_size=0.1,
                 data_parent_dir=None,
                 relabel_nodes=False,
                 ):

        assert split in ['train', 'val', 'test'], 'Invalid split argument'

        self.encoder = None  # initialize to save when transforming labels for inverse_transform later

        # Assume "data" dir is in the same dir as data.py
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        # Can also specify another location.
        if data_parent_dir is not None:
            data_path = os.path.join(data_parent_dir, 'data')
        base_path = os.path.join(data_path, dataset + f'-v{version}')

        df_path = f'{base_path}-{split}{("-randomized" if randomize_source_ip else "")}.pkl'
        df = None

        # Check if preprocessed graph exists, else initialize
        if os.path.exists(df_path):
            df = pd.read_pickle(df_path)
        else:
            csv_path = base_path + '.csv'
            dfs = self.preprocess_data(csv_path, randomize_source_ip, test_size, val_size)
            self.save_dataframes(base_path, randomize_source_ip, dfs)
            df = dfs[split]

        # Set the required labels
        self.set_labels(df, multiclass=multiclass)

        # Init Graph
        self.graph = self.convert_to_dgl(df, relabel_nodes=relabel_nodes)

        if not multiclass:
            self.graph.edata['edge_label'] = self.graph.edata['edge_label'].unsqueeze(-1).to(torch.float32)

        self.num_features = self.graph.edata['edge_attr'].shape[-1]
        self.classes = np.unique(df['edge_label'].values)
        self.class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                             classes=self.classes,
                                                                             y=df['edge_label'].values)

        self.id = dataset + f'-v{version}{("-randomized" if randomize_source_ip else "")}{"-multiclass" if len(self.classes) > 2 else ""}'

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph

    @staticmethod
    def preprocess_data(csv_path, randomize_source_ip, test_size, val_size):
        df = pd.read_csv(csv_path)

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Remove NaNs
        df = df.dropna()

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
        scaler = sk.preprocessing.StandardScaler().fit(numeric_data)    # Could be easily exchanged for another sklearn scaler
        numeric_data_normalized = scaler.transform(numeric_data)
        df[numeric_cols] = numeric_data_normalized

        # Summarize Edge Attributes
        df['edge_attr'] = df[numeric_cols].apply(lambda row: torch.tensor(row.values, dtype=torch.float32), axis=1)
        df.drop(columns=numeric_cols, inplace=True)

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

        dfs = {}
        splits = ['train', 'val', 'test']
        for i, df in enumerate([train_df, val_df, test_df]):
            dfs[splits[i]] = df

        return dfs

    @staticmethod
    def save_dataframes(base_path, randomize_source_ip, dfs):
        splits = ['train', 'val', 'test']
        for split in splits:
            df_path = f'{base_path}-{split}{("-randomized" if randomize_source_ip else "")}.pkl'
            dfs[split].to_pickle(df_path)

    def set_labels(self, df, multiclass):
        # Set the edge labels depending on binary or multiclass classification
        if multiclass:
            # Use sklearn for encoding of the strings to calculate CE-Loss later
            labels = df['Attack'].values
            # self.encoder = sk.preprocessing.OneHotEncoder()
            self.encoder = sk.preprocessing.LabelEncoder()
            encoded_label = self.encoder.fit_transform(labels)

            df['edge_label'] = encoded_label
        else:
            df['edge_label'] = df['Label']

        df.drop(columns=['Label', 'Attack'], inplace=True)
        return

    def convert_to_dgl(self, df, relabel_nodes):
        # Create Networkx graph
        g = nx.from_pandas_edgelist(df, source='IPV4_SRC_ADDR', target='IPV4_DST_ADDR', edge_attr=True,
                                    create_using=nx.DiGraph())

        # relabel nodes to check permutation equivariance
        if relabel_nodes:
            node_list = list(g)
            renamed_node_list = copy.deepcopy(node_list)
            random.shuffle(renamed_node_list)
            g = nx.relabel_nodes(g, mapping=dict(zip(node_list, renamed_node_list)), copy=True)

        dgl_graph = dgl.from_networkx(g, edge_attrs=['edge_attr', 'edge_label'])

        # Add all-one-vectors as "dummy" node attributes
        # E-Graphsage-paper: "[…] dimension of all one constant vector is the same as the number of edge features."
        dgl_graph.ndata['node_attr'] = torch.ones(dgl_graph.num_nodes(), dgl_graph.edata['edge_attr'].shape[-1])

        return dgl_graph
