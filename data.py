from configuration import *


class IoTDataset(torch.utils.data.Dataset):
    """
    A IoT NIDS dataset, represented as a single DGLGraph.

    Attributes
    ----------
    graph : dgl.DGLGraph
        The constructed DGL graph containing ``edge_attr`` and ``edge_label``.
    encoder : sklearn.preprocessing.LabelEncoder or None
        Encoder used when performing multiclass label encoding.
    num_features : int
        Number of edge features in ``edge_attr``.
    classes : ndarray
        Unique label values present in the dataset split.
    class_weights : ndarray
        Class weights computed with ``class_weight='balanced'`` for loss weighting.
    id : str
        String identifier containing dataset, version, and flags (e.g.,
        randomized IP or multiclass).

    Notes
    ------
    This docstring was created with assistance from ChatGPT.
    """

    def __init__(self,
                 dataset='NF-BoT-IoT',
                 version=1,
                 split='train',
                 multiclass=False,
                 randomize_source_ip=True,
                 test_size=0.2,  # Fixed for this project
                 val_size=0.1,   # Fixed for this project
                 data_parent_dir=None,
                 relabel_nodes=False,
                 g = None
                 ):
        """
        Initializes a graph dataset object by loading or preprocessing the raw NetFlow data,
        generating DGL graphs, setting labels, and computing class statistics.

        Parameters
        ----------
        dataset : str, optional
            Name of the dataset, used to form the base file path.
        version : int, optional
            Dataset version number appended to the file name.
        split : {'train', 'val', 'test'}, optional
            Dataset split to load. If preprocessed pickle files exist, they are loaded.
            Otherwise, the raw CSV file is processed and saved.
        multiclass : bool, optional
            If True, the ``'Attack'`` column is encoded using ``LabelEncoder`` and
            multiclass classification is assumed. If False, binary labels are used.
        randomize_source_ip : bool, optional
            If True, source IP addresses are replaced by randomized private-network
            IPv4 addresses during preprocessing.
        test_size : float, optional
            Fraction of samples allocated to the test split during preprocessing.
        val_size : float, optional
            Fraction of samples allocated to the validation split during preprocessing.
        data_parent_dir : str or None, optional
            Optional directory overriding the default location of the ``data`` folder.
            If ``None``, the ``data`` directory is assumed to be located next to this file.
        relabel_nodes : bool, optional
            If True, graph node identifiers are randomly permuted after graph creation
            (useful for testing permutation equivariance).
        g : int or None, optional
            Random seed for reproducibility.

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """

        assert split in ['train', 'val', 'test'], 'Invalid split argument'

        self.encoder = None  # initialize to save when transforming labels for inverse_transform later

        # Assume "data" dir is in the same dir as data.py
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        # Can also specify another location.
        if data_parent_dir is not None:
            data_path = os.path.join(data_parent_dir, 'data')
        base_path = os.path.join(data_path, dataset + f'-v{version}')

        df_path = f'{base_path}-{("" if g is None else f"g{g}-")}{split}{("-randomized" if randomize_source_ip else "")}.pkl'
        df = None

        # Check if preprocessed graph exists, else initialize
        if os.path.exists(df_path):
            df = pd.read_pickle(df_path)
        else:
            csv_path = base_path + '.csv'
            dfs = self.preprocess_data(csv_path, randomize_source_ip, test_size, val_size)
            self.save_dataframes(base_path, randomize_source_ip, dfs, g)
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
        """
        Returns the number of samples in our dataset.

        Returns
        -------
        int
            Always ``1``. This dataset stores a single full-batch graph.

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """
        return 1

    def __getitem__(self, idx):
        """
        Retrieve the dataset's single DGL graph.

        Parameters
        ----------
        idx : int
            Ignored. Included for compatibility with dataset interfaces.

        Returns
        -------
        dgl.DGLGraph
            The stored graph representing the entire dataset split.

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """
        return self.graph

    @staticmethod
    def preprocess_data(csv_path, randomize_source_ip, test_size, val_size):
        """
        Load, clean, normalize, and split the raw CSV network-flow data into
        train/validation/test dataframes.

        This function performs several preprocessing steps:
        - Loads the CSV file
        - Removes duplicates and NaN values
        - Optionally randomizes source IP addresses within a private range
        - Merges IP address and port information into single identifiers
        - Normalizes numerical feature columns using ``StandardScaler``
        - Converts numeric features into ``torch.Tensor`` edge attributes
        - Produces stratified train/validation/test splits (if possible)

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing Netflow records.
        randomize_source_ip : bool
            If True, source IP addresses are replaced by randomly generated
            IPv4 addresses in the private range 172.16.0.1 – 172.31.0.1.
        test_size : float
            Fraction of samples to allocate to the test split.
        val_size : float
            Fraction of samples to allocate to the validation split.

        Returns
        -------
        dfs : dict of pandas.DataFrame
            A dictionary containing the preprocessed splits:

            - ``'train'`` : training dataframe
            - ``'val'`` : validation dataframe
            - ``'test'`` : test dataframe

            Each dataframe contains:
            - ``'IPV4_SRC_ADDR'`` and ``'IPV4_DST_ADDR'`` indicating the direction of each flow
            - ``'edge_attr'`` : torch.Tensor of normalized numeric features
            - ``'Label'`` and ``'Attack'``

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """
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
                                                                    test_size=test_size + val_size, shuffle=True,
                                                                    random_state=42)
            test_df, val_df = sk.model_selection.train_test_split(test_df, stratify=test_df['Attack'],
                                                                  test_size=val_size / (test_size + val_size),
                                                                  random_state=42)
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
    def save_dataframes(base_path, randomize_source_ip, dfs, g):
        """
        Saves train/validation/test dataframes to pickle files.

        Parameters
        ----------
        base_path : str
            Base file path (without split suffix). The final filenames will be:
            ``<base_path>-train.pkl``, ``<base_path>-val.pkl``, and ``<base_path>-test.pkl``.
        randomize_source_ip : bool
            If True, a ``"-randomized"`` suffix is appended to each output filename.
        dfs : dict of pandas.DataFrame
            Dictionary containing the splits with keys ``'train'``, ``'val'``, and ``'test'``.
        g : int or None, optional
            Random seed for reproducibility.
        Returns
        -------
        None
            The function writes pickle files to disk and returns nothing.

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """
        splits = ['train', 'val', 'test']
        for split in splits:
            df_path = f'{base_path}-{("" if g is None else f"g{g}-")}{split}{("-randomized" if randomize_source_ip else "")}.pkl'
            dfs[split].to_pickle(df_path)

    def set_labels(self, df, multiclass):
        """
        Assigns encoded edge labels in the dataframe, based on binary or multiclass classification.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing at least the columns ``'Label'`` and
            ``'Attack'``. A new column ``'edge_label'`` will be created.
        multiclass : bool
            If True, the ``'Attack'`` column is label-encoded using
            ``sklearn.preprocessing.LabelEncoder`` for multiclass classification.
            If False, the binary ``'Label'`` column is used directly.

        Returns
        -------
        None
            The function updates ``df`` in-place by adding ``'edge_label'`` and
            dropping the original ``'Label'`` and ``'Attack'`` columns.

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """
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

    @staticmethod
    def convert_to_dgl(df, relabel_nodes):
        """
        Converts a given dataframe containing edge features into a DGL-graph.

        Parameters
        ----------
        df : Pandas DataFrame
            Dataframe containing edge information. Must include at least the
            columns ``'IPV4_SRC_ADDR'`` and ``'IPV4_DST_ADDR'`` as source and
            target node identifiers, as well as edge feature columns used to
            build ``'edge_attr'`` and an ``'edge_label'`` column.
        relabel_nodes : bool
            If True, nodes are randomly relabeled before conversion to DGL in
            order to test permutation equivariance.

        Returns
        -------
        dgl_graph : dgl.DGLGraph
            Directed DGL graph constructed from the dataframe. Edge features are
            stored in ``dgl_graph.edata['edge_attr']`` and labels in
            ``dgl_graph.edata['edge_label']``. A dummy one vector is stored in
            ``dgl_graph.ndata['node_attr']`` with dimensionality
            matching the number of edge features.

        Notes
        ------
        This docstring was created with assistance from ChatGPT.
        """
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
