from configuration import *

def load_data(multiclass=False, randomize_source_ip=True, test_size=0.2, val_size=0.1):

    # Locate data
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    file_path = os.path.join(data_path, 'NF-BoT-IoT-v2.csv')

    # Load data in Pandas Dataframe (only load the first 1000 rows now to allow small test runs)
    df = pd.read_csv(file_path, nrows=100000)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Random mapping of IP addresses
    if randomize_source_ip:
        lower = int(ipaddress.IPv4Address('172.16.0.1'))
        upper = int(ipaddress.IPv4Address('172.31.0.1'))
        ip_ints = np.random.randint(low=lower, high=upper+1, size=len(df), dtype=np.uint32)
        # This loop might be slow, might need to rewrite this for larger datasets using numpy string operations or so
        ips = [str(ipaddress.IPv4Address(int(i))) for i in ip_ints]
        df["IPV4_SRC_ADDR"] = ips

    # Merge information about SRC and DST
    df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].astype(str) + ':' +  df["L4_SRC_PORT"].astype(str)
    df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].astype(str) + ':' + df["L4_DST_PORT"].astype(str)
    df.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT'], inplace=True)

    # Select numeric columns / categorical columns which are already transformed to numbers
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = numeric_cols.drop('Label')       # Drop label, we want to keep those entries {0, 1}

    # Normalize the previously extracted numerical data
    # ––––––––––––––––––––––––––––––––––––––––
    # I suggest using sklearn because this would also allow us to impute missing values
    # in more advanced pipelines or easily exchange the scaler (e.g., scale to [0, 1] or so).
    numeric_data = df[numeric_cols]
    scaler = sk.preprocessing.StandardScaler().fit(numeric_data)
    numeric_data_normalized = scaler.transform(numeric_data)
    df[numeric_cols] = numeric_data_normalized

    # Split the data into train, test and validation datasets.
    # Use a stratified split to ensure attack types and benign traffic are represented equally
    train_df, test_df = sk.model_selection.train_test_split(df, stratify=df['Attack'], test_size=test_size+val_size, shuffle=True)
    test_df, val_df = sk.model_selection.train_test_split(test_df, stratify=test_df['Attack'], test_size=val_size/(test_size+val_size))

    # Create Networkx graphs
    g = nx.from_pandas_edgelist(df, source='IPV4_SRC_ADDR', target='IPV4_DST_ADDR', edge_attr=True, create_using=nx.DiGraph())

    # Convert to PyG, group numerical features as edge attributes.
    # These attributes are the ones that we can later pass to the GNN for learning.
    pyg_graph = torch_geometric.utils.convert.from_networkx(g, group_edge_attrs=numeric_cols.tolist())

    # Add all-one-vectors as "dummy" node attributes
    # E-Graphsage-paper: "[…] dimension of all one constant vector is the same as the number of edge features."
    pyg_graph.node_attr = torch.ones(pyg_graph.num_nodes, pyg_graph.edge_attr.shape[-1])

    # Set the edge labels depending on binary or multiclass classification
    # multiclass = True
    if multiclass:
        # Use sklearn for one-hot encoding of the strings to calculate CE-Loss later.
        labels = np.array(pyg_graph.Attack).reshape(-1, 1)
        ordinal_encoder = sk.preprocessing.OrdinalEncoder()
        ordinal_encoded = ordinal_encoder.fit_transform(labels)

        pyg_graph.edge_label = torch.tensor(ordinal_encoded.squeeze(-1), dtype=torch.long)

        # The code below performs one-hot encoding. However, the PyG-train-test-split does not
        # work with one-hot encodings
        # labels = np.array(pyg_graph.Attack).reshape(-1, 1)
        # one_hot_encoder = sk.preprocessing.OneHotEncoder()
        # one_hot_encoded = one_hot_encoder.fit_transform(labels).toarray()
        #
        # pyg_graph.edge_label = one_hot_encoded
    else:
        pyg_graph.edge_label = pyg_graph.Label

    del pyg_graph.Label, pyg_graph.Attack

    print(pyg_graph)

    # train_test_split = torch_geometric.transforms.RandomLinkSplit(is_undirected=False,
    #                                                               num_val=0.1, num_test=0.2,
    #                                                               key ='edge_label',
    #                                                               add_negative_train_samples=False)
    #
    # train_data, val_data, test_data = train_test_split(pyg_graph)
    #
    # print(train_data)
    # TODO
    #   – Check whether this form of splitting training and test data is suitable for the given task

load_data()
