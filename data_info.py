from configuration import *



def get_dataframe(dataset, version, print_info=False):
    """
    Loads the dataframe for the specified dataset and version, 
    removing duplicates and NaN valus.
    This function also prints dataset information if specified by flag print_info.
    ------------
    input:
    dataset: str, name of the dataset
    version: int, version number of the dataset
    print_info: bool, whether to print dataset information
    ------------
    output:
    df: pandas DataFrame, the loaded dataframe
    ------------

    Notes:
    - The first time this function is called for a specific dataset and version,
        it reads the CSV file, processes it, and saves it as a pickle file for faster
        loading in subsequent calls.
    """

    data_path = os.path.join(os.getcwd(), 'data', dataset + f'-v{version}')
    df_path = f'{data_path}-whole_dataset.pkl'
    
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        csv_path = data_path + '.csv'
        df = pd.read_csv(csv_path)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df.to_pickle(df_path)

    print(f"Dataframe for {dataset}-v{version} loaded")
    if print_info:
        print("##########################")
        print(f"Number of samples: {len(df)}")
        print("Features:", df.columns.tolist())
        print("Data type per feature:")
        print(df.dtypes)
        print("##########################")
        print("Class distribution:")
        print(df['Attack'].value_counts())
        print("##########################")
        print("Sample row:")
        print(df.head(1).to_string(index=False))
    return df

def process_ip_addresses(df, randomize_source_ip, random_seed):
    """
    Processes the IP addresses in the dataframe.
    If randomize_source_ip is True, it randomizes the source IP addresses.
    Merges source and destination IP addresses with their respective ports.
    ------------
    input:
    df: pandas DataFrame, the input dataframe
    randomize_source_ip: bool, whether to randomize source IP addresses
    ------------
    output:
    df: pandas DataFrame, the processed dataframe
    ------------
    """
    if randomized_source_ip:
        lower = int(ipaddress.IPv4Address('172.16.0.1'))
        upper = int(ipaddress.IPv4Address('172.31.0.1'))
        rng = np.random.default_rng(random_seed)
        ip_ints = rng.integers(low=lower, high=upper, size=len(df), endpoint=True)
        df["IPV4_SRC_ADDR"] = list(map(lambda i: str(ipaddress.IPv4Address(int(i))), ip_ints))
    df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"] + ':' + df["L4_SRC_PORT"].astype(str)
    df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"] + ':' + df["L4_DST_PORT"].astype(str)
    df.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT'], inplace=True)
    return df

def getGraphInfo(graph):
    """
    Retrieves information from the graph
    ------------
    input:
    graph: networkx Graph, the input graph
    """
    print("##########################")
    print("Basic Graph Information:")
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("###########################")
    histo = np.array(nx.degree_histogram(graph))
    nonzero_indices = np.flatnonzero(histo)
    print("Degree Information:")
    print("Degree Frequency:")
    print(histo[nonzero_indices])
    print("Degree:")
    print(nonzero_indices)
    print("###########################")
    try:
        _ = nx.find_cycle(graph)
        print("The graph contains cycles.")

    except nx.exception.NetworkXNoCycle:
        print("The graph is acyclic.")
        print("###########################")
        print("Longest path:", nx.dag_longest_path_length(graph))

def getParameters():
    """
    Returns the relevant parameters for getting data information on the NF-BoT-IoT dataset and 
    associated graph.
    ------------
    output:
    dataset: str, name of the dataset
    version: int, version number of the dataset
    randomized_source_ip: bool, whether to randomize source IP addresses
    print_df_info: bool, whether to print dataframe information
    print_graph_info: bool, whether to print graph information
    random_seed: int, random seed for reproducibility
    ------------
    Notes:
    - For non-randomized IP, the graph contains cycles, so longest path is not computed due to computational constraints.
    - For non-randomized, the random seed is irrelevant.
    ------------

    """
    dataset = "NF-BoT-IoT"
    version = 1
    randomized_source_ip = True
    print_df_info = False
    print_graph_info = True
    random_seed = 34
    return dataset, version, randomized_source_ip, print_df_info, print_graph_info, random_seed

# The code here was based on data.py
if __name__ == "__main__":
    dataset, version, randomized_source_ip, print_df_info, print_graph_info, random_seed = getParameters()

    df = get_dataframe(dataset, version, print_df_info)
    df = process_ip_addresses(df, randomized_source_ip, random_seed)
    g = nx.from_pandas_edgelist(df, source='IPV4_SRC_ADDR', target='IPV4_DST_ADDR', edge_attr=True,
                                    create_using=nx.DiGraph())
    if print_graph_info:
        getGraphInfo(g)

    