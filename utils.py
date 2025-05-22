from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import osmnx as ox
import openmatrix as omx
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Tuple

pwd = os.getcwd()

def load_inp_outp(directory: str) -> Tuple[np.ndarray, np.ndarray]:
    inputs, outputs = [], []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".pkl"):
            with open(os.path.join(directory, filename), 'rb') as f:
                data = pickle.load(f)
                inputs.append(data['input'])
                outputs.append(data['output'])
    return np.array(inputs), np.array(outputs)

def load_inp_outp_cap(directory: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs, outputs, capacities = [], [], []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".pkl"):
            with open(os.path.join(directory, filename), 'rb') as f:
                data = pickle.load(f)
                inputs.append(data['input'])
                outputs.append(data['output'])
                capacities.append(data['capacity'])
    return np.array(inputs), np.array(capacities), np.array(outputs)

def import_matrix(matfile: str) -> None:
    with open(matfile, 'r') as f:
        blocks = f.read().split('Origin')[1:]
    
    matrix = {}
    for block in blocks:
        lines = block.split('\n')
        orig = int(lines[0])
        destinations = {}
        for line in lines[1:]:
            if line:
                dest = eval('{'+line.replace(';',',').replace(' ','')+'}')
                destinations.update(dest)
        matrix[orig] = destinations
    
    zones = max(matrix.keys())
    mat = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            mat[i,j] = matrix.get(i+1,{}).get(j+1,0)
    
    with omx.open_file('demand.omx', 'w') as myfile:
        myfile['matrix'] = mat
        myfile.create_mapping('taz', np.arange(zones)+1)

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-5
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask])) if mask.any() else 0.0

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {'MSE': [], 'MAE': [], 'R2': [], 'MAPE': []}
    for sample in range(y_true.shape[0]):
        mask = y_true[sample] != 0
        if not mask.any():
            continue
            
        y_t = y_true[sample][mask]
        y_p = y_pred[sample][mask]
        
        metrics['MSE'].append(mean_squared_error(y_t, y_p))
        metrics['MAE'].append(mean_absolute_error(y_t, y_p))
        metrics['R2'].append(r2_score(y_t, y_p))
        metrics['MAPE'].append(safe_mape(y_true[sample], y_pred[sample]))
    
    return {
        'MSE': np.mean(metrics['MSE']) if metrics['MSE'] else np.nan,
        'RMSE': np.sqrt(np.mean(metrics['MSE'])) if metrics['MSE'] else np.nan,
        'MAE': np.mean(metrics['MAE']) if metrics['MAE'] else np.nan,
        'R2': np.mean(metrics['R2']) if metrics['R2'] else np.nan,
        'MAPE': np.mean([m for m in metrics['MAPE'] if abs(m) < 1]),
        'MedianAE': np.median(metrics['MAE']) if metrics['MAE'] else np.nan
    }

def create_network_df(network_name: str = "SiouxFalls", 
                     root: str = os.path.join(pwd, 'TransportationNetworks')) -> pd.DataFrame:
    net = pd.read_csv(
        os.path.join(root, network_name, f"{network_name}_net.tntp"),
        skiprows=8, 
        sep='\t'
    )
    net.columns = [s.strip().lower() for s in net.columns]
    return net.drop(['~', ';'], axis=1)

def prepare_network_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    nodes = sorted(set(df['init_node']).union(set(df['term_node'])))
    node_index = {n:i for i,n in enumerate(nodes)}
    
    T_0 = np.zeros((len(nodes), len(nodes)))
    C = np.zeros_like(T_0)
    
    for _, row in df.iterrows():
        i, j = node_index[row['init_node']], node_index[row['term_node']]
        T_0[i,j] = row['free_flow_time']
        C[i,j] = row['capacity']
    return T_0, C

def generate_od_matrices(base_matrix: np.ndarray, num_matrices: int) -> np.ndarray:
    return np.array([
        (base_matrix * np.random.uniform(0.5, 1.5, base_matrix.shape)).astype(int)
        for _ in range(num_matrices)
    ])

def generate_capacity_matrices(base_capacities: np.ndarray, 
                             num_matrices: int,
                             disruption_level: str = 'L') -> np.ndarray:
    ranges = {
        'L': (0.8, 1.0),
        'M': (0.5, 1.0),
        'H': (0.2, 1.0)
    }
    low, high = ranges.get(disruption_level, (1.0, 1.0))
    return np.array([
        base_capacities * np.random.uniform(low, high, base_capacities.shape)
        for _ in range(num_matrices)
    ])

def load_paired_data(path: str = "data/sioux/uncongested") -> Tuple[np.ndarray, np.ndarray]:
    directory = os.path.join(pwd, path)
    inputs, outputs = [], []
    
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".pkl"):
            with open(os.path.join(directory, filename), 'rb') as f:
                data = pickle.load(f)
                inputs.append(data['input'])
                outputs.append(data['output'])
    
    input_mat = np.array(inputs)
    output_mat = np.array(outputs)
    print(f"Loaded {len(inputs)} samples\nInput shape: {input_mat.shape}\nOutput shape: {output_mat.shape}")
    return input_mat, output_mat


class SaintPetersburgDatasetGenerator:
    def __init__(self, max_nodes: int = 100, seed: int = 40):
        """
        Initialize the dataset generator.

        Parameters:
            max_nodes (int): Maximum number of nodes allowed in each area.
            seed (int): Random seed for reproducibility.
        """
        self.max_nodes = max_nodes
        self.seed = seed
        np.random.seed(self.seed)

    def get_random_area(self):
        """
        Get a random area in Saint Petersburg with <= max_nodes nodes.

        Returns:
            Tuple[str, ox.graph]: Name of the area and its road network graph.
        """
        while True:
            lat = np.round(np.random.uniform(59.8, 60.1), 6)
            lon = np.round(np.random.uniform(30.1, 30.5), 6)
            point = (lat, lon)
            graph = ox.graph_from_point(point, dist=1000, network_type="drive")

            if len(graph.nodes) <= self.max_nodes:
                area_name = f"Area at ({lat:.4f}, {lon:.4f})"
                return area_name, graph

    def generate_od_matrix(self, num_nodes: int) -> np.array:
        """
        Generate a synthetic OD matrix.

        Parameters:
            num_nodes (int): Number of nodes in the area.

        Returns:
            np.array: OD matrix.
        """
        od_matrix = np.random.randint(0, 100, size=(num_nodes, num_nodes))
        np.fill_diagonal(od_matrix, 0)
        return od_matrix

    def process_graph(self, graph: ox.graph) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the graph to extract nodes, edges, and compute capacities and free-flow times.

        Parameters:
            graph (ox.graph): Road network graph.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Nodes and edges GeoDataFrames.
        """
        nodes, edges = ox.graph_to_gdfs(graph)

        try:
            edges["capacity"] = edges["lanes"].apply(lambda x: int(x) if isinstance(x, str) else 1) * 1000
        except KeyError:
            edges["capacity"] = 1000

        edges["free_flow_time"] = edges["length"] / (
            edges["maxspeed"].apply(lambda x: int(x) if isinstance(x, int) else 50) * 1000 / 3600
        )
        return nodes, edges

    def generate_dataset(self, num_areas: int = 10, save_to_csv=False) -> Dict[str, Dict]:
        """
        Generate datasets for multiple random areas.

        Parameters:
            num_areas (int): Number of areas to generate datasets for.

        Returns:
            Dict[str, Dict]: Dictionary containing datasets for each area.
        """
        datasets = {}

        for _ in range(num_areas):
            area_name = None
            try:
                area_name, graph = self.get_random_area()
            except ox._errors.InsufficientResponseError or ValueError:
                while not area_name:
                    area_name, graph = self.get_random_area()

            nodes, edges = self.process_graph(graph)
            num_nodes = len(nodes)
            od_matrix = self.generate_od_matrix(num_nodes)
            datasets[area_name] = {
                "nodes": nodes,
                "edges": edges,
                "od_matrix": od_matrix,
            }
            if save_to_csv:
                edges[["capacity", "free_flow_time"]].to_csv(f"{area_name}_capacities_times.csv", index=False)
            np.savetxt(f"{area_name}_od_matrix.csv", od_matrix, delimiter=",")

            print(f"Generated dataset for {area_name} with {num_nodes} nodes.")

        return datasets

def vect_to_matrix_flows(adj, flows):
    if not isinstance(adj, np.ndarray) or adj.ndim != 2:
        raise ValueError("adj must be a 2D numpy array")
    if not isinstance(flows, np.ndarray) or flows.ndim != 1:
        raise ValueError("flows must be a 1D numpy array")
    
    nnz = np.count_nonzero(adj)
    if len(flows) != nnz:
        raise ValueError(f"Length of flows ({len(flows)}) must match the number of non-zero elements in adj ({nnz})")
    
    flow_matrix = np.zeros_like(adj, dtype=flows.dtype)
    flow_matrix[adj != 0] = flows
    
    return flow_matrix