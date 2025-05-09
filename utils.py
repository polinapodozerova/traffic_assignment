import osmnx as ox
import openmatrix as omx
import numpy as np
import pandas as pd
import os

from typing import Dict, Tuple

# Function to import OMX matrices
def import_matrix(matfile):
    f = open(matfile, 'r')
    all_rows = f.read()
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        orig = int(orig[0])

        d = [eval('{'+a.replace(';', ',').replace(' ', '') + '}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[orig] = destinations
    zones = max(matrix.keys())
    mat = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            # We map values to a index i-1, as Numpy is base 0
            mat[i, j] = matrix.get(i + 1, {}).get(j + 1, 0)

    index = np.arange(zones) + 1

    myfile = omx.open_file('demand.omx', 'w')
    myfile['matrix'] = mat
    myfile.create_mapping('taz', index)
    myfile.close()


def create_network_df(network_name="SiouxFalls", root='/home/polina/kans/traffic_assignment/TransportationNetworks'):
    # Importing the networks into a Pandas dataframe consists of a single line of code
    # but we can also make sure all headers are lower case and without trailing spaces

    netfile = os.path.join(root, network_name, network_name + '_net.tntp')
    net = pd.read_csv(netfile, skiprows=8, sep='\t')

    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed

    # And drop the silly first andlast columns
    net.drop(['~', ';'], axis=1, inplace=True)
    
    return net

def prepare_network_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготавливает данные сети из DataFrame в матричную форму.
    
    Args:
        df: DataFrame с колонками init_node, term_node, capacity, free_flow_time
        
    Returns:
        Кортеж (T_0, C, D), где:
        T_0 - матрица начальных времен проезда
        C - матрица пропускных способностей
        D - матрица спроса (пока нулевая, нужно заполнить отдельно)
    """
    nodes = sorted(set(df['init_node']).union(set(df['term_node'])))
    n = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}
    
    T_0 = np.zeros((n, n))
    C = np.zeros((n, n))
    
    for _, row in df.iterrows():
        i = node_index[row['init_node']]
        j = node_index[row['term_node']]
        T_0[i, j] = row['free_flow_time']
        C[i, j] = row['capacity']

    return T_0, C


def generate_od_matrices(base_matrix, num_matrices, condition='uncogested'):
    matrices = []
    for _ in range(num_matrices):
        random_factors = np.random.uniform(0.5, 1.5, size=base_matrix.shape)
        # if condition == 'uncongested':
        #     # Flow-capacity ratio < 0.5 - use lower random factors
        #     random_factors = np.random.uniform(0.1, 0.5, size=base_matrix.shape)
        # elif condition == 'moderate':
        #     # Flow-capacity ratio between 0.4-0.8 - use medium random factors
        #     random_factors = np.random.uniform(0.4, 0.8, size=base_matrix.shape)
        # elif condition == 'congested':
        #     # Flow-capacity ratio > 1.0 - use higher random factors
        #     random_factors = np.random.uniform(0.8, 1.5, size=base_matrix.shape)
        # else:
        #     raise ValueError("Invalid condition. Use 'uncongested', 'moderate', or 'congested'")

        new_matrix = base_matrix * random_factors
        new_matrix = new_matrix.astype(int)
        matrices.append(new_matrix)

    return matrices

import numpy as np

def generate_capacity_matrices(base_capacities, num_matrices, disruption_level='L'):
    matrices = []
    
    for _ in range(num_matrices):
        if disruption_level == 'L':
            # Light disruption: 20% max reduction
            scaling_factors = np.random.uniform(0.8, 1.0, size=base_capacities.shape)
        elif disruption_level == 'M':
            # Moderate disruption: 50% max reduction
            scaling_factors = np.random.uniform(0.5, 1.0, size=base_capacities.shape)
        elif disruption_level == 'H':
            # High disruption: 80% max reduction
            scaling_factors = np.random.uniform(0.2, 1.0, size=base_capacities.shape)
        else:
            raise ValueError("Invalid disruption level. Use 'L', 'M', or 'H'")
        
        new_matrix = base_capacities * scaling_factors
        matrices.append(new_matrix)
    
    return matrices


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
