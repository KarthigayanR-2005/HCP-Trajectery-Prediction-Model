import os
import pickle
import numpy as np
import networkx as nx

class WOMDParser:
    """
    Parses Waymo Open Motion Dataset TFRecords or serialized scenarios.
    Extracts scenario_id, agent tracks, map features, and SDC paths.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.mock_file = os.path.join(data_dir, "mock_scenario.pkl")
        self.scenarios = {}
        if os.path.exists(self.mock_file):
            with open(self.mock_file, 'rb') as f:
                self.scenarios = pickle.load(f)
                
    def load_scenario(self, scenario_id):
        """
        Lazy loader for scenarios.
        Complexity: O(1) memory lookup.
        """
        if scenario_id in self.scenarios:
            return self.scenarios[scenario_id]
        
        # If we have real Waymo TFRecord parsing, we would implement it here
        # using waymo_open_dataset.protos.scenario_pb2
        return None
        
    def get_all_scenario_ids(self):
        return list(self.scenarios.keys())

    def build_sdc_route_graph(self, scenario_id):
        """
        Builds sdc_paths graph: for each scenario, extracts all valid SDC future
        route candidates as directed graph nodes/edges (spaced at 2m).
        """
        scenario = self.load_scenario(scenario_id)
        if not scenario:
            return None
            
        sdc_paths = scenario.get("sdc_paths", [])
        G = nx.DiGraph()
        
        node_id = 0
        for path_idx, path in enumerate(sdc_paths):
            prev_node = None
            for pt_idx, pt in enumerate(path):
                # Node attributes: coordinates, path source index
                attr = {
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "path_idx": path_idx,
                    "pt_idx": pt_idx
                }
                curr_node = f"p{path_idx}_n{pt_idx}"
                G.add_node(curr_node, **attr)
                
                if prev_node is not None:
                    # Compute edge length (2m spacing)
                    dist = np.linalg.norm(pt - np.array([G.nodes[prev_node]["x"], G.nodes[prev_node]["y"]]))
                    G.add_edge(prev_node, curr_node, weight=float(dist))
                prev_node = curr_node
                
        return G

if __name__ == "__main__":
    parser = WOMDParser(os.path.join(os.path.dirname(__file__), "waymo"))
    ids = parser.get_all_scenario_ids()
    print(f"Loaded {len(ids)} scenarios from Waymo cache.")
    if ids:
        graph = parser.build_sdc_route_graph(ids[0])
        print(f"Built SDC route graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
