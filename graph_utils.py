import json
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import math
from collections import Counter
import random

with open('data/mapping.json', 'r') as file:
        nearest_node_mapping = json.load(file)
            
def get_original_station_id(mapped_node_id, nearest_node_mapping):
    return nearest_node_mapping.get(mapped_node_id, None)

def load_graph(positions, graph):

    with open(positions, 'r') as file:
        loaded_node_positions = json.load(file)
    # Convert position values from strings to tuples (if necessary)
    loaded_graph = nx.read_graphml(graph)
    loaded_node_positions = {node: tuple(map(float, pos)) for node, pos in loaded_node_positions.items()}
    # positions = nx.spring_layout(loaded_graph)  # or any other layout algorithm
    return loaded_graph, loaded_node_positions

def load_dot_graph(graph):
    loaded_graph = nx.nx_pydot.read_dot(graph)
    positions = nx.spring_layout(loaded_graph)
    return loaded_graph, positions

def save_graph(graph, positions, graph_name, positions_name):
    base_path = 'data/graphs/'
    graphml_file_path = base_path + graph_name + '.graphml'
    nx.write_graphml(graph, graphml_file_path)
    
    with open(base_path + positions_name + '.json', 'w') as file:
        json.dump(positions, file)
    
def display_graph(graph, special_nodes, node_positions):
    # Create a list of colors for the nodes
    node_colors = ['blue' if node in special_nodes else 'red' for node in node_positions]
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(graph, pos=node_positions, node_size=5, node_color=node_colors)
    nx.draw_networkx_edges(graph, pos=node_positions, alpha=0.5, edge_color='grey')
    plt.title("Graph Visualization with Highlighted Nodes")
    plt.axis('off')
    plt.show()

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def find_degeneracy(graph):
    # Create a copy of the graph so the original graph is not modified
    G = graph.copy()
    
    degeneracy = 0
    while G.number_of_nodes() > 0:
        # Find the node with the minimum degree
        min_degree_node = min(G.degree, key=lambda x: x[1])[0]
        
        # Update the degeneracy if necessary
        degeneracy = max(degeneracy, G.degree[min_degree_node])
        
        # Remove the node with the minimum degree
        G.remove_node(min_degree_node)
    
    return degeneracy

def display_info(graph):
    max_degree = max(dict(graph.degree()).values())
    num_edges = graph.number_of_edges()
    num_vertices = graph.number_of_nodes()
    
    graph_degeneracy = find_degeneracy(graph)
    is_degenerate = graph_degeneracy < max(dict(graph.degree()).values())
    
    original_station_id = get_original_station_id("23240264", nearest_node_mapping)
    print(f"Maximum Degree: {max_degree}")
    print(f"Number of Edges: {num_edges}")
    print(f"Number of Vertices (Nodes): {num_vertices}")
    
    print(f"Graph Degeneracy: {graph_degeneracy}")
    print(f"Is the graph degenerate? {'Yes' if is_degenerate else 'No'}")
    
    print(f"Original service station node ID: {original_station_id}")

def update_graph_with_distances(graph, node_positions_file):
    # Load node positions from JSON file
    with open(node_positions_file, 'r') as file:
        node_positions = json.load(file)

    # Update the graph with edge weights based on haversine distance
    for u, v in graph.edges():
        if u in node_positions and v in node_positions:
            lat1, lon1 = node_positions[u]
            lat2, lon2 = node_positions[v]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            graph[u][v]['weight'] = distance

def filter_graph_by_distance(graph, max_distance):
    """
    Apply Floyd-Warshall algorithm on the graph and keep edges with a distance <= max_distance.
    Return a NetworkX graph.
    """
    # Run Floyd-Warshall algorithm to find shortest paths
    path_lengths = nx.floyd_warshall(graph)

    # Create a new NetworkX graph
    filtered_graph = nx.Graph()

    # Add edges based on distance
    for node1 in path_lengths:
        for node2, distance in path_lengths[node1].items():
            if node1 != node2 and distance <= max_distance:
                filtered_graph.add_edge(node1, node2)

    return filtered_graph

def greedy_dominating_set_nx(graph):
    not_covered = set(graph.nodes)
    dominating_set = []

    while not_covered:
        max_len = -1
        max_node = None

        for node in graph.nodes:
            curr_len = sum(1 for neighbor in graph.neighbors(node) if neighbor in not_covered)
            if curr_len > max_len:
                max_len = curr_len
                max_node = node

        if max_node is not None:
            dominating_set.append(max_node)
            not_covered.discard(max_node)
            for neighbor in graph.neighbors(max_node):
                not_covered.discard(neighbor)

            # Remove the node and its edges
            graph.remove_node(max_node)

    # print(f"dominating set: {dominating_set}")
    return dominating_set

def greedy_k_dominating_set_nx(graph, k):
    not_covered = set(graph.nodes)
    dominating_set = []

    while not_covered:
        max_cover = -1
        max_node = None

        for node in graph.nodes:
            cover = sum(1 for neighbor in nx.single_source_shortest_path_length(graph, node, cutoff=k) if neighbor in not_covered)
            if cover > max_cover:
                max_cover = cover
                max_node = node

        if max_node is not None:
            dominating_set.append(max_node)
            for neighbor in nx.single_source_shortest_path_length(graph, max_node, cutoff=k):
                not_covered.discard(neighbor)

            # Remove the node and its edges
            graph.remove_node(max_node)

    return dominating_set

def brute_force_dominating_set(graph):
    all_vertices = list(graph.nodes)
    dominating_set = None
    min_dominating_size = float('inf')

    for i in range(2**len(all_vertices)):
        candidate_set = {vertex for j, vertex in enumerate(all_vertices) if (i >> j) & 1}

        if nx.is_dominating_set(graph, candidate_set) and len(candidate_set) < min_dominating_size:
            dominating_set = candidate_set
            min_dominating_size = len(candidate_set)

    return dominating_set

def is_dominating_set(graph, dominating_set, printing=False):
    #get all vertices of the graph
    graph_vertices = set(graph.nodes())

    # Create a set to keep track of uncovered vertices
    uncovered_vertices = set(graph_vertices)

    # Iterate over the dominating set
    for vertex in dominating_set:
        if vertex in graph_vertices:
            # Remove the current vertex and its neighbors from the uncovered set
            uncovered_vertices.discard(vertex)
            uncovered_vertices.difference_update(graph[vertex])
        #else:
            #print("Waring: {vertex} is not in the graph")

    # If all vertices are covered, it's a valid Dominating Set
    '''if uncovered_vertices:
        print(f"#002 uncovered_vertices: {uncovered_vertices}")
    '''
    return not bool(uncovered_vertices)

