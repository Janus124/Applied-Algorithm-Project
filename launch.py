import json
import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import math
from collections import Counter
import random
import graph_utils as gu

# gu.display_info(loaded_graph)
# nx.nx_pydot.write_dot(loaded_graph, "graph.dot")

def main():

    # Create the parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    # Add the command argument
    parser.add_argument('-s', '--solver', choices=['KDOM', 'BRUTE', 'BENCH'], help='Select the -s followed by: KDOM, BRUTE, or BENCH')

    # Add the optional graph argument
    parser.add_argument('-g', '--graph', default='france', help='Specify the graph file (default: "france")')

    # Add the k-value argument specific to -KDOM command
    parser.add_argument('-k', '--kvalue', type=int, default=2, help='Value of k for k-dominating set (used with -KDOM)')

    # Parse the arguments
    args = parser.parse_args()
    
    location = 'france'

    raw_positions = "data/graphs/" + location + "-final-5km.json"
    raw_graph = "data/graphs/" + location + "-final-5km.graphml"

    # loading raw data
    loaded_graph, loaded_node_positions = gu.load_graph(raw_positions, raw_graph)
    filtered_positions_path = "data/graphs/filtered_positions.json"
    filtered_graph_path = "data/graphs/filtered_graph.graphml"

    # Check for command and handle accordingly
    if args.graph is not None:
        path = args.graph
        filtered_graph, filtered_positions = gu.load_dot_graph(path)
    else:
        filtered_graph, filtered_positions = gu.load_graph(filtered_positions_path, filtered_graph_path)
        
    if args.solver == 'KDOM':
        if args.kvalue is not None:
            print(f"Running K-Dominating Set Algorithm with k={args.kvalue}...")
            # Code for k-dominating set
            k = args.kvalue
            dominating_set = gu.greedy_k_dominating_set_nx(filtered_graph.copy(), k)
        else:
            parser.error("-KDOM requires -k <KVALUE>")
            
    elif args.solver == 'BRUTE':
        print("Brute force method selected")
        # Code for brute force
        print("Running Brute Force Algorithm...")
        # Your code for BRUTE command
        dominating_set = gu.brute_force_dominating_set(filtered_graph.copy())
    
    elif args.solver == 'BENCH':
        print("Benchmarking selected")
        # Code for benchmarking
        dominating_set = nx.dominating_set(filtered_graph.copy())

    print(gu.is_dominating_set(filtered_graph, dominating_set))
    print(len(dominating_set))
    
    gu.display_graph(loaded_graph, dominating_set, filtered_positions)

if __name__ == "__main__":
    main()
