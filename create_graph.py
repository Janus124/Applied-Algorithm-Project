import json
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
from collections import Counter

# Load the data from JSON files
file_path_ways = 'data/aquitaine.json'
file_path_services = 'data/aquitaine-services.json'

with open(file_path_ways, 'r') as file:
    aquitaine_ways_data = json.load(file)

with open(file_path_services, 'r') as file:
    aquitaine_services_data = json.load(file)

# Extract nodes and ways from the highway data
all_nodes = [element for element in aquitaine_ways_data['elements'] if element['type'] == 'node']
ways_in_aquitaine_ways = [element for element in aquitaine_ways_data['elements'] if element['type'] == 'way']
node_positions = {node['id']: (node['lon'], node['lat']) for node in all_nodes}

# Extract the first node of each service station
service_stations_nodes = [element for element in aquitaine_services_data['elements'] if element['type'] == 'node']
service_stations = [element['nodes'][0] for element in aquitaine_services_data['elements'] if element['type'] == 'way']
service_node_positions = {node['id']: (node['lon'], node['lat']) for node in service_stations_nodes}

# Identifying nodes that appear in multiple highway ways
node_appearances = Counter(node for way in ways_in_aquitaine_ways for node in way['nodes'])
junction_nodes = {node for node, count in node_appearances.items() if count > 1}

# Haversine formula function
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Find the nearest highway node for each service station node
def find_nearest_highway_node(station_node_id, highway_nodes, node_positions, service_node_positions):
    min_distance = float('inf')
    nearest_highway_node_id = None
    station_lat, station_lon = service_node_positions[station_node_id]

    for highway_node_id in highway_nodes:
        highway_lat, highway_lon = node_positions[highway_node_id['id']]
        distance = haversine_distance(station_lat, station_lon, highway_lat, highway_lon)
        if distance < min_distance:
            min_distance = distance
            nearest_highway_node_id = highway_node_id
    return nearest_highway_node_id

# Creating the initial junction graph
junction_graph = nx.Graph()

for junction_node in junction_nodes:
    junction_graph.add_node(junction_node, pos=node_positions.get(junction_node, (0, 0)))

# Adding edges between junctions
for way in ways_in_aquitaine_ways:
    way_nodes = way['nodes']
    junction_nodes_in_way = [node for node in way_nodes if node in junction_nodes]

    for i in range(len(junction_nodes_in_way) - 1):
        junction_graph.add_edge(junction_nodes_in_way[i], junction_nodes_in_way[i + 1])

nearest_node_mapping={}

# Update the graph by adding service stations and connecting them to their nearest highway nodes
for station_node_id in service_stations:
    nearest_highway_node_id = find_nearest_highway_node(station_node_id, all_nodes, node_positions, service_node_positions)
    if nearest_highway_node_id is not None:
        junction_graph.add_node(nearest_highway_node_id['id'], pos=(nearest_highway_node_id['lon'], nearest_highway_node_id['lat']))
        # Store the mapping
        nearest_node_mapping[nearest_highway_node_id['id']] = station_node_id
            
with open('mapping.json', 'w') as file:
    json.dump(nearest_node_mapping, file)
    
non_junction_nodes = set(junction_graph.nodes()) - junction_nodes
# print(non_junction_nodes)

positions = nx.get_node_attributes(junction_graph, 'pos')

# plt.figure(figsize=(10, 8))

# nx.draw_networkx_nodes(junction_graph, pos=positions, nodelist=junction_nodes, node_size=15, node_color='yellow')
# nx.draw_networkx_nodes(junction_graph, pos=positions, nodelist=non_junction_nodes, node_size=15, node_color='blue')

# nx.draw_networkx_edges(junction_graph, pos=positions, edge_color='red', width=2, alpha=0.5)

# plt.title("Graph of Highway Junctions")
# plt.axis('off')
# plt.show()


# --------------------------------------graph simplification with clustering--------------------------------------

# Haversine formula to calculate the distance between two lat-lon points on the Earth
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Function to determine if two nodes are within a certain radius
def nodes_within_radius(node1, node2, radius, node_positions):
    lat1, lon1 = node_positions[node1]
    lat2, lon2 = node_positions[node2]
    return haversine_distance(lat1, lon1, lat2, lon2) <= radius
# Assuming non_junction_nodes is a set or list of non-junction node IDs

def cluster_nodes(graph, radius, node_positions, non_junction_nodes):
    clusters = {}
    node_to_cluster = {}
    cluster_is_non_junction = {}  # Track whether a cluster contains non-junction nodes

    for node in graph.nodes():
        if node in node_to_cluster:
            continue

        cluster_id = len(clusters)
        clusters[cluster_id] = [node]
        node_to_cluster[node] = cluster_id
        cluster_is_non_junction[cluster_id] = node in non_junction_nodes

        for other_node in graph.nodes():
            if other_node != node and nodes_within_radius(node, other_node, radius, node_positions):
                clusters[cluster_id].append(other_node)
                node_to_cluster[other_node] = cluster_id
                if other_node in non_junction_nodes:
                    cluster_is_non_junction[cluster_id] = True

    simplified_graph = nx.Graph()

    for cluster_id, cluster_nodes in clusters.items():
        rep_node = cluster_nodes[0]
        simplified_graph.add_node(rep_node, is_non_junction=cluster_is_non_junction[cluster_id])
        
        for node1, node2 in graph.edges():
            cluster1 = node_to_cluster[node1]
            cluster2 = node_to_cluster[node2]
            if cluster1 != cluster2:
                rep_node1 = clusters[cluster1][0]
                rep_node2 = clusters[cluster2][0]
                simplified_graph.add_edge(rep_node1, rep_node2)

    return simplified_graph, node_to_cluster, cluster_is_non_junction


# Set a radius value for clustering
radius = 5  # Set your clustering radius in kilometers
simplified_graph, node_clusters, cluster_is_non_junction = cluster_nodes(junction_graph, radius, node_positions, non_junction_nodes)

connected_components = sorted(nx.connected_components(simplified_graph), key=len, reverse=True)
# Select the largest connected component
largest_component = connected_components[0]
# Create a new graph from the largest connected component
largest_subgraph = simplified_graph.subgraph(largest_component).copy()


def remove_nodes_connect_neighbors(graph, nodes_to_remove):
    for node in nodes_to_remove:
        neighbors = list(graph.neighbors(node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if not graph.has_edge(neighbors[i], neighbors[j]):
                    graph.add_edge(neighbors[i], neighbors[j])
        graph.remove_node(node)

        
new_junctions = [node for node in largest_subgraph.nodes() if not largest_subgraph.nodes[node].get('is_non_junction', False)]
remove_nodes_connect_neighbors(largest_subgraph, new_junctions)

# Determine node colors based on cluster type
# node_colors = ['green' if largest_subgraph.nodes[node]['is_non_junction'] else 'red' for node in largest_subgraph.nodes()]
node_colors = ['green' if largest_subgraph.nodes[node].get('is_non_junction', True) else 'red' for node in largest_subgraph.nodes()]
simplified_positions = {node: node_positions[node] for node in largest_subgraph.nodes()}

plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(largest_subgraph, pos=simplified_positions, node_size=10, node_color=node_colors)
nx.draw_networkx_edges(largest_subgraph, pos=simplified_positions, alpha=0.5)

plt.title("Simplified Graph with Differentiated Clusters")
plt.axis('off')
plt.show()

nx.write_graphml(largest_subgraph, 'aquitaine-final-5km.graphml')

with open('aquitaine-final-5km.json', 'w') as file:
    json.dump(simplified_positions, file)
    

# with open('pos.json', 'r') as file:
#     loaded_node_positions = json.load(file)

# # Convert position values from strings to tuples (if necessary)
# loaded_graph = nx.read_graphml('zebi.graphml')
# loaded_node_positions = {node: tuple(map(float, pos)) for node, pos in loaded_node_positions.items()}

# # positions = nx.spring_layout(loaded_graph)  # or any other layout algorithm

# # Visualization using the loaded positions
# plt.figure(figsize=(10, 8))
# nx.draw_networkx_nodes(loaded_graph, pos=loaded_node_positions, node_size=20)
# nx.draw_networkx_edges(loaded_graph, pos=loaded_node_positions, alpha=0.5)
# plt.title("Graph Visualization with Loaded Positions")
# plt.axis('off')
# plt.show()