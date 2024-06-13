import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy


class Graph:

    def __init__(self, adjacency_list, patches_filled, c):
        self.c = c

        self.adj = adjacency_list   # neighbours for each node, -1 means no neighbour
        self.mask = patches_filled  # mask to determine if a node is considered

        # mask with only the nodes that actually have an edge (except food nodes)
        food_nodes = self.food_nodes()
        self.actual_mask = np.array([
            node in food_nodes or any([neighbour != -1 for neighbour in neighbours])
            for node, neighbours in enumerate(self.adj)
        ])
        self.nodes = list(np.where(self.actual_mask)[0])  # indices of nodes in adj list

        # save whether the network is fully connected
        self.connected = self.fullyconnected()


    def coord_to_node(self, coord):
        """Convert a coordinate (of food) to the node index in the adjacency list."""
        # make sure to use a hashable coordinate type
        coord = tuple(coord)

        # use a cache dict to prevent the expensive lookup whenever possible
        try:
            if coord in self.coord_to_node_cache:
                return self.coord_to_node_cache[coord]
        except AttributeError:
            self.coord_to_node_cache = dict()

        # lookup the coordinate in the big coordinate list
        for index, c in enumerate(self.c.all_coordinates_unscaled):
            if np.array_equal(coord, c):
                self.coord_to_node_cache[coord] = index
                return index

        raise LookupError


    def food_nodes(self):
        """Give a list of all adjacency list indices corresponding to food sources."""
        return list(map(self.coord_to_node, self.c.foods_unscaled))


    def neighbours(self, node):
        return self.adj[node][self.adj[node] != -1]


    def cost(self):
        """The cost of the graph is defined as the number of edges."""
        return (self.adj >= 0).sum() // 2


    def fullyconnected(self, only_foods=False):
        """Check if the graph is fully connected."""
        nodes_out = self.food_nodes() if only_foods else copy.deepcopy(self.nodes)
        visited = []
        canvisit = [nodes_out.pop()]

        while len(nodes_out) > 0 and len(canvisit) != 0:
            node = canvisit.pop()
            visited.append(node)

            for neighbour in self.neighbours(node):
                if not neighbour in canvisit and not neighbour in visited:
                    canvisit.append(neighbour)
                    if neighbour in nodes_out:
                        nodes_out.remove(neighbour)

        return len(nodes_out) == 0


    def mst(self, G, terminal_nodes):
        """Compute the size of a minimum rectilinear steiner spanning tree
        between the terminal nodes."""
        steiner_tree = nx.algorithms.approximation.steiner_tree(G, terminal_nodes)
        return steiner_tree.number_of_edges()


    def mst_actual(self):
        """Compute the size of a minimum spanning tree between the food sources."""
        assert self.connected, 'only compute the actual MST for fully connected graphs'
        return self.mst(nx.Graph(self.edges()), self.food_nodes())


    def mst_perfect(self):
        """Compute the size of a minimum spanning tree between the food sources,
        NOT considering the actual network!"""
        # TODO remove this when the experiments are ran after the 13th of June
        self.c.height_unscaled = 2 * self.c.wall_num_height + 1
        self.c.width_unscaled = 2 * self.c.wall_num_width + 1

        # construct a 2d grid graph
        G = nx.grid_2d_graph(range(self.c.height_unscaled), range(self.c.width_unscaled))

        # remove nodes where there are walls
        for y in range(self.c.height_unscaled):
            for x in range(self.c.width_unscaled):
                if y % 2 == 1 and x % 2 == 1:
                    G.remove_node((y, x))

        # relabel the nodes from coordinates to node indices
        nx.relabel_nodes(G, {coord: self.coord_to_node(coord) for coord in G.nodes}, copy=False)
        return self.mst(G, self.food_nodes())


    def edges(self):
        """Return a list of all edges in the graph as [(node1, node2)]"""
        edges = set()

        for node in range(len(self.adj)):
            neighbours = self.neighbours(node)

            for neighbour in neighbours:
                edge = tuple(sorted([node, neighbour]))

                if edge not in edges:
                    edges.add(edge)

        return edges


    def fault_tolerance(self):
        """Percentage of edges that, when removed, disconnect any of the food sources."""
        assert self.connected, 'only compute fault tolerance for fully connected graphs'

        num_connected = 0
        edges = self.edges()

        for node1, node2 in edges:
            node1_index_node2 = np.where(self.adj[node1] == node2)[0][0]
            node2_index_node1 = np.where(self.adj[node2] == node1)[0][0]

            # remove the edge
            self.adj[node1][node1_index_node2] = -1
            self.adj[node2][node2_index_node1] = -1

            # check connectedness
            if self.fullyconnected(only_foods=True):
                num_connected += 1

            # restore the edge
            self.adj[node1][node1_index_node2] = node2
            self.adj[node2][node2_index_node1] = node1

        return num_connected / len(edges)