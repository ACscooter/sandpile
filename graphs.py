""" A script that contains a bunch of Graph utilities.

@author Antares
@date   2/11/2018
"""

from simulation.utils import UpdatablePQ
from abc import ABCMeta, abstractmethod

import matplotlib as mpl
import matplotlib.axes as axes
import networkx as nx
import numpy as np

import math
import collections


# -------------------- GRAPH UTILITIES --------------------


def apex_one_clique(n):
    """ Returns an N-clique with each vertex connected to
    an added apex node.
    """
    g = nx.complete_graph(n)
    g.add_node(n)
    g.add_edges_from([(i, n) for i in range(n)])
    return g

def apex_erdos_renyi(n, p):
    """ Returns a G(N, P) graph with each vertex connected
    to an added apex node.
    """
    g = nx.erdos_renyi_graph(n, p)
    g.add_node(n)
    g.add_edges_from([(i, n) for i in range(n)])
    return g


# --------------- PATH MULTIGRAPH FUNCTIONS ---------------


def constant_path_multigraph(k, d):
    """ Creates a path multi-graph with K nodes. Between
    each node pair there are D multi-edges. Finally, an
    apex node is placed that connects to each vertex on
    the path.
    """
    # Helper draws D edges between each node pair
    def _path_drawer(g):
        for i in range(2, k+1):
            for j in range(d):
                g.add_edge(i-1, i)
    return path_multigraph(k, _path_drawer)


def constant_sink_path_multigraph(k, d):
    """ Creates a path multi-graph with K nodes. Between
    each node pair there are D multi-edges. Finally, an
    apex node is placed that connects to each vertex on
    the path.

    The last pair of nodes have D^2 multi-edges to simulate
    a sink.
    """
    # Helper draws D edges between each node pair
    def _path_drawer(g):
        for i in range(2, k):
            for j in range(d):
                g.add_edge(i-1, i)
        for i in range(d ** 2):
            g.add_edge(k-1, k)
    return path_multigraph(k, _path_drawer)

def digging_for_an_example(k):
    """ Maybe this will help us prove an O(mn) bound. """
    # Helper draws D edges between each node pair
    def _path_drawer(g):
        for i in range(2, k):
            for j in range(k ** 2):
                g.add_edge(i-1, i)
        for i in range(25 * (k ** 3)):
            g.add_edge(k-1, k)
    return path_multigraph(k, _path_drawer)

def digging_for_an_upperbound(k, a, b):
    """ Maybe this will help us prove an O(mn) bound. """
    # Helper draws D edges between each node pair
    def _path_drawer(g):
        for i in range(2, k):
            num_edges = np.random.randint(a, high=b+1)
            print("> Randomly Generated {}".format(num_edges))
            for j in range(num_edges):
                g.add_edge(i-1, i)
        for i in range(25 * (k ** 3)):
            g.add_edge(k-1, k)
    return path_multigraph(k, _path_drawer)

def digging_for_a_counter(d):
    """ Makes a path graph with like 4d nodes on the
    path each with degree d in between except for the
    last vertex which has d^3 edges in between.
    """
    n = 4 * d + 4
    def _path_drawer(g):
        for i in range(2, n):
            for j in range(d):
                g.add_edge(i-1, i)
        for i in range(d ** 5):
            g.add_edge(n-1, n)
    return path_multigraph(n, _path_drawer)

def hill_path(k, d, a):
    """ Creates a path with K nodes. The number of
    edges between pair of nodes starts at D and
    increases by an additive factor of A as you head
    toward the center. As you move away from the center
    it decreases by A.
    """
    def _path_drawer(g):
        # Used to calculate the center. The +1 is so that
        # we dont have to differentiate between odd and
        # even
        mid_bottom = math.floor((k + 1) / 2)
        mid_top = math.ceil((k + 1) / 2)

        # Add upslope edges
        counter = d
        for i in range(2, mid_bottom + 1):
            for j in range(counter):
                g.add_edge(i-1, i)
            counter += a

        # Add center edges
        if mid_bottom != mid_top:
            for j in range(counter):
                g.add_edge(mid_bottom, mid_top)

        # Add downslope edges
        counter -= a
        for i in range(mid_top + 1, k + 1):
            for j in range(counter):
                g.add_edge(i-1, i)
            counter -= a
    return path_multigraph(k, _path_drawer)

def sharp_hill_path(k, d, a):
    """ Creates a hill_path except the central nodes have
    d^3 edges.
    """
    if k < 4:
        raise Exception("Pick a larger k")
    def _path_drawer(g):
        # Used to calculate the center. The +1 is so that
        # we dont have to differentiate between odd and
        # even
        mid_bottom = math.floor((k + 1) / 2)
        mid_top = math.ceil((k + 1) / 2)

        # Add upslope edges
        counter = d
        for i in range(2, mid_bottom):
            for j in range(counter):
                g.add_edge(i-1, i)
            counter += a

        # Add the center edges
        for j in range(d ** 3):
            g.add_edge(mid_bottom-1, mid_bottom)
            g.add_edge(mid_top, mid_top+1)
            if mid_bottom != mid_top:
                g.add_edge(mid_bottom, mid_top)

        # Add downslope edges
        counter -= a
        for i in range(mid_top + 2, k + 1):
            for j in range(counter):
                g.add_edge(i-1, i)
            counter -= a
    return path_multigraph(k, _path_drawer)

def random_sharp_hill(k, d):
    """ Adds a random increasing sequence of edges to the
    center. Then d^3 edges, then a random decreasing
    sequence of edges.
    """
    if k < 4:
        raise Exception("Pick a larger k")
    def _path_drawer(g):
        # Used to calculate the center. The +1 is so that
        # we dont have to differentiate between odd and
        # even
        mid_bottom = math.floor((k + 1) / 2)
        mid_top = math.ceil((k + 1) / 2)
        lower_bound = d
        upper_bound = 5 * (d ** 2)

        # Add upslope edges
        for i in range(2, mid_bottom):
            degree = np.random.randint(lower_bound, upper_bound)
            for j in range(degree):
                g.add_edge(i-1, i)
            lower_bound = degree

        # Add the center edges
        for j in range(d ** 3):
            g.add_edge(mid_bottom-1, mid_bottom)
            g.add_edge(mid_top, mid_top+1)
            if mid_bottom != mid_top:
                g.add_edge(mid_bottom, mid_top)

        # Add downslope edges
        lower_bound = d
        for i in range(mid_top + 2, k + 1):
            degree = np.random.randint(lower_bound, upper_bound)
            for j in range(degree):
                g.add_edge(i-1, i)
            upper_bound = degree
    return path_multigraph(k, _path_drawer)

def degree_list_multigraph(degrees):
    """ Creates a path multi-graph where vertices i, i+1
    degrees[i] edges between.
    """
    def _path_drawer(g):
        for i in range(1, len(degrees) + 1):
            for j in range(degrees[i-1]):
                g.add_edge(i-1, i)
    return path_multigraph(len(degrees) + 1, _path_drawer)

def double_path_multigraph(k):
    """ Creates a path multi-graph with K nodes. Between
    each node there are double the multi-edges than the
    previous node pair. Finally, an apex node is placed
    that connects to each vertex on the path.
    """
    # Helper draws D edges between each node pair
    def _path_drawer(g):
        for i in range(2, k):
            for j in range(d):
                g.add_edge(i-1, i)
        for i in range(d ** 2):
            g.add_edge(k-1, k)
    return path_multigraph(k, _path_drawer)

def random_path_multigraph(k, a, b):
    """ Creates a path multi-graph with K nodes and a
    number chosen u.a.r in [a, b] of multi-edges between
    each node pair.
    """
    # Helper draws D edges between each node pair
    def _path_drawer(g):
        for i in range(2, k+1):
            num_edges = np.random.randint(a, high=b+1)
            print("> Randomly Generated {}".format(num_edges))
            for j in range(num_edges):
                g.add_edge(i-1, i)
    return path_multigraph(k, _path_drawer)

def path_multigraph(k, path_drawer):
    """ Creates a path multi-graph with K nodes. The
    multi-edges between each pair of nodes are decided
    by function PATH_DRAWER which takes in a networkx
    MultiGraph instance and mutates it.
    """
    g = nx.MultiGraph()
    g.add_nodes_from([i for i in range(k+1)])

    # Draw the path edges
    path_drawer(g)

    # Connect the apex
    for i in range(1, k+1):
        g.add_edge(0, i)

    return g


# -------------------- DRAWABLE GRAPHS --------------------


class DrawableGraph(object):
    """ A drawable graph is a graph with a method draw that
    returns a matplotlib.pyplot instance that can be
    rendered
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def graph(self):
        """ Returns the networkx.graph instance. """
        raise NotImplementedError

    @abstractmethod
    def layout(self):
        """ Returns a map from node i to its position in
        the matplotlib plot.
        """
        raise NotImplementedError

    @abstractmethod
    def draw_nodes(self, node_colors=None):
        """ Returns a list of matplotlib PathCollection
        instances representing the nodes of the
        DrawableGraph. An optional argument NODE_COLORS can
        be provided to color the nodes.

        NODE_COLORS is an array where the i-th element
        denotes the i-th node's color. The values are meant
        to be between v_min and v_max of a color_map
        """
        raise NotImplementedError

    @abstractmethod
    def draw_edges(self, highlighted_edges=None):
        """ Returns a list of matplotlib LineCollection
        instances representing the edges of the
        DrawableGraph. An argument HIGHLIGHTED_EDGES can
        be provided to highlight certain nodes.

        HIGHLIGHTED_EDGES is a list of edges in the graph
        """
        raise NotImplementedError

    @abstractmethod
    def draw_labels(self, node_labels):
        """ Returns a list of matplotlib Text instances
        containing the labels for each node. NODE_LABELS
        is a map from node i to its label.
        """
        raise NotImplementedError

    @abstractmethod
    def draw_patches(self):
        """ Returns a list of matplotlib patches that act
        as embelishments to the plot.
        """
        raise NotImplementedError


class ApexBridgeGraph(DrawableGraph):
    """ A wrapper class that contains the networkx.graph
    instance constructed from a single apex node connected
    to a G(n1, p1) component, bridge node, and aG(n2, p2)
    component. The two G(n, p) components have each vertex
    connected to the bridge node.
    """

    def __init__(self, n1, p1, n2, p2):
        """ Creates a new ApexBridgeGraph with parameters
        n1, p1 and n2, p2 into the Erdos Renyi clusters.
        This object uses the following instance variables
        _n1     -   The number of nodes in the first
                    G(n, p) cluster
        _p1     -   The edge probability in the first
                    G(n, p) cluster
        _n2     -   The number of nodes in the second
                    G(n, p) cluster
        _p2     -   The edge probability in the second
                    G(n, p) cluster
        _graph  -   The networkx.graph instance
                    rendered
        _gnp_1  -   List of nodes in the first G(n, p)
                    cluster
        _gnp_2  -   List of nodes in the second G(n, p)
                    cluster
        _bridge -   The bridge node
        _apex   -   The apex node
        """
        self._n1 = n1
        self._n2 = n2
        self._p1 = p1
        self._p2 = p2
        self._init_graph()

    # Initializer methods

    def _init_graph(self):
        """ Initializes the networkx.graph instance. """
        g1 = nx.erdos_renyi_graph(self._n1, self._p1)
        g2 = nx.erdos_renyi_graph(self._n2, self._p2)
        g = nx.disjoint_union(g1, g2)

        # Assign the G(n, p) nodes
        self._gnp_1 = list(range(0, len(g1)))
        self._gnp_2 = list(range(len(g1), len(g1) + len(g2)))

        # Add the bridge node
        self._bridge = len(g1) + len(g2)
        g.add_node(self._bridge)
        g.add_edges_from([(self._bridge, i) for i in range(len(g1) + len(g2))])

        # Add the apex node
        self._apex = len(g1) + len(g2) + 1
        g.add_node(self._apex)
        g.add_edges_from([(self._apex, i) for i in range(len(g1) + len(g2) + 1)])
        self._graph = g

    def graph(self):
        """ Returns the networkx.graph instance. """
        return self._graph

    def layout(self):
        """ Returns a node layout for the graph with the
        clusters placed in a circular layout of radius R1
        and R2 respectively.
        """
        if not self._graph:
            raise Exception("Cannot create layout with empty graph!")

        # Drawing Constants
        r1 = 4 * self._n1
        r2 = 4 * self._n2

        positions = {}
        positions[self._bridge] = np.array([0, 0])
        positions[self._apex] = np.array([0, 0])

        # Assign circular positions
        c1 = position_circular(self._graph, self._gnp_1, r1)
        c2 = position_circular(self._graph, self._gnp_2, r2)
        positions.update(c1)
        positions.update(c2)

        # Move everything to the right place and calculate
        # offsets
        gnp_1_trans = translate_positions(positions, self._gnp_1, -3*r1, 0)
        gnp_2_trans = translate_positions(positions, self._gnp_2, 3*r2, 0)
        apex_trans = translate_positions(positions, [self._apex], 0, -2*r1 - 2*r2)
        offsets = {
            'gnp_1': gnp_1_trans,
            'gnp_2': gnp_2_trans,
            'apex': apex_trans
        }

        return positions, offsets

    def draw_nodes(self, node_colors=None):
        """ Returns a PathCollection object with all the
        drawn nodes as well as the relevent offsets. Will
        color the nodes NODE_COLORS if given. The
        NODE_COLORS is an array where the ith entry
        contains the color of the ith node.
        """
        if not self._graph:
            raise Exception("Cannot draw nodes with an empty graph!")
        # Drawing Constants
        node_cmap = mpl.cm.get_cmap('RdYlBu')
        node_size = 300

        # Draw all the nodes
        layout, offsets = self.layout()
        if not node_colors:
            nodes = nx.draw_networkx_nodes(self._graph, layout,
                node_size=node_size
            )
        else:
            nodes = nx.draw_networkx_nodes(self._graph, layout,
                node_color=node_colors,
                cmap=node_cmap,
                vmin=0.0,
                vmax=1.0,
                node_size=node_size
            )
        return [nodes], offsets

    def draw_edges(self, highlighted_edges=None):
        """ Returns a list of LineCollection objects with
        all the drawn edges. The given list of edges in
        HIGHLIGHTED_EDGES will be highlighted red.
        """
        if not self._graph:
            raise Exception("Cannot draw edges with an empty graph!")

        # Drawing constants
        edge_width = 1.0

        # Draw all the edges
        layout, offset = self.layout()
        collection = []
        edges = nx.draw_networkx_edges(self._graph, layout,
            width=edge_width,
            alpha=0.5
        )
        collection.append(edges)
        if highlighted_edges:
            red_edges = nx.draw_networkx_edges(self._graph, layout,
                edgelist=highlighted_edges,
                edge_color="red",
                width=edge_width * 2,
                alpha=1.0
            )
            collection.append(red_edges)
        return collection

    def draw_labels(self, node_labels):
        """ Returns a list of node labels as matplotlib
        text instances. NODE_LABELS is a map from i to the
        i-th node's label.
        """
        if not self._graph:
            raise Exception("Cannot draw labels with an empty graph!")

        # Drawing constants
        font_size = 7

        # Draw the labels
        layout, offsets = self.layout()
        labels = nx.draw_networkx_labels(self._graph, layout, node_labels,
            font_size=font_size
        )
        return [labels[node] for node in labels]

    def draw_patches(self):
        """ Returns a list of Circle patches that highlight
        the different components of the apex bridge graph.
        """
        if not self._graph:
            raise Exception("Cannot create layout with empty graph!")

        # Drawing constants
        cluster_cmap = mpl.cm.get_cmap('tab20c')
        cluster_spacing = 20.0
        cluster_alpha = 0.5

        # Draw the circles
        layout, offsets = self.layout()

        gnp_1_radius = r1 + cluster_spacing / 2
        gnp_2_radius = r2 + cluster_spacing / 2
        apex_radius = cluster_spacing / 2

        gnp_1_center = offsets['gnp_1']
        gnp_2_center = offsets['gnp_2']
        apex_center = offsets['apex']

        gnp_1_color = mpl.colors.rgb2hex(cluster_cmap(0))
        gnp_2_color = mpl.colors.rgb2hex(cluster_cmap(4))
        apex_color = mpl.colors.rgb2hex(cluster_cmap(8))

        gnp_1_circle = mpl.patches.Circle(gnp_1_center, gnp_1_radius, color=gnp_1_color, alpha=cluster_alpha)
        gnp_2_circle = mpl.patches.Circle(gnp_2_center, gnp_2_radius, color=gnp_2_color, alpha=cluster_alpha)
        apex_circle = mpl.patches.Circle(apex_center, apex_radius, color=apex_color, alpha=cluster_alpha)

        # Return the circles
        return [gnp_1_circle, gnp_2_circle, apex_circle]


# ---------------- GRAPH DRAWING UTILITIES ----------------


def position_circular(graph, vertices, r):
    """ Returns a circular layout for a subset of nodes
    in self._graph with radius R.
    """
    subgraph = graph.subgraph(vertices)
    layout = nx.circular_layout(subgraph, scale=r)
    return layout

def translate_positions(layout, vertices, x, y):
    """ Translates the position of VERTICES in the given
    LAYOUT by (X, Y) and returns the translation matrix.

    NOTE: Mutates LAYOUT
    """
    translate = np.array([x, y])
    for v in vertices:
        layout[v] = layout[v] + translate
    return translate
