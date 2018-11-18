""" A simulation harness is a wrapper object that takes in
a Networkx.graph instance and wraps it with any functionality
necessary for the intended simulation to function.

When writing a simulation harness, make sure to also store
enough meta-data so that Observer objects can make but not
much more.

@author Antares
@date   3/2/2018
"""

from simulation.utils import UpdatablePQ
from simulation.chip import Chip

import networkx as nx


# ------------ ROUND BASED CHIP FIRING HARNESS ------------


class RoundBasedChipFiring(object):
    """ A Graph harness for a round based chip firing
    simulation. Each round consists of firing all firable
    vertices once.
    """

    def __init__(self, graph, chip_counts={}):
        """ Creates a new RoundBasedChipFiring harness. The
        variable CHIP_COUNTS is an option that maps vertex
        to the number of chips to place on that vertex.

        One round of the simulation consists of dropping a
        chip at a vertex then firing all nodes that become
        firable. The rounds continue until the graph
        becomes recurrent.

        _graph: A NetworkX graph for the chip-firing game
                to be executed on.
        _fired: A map from vertex to the number of times
                it has fired a chip.
        _prev:  The previous node fired in the simulation.
        _chips: A list of all the chips currently on the
                graph.
        _queue: An UpdatablePQ mapping vertex to its chips
                count. This is used to determine the next
                node that's fired
        """
        self._graph = graph
        self._fired = {u : 0 for u in graph.nodes()}
        self._prev = None
        self._chips = []
        self._queue = UpdatablePQ()

        # Initialize the graph and set a new round
        self._initialize(chip_counts)
        self.next_tick()

    # Properties

    def chip_counts(self):
        """ Returns a dict mapping node to chip count. """
        return {u : self._graph.nodes[u]['chip_count'] for u in self._graph.nodes()}

    def degrees(self):
        """ Returns the chip counts and degrees. """
        return {u : self._graph.degree[u] for u in self._graph.nodes()}

    def graph(self):
        """ Returns the graph wrapped by this harness. """
        return self._graph

    def chips(self):
        """ Returns a list of all chips on this graph. """
        return self._chips

    def fireable(self, vertex):
        """ Returns true if VERTEX can be fired. This means
        it has at least as many chips as its neighbors.
        """
        return self._graph.nodes[vertex]['chip_count'] >= self._graph.degree(vertex)

    def recurrent(self):
        """ Returns true if the graph is recurrent. """
        return all(self._fired[v] >= 1 for v in self._graph.nodes())

    # Setter and getter methods

    def prev_fire_node(self):
        """ Returns the last vertex that fired its chips. """
        return self._prev

    def next_fire_node(self):
        """ Returns the next vertex that can fire its
        chips or None if no vertices can be fired.
        """
        if len(self._queue) <= 0:
            return None
        vertex = self._queue.smallest()
        if not self.fireable(vertex):
            return None
        return vertex

    def fire_node(self, vertex):
        """ Fire all chips on VERTEX. Can only be done if
        the VERTEX is fireable. Otherwise an exception is
        thrown.
        """
        if not self.fireable(vertex):
            raise Exception("Vertex {} is not fireable!".format(vertex))
        if vertex not in self._graph.nodes():
            raise Exception("Graph does not have vertex {}!".format(vertex))

        # Try looping through neighbors as if the graph is
        # a nx.MultiGraph. If it is not, then treat it as a
        # nx.Graph
        try:
            for u, v, k in self._graph.edges(vertex, keys=True):
                # Update metadata
                self._graph.edges[u, v, k]['fired'] += 1
                self._graph.nodes[u]['chip_count'] -= 1
                self._graph.nodes[v]['chip_count'] += 1
                if v in self._queue:
                    self._queue[v] -= 1
        except TypeError:
            for u, v in self._graph.edges(vertex):
                # Update metadata
                self._graph.edges[u, v]['fired'] += 1
                self._graph.nodes[u]['chip_count'] -= 1
                self._graph.nodes[v]['chip_count'] += 1
                if v in self._queue:
                    self._queue[v] -= 1

        # Increment VERTEX's fired count, make it the previously
        # fired node and remove VERTEX from queue
        self._fired[vertex] += 1
        self._prev = vertex
        self._queue.pop(vertex)

    def drop_chip(self, vertex, chip_count=1):
        """ Adds CHIP_COUNT number of chips to VERTEX
        and, updates the VERTEX's chip count and updates
        self._queue if VERTEX is in the map.
        """
        self._graph.nodes[vertex]['chip_count'] += chip_count
        for i in range(chip_count):
            new_chip = Chip(vertex)
            self._chips.append(new_chip)
            self._graph.nodes[vertex]['chips'].append(new_chip)

        if vertex in self._queue:
            self._queue[vertex] -= chip_count

    def next_tick(self):
        """ Begins a new round by adding all nodes to
        self._queue so that they can be fired again.
        """
        for v in self._graph.nodes():
            if v not in self._queue:
                chip_count = self._graph.nodes[v]['chip_count']
                self._queue[v] = self._graph.degree[v] - chip_count

        # Reset all the chips
        for chip in self._chips:
            chip.next_tick()

        # Reset the fired attribute on the graph
        nx.set_edge_attributes(self._graph, 0, 'fired')

    # Other methods

    def _initialize(self, chip_counts):
        """ Initializes all aspects of the harness. """
        # Initialize the chip counts on the graph
        nx.set_node_attributes(self._graph, 0, 'chip_count')
        for v in self._graph.nodes():
            self._graph.nodes[v]['chips'] = []

        for v in chip_counts:
            self._graph.nodes[v]['chip_count'] = chip_counts[v]

            # Add new chips to the vertex
            for i in range(chip_counts[v]):
                new_chip = Chip(v)
                self._chips.append(new_chip)
                self._graph.nodes[v]['chips'].append(new_chip)

        # Creates the queue for the graph
        for v in self._graph.nodes():
            if v not in self._queue:
                chip_count = self._graph.nodes[v]['chip_count']
                self._queue[v] = self._graph.degree[v] - chip_count

        # Make a "fired" attribute for all edges that count
        # how many chips were fired over that edge
        nx.set_edge_attributes(self._graph, 0, 'fired')


# ------------- CHIP FIRING ON SPECIAL GRAPHS -------------


class ApexBridgeChipFiring(RoundBasedChipFiring):
    """ A Round Based Chip Firing harness where the graph
    is an apex bridge graph. The apex bridge family of
    graphs are those with two clusters separated by
    a bridge node. Every node, including the bridge node
    is then connected to the apex.

    The two clusters are constructed from the G(n, p)
    model.
    """

    def __init__(self, n1, p1, n2, p2, chip_counts={}):
        """ Creates a new ApexBridgeChipFiring graph. This
        contains the following instance variables.

        _n1:        The number of nodes in the first
                    cluster.
        _n2:        The number of nodes in the second
                    cluster.
        _cluster1:  The nodes in the first cluster.
        _cluster2:  The nodes in the second cluster.
        _apex:      The apex node.
        _bridge:    The bridge node.
        """
        self._n1 = n1
        self._n2 = n2

        # Set the graph instance variables
        self._apex = 0
        self._cluster1 = list(range(1, n1 + 1))
        self._cluster2 = list(range(n1 + 1, n1 + n2 + 1))
        self._bridge = n1 + n2 + 1

        # Initialize the graph and call the super-constructor
        graph = self._init_graph(p1, p2)
        super(ApexBridgeChipFiring, self).__init__(graph, chip_counts=chip_counts)

    def _init_graph(self, p1, p2):
        """ Returns an Apex Bridge graph where the first
        cluster is sampled from G(self._n1, P1) and the
        second sampled from G(self._n2, P2).
        """
        # Create a graph with only the apex
        apex_graph = nx.Graph()
        apex_graph.add_node(self._apex)

        # Create a graph with only the bridge
        bridge_graph = nx.Graph()
        bridge_graph.add_node(self._bridge)

        # Create the two cluster graphs
        g1 = nx.erdos_renyi_graph(self._n1, p1)
        g2 = nx.erdos_renyi_graph(self._n2, p2)

        # Create the final graph
        g = nx.disjoint_union(apex_graph, g1)
        g = nx.disjoint_union(g, g2)
        g = nx.disjoint_union(g, bridge_graph)

        # Connect the apex and bridge nodes
        for v in range(1, self._bridge + 1):
            g.add_edge(self._apex, v)

        # Connect the bridge to the two communities
        for v in self._cluster1:
            g.add_edge(self._bridge, v)
        for v in self._cluster2:
            g.add_edge(self._bridge, v)

        return g
