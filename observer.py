""" A script containing multiple Observer objects.

@author Antares
@date   2/11/2018
"""

from collections import Counter

import matplotlib as mpl
import networkx as nx

import os

# ------------------------ MANAGER ------------------------


class ObserverManager(object):
    """ An ObserverManager aggregates multiple observers
    that may need to communicate with one another while
    a simulation is running.
    """

    def __init__(self, observers=[], parameters={}):
        """ Initializes an ObserverManager. If provided
        OBSERVERS, all the observers will be added to the
        ObserverManager.

        _observers:     The observers being managed
        _parameters:    The parameters that are passed in
                        to observer classes upon their
                        resolution.
        _order:         A list of observer NAMEs that denote
                        the order in which observers should
                        be updated. Some may have dependencies
                        on one another.
        _is_terminated: If the ObserverManager has been
                        terminated

        Note that the optional argument PARAMETERS should be
        a dict mapping <observer name>: { <observer constructor params> }
        """
        # Add the observers
        self._observers = {}
        self._parameters = parameters
        self._order = []
        self._is_terminated = False
        self.add_observer_from_list(observers)

    # Observer methods

    def add_observer(self, observer):
        """ Adds the OBSERVER to the manager and associates
        SELF as the manager of the OBSERVER. If the observer
        is not manageable when added to then an exception
        will be raised
        """
        name = observer.NAME
        self._observers[name] = observer
        observer.add_manager(self)

        # Resolve any dependencies for the added observer
        self._resolve_dependencies()
        self._resolve_order()

    def add_observer_from_list(self, observers):
        """ Adds all the OBSERVERS and associates SELF as the
        manager for each observer. Will raise an exception if
        any added observer is not manageable.
        """
        for obs in observers:
            name = obs.NAME
            self._observers[name] = obs
            obs.add_manager(self)

        # Resolve any dependencies for the added observers
        self._resolve_dependencies()
        self._resolve_order()

    def has_observer(self, name):
        """ Returns true if NAME is an observer being
        managed.
        """
        return name in self._observers

    # Simulation workflow methods

    def start(self, graph_harness):
        """ Calls start on each of its observers with
        the argument GRAPH_HARNESS.
        """
        for obs in self._order:
            self._observers[obs].start(graph_harness)

    def observe(self, graph_harness):
        """ Calls observe on GRAPH_HARNESS on each of its
        observers.
        """
        for obs in self._order:
            self._observers[obs].observe(graph_harness)

    def next_tick(self):
        """ Calls next_tick on each of its observers. """
        for obs in self._order:
            self._observers[obs].next_tick()

    def terminate(self):
        """ Calls terminate on each of its observers and
        then terminates itself.
        """
        for obs in self._order:
            self._observers[obs].terminate()
        self._is_terminated = True

    # Data query methods

    def all_data(self):
        """ Returns a dictionary with data from every
        observer.
        """
        all_data = {}
        for obs in self._order:
            all_data[obs] = self._observers[obs]._data
        return all_data

    def get_data(self, name):
        """ Returns data from the observer with NAME. """
        if name not in self._observers:
            raise Exception("Observer {} is not in this manager!".format(name))
        return self._observers[name]._data

    def get_final_data(self, name):
        """ Returns the final data observed by NAME. """
        if name not in self._observers:
            raise Exception("Observer {} is not in this manager!".format(name))
        return self._observers[name]._data['final_data']

    def get_observation(self, name, round_number):
        """ Returns data observed on ROUND_NUMBER from the
        observer with NAME. Returns None if NAME does not
        have data associated with ROUND_NUMBER.
        """
        if name not in self._observers:
            raise Exception("Observer {} is not in this manager!".format(name))
        data = self.get_data(name)
        if round_number not in data['observations']:
            return None
        return data['observations'][round_number]

    # Other methods

    def _resolve_dependencies(self):
        """ Iterates through each observer and adds any
        observer dependencies not currently captured by
        this manager.
        """
        # A function to build the set of dependencies. The
        # argument OBS should always be an instance of the
        # observer class, but DEPEPDENCIES should always
        # be the class.
        def build_dependencies(obs, new_observers):
            new_dep = obs.dependencies()
            for obs_class in new_dep:
                if obs_class.NAME in self._observers:
                    continue

                # Create the observer
                new_obs = None
                if obs_class.NAME in self._parameters:
                    obs_args = self._parameters[obs_class.NAME]
                    new_obs = obs_class(**obs_args)
                else:
                    new_obs = obs_class()

                # Add the new observer and recurse
                new_obs.add_manager(self)
                new_observers[obs_class.NAME] = new_obs
                build_dependencies(new_obs, new_observers)

        # Build the dependency set for all observers
        new_observers = {}
        for name in self._observers:
            build_dependencies(self._observers[name], new_observers)

        # Update the current set of observers
        self._observers.update(new_observers)

    def _resolve_order(self):
        """ Iterates through the observers and recalculates
        the correct order that observers should be updated
        to ensure that dependencies are respected.
        """
        # Add observers as nodes to a DiGraph.
        observer_node = {}
        G = nx.DiGraph()
        for i, name in enumerate(self._observers):
            observer_node[name] = i
            G.add_node(i)
            G.nodes[i]['name'] = name

        # Create the dependency edges
        for name in self._observers:
            curr_node = observer_node[name]
            obs = self._observers[name]

            # Create all the edges x -> what x depends on
            dep_nodes = [observer_node[obs_class.NAME] for obs_class in obs.dependencies()]
            dep_edges = [(curr_node, i) for i in dep_nodes]
            G.add_edges_from(dep_edges)

        # Create the correct update order. The order is
        # reversed since the sink should be iterated first
        order = list(nx.topological_sort(G))
        dependencies = [G.nodes[i]['name'] for i in order]
        dependencies.reverse()
        self._order = dependencies


# ------------------------ OBSERVER -----------------------

# ALL OBSERVERS MUST HAVE THE SAME PATTERN OF self._data
# ALL OBSERVER CONSTRUCTORS MUST HAVE keyword arguments only
# THE DEPENDENCIES method should always return a list of classes


class Observer(object):
    """ An Observer object carries any necessary data-structures
    for the purpose of recording the state of a chip firing
    simulation. It must have an observe(networkx.Graph) method
    which updates internal datastructures based on observations
    made from the networkx.Graph instance.
    """

    def __init__(self):
        """ Initializes an observer. An Observer contains
        the following instance variables.

        _round_number:  The round that the observer is on
        _is_terminated: If the observer has been terminated
        _manager:       The manager managing this observer
        _data:          The data that the observer records
        """
        self._round_number = 0
        self._is_terminated = False
        self._manager = None

        # Initialize the internal data structure
        self._data = {}
        self._data['observations'] = {}
        self._data['final_data'] = {}

    # Management methods

    def dependencies(self):
        """ Returns all class this observer depends on. """
        return []

    def add_manager(self, manager):
        """ Adds MANAGER as a manager for SELF. """
        self._manager = manager

    # Data collection methods

    def start(self, graph_harness):
        """ Stores any state from GRAPH_HARNESS that will
        be useful to have during the entire simulation.
        This method should only be called once at the
        beginning.
        """
        if self._is_terminated:
            raise Exception("Observer already terminated!")

    def observe(self, graph_harness):
        """ Records the necessary state from the GRAPH_HARNESS
        instance GRAPH and the FIRED_VERTEX then updates any
        internal datastructures.
        """
        if self._is_terminated:
            raise Exception("Observer already terminated!")

    def next_tick(self):
        """ Updates any internal datastructures to annotate
        the start of a new iteration of the simulation being
        observed.
        """
        if self._is_terminated:
            raise Exception("Observer already terminated!")

    def terminate(self):
        """ Terminates the observer and calculates any
        culminating statistics from the observations made.
        """
        if self._is_terminated:
            raise Exception('Observer already terminated!')
        self._is_terminated = True


# ---------------- OBSERVER IMPLEMENTATIONS ---------------


# To create an observer you need to override all observation methods
# override manageable if it depends on other observers and provide
# a static variable NAME


class EdgeFiringObserver(Observer):
    """ Keeps track of all the edges that have fired during
    a round of net firings.

    The data dictionary is organized as follows
    {
        'observations': {
            <round_number>: {
                'fired_edges': [<all fired edges>]
            }
        }
        'final_data': { <no data> }
    }

    Note that edges will take the form of (u, v, k) where
    k is the key of the edge if the graph being observed
    is a multi-graph
    """

    NAME = "fired_edges"

    def __init__(self):
        """ Initializes an EdgeFiringObserver. """
        super(EdgeFiringObserver, self).__init__()

        # Create a new data entry
        self._data['observations'][self._round_number] = {}
        self._data['observations'][self._round_number]['fired_edges'] = []

    def next_tick(self):
        """ Increments the current round and creates a new
        data entry.
        """
        super(EdgeFiringObserver, self).next_tick()

        # Create the new entry
        self._round_number += 1
        self._data['observations'][self._round_number] = {}
        self._data['observations'][self._round_number]['fired_edges'] = []

    def observe(self, graph_harness):
        """ Records all edge firings made by firing on the
        FIRING_GRAPH.
        """
        super(EdgeFiringObserver, self).observe(graph_harness)

        # Append all fired edges
        graph = graph_harness.graph()
        fired_node = graph_harness.prev_fire_node()
        round_data = self._data['observations'][self._round_number]['fired_edges']
        try:
            for u, v, k in graph.edges(fired_node, keys=True):
                if graph.edges[u, v, k]['fired'] > 0:
                    round_data.append((u, v, k))
        except TypeError:
            for u, v in graph.edges(fired_node):
                if graph.edges[u, v]['fired'] > 0:
                    round_data.append((u, v))


class NetEdgeFiringObserver(Observer):
    """ Keeps track of all net fired edges during a round
    of net firings.

    The data dictionary is organized as follows
    {
        'observations': {
            <round_number>: {
                'fired_edges': [<all net fired edges>]
            }
        }
        'final_data': { <no data> }
    }
    """

    NAME = "net_fired_edges"

    def dependencies(self):
        """ Returns the list of dependencies. The classes
        that this depends on are the following.

        EdgeFiringObserver: Required for counting the number
                            of net firings.
        """
        return [EdgeFiringObserver]

    def next_tick(self):
        """ Increments the current round and creates a new
        data entry.
        """
        super(NetEdgeFiringObserver, self).next_tick()

        # Build all the net fired edges and increment round
        if not self._manager:
            raise Exception("NetEdgeFiringObserver does not have manager!")

        fired_data = self._manager.get_observation(EdgeFiringObserver.NAME, self._round_number)
        fired_once = {}
        new_entry = {}

        for e in fired_data['fired_edges']:
            rev_e = (e[1], e[0])

            # Build the extended edge if its the harness
            # uses a multigraph
            if len(e) == 3:
                rev_e = (e[1], e[0], e[2])

            # Check if the reverse edge is in fired_once
            fired_once[e] = (e not in fired_once) and (rev_e not in fired_once)
            if rev_e in fired_once:
                fired_once[rev_e] = (e not in fired_once) and (rev_e not in fired_once)

        # Add all the net fired edges
        new_entry['fired_edges'] = [e for e in fired_once if fired_once[e]]

        # Add the data and increment the round number
        self._data['observations'][self._round_number] = new_entry
        self._round_number += 1


class NetEdgeFiringCounter(Observer):
    """ Counts the number of net fired edges furing a round
    of net firings.

    The data dictionary is organized as follows
    {
        'observations': {
            <round_number>: {
                'count': <The number of fired edges>
            }
        }
        'final_data': {
            'total_firings': <the total number of net firings>
        }
    }
    """

    NAME = 'net_fired_edges_counter'

    def dependencies(self):
        """ Returns the list of dependencies. The classes
        that this depends on are the following.

        NetEdgeFiringObserver:  Required for counting the
                                number of net firings.
        """
        return [NetEdgeFiringObserver]

    def next_tick(self):
        """ Increments the current round and creates a new
        data entry.
        """
        super(NetEdgeFiringCounter, self).next_tick()

        # calculate all net fired edges and increment round
        if not self._manager:
            raise Exception("NetEdgeFiringCounter does not have manager!")

        data = self._manager.get_observation(NetEdgeFiringObserver.NAME, self._round_number)
        new_entry = {
            'count': len(data['fired_edges'])
        }

        # Add the data and increment the round number
        self._data['observations'][self._round_number] = new_entry
        self._round_number += 1

    def terminate(self):
        """ Terminates the observer and calculates the
        total number of net firings made.
        """
        if self._is_terminated:
            raise Exception('Observer already terminated!')

        total_firings = 0
        for i in self._data['observations']:
            total_firings += self._data['observations'][i]['count']

        # Set the observed to be terminated
        self._data['final_data']['total_firings'] = total_firings
        self._is_terminated = True


class NodeFiringObserver(Observer):
    """ An observer that records all the nodes that were
    fired in a round of net firings.

    The data dictionary is organized as follows.
    {
        'observations': {
            <round_number>: {
                'fired_nodes': [<nodes that fired chips>]
            }
        }
        'final_data': { <no data> }
    }
    """

    NAME = 'fired_nodes'

    def __init__(self):
        super(NodeFiringObserver, self).__init__()

        # Create the new entry
        self._data['observations'][self._round_number] = {}
        self._data['observations'][self._round_number]['fired_nodes'] = []

    def next_tick(self):
        """ Increments the round number and adds a new list
        in the observations map.
        """
        super(NodeFiringObserver, self).next_tick()

        # Create the new entry
        self._round_number += 1
        self._data['observations'][self._round_number] = {}
        self._data['observations'][self._round_number]['fired_nodes'] = []

    def observe(self, graph_harness):
        """ Appends the layout of the current graph to the
        data list.
        """
        super(NodeFiringObserver, self).observe(graph_harness)

        # Append the previously fired node
        graph = graph_harness.graph()
        fired_node = graph_harness.prev_fire_node()
        round_data = self._data['observations'][self._round_number]
        round_data['fired_nodes'].append(fired_node)


class NodeFiringStat(Observer):
    """ An observer that calculates statistics over the
    fired nodes in the graph.

    The data dictionary is organized as follows.
    {
        'observations': { <no data> }
        'final_data': {
            'total': <the number of times a node fired>
            'max_node': <the node with most firings>
            'max_firing': <the maximum number of firings made by a node>
            'firings_per_node': {
                <node>: <number chip firings>
            }
        }
    }
    """

    NAME = 'fired_nodes_stat'

    def dependencies(self):
        """ Returns the list of dependencies. The classes
        that this depends on are the following.

        NodeFiringObserver: Required to calculate stats on
                            number of times a node fired
                            its chips.
        """
        return [NodeFiringObserver]

    def terminate(self):
        """ Terminates the observer and calculates the
        total number of net firings made.
        """
        if self._is_terminated:
            raise Exception('Observer already terminated!')

        # Get the statistics
        firings_per_node = self._firings_per_node()

        # Get the max node. If no nodes fired, then make it -1
        max_pairs = firings_per_node.most_common(1)
        if len(max_pairs) == 0:
            max_node = -1
            max_firing = -1
        else:
            max_node, max_firing = max_pairs[0]

        total = sum([firings_per_node[node] for node in firings_per_node])

        # Build the final data map
        final_data = {
            'total': total,
            'max_node': max_node,
            'max_firing': max_firing,
            'firings_per_node': dict(firings_per_node)
        }

        # Store the final data and terminate the observer
        self._data['final_data'] = final_data
        self._is_terminated = True

    def _firings_per_node(self):
        """ Returns a collections.Counter mapping nodes to
        the number of times they fired their chips.
        """
        if not self._manager:
            raise Exception("NetEdgeFiringObserver does not have manager!")

        node_data = self._manager.get_data(NodeFiringObserver.NAME)
        node_observations = node_data['observations']

        all_fires = []
        for i in node_observations:
            round_data = node_observations[i]
            all_fires += round_data['fired_nodes']

        return Counter(all_fires)


# --------------------- CHIP OBSERVER ---------------------


class ChipCounter(Observer):
    """ An observer that tracks how many chips are at each
    node during the simulation. Introduces the following
    instance variables.

    _harness:   Stores the graph harness that is being
                observed.

    The data dictionary is stored as follows.
    {
        'observations': {
            <round_number>: {
                'chip_counts': [<number of chips on vertex i>]
            }
        },
        'final_data': { <no data> }
    }
    """

    NAME = 'chip_counter'

    def start(self, graph_harness):
        """ Starts the NetChipObserver by associating the
        GRAPH_HARNESS to self._harness.
        """
        self._harness = graph_harness

    def next_tick(self):
        """ Net fires all the chips in the preceding round.
        Note that the harness is NOT responsible for calling
        next_tick on the chips. The simulation code will
        call it instead.
        """
        super(ChipCounter, self).next_tick()

        # Get the graph information
        graph = self._harness.graph()
        chip_counts = self._harness.chip_counts()

        # Store the counts
        counts = [chip_counts[u] for u in graph.nodes()]
        self._data['observations'][self._round_number] = {}
        self._data['observations'][self._round_number]['chip_counts'] = counts

        # Increment the round number
        self._round_number += 1


class NetChipObserver(Observer):
    """ An observer that tracks the net movement of chips
    during a simulation. Introduces the following instance
    variables.

    _harness:   Stores the graph harness that is being
                observed.
    _graph:     Used to track the movements of the chips

    The data dictionary is stored as follows.
    {
        'observations': { <no data> },
        'final_data': {
            <chip_number>: [<the nodes the chip_number traveled>]
        }
    }

    NOTE:   This isn't really an observer since it changes
            the state of the given GRAPH_HARNESS. That's
            probably bad... but whatever.
    """

    NAME = 'net_chip_history'

    def dependencies(self):
        """ Returns the list of dependencies. The classes
        that this depends on are the following.

        NetEdgeFiringObserver:  Required to determine where
                                to net move the chips.
        """
        return [NetEdgeFiringObserver]

    def start(self, graph_harness):
        """ Starts the NetChipObserver by associating the
        GRAPH_HARNESS to self._harness.
        """
        self._harness = graph_harness

    def next_tick(self):
        """ Net fires all the chips in the preceding round.
        Note that the harness is NOT responsible for calling
        next_tick on the chips. The simulation code will
        call it instead.
        """
        super(NetChipObserver, self).next_tick()

        # Build all the net fired edges and increment round
        if not self._manager:
            raise Exception("NetEdgeFiringObserver does not have manager!")

        # Net fire all the chips
        fired_data = self._manager.get_observation(NetEdgeFiringObserver.NAME, self._round_number)
        for e in fired_data['fired_edges']:
            # Get the chip lists for the vertices adjacent
            # to the edge
            u_chips = self._harness.graph().nodes[e[0]]['chips']
            v_chips = self._harness.graph().nodes[e[1]]['chips']

            # Fire the chip
            chip = u_chips.pop(0)
            chip.fire(e[1])
            v_chips.append(chip)

        # Increment the round number
        self._round_number += 1

    def terminate(self):
        """ Calculates the observations, final data and
        terminates the observer.
        """
        if self._is_terminated:
            raise Exception('Observer already terminated!')

        # Calculate the history of all chips
        data = {i : c.get_history() for i, c in enumerate(self._harness.chips())}
        self._data['final_data'] = data

        # Set terminated to true.
        self._is_terminated = True
