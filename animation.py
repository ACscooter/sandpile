""" This file contains a number of rendering and animation
observer implementations. Code that renders the graph as an
image or animation is implemented as an observer as the
view can then be updated at each iteration of the
simuation.

Rendering specific graphs can be done by implementing a
specific observer for that graph design.

General idea for implementing a Plotter for a specific class
of graphs.

The base case is GraphPlotter which sets up how all the data
should be stored. Then extend from GraphPlotter in various
ways to implement the necessary methods. Finally, the actual
graph plotter that the user instantiates implements the layout
of the nodes.

REMEMBER when creating the final observer, you're going to
need to call SUPER on the dependencies method

@author Antares
@date   3/5/2018
"""
from matplotlib import cm
from copy import copy
from abc import ABCMeta, abstractmethod

from simulation.observer import Observer, NetEdgeFiringCounter, NetEdgeFiringObserver
from simulation.harness import ApexBridgeChipFiring

import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np


# ------------ LAYOUT OBSERVER IMPLEMENTATIONS ------------


class GraphPlotter(Observer):
    """ An observer that records how the graph should be
    drawn after each round of net firings. This observer
    does not store any human-readable data, but instead
    records matplotlib objects that can be used for
    plotting. Introduces the following instance
    variables.

    _harness:   The harness that is to be rendered by
                this observer.

    Plotters have four 5 additional methods for drawing
    each of the components in the rendering.
    - draw_nodes()
    - draw_edges()
    - draw_labels()
    - draw_annotations()
    - draw_patches()

    The data dictionary is organized as follows.
    {
        'observations': {
            <round number>: {
                'nodes': [<matplotlib PathCollection>],
                'edges': [<matplotlib LineCollection>],
                'node_labels': [<matplotlib Text>],
                'edge_labels': [<matplotlib Text>],
                'annotations': [<matplotlib Text>],
                'patches': [<matplotlib Patches>]
            }
        }
        'final_data': {
            'nodes': [<matplotlib PathCollection>],
            'edges': [<matplotlib LineCollection>],
            'node_labels': [<matplotlib Text>],
            'edge_labels': [<matplotlib Text>],
            'annotations': [<matplotlib Text>],
            'patches': [<matplotlib Patches>]
        }
    }

    -   NODES contains all the node positions.
    -   EDGES contains all the edge positions.
    -   LABELS contains all the node label positions.
    -   ANNOTATIONS contains any text that annotates
        the entire graph.
    -   PATCHES contains any other arbitrary shapes.
    """

    __metaclass__ = ABCMeta

    NAME = "graph_plotter"

    def start(self, graph_harness):
        """ The graph plotter starts tracking information
        when it is associated with a GRAPH_HARNESS.

        NOTE:   If you want to plot the original plot, you
                can do so here.
        """
        self._harness = graph_harness

    def _draw(self, terminate=False):
        """ A private helper method that returns a dict
        with all the rendered components.
        """
        new_plot = {
            'nodes': self.draw_nodes(self._harness, terminate=terminate),
            'edges': self.draw_edges(self._harness, terminate=terminate),
            'node_labels': self.draw_node_labels(self._harness, terminate=terminate),
            'edge_labels': self.draw_edge_labels(self._harness, terminate=terminate),
            'annotations': self.draw_annotations(self._harness, terminate=terminate),
            'patches': self.draw_patches(self._harness, terminate=terminate)
        }
        return new_plot

    def next_tick(self):
        """ Increment the round number and add a new list of
        plots to the plotter's data map.
        """
        super(GraphPlotter, self).next_tick()
        if self._harness is None:
            raise Exception("Graph Plotter needs a harness.")

        # Create a new plot
        self._data['observations'][self._round_number] = self._draw()

        # Increment the round number
        self._round_number += 1

    def terminate(self):
        """ Terminates the plotter by drawing a final plot
        of the graph and setting is_terminated to true.
        """
        if self._is_terminated:
            raise Exception('Observer already terminated!')

        # Create the last plot
        self._data['final_data'] = self._draw(terminate=True)
        self._is_terminated = True

    @abstractmethod
    def draw_nodes(self, graph_harness, terminate=False):
        """ Returns a list of matplotlib PathCollection
        instances representing the nodes of the graph in
        the GRAPH_HARNESS.

        This MUST be overriden
        """
        raise NotImplementedError

    @abstractmethod
    def draw_edges(self, graph_harness, terminate=False):
        """ Returns a list of matplotlib LineCollection
        instances representing the edges of the graph in
        the GRAPH_HARNESS.

        This MUST be overriden
        """
        raise NotImplementedError

    def draw_node_labels(self, graph_harness, terminate=False):
        """ Returns a list of matplotlib Text instances
        containing the labels for each node.
        """
        return []

    def draw_edge_labels(self, graph_harness, terminate=False):
        """ Returns a list of matplotlib Text instances
        containing the labels for each edge.
        """
        return []

    def draw_annotations(self, graph_harness, terminate=False):
        """ Returns a list of matplotlib Text instances
        containing any text annotations to be made to
        the rendering.
        """
        return []

    def draw_patches(self, graph_harness, terminate=False):
        """ Returns a list of matplotlib Patch instances
        that embelish the plot in some way.
        """
        return []

    @abstractmethod
    def layout(self, graph_harness, terminate=False):
        """ Returns a networkx graph layout which is a dict
        mapping nodes to 2D numpy vectors denoting X and Y
        coordinate positions.

        This MUST be overriden
        """
        raise NotImplementedError


# ------------ VARIOUS PLOTTER IMPLEMENTATIONS ------------


class NetEdgeFiringPlotter(GraphPlotter):
    """ An abstract plotter that highlights net edge firings
    made during one round of the simulation and draws a
    textbox with that round's information.

    This class implements the following methods from the
    GraphPlotter super class
    -   draw_edges()
    -   draw_annotations()
    """

    NAME = "net_edge_firing_plotter"

    # Edge drawing constants
    EDGE_WIDTH = 1.0
    EDGE_ALPHA = 0.5
    HIGHLIGHT_COLOR = 'red'

    # Text box drawing constants
    FONT_SIZE = 9
    BOX_COLOR = 'wheat'
    BOX_STYLE = 'square'
    BOX_ALPHA = 0.5
    BOX_LOC = 4

    def __init__(self):
        """ Creates a new NetEdgeFiringPlotter. We add an
        additional instance variable to the super constructor

        _edges_fired:   The total number of edges that have
                        fired during this simulation.
        """
        super(NetEdgeFiringPlotter, self).__init__()
        self._edges_fired = 0

    def dependencies(self):
        """ Returns the list of dependencies. This plotter
        depends on the following observers.

        NetEdgeFiringObserver:  Required for plotting edges
                                that were net edge fired.
        """
        return [NetEdgeFiringCounter, NetEdgeFiringObserver]

    def draw_edges(self, graph_harness, terminate=False):
        """ Renders the edges of the graph in GRAPH_HARNESS.
        Edges are colored black as default and highlighted
        HIGHLIGHT_COLOR if there's a net firing.
        """
        if not self._manager:
            raise Exception("NetEdgeFiringCounter does not have manager!")

        # Draw all the edges
        layout = self.layout(graph_harness, terminate=terminate)
        collection = []
        edges = nx.draw_networkx_edges(graph_harness.graph(), layout,
            width=NetEdgeFiringPlotter.EDGE_WIDTH,
            alpha=NetEdgeFiringPlotter.EDGE_ALPHA
        )
        collection.append(edges)

        # Draw any highlights
        if terminate:
            red_edges = nx.draw_networkx_edges(graph_harness.graph(), layout,
                edge_color=NetEdgeFiringPlotter.HIGHLIGHT_COLOR,
                width=NetEdgeFiringPlotter.EDGE_WIDTH * 2,
                alpha=NetEdgeFiringPlotter.EDGE_ALPHA * 2
            )
            collection.append(red_edges)
        else:
            obs_data = self._manager.get_observation(NetEdgeFiringObserver.NAME, self._round_number)
            highlights = obs_data['fired_edges']
            if len(highlights) > 0:
                red_edges = nx.draw_networkx_edges(graph_harness.graph(), layout,
                    edgelist=highlights,
                    edge_color=NetEdgeFiringPlotter.HIGHLIGHT_COLOR,
                    width=NetEdgeFiringPlotter.EDGE_WIDTH * 2,
                    alpha=NetEdgeFiringPlotter.EDGE_ALPHA * 2
                )
                collection.append(red_edges)

        # Return the edges
        return collection

    def draw_annotations(self, graph_harness, terminate=False):
        """ Draws a legend in the plot with the number of
        net firings during the current round, the total
        number of net firings, and a note when the graph
        becomes recursive.
        """
        if not self._manager:
            raise Exception("NetEdgeFiringCounter does not have manager!")

        # Get the number of net fired edges
        fired_count = 0
        if terminate:
            obs_data = self._manager.get_final_data(NetEdgeFiringCounter.NAME)
            fired_count = obs_data['total_firings']
        else:
            obs_data = self._manager.get_observation(NetEdgeFiringCounter.NAME, self._round_number)
            fired_count = obs_data['count']

         # NOTE Putting this here is a bit weird since the
         #      call should technically go somewhere like
         #      next_tick.
        self._edges_fired += fired_count

        # Generate the text
        text = "Round Number: {}\nCurrent Net Firings: {}\nTotal Net Firings: {}".format(
            self._round_number,
            fired_count,
            self._edges_fired
        )
        if terminate:
            text += "\nRECURRENT"

        font_prop = dict(size=NetEdgeFiringPlotter.FONT_SIZE)
        text_box = mpl.offsetbox.AnchoredText(text,
            loc=NetEdgeFiringPlotter.BOX_LOC,
            prop=font_prop
        )
        text_box.patch.set(
            boxstyle=NetEdgeFiringPlotter.BOX_STYLE,
            facecolor=NetEdgeFiringPlotter.BOX_COLOR,
            alpha=NetEdgeFiringPlotter.BOX_ALPHA
        )
        return [text_box]


class ChipCountNodePlotter(GraphPlotter):
    """ An abstract plotter that colors the nodes according
    to the number of chips the node has and renders a label
    for each node noting how many chips it has until it
    fires.

    This class implements the following methods from the
    GraphPlotter super class
    -   draw_nodes()
    -   draw_node_labels()
    """

    NAME = 'chip_count_node_plotter'

    # Node drawing constants
    NODE_SIZE = 300
    NODE_CMAP = cm.get_cmap('RdYlBu')
    VMIN = 0.0
    VMAX = 1.0

    # Node label contants
    FONT_SIZE = 9

    def draw_nodes(self, graph_harness, terminate=False):
        """ Returns a PathCollection object with each node
        colored via the number of chips it has. The color
        value corresponds to #chips / degree.
        """
        # Get the node colors
        chip_counts = graph_harness.chip_counts()
        degrees = graph_harness.degrees()
        colors = [chip_counts[v] / degrees[v] if degrees[v] > 0 else 0 for v in graph_harness.graph().nodes()]

        # Draw all the nodes
        layout = self.layout(graph_harness, terminate=terminate)
        nodes = nx.draw_networkx_nodes(graph_harness.graph(), layout,
            node_color=colors,
            cmap=ChipCountNodePlotter.NODE_CMAP,
            vmin=ChipCountNodePlotter.VMIN,
            vmax=ChipCountNodePlotter.VMAX,
            node_size=ChipCountNodePlotter.NODE_SIZE
        )

        # Copy the nodes because networkx automatically
        # draws it to a matplotlib figure
        return [nodes]

    def draw_node_labels(self, graph_harness, terminate=False):
        """ Returns a list of matplotlib Text instances
        containing the labels for each node.
        """
        # Get the node labels
        chip_counts = graph_harness.chip_counts()
        degrees = graph_harness.degrees()
        node_labels = {v : "{}/{}".format(chip_counts[v], degrees[v]) for v in graph_harness.graph().nodes()}

        # Draw all the nodes
        layout = self.layout(graph_harness, terminate=terminate)
        labels = nx.draw_networkx_labels(graph_harness.graph(), layout, node_labels,
            font_size=ChipCountNodePlotter.FONT_SIZE
        )

        # Copy the labels because networkx automatically
        # draws it to a matplotlib figure
        return [labels[v] for v in graph_harness.graph().nodes()]


# ------------- THE APEX BRIDGE GRAPH PLOTTER -------------


class ApexBridgeGraphPlotter(NetEdgeFiringPlotter, ChipCountNodePlotter):
    """ Renders an Apex Bridge Graph. Implements the following
    methods from the GraphPlotter super class.
    -   layout()
    -   draw_patches()

    This also inherits the following methods its subclasses
    -   draw_nodes()
    -   draw_node_labels()
    -   draw_edges()
    -   draw_annotations()

    NOTE:   ApexBridgeGraphPlotter requires that the given
            graph_harness is an instance of ApexBridgeChipFiring
    """

    NAME = 'apex_bridge_plotter'

    # Layout constants
    CLUSTER_RADIUS = 4
    CLUSTER_SCALE = 3

    # Cluster highlight constants
    CLUSTER_CMAP = cm.get_cmap('tab20c')
    CLUSTER_SPACING = 25.0
    CLUSTER_ALPHA = 0.5

    def dependencies(self):
        """ Returns all dependencies required by this
        ApexBridgeGraphPlotter instance by unioning the two
        dependency sets from NetEdgeFiringPlotter and
        ChipCountNodePlotter.
        """
        return [NetEdgeFiringCounter, NetEdgeFiringObserver]

    def draw_patches(self, graph_harness, terminate=False):
        """ Renders patches that highlight each of the two
        clusters.
        """
        positions, offsets = self._layout(graph_harness, terminate=False)

        # Calculate the radii of the patches
        r1 = ApexBridgeGraphPlotter.CLUSTER_RADIUS * graph_harness._n1 + ApexBridgeGraphPlotter.CLUSTER_SPACING / 2
        r2 = ApexBridgeGraphPlotter.CLUSTER_RADIUS * graph_harness._n2 + ApexBridgeGraphPlotter.CLUSTER_SPACING / 2

        # Get the cluster patch colors
        c1_color = mpl.colors.rgb2hex(ApexBridgeGraphPlotter.CLUSTER_CMAP(0))
        c2_color = mpl.colors.rgb2hex(ApexBridgeGraphPlotter.CLUSTER_CMAP(4))

        # Draw the patches and return
        c1 = mpl.patches.Circle(offsets['cluster1'], r1,
            color=c1_color,
            alpha=ApexBridgeGraphPlotter.CLUSTER_ALPHA
        )
        c2 = mpl.patches.Circle(offsets['cluster2'], r2,
            color=c2_color,
            alpha=ApexBridgeGraphPlotter.CLUSTER_ALPHA
        )
        return [c1, c2]

    def layout(self, graph_harness, terminate=False):
        """ Returns the node layout of the Apex Bridge
        Graph. The two cliques are placed in a circular
        layout with the bridge node between and the
        apex placed below it.
        """
        positions, layout = self._layout(graph_harness, terminate=False)
        return positions

    def _layout(self, graph_harness, terminate=False):
        """ Returns the layout and a dictionary of offsets
        to be used when rendering the other pieces of the
        graph.
        """
        if not isinstance(graph_harness, ApexBridgeChipFiring):
            raise Exception("The graph harness must be an instance of ApexBridgeChipFiring")

        # Calculate the radii of the clusters
        r1 = ApexBridgeGraphPlotter.CLUSTER_RADIUS * graph_harness._n1
        r2 = ApexBridgeGraphPlotter.CLUSTER_RADIUS * graph_harness._n2

        # Create the offsets
        offsets = {
            'cluster1': np.array([-r1 * ApexBridgeGraphPlotter.CLUSTER_SCALE, 0]),
            'cluster2': np.array([r2 * ApexBridgeGraphPlotter.CLUSTER_SCALE, 0]),
            'apex': np.array([0, -2 * (r1 + r2)])
        }

        # Create the positions
        positions = {}
        positions[graph_harness._bridge] = np.array([0, 0])
        positions[graph_harness._apex] = offsets['apex']

        # Assign circular positions
        c1 = nx.circular_layout(graph_harness._cluster1, scale=r1, center=offsets['cluster1'])
        c2 = nx.circular_layout(graph_harness._cluster2, scale=r2, center=offsets['cluster2'])
        positions.update(c1)
        positions.update(c2)

        # Return the positions and offsets
        return positions, offsets


# ----- RENDER AND ANIMATION OBSERVER IMPLEMENTATIONS -----


class RenderObserver(Observer):
    """ Renders a graph after each round. The renderer is
    built from a Plotter instance and simply renders all
    the fields from a GraphPlotter's data structure each
    round. This observer does not save any data.
    """

    NAME = "render_observer"

    def __init__(self, graph_plotter=GraphPlotter, save_loc=None):
        """ Creates a new RenderObserver. This object
        introduces the following instance variables.

        _plotter:   The graph plotter class that this render
                    observer depends on.
        _save_loc:  The location of where to solve the
                    rendered plots to. By default this value
                    is None.
        """
        super(RenderObserver, self).__init__()
        self._plotter = graph_plotter
        self._save_loc = save_loc

    def dependencies(self):
        """ Returns a list of dependencies. The only
        dependency for a RenderObserver is the given
        Plotter class.
        """
        return [self._plotter]

    def next_tick(self):
        """ Renders a single plot at the end of the
        previous round.
        """
        super(RenderObserver, self).next_tick()
        if not self._manager:
            raise Exception("RenderObserver does not have manager!")

        # Render the most recent observations
        shapes = self._manager.get_observation(self._plotter.NAME, self._round_number)
        curr_plt = self._render(shapes)
        curr_plt.show()

        # Save it if save_loc provided
        if self._save_loc:
            self._save(curr_plt)

        # Increment the round_number
        self._round_number += 1

    def terminate(self):
        """ Renders the final plot and terminates the
        observer.
        """
        if self._is_terminated:
            raise Exception('Observer already terminated!')
        if not self._manager:
            raise Exception("RenderObserver does not have manager!")

        # Plot the final plot
        shapes = self._manager.get_final_data(self._plotter.NAME)
        final_plt = self._render(shapes)
        final_plt.show()
        if self._save_loc:
            self._save(final_plt, terminate=True)

        # Terminate this observer
        self._is_terminated = True

    def _render(self, shapes):
        """ Returns a rendered version of the graph using
        observations stored in SHAPES. This will automatically
        import matplotlib and return the pyplot instance.
        """
        # fig = plt.figure()
        ax = plt.gca()
        # ax = fig.add_subplot(1, 1, 1)

        # Construct the image
        ax.set_axis_off()
        ax.set_aspect('equal')
        # ax.set_xlim(-70, 120)
        # ax.set_ylim(-80, 40)

        # Draw all the shapes
        for nodes in shapes['nodes']:
            ax.add_collection(nodes)
        for edges in shapes['edges']:
            ax.add_collection(edges)
        for label in shapes['node_labels']:
            ax.add_artist(label)
        for label in shapes['edge_labels']:
            ax.add_artist(label)
        for annotation in shapes['annotations']:
            ax.add_artist(annotation)
        for patch in shapes['patches']:
            ax.add_artist(patch)
        return plt

    def _save(self, plt, terminate=False):
        """ Saves the PLT to self._save_loc + the value
        of self._counter.
        """
        if terminate:
            path = os.path.join(self._save_loc, "image-final.png")
            plt.savefig(path)
        else:
            path = os.path.join(self._save_loc, "image-{}.png".format(self._counter))
            plt.savefig(path)
