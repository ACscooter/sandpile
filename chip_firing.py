""" A couple of functions that run the naive chip firing
simulation on general graphs.

@author Antares Chen
@date   2/11/2018
"""

import networkx as nx
import numpy as np


# ---------------- ROUND-BASED SIMULATION -----------------


def run(harness, chip_dropper, manager):
    """ Runs a round based net firing simulation on the
    given simulation HARNESS. The CHIP_DROPPER determines
    where the chips are dropped each round and the
    observer MANAGER records any necessary measurements.

    NOTE: This mutates the graph in HARNESS
    """
    # Start all the observers
    manager.start(harness)

    # Run the simulation while the graph is not recurrent
    while not harness.recurrent():
        drop_vertex = chip_dropper.next_vertex()
        _run_round(harness, drop_vertex, manager)

    # Terminate the observer
    manager.terminate()

def _run_round(harness, drop_vertex, manager):
    """ Runs one round of the net firing simulation by
    dropping a chip on DROP_VERTEX in the HARNESS. The
    observer MANAGER then takes any required
    measurements
    """
    # Drop the chip
    harness.drop_chip(drop_vertex)

    # Loop through all fireable vertices once
    next_vertex = harness.next_fire_node()
    while next_vertex is not None:
        harness.fire_node(next_vertex)
        manager.observe(harness)
        next_vertex = harness.next_fire_node()

    # Iterate the manager and reset the graph
    manager.next_tick()
    harness.next_tick()
