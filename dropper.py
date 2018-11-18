""" A script that defines all chip dropppers.

@author Antares
@date   2/11/2018
"""

from abc import ABCMeta, abstractmethod


# --------------------- CHIP DROPPERS ---------------------


class ChipDropper(object):
    """ A ChipDropper is an iterator over nodes in a
    networkx.graph instance. Each node emitted signifies
    that a chip is to be dropped on that node.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def next_vertex(self):
        """ Returns the next vertex to recieve a chip and
        advances the iterator.
        """
        raise NotImplementedError


# ---------------- DROPPER IMPLEMENTATIONS ----------------


class CycleDropper(object):
    """ Drops a chip on each vertex given to the dropper
    then repeats.
    """

    def __init__(self, vertices):
        """ Creates a CycleDropper which cycles through
        vertices in VERTICES.
        """
        self._vertices = vertices
        self._index = 0

    def next_vertex(self):
        """ Returns the apex node. """
        next_vertex = self._vertices[self._index]
        self._index = (self._index + 1) % len(self._vertices)
        return next_vertex


class ApexDropper(CycleDropper):
    """ Repeatedly drops a chip on a single vertex. """

    def __init__(self, apex):
        super(ApexDropper, self).__init__([apex])
