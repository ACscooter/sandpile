""" A chip that is fired around the graph.

@author Antares
@date   3/13/2018
"""

class Chip(object):
    """ A chip that only moves along net fired edges. """

    def __init__(self, start, **kwargs):
        """ Creates a new chip that stores the nodes it
        net visited in a per-round basis. The node is
        placed at START to begin its journey.

        _properties:    Any book keeping information that
                        needs to be stored with the chip.
        _history:       A dict from round number to the
                        vertices the chip visits on that
                        round.
        _curr_round:    The current round the chip is on.
        _curr_history:  The current nodes that the chip has
                        visited during this round.
        """
        self._properties = kwargs # Maybe of use later
        self._history = {}
        self._curr_round = 0
        self._curr_history = [start]

    # GETTER METHODS

    def get_data(self):
        """ Returns this chip's raw data. """
        return self._history

    def get_history(self):
        """ Returns the chip's entire history of vertices
        visited.
        """
        history = []
        for i in range(self._curr_round):
            history += self._history[i]
        return history

    def get_round(self, round_num):
        """ Returns the vertices visited by this chip during
        round ROUND_NUM.
        """
        return self._history[round_num]

    # ACTION METHODS

    def fire(self, vertex):
        """ Fires the chip to VERTEX. """
        self._curr_history.append(vertex)

    def next_tick(self):
        """ Appends the current round's history to the
        overall history and increments the round number.
        """
        self._history[self._curr_round] = self._curr_history

        # Reset the round's history and increment the round
        self._curr_history = []
        self._curr_round += 1
