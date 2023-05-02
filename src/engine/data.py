
class Cell:
    def __init__(self, x, y, permeability, depth, height, porosity):
        self.x = x
        self.y = y
        self.permeability = permeability
        self.reservoir_depth = depth
        self.reservoir_height = height
        self.reservoir_porosity = porosity  # greater then 0, lower than 1. 1 corresponds to clear reservoir
        self.oil_amount = self.reservoir_height * self.reservoir_porosity
        self.max_oil_amount = self.reservoir_height * self.reservoir_porosity
        self.neighbors = None

    def _set_neighbors(self, neighbors):
        self.neighbors = tuple(neighbors)

    def get_neighbor_idx(self, x, y):
        if 0 == abs(x - self.x) + abs(y - self.y) > 1:
            return None
        if x < self.x:
            return 0
        if x > self.x:
            return 1
        if y < self.y:
            return 2
        if y > self.y:
            return 3


class Well:
    def __init__(self, x, y):
        self.x = x
        self.y = y
