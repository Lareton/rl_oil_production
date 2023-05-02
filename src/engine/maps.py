from collections import deque
#from queue import PriorityQueue

import numpy as np
from src.engine.data import Well, Cell


class WellMap:
    """
    Stores information about all wells on the field
    """

    def __init__(self, width, height, intersection_radius=3):
        self.wells = []
        self.width = width
        self.height = height
        self.intersection_radius = intersection_radius

    def add_well(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            for w in self.wells:
                if abs(w.x - x) < self.intersection_radius or abs(w.y - y) < self.intersection_radius:
                    return False
            self.wells.append(Well(x, y))
            return True
        return False

    def get_outflows(self):
        return [(w.x, w.y) for w in self.wells]

    def get_dist_to_closest(self, x, y):
        return np.min([(x - w.x)**2 + (y - w.y)**2] for w in self.wells)**0.5

    def get_wells_map(self):
        map = np.zeros((self.height, self.width, 1))
        for w in self.wells:
            map[w.y, w.x] = 1
        return map


class FieldMap:
    """
    Stores basic information about oil field
    """

    def __init__(self, np_map: np.ndarray):
        self.width = np_map.shape[1]
        self.height = np_map.shape[0]
        self.map = [[
            Cell(x, y, cell[0], cell[1], cell[2], cell[3])
            for x, cell in enumerate(row)
        ] for y, row in enumerate(np_map)]

        for x in range(np_map.shape[1]):
            for y in range(np_map.shape[0]):
                cell = self.map[y][x]
                neighbors = []
                if cell.x > 0:
                    neighbors.append(self.map[cell.y][cell.x - 1])
                else:
                    neighbors.append(None)
                if cell.x < self.width - 1:
                    neighbors.append(self.map[cell.y][cell.x + 1])
                else:
                    neighbors.append(None)
                if cell.y > 0:
                    neighbors.append(self.map[cell.y - 1][cell.x])
                else:
                    neighbors.append(None)
                if cell.y < self.height - 1:
                    neighbors.append(self.map[cell.y + 1][cell.x])
                else:
                    neighbors.append(None)
                cell._set_neighbors(neighbors)

    def get_drilling_cost_map(self):
        map = np.zeros((self.height, self.width, 1))
        for row in self.map:
            for cell in row:
                map[cell.y, cell.x] = cell.reservoir_depth * cell.permeability
        return map

    def get_parameters_map(self):
        map = np.zeros((self.height, self.width, 6))
        for row in self.map:
            for cell in row:
                map[cell.y, cell.x, 0] = cell.reservoir_depth
                map[cell.y, cell.x, 1] = cell.permeability
                map[cell.y, cell.x, 2] = cell.reservoir_height
                map[cell.y, cell.x, 3] = cell.reservoir_porosity
                map[cell.y, cell.x, 4] = cell.oil_amount
                map[cell.y, cell.x, 5] = cell.max_oil_amount
        return map


class FlowMap:
    """
    Stores information about oil flows
    """
    def __init__(self, field_map: FieldMap, well_map: WellMap, oil_viscosity=0.5, well_power=2, well_power_reduction=0.95):
        self.field_map = field_map
        self.well_map = well_map
        self.oil_viscosity = oil_viscosity  # Determines how fast flow can be, normally between 0 and 1
        self.well_power = well_power
        self.well_power_reduction = well_power_reduction
        self.well_flow_map = None
        self.flow_limit = None

    def update(self):
        """
        This method should be called after a new wells were added
        :return:
        """

        # Contains information about outflow and inflow speed
        self.flow_limit = np.zeros((self.field_map.height, self.field_map.width, 4))

        cells = [cell for row in self.field_map.map for cell in row]

        # Compute maximum flow capacity
        for cell in cells:
            for i, other in enumerate(cell.neighbors):
                if other is None:
                    continue
                self.flow_limit[cell.y, cell.x, i] = self._compute_flow_limit(cell, other)

        # Compute flow between cells
        # Normally oil flows from top cells to bottom ones
        self.well_flow_map = np.zeros((self.field_map.height, self.field_map.width, 4))
        for well in self.well_map.wells:
            coefs = self.flow_limit[well.y, well.x]
            coefs /= max(np.sum(coefs), 1e-9)
            self.well_flow_map[well.y, well.x] += coefs

        for well in self.well_map.wells:
            self._spread_oil_extraction_flow(self.well_flow_map, well.x, well.y)
        pass

    def step(self, steps):
        """
        Update field map for given number of steps
        """
        total_extracted_oil = 0
        for _ in range(steps):
            total_extracted_oil += self._step()
        return total_extracted_oil

    def _step(self):
        """
        Perform one step of flow simulation
        """
        # compute downstream flow map and final flow map
        downstream_flow_map = self._compute_downstream_flow_map()
        total_flow_map = np.minimum(self.well_flow_map * self.well_power + downstream_flow_map, self.flow_limit) * self.oil_viscosity

        cells = [cell for row in self.field_map.map for cell in row]

        # resolve conflicts
        for cell in cells:
            for i, other in enumerate(cell.neighbors):
                if other is None:
                    continue
                j = other.get_neighbor_idx(cell.x, cell.y)
                if total_flow_map[cell.y, cell.x, i] > 0 and total_flow_map[other.y, other.x, j] > 0:
                    first, second = total_flow_map[cell.y, cell.x, i], total_flow_map[other.y, other.x, j]
                    total_flow_map[cell.y, cell.x, i] = max(0, first - second)
                    total_flow_map[other.y, other.x, j] = max(0, second - first)

        # Extract oil before performing a step
        extracted_oil = 0
        for well in self.well_map.wells:
            cell = self.field_map.map[well.y][well.x]
            extracted = min(cell.oil_amount, self.oil_viscosity * self.well_power)
            cell.oil_amount -= extracted
            extracted_oil += extracted

        # Update cell oil values
        sorted_cells = sorted(cells, key=lambda cell: np.sum(total_flow_map[cell.y, cell.x]))
        for cell in sorted_cells:
            max_total_inflow_amount = cell.max_oil_amount - cell.oil_amount
            inflow_amounts = np.zeros(4)
            for i, other in enumerate(cell.neighbors):
                if other is None:
                    continue
                inflow_amounts[i] = min(other.oil_amount, total_flow_map[cell.y, cell.x, i])

            total_inflow_amount = np.sum(inflow_amounts)
            if total_inflow_amount > max_total_inflow_amount and total_inflow_amount > 1e-9:
                inflow_amounts = inflow_amounts * max_total_inflow_amount / total_inflow_amount
                total_inflow_amount = max_total_inflow_amount
            cell.oil_amount += total_inflow_amount

            for i, other in enumerate(cell.neighbors):
                if other is None:
                    continue
                other.oil_amount -= inflow_amounts[i]

        return extracted_oil

    @staticmethod
    def _compute_flow_limit(cell, other):
        interception = min(cell.reservoir_depth + cell.reservoir_height,
                           other.reservoir_depth + other.reservoir_height) - max(cell.reservoir_depth,
                                                                                 other.reservoir_depth)
        if interception > 0:
            return cell.reservoir_porosity * other.reservoir_porosity * interception

    def _spread_oil_extraction_flow(self, well_flow_map, x, y):
        visits = np.zeros(well_flow_map.shape[:2], dtype=int)
        pending = np.zeros(well_flow_map.shape[:2], dtype=bool)
        delta_map = np.zeros_like(well_flow_map)

        queue = deque()
        queue.append((x, y))
        pending[y, x] = True
        delta_map[y, x] = well_flow_map[y, x]
        while len(queue) > 0:
            x, y = queue.popleft()
            pending[y, x] = False

            coords_to_visit = []
            cell = self.field_map.map[y][x]
            for i, other in enumerate(cell.neighbors):
                if other is None:
                    continue

                j = other.get_neighbor_idx(cell.x, cell.y)

                # Resolve conflicts
                if well_flow_map[other.y, other.x, j] > 0:
                    if well_flow_map[cell.y, cell.x, i] > 0:
                        left, right = well_flow_map[other.y, other.x, j], well_flow_map[cell.y, cell.x, i]
                        well_flow_map[other.y, other.x, j] = max(0, left - right)
                        well_flow_map[cell.y, cell.x, i] = max(0, right - left)
                    left, right = well_flow_map[other.y, other.x, j], delta_map[cell.y, cell.x, i]
                    well_flow_map[other.y, other.x, j] = max(0, left - right)
                    delta_map[cell.y, cell.x, i] = max(0, right - left)

                if well_flow_map[other.y, other.x, j] > 0:
                    continue

                # Add extraction power
                coefs = np.copy(self.flow_limit[other.y, other.x])
                coefs[j] = 0
                coefs /= max(np.sum(coefs), 1e-9)

                flow_power = delta_map[cell.y, cell.x, i] * self.well_power_reduction
                delta_map[other.y, other.x] += flow_power * coefs
                coords_to_visit.append((other.x, other.y))

            well_flow_map[cell.y, cell.x] += delta_map[cell.y, cell.x]
            delta_map[cell.y, cell.x] = 0

            for _x, _y in coords_to_visit:
                if not pending[_y, _x] and visits[_y, _x] < 8:
                    pending[_y, _x] = True
                    visits[_y, _x] += 1
                    queue.append((_x, _y))
                #if visits[_y, _x] >= 8:
                #    print("Reached visits limit")

    def _compute_downstream_flow_map(self):
        downstream_flow_map = np.zeros((self.field_map.height, self.field_map.width, 4))
        max_outflow_map = np.zeros((self.field_map.height, self.field_map.width, 4))  # in pure oil
        max_inflow_map = np.zeros((self.field_map.height, self.field_map.width, 4))

        cells = [cell for row in self.field_map.map for cell in row]
        for cell in cells:
            if cell.max_oil_amount < 1e-9:
                continue

            oil_level_this = cell.reservoir_height * (
                    1 - cell.oil_amount / cell.max_oil_amount) + cell.reservoir_depth

            ls_out = []
            cs_out = np.ones((4, 4)) / cell.reservoir_porosity
            ids_out = []

            ls_in = []
            cs_in = np.ones((4, 4)) / cell.reservoir_porosity
            ids_in = []

            for i, other in enumerate(cell.neighbors):
                if other is None or self.flow_limit[cell.y, cell.x, i] < 1e-9:
                    continue
                oil_level_other = other.reservoir_height * (
                        1 - other.oil_amount / other.max_oil_amount) + other.reservoir_depth
                oil_level_intersection = (min(cell.reservoir_depth + cell.reservoir_height,
                                              other.reservoir_depth + other.reservoir_height,
                                              max(oil_level_other, oil_level_this))
                                          - max(cell.reservoir_depth, other.reservoir_depth,
                                                min(oil_level_this, oil_level_other)))

                if oil_level_intersection > 0:
                    if oil_level_this < oil_level_other:
                        cs_out[len(ls_out), len(ls_out)] += 1 / other.reservoir_porosity
                        ls_out.append(oil_level_intersection)
                        ids_out.append(i)
                    elif oil_level_this > oil_level_other:
                        cs_in[len(ls_in), len(ls_in)] += 1 / other.reservoir_porosity
                        ls_in.append(oil_level_intersection)
                        ids_in.append(i)

            ls_out = np.array(ls_out)
            cs_out = cs_out[:len(ls_out)][:, :len(ls_out)]
            outflows_cap = np.linalg.inv(cs_out).dot(ls_out)
            for i, oc in zip(ids_out, outflows_cap):
                max_outflow_map[cell.y, cell.x, i] = oc

            ls_in = np.array(ls_in)
            cs_in = cs_in[:len(ls_in)][:, :len(ls_in)]
            inflows_cap = np.linalg.inv(cs_in).dot(ls_in)
            for i, oc in zip(ids_in, inflows_cap):
                max_inflow_map[cell.y, cell.x, i] = oc

        for cell in cells:
            for i, other in enumerate(cell.neighbors):
                if other is None or self.flow_limit[cell.y, cell.x, i] < 1e-9:
                    continue
                downstream_flow_map[cell.y, cell.x, i] = other.reservoir_porosity * cell.reservoir_porosity * min(
                    max_inflow_map[cell.y, cell.x, i],
                    max_outflow_map[other.y, other.x, 2 * (i // 2) + (i + 1) % 2]
                )
        return downstream_flow_map