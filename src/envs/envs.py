import numpy as np

from src.engine.maps import FieldMap, WellMap, FlowMap
from src.utils.map_generation.generate_simple import generate_simple
from src.utils.map_generation.generate_linked_graph import generate_graph
from src.render.render import render, RenderSettings


class BaseBlackOilEnv:
    def __init__(self, w, h, wells, days_per_well, well_power=1., wpr=0.9, oil_cost=0.05, well_cost=0.1, well_base_cost=0.2):
        self.field_map = None
        self.well_map = None
        self.flow_map = None
        self.w, self.h = w, h
        self.steps = 0
        self.days_per_well = days_per_well
        self.wells = wells
        self.well_power = well_power
        self.wpr = wpr
        self.oil_cost = oil_cost
        self.well_cost = well_cost
        self.well_base_cost = well_base_cost
        self.observation = None

    def _generate_map(self, w, h):
        pass

    def render(self, render_settings=None):
        if render_settings is None:
            render_settings = RenderSettings()
        render(self.field_map, self.well_map, render_settings)

    def reset(self):
        """
        Begins a new episode of a simulation.
        :return: First observation
        """
        self.steps = 0
        self.field_map = FieldMap(self._generate_map(self.w, self.h))
        self.well_map = WellMap(self.w, self.h, 2)
        self.flow_map = FlowMap(self.field_map, self.well_map, 0.6 / 5, self.well_power, self.wpr)
        self.observation = self._build_observation()
        return self.observation

    def step(self, action):
        """
        Places a well in coordinates corresponding to an action. Then simulates several days of oil extraction.
        Returns new observation, reward, done and info.
        Observation is a 3D tensor with first two dimensions corresponding to rows and columns. Third dimension is a features of specific cell.
        Reward is computed by summing well placement cost and gain from selling extracted oil.
        Done indicates an end of the episode.
        Info contains additional information about an environment.
        :param action:
        :return: observation, reward, done and info.
        """
        x, y = action
        drilling_cost = self.field_map.get_drilling_cost_map()[y, x]

        r_placement = - (drilling_cost * self.well_cost + self.well_base_cost) if self.well_map.add_well(x, y) else 0
        self.flow_map.update()
        r_extraction = self.flow_map.step(5 * self.days_per_well) * self.oil_cost
        self.steps += 1

        done = (self.steps >= self.wells)

        self.observation = self._build_observation()

        return self.observation, r_extraction + r_placement, done

    def _build_observation(self):
        return np.concatenate([self.field_map.get_parameters_map(),
                               self.field_map.get_drilling_cost_map(),
                               self.well_map.get_wells_map()], axis=-1)


class DummyBlackOilEnv(BaseBlackOilEnv):
    def __init__(self):
        super().__init__(25, 25, 4, 30)

    def _generate_map(self, w, h):
        return generate_simple(w, h, 2, 0.5, 4, 2)


class BlackOilEnv(BaseBlackOilEnv):
    def __init__(self, w=80, h=40, wells=8, days=30):
        super().__init__(w, h, wells, days, 0.9, 0.8, 0.0523, 0.1127, 0.271)

    def _generate_map(self, w, h):
        return generate_graph(w, h, 8, 20, 1, 3, 0.8, 2.5, 0.5, 2.5, 0.15, 1.5)
