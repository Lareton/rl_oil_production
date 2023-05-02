import plotly
import plotly.graph_objects as go
from src.engine.maps import FieldMap, WellMap
from .utils import build_ground_mesh, build_well_mesh, build_oil_mesh, build_reservoir_mesh, build_porosity_volume


class RenderSettings:
    display_porosity = True
    porosity_opacity = 0.1
    porosity_surface_num = 21

    display_ground = True
    ground_opacity = 0.999

    display_wells = True
    well_opacity = 0.75

    oil_opacity = 0.75
    reservoir_opacity = 0.15


def render(field_map:FieldMap, well_map:WellMap, render_settings: RenderSettings):
    data = []
    data.append(build_oil_mesh(field_map, render_settings.oil_opacity))
    data.append(build_reservoir_mesh(field_map, render_settings.reservoir_opacity, not render_settings.display_ground))
    if render_settings.display_wells:
        data.extend(build_well_mesh(well_map, field_map, render_settings.well_opacity))
    # TODO: Fix porosity render
    #if render_settings.display_porosity:
    #    data.append(build_porosity_volume(field_map, render_settings.porosity_opacity, render_settings.porosity_surface_num))
    if render_settings.display_ground:
        data.append(build_ground_mesh(field_map, render_settings.ground_opacity))
    fig = go.Figure(data=data)
    fig.update_layout(
        scene_aspectmode='manual', scene_aspectratio=dict(x=field_map.width/field_map.height, y=1, z=1)
    )
    fig.show()