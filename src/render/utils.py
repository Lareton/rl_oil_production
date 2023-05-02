from src.engine.maps import FieldMap, WellMap
import plotly.graph_objects as go
from plotly import colors as clrs
import numpy as np


def build_ground_mesh(field_map: FieldMap, opacity=0.4):
    num_cells = field_map.width * field_map.height
    z_upper = np.zeros(num_cells)
    z_lower = np.array([-cell.reservoir_depth for row in field_map.map for cell in row])
    intensity = np.array([cell.permeability for row in field_map.map for cell in row])
    return _build_mesh_coords(z_upper, z_lower, intensity, field_map, opacity, (0.5, 2.5),
                              ((0.7, 0.4, 0.2, 0.5), (0.4, 0.4, 0.4, 1.)), "Проницаемость", True)


def build_reservoir_mesh(field_map: FieldMap, opacity=0.25, render_top=True):
    z_upper = np.array([-cell.reservoir_depth for row in field_map.map for cell in row])
    z_lower = np.array([-cell.reservoir_depth-cell.reservoir_height for row in field_map.map for cell in row])
    return _build_mesh_coords(z_upper, z_lower, None, field_map, opacity, None,
                              ((0.8, 0.5, 0.3), ), None, False, not  render_top)


def build_oil_mesh(field_map: FieldMap, opacity=0.75):
    z_upper = np.array([-cell.reservoir_depth-cell.reservoir_height * (1 - cell.oil_amount / max(1e-9, cell.max_oil_amount)) for row in field_map.map for cell in row])
    z_lower = np.array([-cell.reservoir_depth-cell.reservoir_height for row in field_map.map for cell in row])
    intensity = np.array([(min(0.8, cell.oil_amount / max(1e-9, cell.max_oil_amount)) / 0.8 if cell.max_oil_amount > 1e-3 else 0.)
                          for row in field_map.map for cell in row])
    return _build_mesh_coords(z_upper, z_lower, intensity, field_map, opacity, (0, 1),
                              ((0., 0., 0., 0.), (0., 0., 0., 1.)), None)


def build_porosity_volume(field_map: FieldMap, opacity=0.1, surface_count=21):
    x = np.array([cell.x for row in field_map.map for cell in row])
    y = np.array([cell.y for row in field_map.map for cell in row])
    z_upper = np.array([-cell.reservoir_depth for row in field_map.map for cell in row])
    z_lower = np.array([-cell.reservoir_depth-cell.reservoir_height for row in field_map.map for cell in row])
    intensity = np.array([cell.reservoir_porosity for row in field_map.map for cell in row])

    x = np.concatenate([x, x, x])
    y = np.concatenate([y, y, y])
    z = np.concatenate([z_upper, z_lower, (z_upper + z_lower) / 2])
    intensity = np.concatenate([np.zeros_like(intensity), np.zeros_like(intensity), intensity])

    volume = go.Volume(
        x=x, y=y, z=z, value=intensity, opacity=opacity, surface_count=surface_count,
        isomin=0, isomax=1, opacityscale=[[0, 0], [1, 1]], colorscale=
        [[0., clrs.label_rgb((0.7, 0.4, 0.2))], [1., clrs.label_rgb((0.7, 0.4, 0.2))]]
    )
    return volume


def build_well_mesh(well_map: WellMap, field_map:FieldMap, opacity=0.25):
    x = [well.x for well in well_map.wells]
    y = [well.y for well in well_map.wells]
    lines = []
    for _x, _y in zip(x, y):
        line = go.Scatter3d(
            x=[_x, _x], y=[_y, _y], z=[0.5, -field_map.map[_y][_x].reservoir_depth],
            mode="lines",
            line=dict(
                color=clrs.label_rgb((0.3, 0.3, 0.9)),
                width=5
            ), showlegend=False
        )
        lines.append(line)
    return lines


def _build_mesh_coords(z_upper, z_lower, intensity, field_map:FieldMap, opacity, intensity_lims,
                       colors, label=None, no_bounds=False, no_top=False):
    num_cells = field_map.width * field_map.height
    x = np.array([cell.x for row in field_map.map for cell in row])
    y = np.array([cell.y for row in field_map.map for cell in row])
    x = np.concatenate([x, x])
    y = np.concatenate([y, y])
    if intensity is not None:
        intensity = np.concatenate([intensity, intensity])
    z = np.concatenate([z_upper, z_lower])

    ### Build a list of triangles

    # Triangles for lower part
    triangles = [(y * field_map.width + x + num_cells,
                  (y + 1) * field_map.width + x + num_cells,
                  y * field_map.width + x+1 + num_cells)
                 for y in range(field_map.height-1) for x in range(field_map.width-1)]
    triangles += [(y * field_map.width + x + num_cells,
                   (y - 1) * field_map.width + x + num_cells,
                   y * field_map.width + x-1 + num_cells)
                  for y in range(1, field_map.height) for x in range(1, field_map.width)]

    # Triangles for upper part
    if not no_top:
        triangles += [(i - num_cells, j - num_cells, k - num_cells) for i, j, k in triangles]

    # Triangles for connection between lower and upper parts
    if not no_bounds:
        for i in (0, 1):
            for j in (0, 1):
                triangles += [(i * (field_map.height - 1) * field_map.width + x + num_cells,
                               i * (field_map.height - 1) * field_map.width + x + j + j * num_cells,
                               i * (field_map.height - 1) * field_map.width + x + 1)
                              for x in range(field_map.width-1)]

                triangles += [(y * field_map.width + i * (field_map.width - 1) + num_cells,
                               (y + j) * field_map.width + i * (field_map.width - 1) + j * num_cells,
                               (y + 1) * field_map.width + i * (field_map.width - 1))
                              for y in range(field_map.height - 1)]

    if intensity is not None:
        mesh = go.Mesh3d(
            x=x, y=y, z=z, intensity=intensity,
            i=[i for (i, _, _) in triangles],
            j=[j for (_, j, _) in triangles],
            k=[k for (_, _, k) in triangles],
            colorscale=[
                [0., (clrs.label_rgb(colors[0]) if len(colors[0]) != 4 else _label_rgba(*colors[0]))],
                [1., (clrs.label_rgb(colors[1]) if len(colors[1]) != 4 else _label_rgba(*colors[1]))]
            ],
            colorbar_title=label,
            showscale=(label is not None),
            opacity=opacity,
            cmin=intensity_lims[0],
            cmax=intensity_lims[1]
        )
    else:
        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=[i for (i, _, _) in triangles],
            j=[j for (_, j, _) in triangles],
            k=[k for (_, _, k) in triangles],
            color=clrs.label_rgb(colors[0]),
            opacity=opacity,
        )
    return mesh


def _label_rgba(r, g, b, a):
    return f"rgba({int(255*r)}, {int(255*g)}, {int(255*b)}, {a:.3f})"