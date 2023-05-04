import plotly.subplots as sp
import plotly.graph_objects as go
from src.engine.maps import FieldMap, WellMap
from .utils import build_ground_mesh, build_well_mesh, build_oil_mesh, build_reservoir_mesh, build_porosity_volume
import numpy as np


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


def render(pre_oil_amount, state, actions, env):
    reservoir_depth = state[:, :, 0]
    premeability = state[:, :, 1]
    reservoir_height = state[:, :, 2]
    reservoir_porosity = state[:, :, 3]
    oil_amount  = state[:, :, 4]
    diff_oil_amount = np.array(pre_oil_amount) -  np.array(state[:, :, 4]) 
    
    fig = sp.make_subplots(rows=6, cols=1, vertical_spacing = 0.05,
                        subplot_titles=("Глубина залегания", "Проницаемость", "Высота резервуара", "Пористость", "Количество нефти", "Именение нефти"))
    

    fig.add_trace(go.Heatmap(z=reservoir_depth), row=1, col=1)
    fig.add_trace(go.Heatmap(z=premeability), row=2, col=1)
    fig.add_trace(go.Heatmap(z=reservoir_height), row=3, col=1)
    fig.add_trace(go.Heatmap(z=reservoir_porosity), row=4, col=1)
    fig.add_trace(go.Heatmap(z=oil_amount), row=5, col=1)
    fig.add_trace(go.Heatmap(z=diff_oil_amount), row=6, col=1)

    fig.update_traces(showscale=False)

    fig.add_trace(go.Scatter(x=[action[0] for action in actions], y=[action[1] for action in actions], marker_color='rgba(0, 255, 0, .9)', marker_symbol="square", marker_size = 10, mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[action[0] for action in actions], y=[action[1] for action in actions], marker_color='rgba(0, 255, 0, .9)', marker_symbol="square", marker_size = 10, mode='markers'), row=2, col=1)
    fig.add_trace(go.Scatter(x=[action[0] for action in actions], y=[action[1] for action in actions], marker_color='rgba(0, 255, 0, .9)', marker_symbol="square", marker_size = 10, mode='markers'), row=3, col=1)
    fig.add_trace(go.Scatter(x=[action[0] for action in actions], y=[action[1] for action in actions], marker_color='rgba(0, 255, 0, .9)', marker_symbol="square", marker_size = 10, mode='markers'), row=4, col=1)
    fig.add_trace(go.Scatter(x=[action[0] for action in actions], y=[action[1] for action in actions], marker_color='rgba(0, 255, 0, .9)', marker_symbol="square", marker_size = 10, mode='markers'), row=5, col=1)
    fig.add_trace(go.Scatter(x=[action[0] for action in actions], y=[action[1] for action in actions], marker_color='rgba(0, 255, 0, .9)', marker_symbol="square", marker_size = 10, mode='markers'), row=6, col=1)


    fig.update_xaxes(range=[0, env.w - 1])
    fig.update_yaxes(range=[0, env.h - 1])
    fig.update_traces(showlegend=False)


    fig.update_layout(
        autosize=False,
        width=800,
        height=5*400,
        margin=dict(l=10, r=10, b=50, t=50, pad=0)
    )
    fig.show()

def error_func(x, y):
    fig = sp.make_subplots(row_heights=[0.5])
    
    fig.add_trace(go.Scatter(x = x, y = y,
    mode='lines',
    marker_color='rgba(255, 0, 0, .9)',
                    ))
    
    fig.update_layout(title='Фунция ошибки', height=450, width=900)
    fig.show()