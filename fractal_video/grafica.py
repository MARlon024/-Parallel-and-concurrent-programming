import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import os

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Definir el layout de la aplicación
app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # en milisegundos
        n_intervals=0
    ),
    dcc.Graph(id='live-graph')
])

# Definir la función de callback para actualizar el gráfico
@app.callback(Output('live-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    # Leer el archivo CSV
    if os.path.exists('times.csv'):
        df = pd.read_csv('times.csv', names=['Parallel Time', 'Smile Time', 'Concurrent Time'])
    else:
        df = pd.DataFrame(columns=['Parallel Time', 'Smile Time', 'Concurrent Time'])

    # Crear una figura
    fig = go.Figure()

    # Agregar los datos al gráfico
    fig.add_trace(go.Scatter(x=list(range(len(df['Parallel Time']))), y=df['Parallel Time'], mode='lines', name='Parallel Time'))
    fig.add_trace(go.Scatter(x=list(range(len(df['Smile Time']))), y=df['Smile Time'], mode='lines', name='Smile Time'))
    fig.add_trace(go.Scatter(x=list(range(len(df['Concurrent Time']))), y=df['Concurrent Time'], mode='lines', name='Concurrent Time'))

    # Actualizar los títulos de los ejes
    fig.update_layout(title='Comparación de tiempos de procesamiento', xaxis_title='Frame Number', yaxis_title='Time (s)')

    return fig
import dash
from dash.dependencies import Input, Output
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import os

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Definir el layout de la aplicación
app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # en milisegundos
        n_intervals=0
    ),
    dcc.Graph(id='live-graph')
])

# Definir la función de callback para actualizar el gráfico
@app.callback(Output('live-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    # Leer el archivo CSV
    if os.path.exists('times.csv'):
        df = pd.read_csv('times.csv', names=['Parallel Time', 'Smile Time', 'Concurrent Time'])
    else:
        df = pd.DataFrame(columns=['Parallel Time', 'Smile Time', 'Concurrent Time'])

    # Crear una figura
    fig = go.Figure()

    # Agregar los datos al gráfico
    fig.add_trace(go.Scatter(x=list(range(len(df['Parallel Time']))), y=df['Parallel Time'], mode='lines', name='Parallel Time'))
    fig.add_trace(go.Scatter(x=list(range(len(df['Smile Time']))), y=df['Smile Time'], mode='lines', name='Smile Time'))
    fig.add_trace(go.Scatter(x=list(range(len(df['Concurrent Time']))), y=df['Concurrent Time'], mode='lines', name='Concurrent Time'))

    # Actualizar los títulos de los ejes
    fig.update_layout(title='Comparación de tiempos de procesamiento', xaxis_title='Frame Number', yaxis_title='Time (s)')

    return fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
