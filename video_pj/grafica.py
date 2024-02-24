import dash
from dash.dependencies import Input, Output
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import os

# Create the Dash application
app = dash.Dash(__name__)

# Define the layout of the application
app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Graph(id='live-graph')
])

# Define the callback function to update the graph
@app.callback(Output('live-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    # Read the CSV file
    if os.path.exists('times.csv'):
        df = pd.read_csv('times.csv', names=['Parallel Time', 'Smile Time', 'Concurrent Time'])
    else:
        df = pd.DataFrame(columns=['Parallel Time', 'Smile Time', 'Concurrent Time'])

    # Create a figure
    fig = go.Figure()

    # Add the data to the graph
    fig.add_trace(go.Scatter(x=list(range(len(df['Parallel Time']))), y=df['Parallel Time'], mode='lines', name='Parallel Time'))
    fig.add_trace(go.Scatter(x=list(range(len(df['Smile Time']))), y=df['Smile Time'], mode='lines', name='Smile Time'))
    fig.add_trace(go.Scatter(x=list(range(len(df['Concurrent Time']))), y=df['Concurrent Time'], mode='lines', name='Concurrent Time'))

    # Update the titles of the axes
    fig.update_layout(title='Processing Time Comparison', xaxis_title='Frame Number', yaxis_title='Time (s)')

    return fig

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)
