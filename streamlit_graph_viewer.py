import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.title("Interactive Graph Viewer with Plotly")

# Graph
G = nx.random_geometric_graph(10, 0.4)

# Node positions
pos = nx.get_node_attributes(G, 'pos')
edge_x, edge_y = [], []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y,
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        mode='lines')

node_x, node_y = zip(*pos.values())

node_trace = go.Scatter(x=node_x, y=node_y,
                        mode='markers',
                        hoverinfo='text',
                        marker=dict(showscale=False, color='lightblue', size=20),
                        text=[str(node) for node in G.nodes()])

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(showlegend=False,
                                 hovermode='closest',
                                 margin=dict(b=0,l=0,r=0,t=0)))
st.plotly_chart(fig)
