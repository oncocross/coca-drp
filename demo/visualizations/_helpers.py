# visualizations/_helpers.py

"""
This module contains common helper functions used by other modules
within the 'visualizations' package. The leading underscore indicates
it is intended for internal use within this package.
"""

import plotly.graph_objects as go


def create_placeholder_fig(message="Results will be displayed here.") -> go.Figure:
    """Creates an empty Plotly Figure with a centered message for initial UI display."""
    # Create an empty figure object.
    fig = go.Figure()
    
    # Configure the layout of the empty figure.
    fig.update_layout(
        # Make the figure background transparent.
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        
        # Hide the x and y axes completely.
        xaxis={"visible": False},
        yaxis={"visible": False},
        
        # Add a centered text annotation to display the message.
        annotations=[{
            "text": message, "xref": "paper", "yref": "paper",
            "showarrow": False, "font": {"size": 16, "color": "grey"}
        }]
    )
    return fig