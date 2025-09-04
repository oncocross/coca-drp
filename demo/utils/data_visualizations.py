# utils/data_visualizations.py

# --- Core Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pcolors
import networkx as nx

import numpy as np
from typing import Dict, Any

# --- Custom Application Modules ---
# Import the placeholder function from result_visualizations to maintain consistency.
from utils.result_visualizations import create_placeholder_fig

def create_interactive_cell_line_treemap(cell_line_df: pd.DataFrame):
    """
    Creates an interactive treemap to visualize the hierarchical composition of cell lines,
    from tissues to specific cancer types.

    Args:
        cell_line_df (pd.DataFrame): DataFrame with 'tissue' and 'cancer_type' columns.

    Returns:
        go.Figure: An interactive Plotly treemap figure.
    """
    df = cell_line_df[['tissue', 'cancer_type']].dropna().copy()
    
    # --- Data Cleaning: Prevent Plotly Naming Conflicts ---
    # This trick prevents errors when a cancer_type has the same name as a tissue.
    # Appending a zero-width space ('\u200B') makes the string unique internally
    # without changing its visual appearance in the chart.
    all_tissue_names = df['tissue'].unique()
    mask = df['cancer_type'].isin(all_tissue_names)
    df.loc[mask, 'cancer_type'] = df.loc[mask, 'cancer_type'] + '\u200B'
    
    # Group the data to get the counts for each unique tissue/cancer_type combination.
    df_grouped = df.groupby(['tissue', 'cancer_type']).size().reset_index(name='counts')

    # --- Color Palette Generation ---
    # 1. Determine the number of unique top-level categories (tissues) to color.
    num_colors = len(df['tissue'].unique())
    
    # 2. Sample colors from a predefined scale, avoiding the pale end for better visibility.
    custom_palette = pcolors.sample_colorscale(
        'Peach', 
        samplepoints=num_colors, 
        low=0.0, 
        high=0.85, # Use up to 85% of the scale to exclude the lightest colors.
        colortype='rgb'
    )
    custom_palette = custom_palette[::-1] # Reverse for a more intuitive color progression.
    
    # --- Figure Creation ---
    # Create the treemap figure. The 'path' argument defines the hierarchy.
    fig = px.treemap(
        df_grouped,
        path=[px.Constant("Cell Lines"), 'tissue', 'cancer_type'], # Defines the hierarchy: Root -> tissue -> cancer_type
        values='counts',
        color_discrete_sequence=custom_palette
    )

    # --- Custom Text Formatting for Each Hierarchy Level ---
    # This approach is robust as it iterates through the figure's generated data structure.
    total_samples = len(df)
    ordered_templates = []
    for label, parent, value in zip(fig.data[0]['labels'], fig.data[0]['parents'], fig.data[0]['values']):
        if parent == "Cell Lines": # This is a Level 1 item (Tissue).
            # For tissues, display the name and its percentage of the total.
            percentage = value / total_samples
            template = f'<b>{label}</b><br>{percentage:.1%}'
            
        elif parent == "": # This is the top-level root node ("Cell Lines").
            template = "" # The root node itself doesn't need text.
            
        else: # This is a Level 2 item (Cancer Type).
            # For cancer types, display the name and the absolute sample count.
            count = int(value)
            template = f'<b>{label}</b><br>{count} samples'
            
        ordered_templates.append(template)
        
    # --- Layout and Trace Customizations ---
    fig.update_layout(
        margin=dict(t=20, l=0, r=0, b=10),
        plot_bgcolor='#f0f2f6' # Set a light background color.
    )
    
    fig.update_traces(
        maxdepth=2, # Ensure the treemap only displays two levels deep.
        texttemplate=ordered_templates, # Apply the custom text formats generated above.
        textinfo='text',
        pathbar_textfont_size=14,
        # Define a custom hover template for detailed tooltips.
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Sample Count: %{value}<br>"
            "Contribution to %{parent}: %{percentParent:.1%}"
            "<extra></extra>" # Hide the secondary hover box.
        ),
        textposition='middle center',
        insidetextfont={'size': 16, 'color': 'white'},
        marker=dict(cornerradius=5), # Add rounded corners to the rectangles.
        marker_pad=dict(t=5, l=5, r=5, b=5) # Add padding around each rectangle.
    )
    
    return fig

def create_stacked_bar_chart(drug_df: pd.DataFrame, main_axis: str, stack_by: str, top_n_stack: int = None):
    """
    Creates a stacked bar chart from drug data, grouped and stacked by specified criteria.
    Can optionally filter to show only the top N stacking categories.

    Args:
        drug_df (pd.DataFrame): DataFrame with drug information.
        main_axis (str): The column name for the x-axis categories.
        stack_by (str): The column name for stacking the bars (legend items).
        top_n_stack (int, optional): If provided, filters to the top N stacking categories. Defaults to None.

    Returns:
        go.Figure: A Plotly stacked bar chart figure.
    """
    # --- 1. Data Preparation ---
    df_filled = drug_df.copy()
    df_filled['targets'] = df_filled['targets'].fillna('Unknown')
    df_filled['pathway_name'] = df_filled['pathway_name'].fillna('Unknown')

    # Explode the DataFrame to handle drugs with multiple comma-separated pathways or targets.
    # This creates a separate row for each pathway/target of a drug, ensuring accurate counts.
    df_clean = df_filled[['drug_name', 'pathway_name', 'targets']].copy()
    df_clean['pathway_name'] = df_clean['pathway_name'].str.split(r'\s*,\s*')
    df_clean['targets'] = df_clean['targets'].str.split(r'\s*,\s*')
    df_exploded_pathway = df_clean.explode('pathway_name')
    df_exploded = df_exploded_pathway.explode('targets')
    
    # Count the number of unique drugs for each main_axis/stack_by combination.
    df_counts = df_exploded.groupby([main_axis, stack_by])['drug_name'].nunique().reset_index()
    df_counts.rename(columns={'drug_name': 'drug_count'}, inplace=True)
    
    # Filter out generic categories from the main axis for a cleaner chart.
    df_counts = df_counts[~df_counts[main_axis].isin(['Other', 'Unknown', 'Unclassified'])]

    
    # --- 2. Filter by Top N Stacking Categories (if specified) ---
    if top_n_stack:
        # Calculate the total count for each stacking category across all bars.
        stack_totals = df_counts.groupby(stack_by)['drug_count'].sum().sort_values(ascending=False)
        # Identify the names of the top N categories.
        top_n_categories = stack_totals.head(top_n_stack).index
        # Filter the DataFrame to include only data related to these top categories.
        df_filtered = df_counts[df_counts[stack_by].isin(top_n_categories)]
    else:
        df_filtered = df_counts

    # --- 3. Color Map Generation ---
    # Create a consistent color mapping for the stacking categories.
    unique_stacks = df_filtered[stack_by].unique()
    num_colors = len(unique_stacks)
    custom_palette = pcolors.sample_colorscale('Peach', samplepoints=num_colors)[::-1]
    color_map = {category: color for category, color in zip(unique_stacks, custom_palette)}

    # --- 4. Create the Stacked Bar Chart ---
    fig = px.bar(
        data_frame=df_filtered,
        x=main_axis,
        y='drug_count',
        color=stack_by,
        color_discrete_map=color_map,
        labels={
            'drug_count': 'Number of Unique Drugs', 
            'pathway_name': 'Pathways',  
            'targets': 'Targets'
        }
    )
    
    # --- 5. Layout and Style Customizations ---
    fig.update_layout(
        xaxis=dict(categoryorder='total descending', fixedrange=True), # Order bars by total height.
        yaxis=dict(fixedrange=True), # Disable y-axis zoom.
        plot_bgcolor='white',
        margin=dict(t=20, l=0, r=0, b=10),
        xaxis_showline=False, yaxis_showline=False,
        xaxis_showgrid=False, yaxis_showgrid=False,
        legend=dict(title_text='&nbsp;&nbsp;Targets') # Add padding to the legend title.
    )
    
    # Set the 'Unknown' category to be hidden by default in the legend for a cleaner initial view.
    fig.for_each_trace(lambda trace: trace.update(visible="legendonly") if trace.name == "Unknown" else ())
    # Add rounded corners to the bars for a modern aesthetic.
    fig.update_traces(marker=dict(cornerradius=5))

    return fig


def create_3d_cell_line_umap_plot(umap_df, color_by, hover_name):
    """
    Creates an interactive 3D scatter plot from pre-computed cell line UMAP data.
    """
    if umap_df is None or umap_df.empty:
        return create_placeholder_fig("Could not load UMAP data for cell lines.")

    # Use plotly.express.scatter_3d to create the base plot.
    fig = px.scatter_3d(
        umap_df,
        x='UMAP_X',
        y='UMAP_Y',
        z='UMAP_Z',
        color=color_by,
        hover_name=hover_name,
        labels={'color': color_by.replace('_', ' ').title()}
    )

    # Style the plot for a clean, "floating points" look.
    fig.update_layout(
        margin=dict(l=0, r=200, b=0, t=0), # Adjust right margin to fit legend.
        legend=dict(font=dict(size=12)),
        scene=dict(
            # Hide axis lines, ticks, and titles for a minimalist appearance.
            xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            # Make the scene background transparent.
            bgcolor="rgba(0,0,0,0)",
            # Set the initial camera position for an optimal viewing angle on load.
            camera=dict(
                eye=dict(x=0, y=1.3, z=0), # eye: position of the camera.
                center=dict(x=0, y=0, z=0), # center: point the camera is looking at.
                up=dict(x=0.1, y=0, z=1.2) # up: defines the "up" direction for the camera.
            )
        )
    )
    # Make the points smaller and slightly transparent.
    fig.update_traces(marker=dict(size=3, opacity=0.8))

    return fig


def create_3d_drug_umap_plot(umap_df, color_by='pathway_name'):
    """
    Creates an interactive 3D scatter plot from drug UMAP data, correctly handling
    multi-valued categories like 'targets' or 'pathway_name' by exploding the data.
    """
    if umap_df is None or umap_df.empty:
        return create_placeholder_fig("Could not load UMAP data for drugs.")
        
    # --- 1. Data Preparation ---
    # Create a copy and remove rows with missing essential data.
    plot_df = umap_df.dropna(subset=['targets', 'pathway_name']).copy()
    
    # --- 2. Data Transformation (Explode) ---
    # This is crucial for correctly coloring points with multiple categories.
    # Convert comma-separated strings into lists.
    plot_df['targets'] = plot_df['targets'].astype(str).str.split(r'\s*,\s*')
    plot_df['pathway_name'] = plot_df['pathway_name'].astype(str).str.split(r'\s*,\s*')
    
    # Explode the DataFrame based on the selected coloring category.
    # This duplicates a drug's row for each of its targets/pathways,
    # allowing Plotly to assign a color for each individual category.
    plot_df = plot_df.explode(color_by)
    
    # Filter out generic pathway names.
    plot_df = plot_df[~plot_df['pathway_name'].isin(['Other', '-'])]

    # --- 3. Create the 3D Scatter Plot ---
    fig = px.scatter_3d(
        plot_df,
        x='umap_x',
        y='umap_y',
        z='umap_z',
        color=color_by,
        hover_name='drug_name',
        # Pass full target and pathway info for the hover tooltip.
        custom_data=['targets', 'pathway_name'] 
    )

    # --- 4. Layout and Style Customizations ---
    # Apply the same minimalist styling as the cell line UMAP plot.
    fig.update_layout(
        margin=dict(l=0, r=200, b=0, t=0),
        legend=dict(font=dict(size=12)),
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
            # Adjust camera angle for the drug UMAP.
            camera=dict(
                eye=dict(x=1.25, y=0, z=0), 
                center=dict(x=0, y=0, z=0),
                up=dict(x=0.1, y=1.2, z=0)
            )
        )
    )

    # Define a custom hover template to correctly display multi-valued info.
    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{hovertext}</b>",
            "Targets: %{customdata[0]}",
            "Pathway: %{customdata[1]}",
            "<extra></extra>"
        ]),
        marker=dict(size=3, opacity=0.8)
    )

    return fig