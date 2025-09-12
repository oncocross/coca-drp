# visualizations/data_overview.py
# Contains functions to generate plots for the "Data Overview" section of the UI.

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pcolors

from visualizations import create_placeholder_fig


def create_interactive_cell_line_treemap(cell_line_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive treemap to visualize the hierarchical composition of cell lines.
    """
    df = cell_line_df[['tissue', 'cancer_type']].dropna().copy()
    
    # Append a zero-width space to prevent Plotly path conflicts if a
    # cancer_type has the same name as a tissue.
    all_tissue_names = df['tissue'].unique()
    mask = df['cancer_type'].isin(all_tissue_names)
    df.loc[mask, 'cancer_type'] = df.loc[mask, 'cancer_type'] + '\u200B'
    
    # Group data to get counts for each hierarchy level.
    df_grouped = df.groupby(['tissue', 'cancer_type']).size().reset_index(name='counts')

    # Generate a custom color palette.
    num_colors = len(df['tissue'].unique())
    custom_palette = pcolors.sample_colorscale('Peach', samplepoints=num_colors, low=0.0, high=0.85, colortype='rgb')
    custom_palette = custom_palette[::-1]
    
    # Create the treemap figure.
    fig = px.treemap(
        df_grouped,
        path=[px.Constant("Cell Lines"), 'tissue', 'cancer_type'],
        values='counts',
        color_discrete_sequence=custom_palette
    )

    # Generate custom text templates for each level of the hierarchy.
    total_samples = len(df)
    ordered_templates = []
    for label, parent, value in zip(fig.data[0]['labels'], fig.data[0]['parents'], fig.data[0]['values']):
        if parent == "Cell Lines": # Level 1 (Tissue)
            percentage = value / total_samples
            template = f'<b>{label}</b><br>{percentage:.1%}'
        elif parent == "": # Root node
            template = ""
        else: # Level 2 (Cancer Type)
            count = int(value)
            template = f'<b>{label}</b><br>{count} samples'
        ordered_templates.append(template)
        
    # Apply layout and trace customizations.
    fig.update_layout(margin=dict(t=20, l=0, r=0, b=10), plot_bgcolor='#f0f2f6')
    fig.update_traces(
        maxdepth=2,
        texttemplate=ordered_templates,
        textinfo='text',
        pathbar_textfont_size=14,
        hovertemplate=("<b>%{label}</b><br>"
                       "Sample Count: %{value}<br>"
                       "Contribution to %{parent}: %{percentParent:.1%}"
                       "<extra></extra>"),
        textposition='middle center',
        insidetextfont={'size': 16, 'color': 'white'},
        marker=dict(cornerradius=5),
        marker_pad=dict(t=5, l=5, r=5, b=5)
    )
    
    return fig

def create_stacked_bar_chart(drug_df: pd.DataFrame, main_axis: str, stack_by: str, top_n_stack: int = 20) -> go.Figure:
    """
    Creates a stacked bar chart from drug data, optionally filtered to the top N categories.
    """
    # --- 1. Data Preparation ---
    df_filled = drug_df.copy()
    df_filled['targets'] = df_filled['targets'].fillna('Unknown')
    df_filled['pathway_name'] = df_filled['pathway_name'].fillna('Unknown')

    # Explode the DataFrame to handle drugs with multiple comma-separated values.
    df_clean = df_filled[['drug_name', 'pathway_name', 'targets']].copy()
    df_clean['pathway_name'] = df_clean['pathway_name'].str.split(r'\s*,\s*')
    df_clean['targets'] = df_clean['targets'].str.split(r'\s*,\s*')
    df_exploded_pathway = df_clean.explode('pathway_name')
    df_exploded = df_exploded_pathway.explode('targets')
    
    # Count unique drugs for each group.
    df_counts = df_exploded.groupby([main_axis, stack_by])['drug_name'].nunique().reset_index(name='drug_count')
    df_counts = df_counts[~df_counts[main_axis].isin(['Other', 'Unknown', 'Unclassified'])]

    # --- 2. Filter by Top N Stacking Categories ---
    if top_n_stack:
        stack_totals = df_counts.groupby(stack_by)['drug_count'].sum().sort_values(ascending=False)
        top_n_categories = stack_totals.head(top_n_stack).index
        df_filtered = df_counts[df_counts[stack_by].isin(top_n_categories)]
    else:
        df_filtered = df_counts

    # --- 3. Color Map Generation ---
    unique_stacks = df_filtered[stack_by].unique()
    custom_palette = pcolors.sample_colorscale('Peach', samplepoints=len(unique_stacks))[::-1]
    color_map = {category: color for category, color in zip(unique_stacks, custom_palette)}

    # --- 4. Create the Bar Chart ---
    fig = px.bar(
        data_frame=df_filtered, x=main_axis, y='drug_count',
        color=stack_by, color_discrete_map=color_map,
        labels={'drug_count': 'Number of Unique Drugs', 'pathway_name': 'Pathways', 'targets': 'Targets'}
    )
    
    # --- 5. Layout and Style Customizations ---
    fig.update_layout(
        xaxis=dict(categoryorder='total descending', fixedrange=True),
        yaxis=dict(fixedrange=True),
        plot_bgcolor='white', margin=dict(t=20, l=0, r=0, b=10),
        xaxis_showline=False, yaxis_showline=False,
        xaxis_showgrid=False, yaxis_showgrid=False,
        legend=dict(title_text='&nbsp;&nbsp;Targets')
    )
    # Hide 'Unknown' category by default in the legend.
    fig.for_each_trace(lambda trace: trace.update(visible="legendonly") if trace.name == "Unknown" else ())
    fig.update_traces(marker=dict(cornerradius=5))

    return fig


def create_3d_cell_line_umap_plot(umap_df: pd.DataFrame, color_by: str, hover_name: str) -> go.Figure:
    """Creates an interactive 3D scatter plot from pre-computed cell line UMAP data."""
    if umap_df is None or umap_df.empty:
        return create_placeholder_fig("Could not load UMAP data for cell lines.")

    fig = px.scatter_3d(
        umap_df, x='UMAP_X', y='UMAP_Y', z='UMAP_Z',
        color=color_by, hover_name=hover_name,
        labels={'color': color_by.replace('_', ' ').title()}
    )

    # Style the plot for a clean, "floating points" look.
    fig.update_layout(
        margin=dict(l=0, r=200, b=0, t=0),
        legend=dict(font=dict(size=12)),
        scene=dict(
            # Configure axes for a minimalist appearance.
            xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
            # Set the initial camera angle.
            camera=dict(eye=dict(x=0, y=1.3, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0.1, y=0, z=1.2))
        )
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))

    return fig


def create_3d_drug_umap_plot(umap_df: pd.DataFrame, color_by: str = 'pathway_name') -> go.Figure:
    """Creates a 3D scatter plot from drug UMAP data, handling multi-valued categories."""
    if umap_df is None or umap_df.empty:
        return create_placeholder_fig("Could not load UMAP data for drugs.")
        
    plot_df = umap_df.dropna(subset=['targets', 'pathway_name']).copy()
    
    # Explode dataframe to handle multiple comma-separated categories per drug.
    plot_df['targets'] = plot_df['targets'].astype(str).str.split(r'\s*,\s*')
    plot_df['pathway_name'] = plot_df['pathway_name'].astype(str).str.split(r'\s*,\s*')
    plot_df = plot_df.explode(color_by)
    plot_df = plot_df[~plot_df['pathway_name'].isin(['Other', '-'])]

    # Create the 3D scatter plot.
    fig = px.scatter_3d(
        plot_df, x='umap_x', y='umap_y', z='umap_z',
        color=color_by, hover_name='drug_name',
        custom_data=['targets', 'pathway_name']
    )

    # Apply minimalist styling.
    fig.update_layout(
        margin=dict(l=0, r=200, b=0, t=0),
        legend=dict(font=dict(size=12)),
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, showline=False, showticklabels=False, title='', backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.25, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0.1, y=1.2, z=0))
        )
    )
    # Define a custom hover template.
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