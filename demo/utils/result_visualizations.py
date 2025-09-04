# utils/result_visualizations.py

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
import statsmodels.api as sm

# A dictionary to map user-facing dropdown options to the actual DataFrame column names.
# This allows the UI to show "Cancer Type" while the code uses "cancer_type".
group_column_map = {
    'Tissue': 'tissue',
    'Cancer type': 'cancer_type',
    'Target': 'targets',
    'Pathway': 'pathway_name',
}

def create_placeholder_fig(message="Results will be displayed here after running a prediction."):
    """
    Creates an empty Plotly Figure with a centered message.
    This is used as the initial state for all plot areas in the UI.

    Args:
        message (str): The message to display in the center of the plot area.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure()
    # Update layout to be transparent and hide all axes for a clean look.
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        # Add a text annotation in the center of the figure.
        annotations=[{
            "text": message, "xref": "paper", "yref": "paper",
            "showarrow": False, "font": {"size": 16, "color": "grey"}
        }]
    )
    return fig


def create_ic50_comp_plot(df, threshold, group_by):
    """
    Creates a donut chart showing the composition of sensitive cell lines.
    It aggregates categories that constitute less than 3% into an 'Others' slice
    and provides a detailed breakdown in the hover tooltip.

    Args:
        df (pd.DataFrame): The prediction DataFrame.
        threshold (float): The lnIC50 threshold to define "sensitive" cell lines.
        group_by (str): The user-selected category to group by (e.g., 'Tissue').

    Returns:
        tuple: A tuple containing the (Plotly figure, informational text string).
    """
    # Return a placeholder if the prediction has not been run yet.
    if df is None or df.empty:
        return create_placeholder_fig(), "Waiting for results..."

    # Filter the DataFrame to include only sensitive cell lines based on the threshold.
    filtered_df = df[df['pred_lnIC50'] <= threshold]

    # Return a placeholder if no data meets the filtering criteria.
    if filtered_df.empty:
        return create_placeholder_fig("No data matches the criteria."), "No data matches the criteria."

    # --- Core Charting Logic ---
    # 1. Calculate counts and percentages for the specified group.
    group_col = group_column_map[group_by]
    composition_counts = filtered_df[group_col].value_counts()
    composition_percent = filtered_df[group_col].value_counts(normalize=True) * 100
    
    # 2. Identify small categories to be grouped into 'Others'.
    above_threshold_labels = composition_percent[composition_percent >= 3].index
    below_threshold_labels = composition_percent[composition_percent < 3].index

    # 3. Prepare data lists for plotting and for custom hover information.
    plot_labels, plot_values, custom_hover_data = [], [], []

    # 3a. Add data for major categories (>= 3%).
    for label in above_threshold_labels:
        plot_labels.append(label)
        plot_values.append(composition_percent[label])
        hover_text = (
            f"<b>{label}</b><br>"
            f"Count: {composition_counts[label]}<br>"
            f"Percent: {composition_percent[label]:.2f}%"
        )
        custom_hover_data.append(hover_text)

    # 3b. Aggregate and add data for the 'Others' category if it exists.
    if not below_threshold_labels.empty:
        others_percent_sum = composition_percent[below_threshold_labels].sum()
        others_count_sum = composition_counts[below_threshold_labels].sum()
        
        plot_labels.append('Others')
        plot_values.append(others_percent_sum)
        
        # Create a detailed breakdown for the 'Others' hover tooltip.
        others_detail_text = "".join([f"<br> - {label}: {composition_counts[label]} ({composition_percent[label]:.2f}%)" for label in below_threshold_labels])
        others_hover_text = (
            f"<b>Others (Total)</b><br>"
            f"Count: {others_count_sum}<br>"
            f"Percent: {others_percent_sum:.2f}%<br>"
            f"--------------------"
            f"{others_detail_text}"
        )
        custom_hover_data.append(others_hover_text)

    # 4. Create the donut chart using Plotly Graph Objects.
    fig = go.Figure(data=[go.Pie(
        labels=plot_labels,
        values=plot_values,
        customdata=custom_hover_data,
        hovertemplate='%{customdata}<extra></extra>', # Use custom data for rich hover info.
        hole=.5, # This parameter turns the pie chart into a donut chart.
        sort=False,
        direction='clockwise'
    )])
    
    # Update trace properties for better text display.
    fig.update_traces(
        texttemplate='%{label}<br>%{percent:.1%}',
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]) # Fix domain to prevent legend from resizing chart.
    )
    
    # Update layout properties for a clean and professional look.
    fig.update_layout(
        height=600, 
        showlegend=True,
        legend_title_text=group_by.replace('_', ' ').title(),
        margin=dict(t=50, b=50, l=50, r=50),
        uniformtext_minsize=9, # Ensures text inside slices is readable.
        uniformtext_mode='hide' # Hides text if it doesn't fit, preventing overlap.
    )

    # Create a summary text to display alongside the plot.
    info_text = f"Out of **{len(df)}** total cell lines, **{len(filtered_df)}** match this condition."
    
    return fig, info_text


def create_ic50_dist_plot(df, threshold, group_by):
    """
    Creates a minimalist violin plot to show the distribution of predicted lnIC50 values,
    grouped by a selected category.

    Args:
        df (pd.DataFrame): The prediction DataFrame.
        threshold (float): The lnIC50 threshold to define sensitive cell lines.
        group_by (str): The column to group the data by.

    Returns:
        tuple: A tuple containing the (Plotly figure, informational text string).
    """
    if df is None or df.empty:
        return create_placeholder_fig(), ""

    filtered_df = df[df['pred_lnIC50'] <= threshold]

    if filtered_df.empty:
        return create_placeholder_fig("No data matches the criteria."), "No data matches the criteria."
    
    # 1. Create a violin plot using Plotly Express.
    fig = px.violin(
        filtered_df,
        x=group_column_map[group_by],
        y='pred_lnIC50',
        color=group_column_map[group_by],
        box=False, # Hide the inner box plot for a cleaner look.
        points="all" # Show all individual data points.
    )
    
    # 2. Customize the traces for a minimalist aesthetic.
    fig.update_traces(
        jitter=0, # Align points vertically along the center.
        pointpos=0, # Center the points within the violin.
        marker=dict(size=3, opacity=0.7),
        fillcolor='rgba(0,0,0,0)', # Make the violin area transparent.
        line_width=1.5, # Keep the outline of the violin shape.
        meanline_visible=True # Show a horizontal line for the mean value.
    )
    
    # 3. Update the layout to remove background colors and disable interaction.
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=group_by.replace('_', ' ').title(),
        yaxis_title="Predicted lnIC50",
        showlegend=False,
        dragmode=False, # Disable zoom and pan.
        height=600,
        margin=dict(t=50, b=20, l=0, r=0)
    )

    # Add light gridlines for better readability of values.
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')

    info_text = f"Out of **{len(df)}** total cell lines, **{len(filtered_df)}** match this condition."
    
    return fig, info_text


def create_drug_similarity_dist_plot(df, score_threshold, group_by):
    """
    Creates a bar chart showing the distribution of similar drugs, grouped by a
    selected category (e.g., Target or Pathway).

    Args:
        df (pd.DataFrame): The similarity DataFrame.
        score_threshold (float): The similarity score threshold for filtering.
        group_by (str): The column to group the drugs by.

    Returns:
        tuple: A tuple containing the (Plotly figure, informational text string).
    """
    if df is None or df.empty:
        return create_placeholder_fig(), "Waiting for results..."

    # Ensure the 'Score' column is treated as a numeric type.
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    
    # 1. Filter the DataFrame based on the similarity score threshold.
    filtered_df = df[df['Score'] >= score_threshold].copy()

    if filtered_df.empty:
        message = f"No similar drugs found with a score of {score_threshold} or higher."
        return create_placeholder_fig(message), message
    
    # 2. Preprocess data for grouping. This handles multi-valued entries.
    group_col = group_column_map[group_by]
    filtered_df[group_col] = filtered_df[group_col].fillna('Unknown')
    # Explode the DataFrame to handle comma-separated values (e.g., "Target1, Target2")
    # by creating a new row for each value.
    df_exploded = filtered_df.assign(**{group_col: filtered_df[group_col].str.split(r'\s*,\s*')})
    df_exploded = df_exploded.explode(group_col)

    # 3. Aggregate the data: count unique drugs and create a drug list for the hover tooltip.
    agg_df = df_exploded.groupby(group_col).agg(
        Count=('drug', 'nunique'),
        Drug_List=('drug', lambda s: '<br>'.join(s.unique())) # Create an HTML-formatted list of drugs.
    ).reset_index().sort_values('Count', ascending=False)
    
    # Exclude generic or uninformative categories from the plot.
    categories_to_exclude = ['Other', 'Unknown', 'Unclassified', '-']
    agg_df = agg_df[~agg_df[group_col].isin(categories_to_exclude)]

    # 4. Create the bar chart using Plotly Express.
    fig = px.bar(
        agg_df,
        x=group_col,
        y='Count',
        color=group_col,
        text_auto=True, # Automatically display the count on top of each bar.
        custom_data=['Drug_List'] # Pass the drug list for use in the hover template.
    )
    
    # 5. Customize the layout and style.
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=group_by,
        yaxis_title="Number of Drugs",
        showlegend=True,
        legend_title_text=group_by.replace('_', ' ').title(),
        dragmode=False,
        height=600,
        margin=dict(t=50, b=20, l=0, r=0)
    )

    # 6. Customize the traces and hover template for rich information display.
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False,
        marker=dict(cornerradius=5), # Apply rounded corners to the bars.
        hovertemplate=(
            f"<b>{group_by}: %{{x}}</b><br>"
            "Drug Count: %{y}<br>"
            "--------------------<br>"
            "<b>Included Drugs:</b><br>"
            "%{customdata[0]}"
            "<extra></extra>" # Hide the secondary hover box for a cleaner look.
        )
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')

    # Calculate the total number of unique drugs displayed in the plot.
    num_filtered_drugs = agg_df['Count'].sum()
    
    info_text = f"Out of **{len(df)}** total drugs, **{num_filtered_drugs}** match this condition."
    
    return fig, info_text


def create_ic50_scatter_plot(pred_df, ic50_df, selected_drug_name):
    """
    Creates a scatter plot showing the lnIC50 correlation between the user's
    compound and a selected reference drug from the internal dataset.

    Args:
        pred_df (pd.DataFrame): The prediction DataFrame for the input compound.
        ic50_df (pd.DataFrame): The reference DataFrame with known IC50 values.
        selected_drug_name (str): The name of the reference drug to compare against.

    Returns:
        go.Figure: A Plotly scatter plot figure.
    """
    if pred_df is None or selected_drug_name is None:
        return create_placeholder_fig("Select a drug from the list above to compare.")

    # --- Pre-filter reference data to handle duplicate drugs from different sources ---
    # The raw data contains drugs tested in multiple screens (e.g., GDSC1, GDSC2).
    # This logic prioritizes GDSC2 data and removes duplicates to ensure a clean comparison.
    col_meta = pd.DataFrame({'full_name': ic50_df.columns})
    parts = col_meta['full_name'].str.split(';', expand=True)
    col_meta['drug_name'] = parts[1]
    col_meta['database'] = parts[2]
    
    drugs_in_gdsc2 = set(col_meta[col_meta['database'] == 'GDSC2']['drug_name'])
    cols_to_keep = col_meta[
        (col_meta['database'] == 'GDSC2') | 
        (~col_meta['drug_name'].isin(drugs_in_gdsc2))
    ]['full_name'].tolist()
    
    ic50_df_filtered = ic50_df[cols_to_keep]
    
    # 1. Find the full column name in the reference df that corresponds to the selected drug.
    try:
        target_col = next(col for col in ic50_df_filtered.columns if selected_drug_name in col.split(';'))
    except StopIteration:
        return create_placeholder_fig(f"Could not find data for '{selected_drug_name}'.")

    # 2. Prepare the data for merging.
    # X-axis data: The reference drug's known lnIC50 values.
    ref_drug_ic50 = ic50_df_filtered[[target_col]].rename(columns={target_col: 'Reference_lnIC50'})
    
    # Y-axis data: The input compound's predicted lnIC50 values.
    pred_drug_ic50 = pred_df.drop_duplicates(subset=['cell_lines']).set_index('cell_lines')[['pred_lnIC50']].rename(columns={'pred_lnIC50': 'Predicted_lnIC50'})
    
    # 3. Merge the two dataframes on the cell line index and drop rows with missing values.
    merged_df = pd.merge(ref_drug_ic50, pred_drug_ic50, left_index=True, right_index=True).dropna()

    if len(merged_df) < 2:
        return create_placeholder_fig("Not enough common cell lines to compare.")
        
    # 4. Create the scatter plot with a regression trendline.
    fig = px.scatter(
        merged_df,
        x='Reference_lnIC50',
        y='Predicted_lnIC50',
        labels={'Reference_lnIC50': f'{selected_drug_name} lnIC50 (Reference)', 'Predicted_lnIC50': 'Your Compound lnIC50 (Predicted)'},
        trendline="ols", # Add an Ordinary Least Squares regression trendline.
        trendline_color_override="red",
        hover_data=[merged_df.index] # Show cell line name on hover.
    )

    # 5. Apply standard styling for consistency.
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
        height=600,
        margin=dict(t=50, b=20, l=0, r=0)
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')

    return fig


def create_external_ic50_scatter_plot(pred_df, ic50_df, selected_drug_name):
    """
    Creates a scatter plot showing the lnIC50 correlation between the user's
    compound and a selected reference drug from the EXTERNAL dataset.

    Args:
        pred_df (pd.DataFrame): The prediction DataFrame for the input compound.
        ic50_df (pd.DataFrame): The external reference DataFrame with known IC50 values.
        selected_drug_name (str): The name of the reference drug to compare against.

    Returns:
        go.Figure: A Plotly scatter plot figure.
    """
    if pred_df is None or selected_drug_name is None:
        return create_placeholder_fig("Select a drug to compare.")

    # 1. Check if the selected drug exists in the external reference DataFrame.
    if selected_drug_name not in ic50_df.columns:
        return create_placeholder_fig(f"Could not find data for '{selected_drug_name}'.")

    # 2. Prepare the data for merging.
    # X-axis data: The external reference drug's lnIC50 values.
    ref_drug_ic50 = ic50_df[[selected_drug_name]].rename(columns={selected_drug_name: 'Reference_lnIC50'})
    
    # Y-axis data: The input compound's predicted lnIC50 values.
    pred_drug_ic50 = pred_df.drop_duplicates(subset=['cell_lines']).set_index('cell_lines')[['pred_lnIC50']].rename(columns={'pred_lnIC50': 'Predicted_lnIC50'})
    
    # 3. Merge the two dataframes on the cell line index and drop missing values.
    merged_df = pd.merge(ref_drug_ic50, pred_drug_ic50, left_index=True, right_index=True).dropna()

    if len(merged_df) < 2:
        return create_placeholder_fig("Not enough common cell lines to compare.")
        
    # 4. Create the scatter plot with a regression trendline.
    fig = px.scatter(
        merged_df,
        x='Reference_lnIC50',
        y='Predicted_lnIC50',
        labels={'Reference_lnIC50': f'{selected_drug_name} lnIC50 (Reference)', 'Predicted_lnIC50': 'Your Compound lnIC50 (Predicted)'},
        trendline="ols",
        trendline_color_override="red",
        hover_data=[merged_df.index]
    )

    # 5. Apply standard styling.
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
        height=600,
        margin=dict(t=50, b=20, l=0, r=0)
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')

    return fig