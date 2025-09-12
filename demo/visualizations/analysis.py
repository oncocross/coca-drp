# visualizations/analysis.py
# Contains functions to generate plots for the main analysis tabs.

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ._helpers import create_placeholder_fig

# Maps user-facing UI labels to actual DataFrame column names.
GROUP_COLUMN_MAP = {
    'Tissue': 'tissue',
    'Cancer type': 'cancer_type',
    'Target': 'targets',
    'Pathway': 'pathway_name',
    'MOA': 'moa'
}


def create_cell_line_ic50_comp_plot(pred_df: pd.DataFrame, threshold: float, group_by: str) -> tuple[go.Figure, str]:
    """Creates a donut chart showing the composition of sensitive cell lines."""
    if pred_df is None or pred_df.empty:
        return create_placeholder_fig(), "Waiting for results..."

    # Filter for sensitive cell lines based on the lnIC50 threshold.
    sensitive_df = pred_df[pred_df['pred_lnIC50'] <= threshold]
    if sensitive_df.empty:
        return create_placeholder_fig("No data matches the criteria."), "No matching data found."

    # --- Data Preparation ---
    group_col = GROUP_COLUMN_MAP[group_by]
    counts = sensitive_df[group_col].value_counts()
    percents = sensitive_df[group_col].value_counts(normalize=True) * 100

    # Aggregate slices representing <3% into an 'Others' category.
    main_labels_series = percents[percents >= 3]
    other_labels_series = percents[percents < 3]
    
    # Prepare lists for plot data, including rich hover text.
    full_labels = list(main_labels_series.index)
    plot_values = list(main_labels_series.values)
    custom_hover_data = []

    for label in main_labels_series.index:
        hover_text = (f"<b>{label}</b><br>"
                      f"Count: {counts[label]}<br>"
                      f"Percent: {percents[label]:.2f}%")
        custom_hover_data.append(hover_text)

    if not other_labels_series.empty:
        full_labels.append('Others')
        plot_values.append(other_labels_series.sum())
        others_detail = "".join([f"<br> - {label}: {counts[label]} ({percents[label]:.2f}%)" for label in other_labels_series.index])
        others_hover_text = (f"<b>Others (Total)</b><br>"
                             f"Count: {counts[other_labels_series.index].sum()}<br>"
                             f"Percent: {other_labels_series.sum():.2f}%<br>"
                             f"--------------------{others_detail}")
        custom_hover_data.append(others_hover_text)

    # Truncate long labels for on-chart display.
    def truncate_label(label, max_len=25):
        return label if len(label) <= max_len else label[:max_len-3] + "..."
    display_labels = [truncate_label(label) for label in full_labels]
    
    # Create the plot using full labels for the legend and truncated labels for on-chart text.
    fig = go.Figure(data=[go.Pie(
        labels=full_labels,     # Full labels for the interactive legend.
        text=display_labels,    # Truncated labels for the on-chart display.
        values=plot_values,
        customdata=custom_hover_data,
        hovertemplate='%{customdata}<extra></extra>',
        hole=.5, sort=False, direction='clockwise'
    )])
    
    # Configure on-chart text to be auto-positioned and use the truncated 'text' property.
    fig.update_traces(texttemplate='%{text}<br>%{percent:.1%}', textposition='auto')
    fig.update_layout(
        height=600, showlegend=True,
        legend_title_text=group_by.replace('_', ' ').title(),
        margin=dict(t=50, b=50, l=50, r=50),
        uniformtext_minsize=9, uniformtext_mode='hide'
    )
        
    info_text = f"Out of **{len(pred_df)}** total cell lines, **{len(sensitive_df)}** unique cell lines matching the criteria."
    return fig, info_text


def create_cell_line_ic50_dist_plot(pred_df: pd.DataFrame, threshold: float, group_by: str) -> tuple[go.Figure, str]:
    """Creates a minimalist violin plot of predicted lnIC50 values."""
    if pred_df is None or pred_df.empty:
        return create_placeholder_fig(), ""

    sensitive_df = pred_df[pred_df['pred_lnIC50'] <= threshold]
    if sensitive_df.empty:
        return create_placeholder_fig("No data matches the criteria."), "No matching data found."

    group_col = GROUP_COLUMN_MAP[group_by]
    fig = px.violin(
        sensitive_df, x=group_col, y='pred_lnIC50',
        color=group_col, box=False, points="all"
    )
    # Customize traces for a cleaner aesthetic.
    fig.update_traces(
        jitter=0, pointpos=0, marker=dict(size=3, opacity=0.7),
        fillcolor='rgba(0,0,0,0)', line_width=1.5, meanline_visible=True
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=group_by.replace('_', ' ').title(), yaxis_title="Predicted lnIC50",
        showlegend=False, dragmode=False, height=600, margin=dict(t=50, b=20, l=0, r=0)
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_xaxes(showgrid=False)
    
    info_text = f"Out of **{len(pred_df)}** total cell lines, **{len(sensitive_df)}** unique cell lines matching the criteria."
    return fig, info_text


def create_drug_similarity_dist_plot(sim_df: pd.DataFrame, score_threshold: float, group_by: str) -> tuple[go.Figure, str]:
    """Creates a bar chart showing the distribution of similar drugs."""
    if sim_df is None or sim_df.empty:
        return create_placeholder_fig(), "Waiting for results..."

    sim_df['Score'] = pd.to_numeric(sim_df['Score'], errors='coerce')
    filtered_df = sim_df[sim_df['Score'] >= score_threshold].copy()

    if filtered_df.empty:
        message = f"No drugs found with score ≥ {score_threshold}."
        return create_placeholder_fig(message), message

    # Preprocess data to handle multiple comma-separated values per drug.
    group_col = GROUP_COLUMN_MAP[group_by]
    filtered_df[group_col] = filtered_df[group_col].fillna('Unknown').str.split(r'\s*,\s*')
    exploded_df = filtered_df.explode(group_col)

    # Aggregate data to count unique drugs per category.
    agg_df = exploded_df.groupby(group_col).agg(
        Count=('drug', 'nunique'),
        Drug_List=('drug', lambda s: '<br>'.join(s.unique()))
    ).reset_index().sort_values('Count', ascending=False)
    
    # Exclude generic categories for a cleaner plot.
    agg_df = agg_df[~agg_df[group_col].isin(['Other', 'Unknown', 'Unclassified', '-'])]

    fig = px.bar(
        agg_df, x=group_col, y='Count', color=group_col,
        text_auto=True, custom_data=['Drug_List']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=group_by, yaxis_title="Number of Drugs",
        legend_title_text=group_by.replace('_', ' ').title(),
        dragmode=False, height=600, margin=dict(t=50, b=20, l=0, r=0)
    )
    # Customize traces and hover template for rich information display.
    fig.update_traces(
        textposition="outside", marker=dict(cornerradius=5),
        hovertemplate=(f"<b>{group_by}: %{{x}}</b><br>"
                       "Drug Count: %{y}<br>"
                       "--------------------<br>"
                       "<b>Included Drugs:</b><br>%{customdata[0]}"
                       "<extra></extra>")
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
    info_text = f"Out of **{len(sim_df)}** total drugs, **{len(filtered_df)}** unique drugs matching the criteria."
    return fig, info_text


def create_drug_ic50_correlation_plot(pred_df: pd.DataFrame, ref_ic50_df: pd.DataFrame, selected_drug: str) -> go.Figure:
    """
    Creates a scatter plot correlating predicted vs. reference lnIC50 values.
    Handles both internal (GDSC) and external (DRH) data sources.
    """
    if pred_df is None or ref_ic50_df is None or not selected_drug:
        return create_placeholder_fig("Select a drug to compare.")
    
    # Infer data source and apply GDSC-specific deduplication logic if needed.
    is_internal_data = any(';' in col for col in ref_ic50_df.columns)
    if is_internal_data:
        col_meta = pd.DataFrame({'full_name': ref_ic50_df.columns})
        parts = col_meta['full_name'].str.split(';', expand=True)
        col_meta['drug_name'] = parts[1]
        col_meta['database'] = parts[2]
        
        drugs_in_gdsc2 = set(col_meta[col_meta['database'] == 'GDSC2']['drug_name'])
        cols_to_keep = col_meta[
            (col_meta['database'] == 'GDSC2') | 
            (~col_meta['drug_name'].isin(drugs_in_gdsc2))
        ]['full_name'].tolist()
        ref_ic50_df_processed = ref_ic50_df[cols_to_keep]
    else:
        ref_ic50_df_processed = ref_ic50_df

    # Find the corresponding column for the selected drug.
    target_col = None
    if selected_drug in ref_ic50_df_processed.columns:
        target_col = selected_drug
    else:
        try:
            target_col = next(col for col in ref_ic50_df_processed.columns if selected_drug in col.split(';'))
        except StopIteration:
            return create_placeholder_fig(f"Data not found for '{selected_drug}'.")
    
    # Prepare data for plotting by merging prediction and reference data.
    ref_drug_ic50 = ref_ic50_df_processed[[target_col]].rename(columns={target_col: 'Reference_lnIC50'})
    pred_data_for_merge = pred_df[['cell_lines', 'tissue', 'cancer_type', 'pred_lnIC50']].drop_duplicates(subset=['cell_lines']).set_index('cell_lines')
    pred_data_for_merge = pred_data_for_merge.rename(columns={'pred_lnIC50': 'Predicted_lnIC50'})
    
    merged_df = pd.merge(ref_drug_ic50, pred_data_for_merge, left_index=True, right_index=True).dropna()

    if len(merged_df) < 2:
        return create_placeholder_fig("Not enough common cell lines to compare.")

    fig = px.scatter(
        merged_df,
        x='Reference_lnIC50',
        y='Predicted_lnIC50',
        labels={'Reference_lnIC50': f'{selected_drug} lnIC50 (Reference)', 'Predicted_lnIC50': 'Input SMILES lnIC50 (Predicted)'},
        trendline="ols",
        trendline_color_override="red",
        hover_name=merged_df.index,
        custom_data=['tissue', 'cancer_type']
    )

    # Customize the hover template to show detailed cell line information.
    fig.update_traces(
        hovertemplate=(
            f"<b>%{{hovertext}}</b><br><br>"
            f"Tissue: %{{customdata[0]}}<br>"
            f"Cancer Type: %{{customdata[1]}}<br>"
            f"<b>{selected_drug} lnIC50:</b> %{{x:.3f}}<br>"
            f"<b>Input SMILES lnIC50:</b> %{{y:.3f}}"
            f"<extra></extra>"
        ),
        selector=dict(mode='markers') # Apply only to scatter points, not the trendline.
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        dragmode=False, height=600, margin=dict(t=50, b=20, l=0, r=0)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')

    return fig