# callbacks/event_listener.py
# This module centralizes the registration of all Gradio event listeners.
# It connects UI components (from the layout package) to their corresponding
# callback functions (from other modules in this package).

import gradio as gr
from functools import partial

# Import all handler functions from their respective modules.
from .data_overview import (
    handle_load_more_drug_targets,
    handle_umap_color_group_change
)
from .prediction import (
    handle_molecular_description,
    handle_smiles_input_change,
    handle_prediction_and_plot_generation
)
from .analysis import (
    handle_drug_ic50_correlation_source_switch,
    handle_drug_ic50_correlation_control_selection,
    handle_drug_ic50_correlation_page_change, 
    handle_drug_dist_source_switch,
    handle_drug_dist_plot_update
)

# Import visualization functions that are directly used as event handlers.
from visualizations import (
    create_cell_line_ic50_comp_plot,
    create_cell_line_ic50_dist_plot,
)


def register_event_listeners(components: dict, states: dict):
    """
    Registers all Gradio event listeners for the application.
    """

    # --- Overview Tab Events ---
    components["load_more_btn"].click(fn=handle_load_more_drug_targets, inputs=[components["pathway_top_n_state"]], outputs=[components["pathway_plot"], components["pathway_top_n_state"]])
    components["umap_cell_line_color_radio"].change(fn=partial(handle_umap_color_group_change, data_type="cell_line"), inputs=[components["umap_cell_line_color_radio"]], outputs=[components["cell_line_umap_plot"]])
    components["umap_drug_color_radio"].change(fn=partial(handle_umap_color_group_change, data_type="drug"), inputs=[components["umap_drug_color_radio"]], outputs=[components["drug_umap_plot"]])

    # --- Main Prediction Flow ---
    # Enable/disable predict button based on SMILES validity.
    components["smiles_input"].change(fn=handle_smiles_input_change, inputs=[components["smiles_input"]], outputs=[components["predict_button"]])
    
    # Define input and output lists for the main prediction button's chained events.
    predict_button_outputs_mol_desc = [
        components["overview_accordion"], components["mol_desc_panel"], components["mol_image_output"],
        components["formula_output"], components["weight_output"], components["can_smiles_output"], components["inchikey_output"],
        components["logp_output"], components["tpsa_output"], components["hdonors_output"], components["hacceptors_output"]
    ]
    predict_button_inputs_plots = [
        components["smiles_input"], components["ic50_comp_slider"], components["ic50_comp_group_by_radio"],
        components["dist_slider"], components["dist_group_by_radio"]
    ]
    predict_button_outputs_plots = [
        components["ic50_output"], components["similarity_output"], components["external_similarity_output"],
        states["pred_data"], states["sim_data"], states["external_sim_data"],
        components["ic50_comp_plot"], components["ic50_comp_info_output"], components["ic50_comp_controls"],
        components["ic50_dist_plot"], components["ic50_dist_info_output"], components["ic50_dist_controls"],
        components["dist_plot"], components["dist_info_output"], components["dist_controls"], components["dist_source_radio"], components["dist_source_info_md"],
        components["corr_source_radio"], components["corr_source_info_md"], components["corr_plot"], *components["corr_top_drug_buttons"],
        components["corr_controls"], components["corr_page_state"], components["corr_selected_drug_state"],
        components["corr_selected_drug_info_md"], components["corr_page_info_md"],
        components["corr_prev_button"], components["corr_next_button"]
    ]
    
    # Chain the two main events: 1. Show molecule info, 2. Run prediction and show all results.
    components["predict_button"].click(
        fn=partial(handle_molecular_description, close_accordion=True),
        inputs=[components["smiles_input"]],
        outputs=predict_button_outputs_mol_desc
    ).then(
        fn=handle_prediction_and_plot_generation,
        inputs=predict_button_inputs_plots,
        outputs=predict_button_outputs_plots
    )

    # --- Analysis Plot Control Events ---
    # Helper function to register listeners for slider/radio controls on plots.
    def register_plot_update_listeners(controls, handler_fn, inputs, outputs):
        for control in controls:
            (control.release if isinstance(control, gr.Slider) else control.change)(fn=handler_fn, inputs=inputs, outputs=outputs)

    # Listeners for the two Cell Line analysis plots.
    register_plot_update_listeners([components["ic50_comp_slider"], components["ic50_comp_group_by_radio"]], create_cell_line_ic50_comp_plot, [states["pred_data"], components["ic50_comp_slider"], components["ic50_comp_group_by_radio"]], [components["ic50_comp_plot"], components["ic50_comp_info_output"]])
    register_plot_update_listeners([components["ic50_dist_slider"], components["ic50_dist_group_by_radio"]], create_cell_line_ic50_dist_plot, [states["pred_data"], components["ic50_dist_slider"], components["ic50_dist_group_by_radio"]], [components["ic50_dist_plot"], components["ic50_dist_info_output"]])
    
    # Listeners for the Drug IC50 Correlation plot.
    components["corr_source_radio"].change(fn=handle_drug_ic50_correlation_source_switch, inputs=[components["corr_source_radio"], states["pred_data"], states["sim_data"], states["external_sim_data"]], outputs=[components["corr_source_info_md"], components["corr_plot"], components["corr_selected_drug_info_md"], *components["corr_top_drug_buttons"], components["corr_page_info_md"], components["corr_prev_button"], components["corr_next_button"], components["corr_page_state"], components["corr_selected_drug_state"]])
    
    for btn in components["corr_top_drug_buttons"]:
        btn.click(fn=handle_drug_ic50_correlation_control_selection, inputs=[btn, components["corr_source_radio"], states["pred_data"], states["sim_data"], states["external_sim_data"]] + components["corr_top_drug_buttons"], outputs=[components["corr_plot"], components["corr_selected_drug_info_md"], *components["corr_top_drug_buttons"], components["corr_selected_drug_state"]])
    components["corr_prev_button"].click(fn=partial(handle_drug_ic50_correlation_page_change, "prev"), inputs=[components["corr_source_radio"], components["corr_page_state"], states["sim_data"], states["external_sim_data"], components["corr_selected_drug_state"]], outputs=[components["corr_page_state"], *components["corr_top_drug_buttons"], components["corr_page_info_md"], components["corr_prev_button"], components["corr_next_button"]])
    components["corr_next_button"].click(fn=partial(handle_drug_ic50_correlation_page_change, "next"), inputs=[components["corr_source_radio"], components["corr_page_state"], states["sim_data"], states["external_sim_data"], components["corr_selected_drug_state"]], outputs=[components["corr_page_state"], *components["corr_top_drug_buttons"], components["corr_page_info_md"], components["corr_prev_button"], components["corr_next_button"]])

    # Listeners for the Drug Similarity Distribution plot.
    dist_plot_update_inputs = [components["dist_source_radio"], states["sim_data"], states["external_sim_data"], components["dist_slider"], components["dist_group_by_radio"]]
    dist_plot_update_outputs = [components["dist_plot"], components["dist_info_output"]]
    components["dist_source_radio"].change(fn=handle_drug_dist_source_switch, inputs=[components["dist_source_radio"]], outputs=[components["dist_group_by_radio"], components["dist_source_info_md"]]).then(fn=handle_drug_dist_plot_update, inputs=dist_plot_update_inputs, outputs=dist_plot_update_outputs)
    register_plot_update_listeners([components["dist_slider"], components["dist_group_by_radio"]], handle_drug_dist_plot_update, dist_plot_update_inputs, dist_plot_update_outputs)