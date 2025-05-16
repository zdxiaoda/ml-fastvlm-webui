# app_fastvlm_ui.py

import os

# import torch # No longer directly needed
# from PIL import Image # No longer directly needed here
import gradio as gr
import atexit

# import json # No longer directly needed
# import glob # No longer directly needed
# import requests # No longer directly needed
# import zipfile # No longer directly needed
# import shutil # No longer directly needed
import threading  # Still used for download wrapper
import queue  # Still used for download wrapper
import time  # Still used for download wrapper

# LLaVA GONE (moved to core_logic)

# --- Core Logic, API, and Model Downloader Imports ---
import core_logic
import model_downloader  # For direct access to MODEL_DOWNLOAD_LINKS in UI
from api import start_api_server, stop_api_server  # For API server control


# --- Initial Setup from Core Logic ---
core_logic.load_translations()  # Ensure translations are loaded

# --- Internationalization Support GONE (moved to core_logic) ---
# LOCALES_DIR, I18N_MESSAGES, DEFAULT_LANG, load_translations(), current_language_state, get_text() GONE

# --- Global variables: Model and Config GONE (moved to core_logic) ---
# MODEL, TOKENIZER, IMAGE_PROCESSOR, CONTEXT_LEN, MODEL_NAME_GLOBAL, MODEL_CONFIG GONE

# --- Default paths and parameters GONE (moved to core_logic) ---
# MODEL_ROOT_DIR, ALL_MODEL_PATHS, DEFAULT_MODEL_PATH, DEFAULT_MODEL_BASE, etc. GONE

# --- Handling generation_config.json GONE (moved to core_logic) ---
# _initial_model_path_expanded, ..., _renamed_generation_config_globally GONE
# _restore_generation_config_on_exit() GONE (core_logic handles its own via its atexit or load/unload)

# Register atexit hook for the final model's config restoration via core_logic
atexit.register(core_logic.restore_generation_config_for_current_model)

# --- Device automatic detection GONE (moved to core_logic) ---
# get_optimal_device() GONE

# --- Model loading functions GONE (moved to core_logic) ---
# load_model_globally(), unload_model_globally() GONE


# --- Gradio's Model Download Wrapper (will be refactored next to use core_logic.get_text) ---
# download_model_gradio_wrapper will be kept but refactored

# --- Gradio's predict function (will be refactored next to use core_logic) ---
# generate_description() / generate_description_i18n() will be refactored into generate_description_ui_wrapper()

# ... (rest of the file will be refactored in subsequent steps)

# --- UI Event Handler Functions (Refactored to use core_logic) ---


def on_lang_change_handler(new_lang, current_api_host, current_api_port):
    core_logic.set_current_language(new_lang)

    is_model_loaded_status = core_logic.is_model_loaded()
    current_button_text_key = (
        "unload_model_button" if is_model_loaded_status else "load_model_button"
    )
    current_button_variant = "stop" if is_model_loaded_status else "secondary"

    # Order of updates must exactly match the `outputs` list in `lang_dropdown_ui.change()`
    # This list will be fully populated when the UI structure is complete.
    updates = (
        gr.update(label=core_logic.get_text("lang_label")),
        (
            gr.update(label=core_logic.get_text("main_interface_tab_label"))
            if APP_UI_STRUCTURE.get("main_tab")
            else gr.update()
        ),  # main_tab label
        (
            gr.update(value=core_logic.get_text("title"))
            if APP_UI_STRUCTURE.get("title_md")
            else gr.update()
        ),
        (
            gr.update(value=core_logic.get_text("desc"))
            if APP_UI_STRUCTURE.get("desc_md")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("select_model"))
            if APP_UI_STRUCTURE.get("model_selector_ui")
            else gr.update()
        ),
        (
            gr.update(
                value=core_logic.get_text(current_button_text_key),
                variant=current_button_variant,
            )
            if APP_UI_STRUCTURE.get("load_unload_button_ui")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("model_load_unload_status_label"))
            if APP_UI_STRUCTURE.get("model_status_text_ui")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("upload_image"))
            if APP_UI_STRUCTURE.get("image_input_ui")
            else gr.update()
        ),
        (
            gr.update(
                label=core_logic.get_text("prompt_label"),
                value=core_logic.get_text("default_prompt"),
            )
            if APP_UI_STRUCTURE.get("prompt_input_ui")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("adv_params"))
            if APP_UI_STRUCTURE.get("adv_params_accordion")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("temperature"))
            if APP_UI_STRUCTURE.get("temperature_ui")
            else gr.update()
        ),
        (
            gr.update(
                label=core_logic.get_text("top_p"),
                info=core_logic.get_text("top_p_info"),
            )
            if APP_UI_STRUCTURE.get("top_p_ui")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("num_beams"))
            if APP_UI_STRUCTURE.get("num_beams_ui")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("conv_mode"))
            if APP_UI_STRUCTURE.get("conv_mode_ui")
            else gr.update()
        ),
        (
            gr.update(value=core_logic.get_text("submit"))
            if APP_UI_STRUCTURE.get("submit_button_ui")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("output"))
            if APP_UI_STRUCTURE.get("output_text_ui")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("select_model_to_download"))
            if APP_UI_STRUCTURE.get("model_download_selector_ui")
            else gr.update()
        ),
        (
            gr.update(value=core_logic.get_text("download_model_button"))
            if APP_UI_STRUCTURE.get("download_button_ui")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("download_status_label"))
            if APP_UI_STRUCTURE.get("download_status_ui")
            else gr.update()
        ),
        # API Tab components
        (
            gr.update(label=core_logic.get_text("api_settings_tab_label"))
            if APP_UI_STRUCTURE.get("api_tab")
            else gr.update()
        ),  # api_tab label
        (
            gr.update(value=core_logic.get_text("api_settings_desc"))
            if APP_UI_STRUCTURE.get("api_desc_md")
            else gr.update()
        ),
        (
            gr.update(
                label=core_logic.get_text("api_host_label"), value=current_api_host
            )
            if APP_UI_STRUCTURE.get("api_host_ui")
            else gr.update()
        ),
        (
            gr.update(
                label=core_logic.get_text("api_port_label"), value=current_api_port
            )
            if APP_UI_STRUCTURE.get("api_port_ui")
            else gr.update()
        ),
        (
            gr.update(value=core_logic.get_text("start_api_button"))
            if APP_UI_STRUCTURE.get("start_api_button_ui")
            else gr.update()
        ),
        (
            gr.update(value=core_logic.get_text("stop_api_button"))
            if APP_UI_STRUCTURE.get("stop_api_button_ui")
            else gr.update()
        ),
        (
            gr.update(label=core_logic.get_text("api_status_label"))
            if APP_UI_STRUCTURE.get("api_status_text_ui")
            else gr.update()
        ),
    )
    return updates


def generate_description_ui_wrapper(
    image_input_pil,
    prompt_str,
    temperature_float,
    top_p_float,
    num_beams_int,
    conv_mode_str,
):
    # Basic input validation, though core_logic might also do some.
    if not core_logic.is_model_loaded():
        return core_logic.get_text("error_model_not_init")
    if image_input_pil is None:  # Gradio Image input type='pil' gives PIL image or None
        return core_logic.get_text("error_no_image")
    if not prompt_str:
        return core_logic.get_text("error_no_prompt")

    # Call the actual inference function from core_logic
    return core_logic.execute_model_prediction(
        image_input_pil=image_input_pil,
        prompt_str=prompt_str,
        temperature_float=temperature_float,
        top_p_float=top_p_float,
        num_beams_int=num_beams_int,
        conv_mode_str=conv_mode_str,
        max_new_tokens=512,  # Or make this configurable in UI
    )


def on_model_change_handler(new_model_path):
    is_currently_loaded = core_logic.is_model_loaded()
    # Determine current button state to revert to if loading new model fails
    current_button_text_key = (
        "unload_model_button" if is_currently_loaded else "load_model_button"
    )
    current_button_variant = "stop" if is_currently_loaded else "secondary"

    if not new_model_path or not os.path.exists(os.path.expanduser(new_model_path)):
        return (
            core_logic.get_text(
                "error_invalid_model_path_selected", path=str(new_model_path)
            ),
            gr.update(
                value=core_logic.get_text(current_button_text_key),
                variant=current_button_variant,
            ),
            gr.update(
                interactive=is_currently_loaded
            ),  # Keep submit button state as per current loaded model
        )

    # core_logic.load_model_globally handles gen_config renaming and restoring.
    success = core_logic.load_model_globally(
        new_model_path, model_base_str=core_logic.DEFAULT_MODEL_BASE
    )

    if success:
        status_msg = core_logic.get_text("model_switched", model_path=new_model_path)
        button_update = gr.update(
            value=core_logic.get_text("unload_model_button"), variant="stop"
        )
        submit_update = gr.update(interactive=True)
    else:
        status_msg = core_logic.get_text(
            "error_model_load_failed", model_path=new_model_path
        )
        # Revert button to its state before the failed load attempt
        button_update = gr.update(
            value=core_logic.get_text(current_button_text_key),
            variant=current_button_variant,
        )
        submit_update = gr.update(
            interactive=is_currently_loaded
        )  # Revert submit interactive state
    return status_msg, button_update, submit_update


def handle_load_unload_click_handler(current_model_path_from_selector):
    if not core_logic.is_model_loaded():  # Current state is "Load Model"
        if not current_model_path_from_selector or not os.path.exists(
            os.path.expanduser(current_model_path_from_selector)
        ):
            status_msg = core_logic.get_text(
                "error_invalid_model_path_selected",
                path=str(current_model_path_from_selector),
            )
            button_update = gr.update(
                value=core_logic.get_text("load_model_button"), variant="secondary"
            )
            submit_update = gr.update(interactive=False)
            return status_msg, button_update, submit_update

        success = core_logic.load_model_globally(
            current_model_path_from_selector, core_logic.DEFAULT_MODEL_BASE
        )
        if success:
            status_msg = core_logic.get_text(
                "model_loaded_success", model_path=current_model_path_from_selector
            )
            button_update = gr.update(
                value=core_logic.get_text("unload_model_button"), variant="stop"
            )
            submit_update = gr.update(interactive=True)
        else:
            status_msg = core_logic.get_text(
                "error_model_load_failed", model_path=current_model_path_from_selector
            )
            button_update = gr.update(
                value=core_logic.get_text("load_model_button"), variant="secondary"
            )
            submit_update = gr.update(interactive=False)
    else:  # Current state is "Unload Model"
        core_logic.unload_model_globally()
        status_msg = core_logic.get_text("model_unloaded_success")
        button_update = gr.update(
            value=core_logic.get_text("load_model_button"), variant="secondary"
        )
        submit_update = gr.update(interactive=False)
    return status_msg, button_update, submit_update


def download_model_gradio_wrapper(
    model_name_to_download, progress=gr.Progress(track_tqdm=True)
):
    # Get current model list and default from core_logic to ensure UI consistency
    current_model_list, current_default_model = core_logic.update_global_model_paths()

    selector_ui_update = {"choices": current_model_list, "value": current_default_model}

    if not model_name_to_download:
        progress(0, desc="")  # Clear progress
        return (
            core_logic.get_text("error_no_model_selected_for_download"),
            gr.update(**selector_ui_update),  # Update dropdown with current paths
        )

    url = model_downloader.MODEL_DOWNLOAD_LINKS.get(model_name_to_download)
    if not url:
        progress(0, desc="")  # Clear progress
        return (
            core_logic.get_text(
                "error_model_url_not_found", model_name=model_name_to_download
            ),
            gr.update(**selector_ui_update),  # Update dropdown with current paths
        )

    update_q = queue.Queue()
    # _perform_download_and_extraction_task is from model_downloader, not core_logic
    thread = threading.Thread(
        target=model_downloader._perform_download_and_extraction_task,
        args=(model_name_to_download, url, update_q),
    )
    thread.start()

    status_message = core_logic.get_text(
        "download_queued", model_name=model_name_to_download
    )

    progress(0, desc=status_message)
    yield status_message, gr.update(**selector_ui_update)

    last_progress_update_time = time.time()
    loop_count = 0  # For occasional full refresh of selector paths from core_logic

    while thread.is_alive() or not update_q.empty():
        loop_count += 1
        try:
            item = update_q.get(timeout=0.05)
            message_type, *payload = item

            if message_type == "status":
                key, kwargs_dict = (
                    payload[0]
                    if isinstance(payload[0], tuple)
                    else (payload[0], payload[1] if len(payload) > 1 else {})
                )
                status_message = core_logic.get_text(key, **kwargs_dict)
                current_progress_val = getattr(progress, "current_progress", 0) or 0
                progress(current_progress_val, desc=status_message)

            elif message_type == "detailed_progress":
                data = payload[0]
                prog_fraction = (
                    (data["downloaded_bytes"] / data["total_bytes"])
                    if data["total_bytes"] > 0
                    else 0
                )
                # Prepare data for get_text more carefully
                fmt_data = {
                    "model_name": data.get("model_name", model_name_to_download),
                    "downloaded_mb": data.get("downloaded_bytes", 0) / (1024 * 1024),
                    "total_mb": data.get("total_bytes", 0) / (1024 * 1024),
                    "speed_mbps": data.get("speed_bps", 0) / (1024 * 1024),
                }
                status_message = core_logic.get_text(
                    "download_detailed_progress", **fmt_data
                )
                current_time = time.time()
                if (
                    current_time - last_progress_update_time >= 0.2
                    or prog_fraction == 1.0
                ):
                    progress(prog_fraction, desc=status_message)
                    last_progress_update_time = current_time

            elif (
                message_type == "selector_update_info"
            ):  # Downloader thread informs of path changes
                update_info = payload[0]
                if "choices" in update_info:
                    selector_ui_update["choices"] = update_info["choices"]
                    core_logic.ALL_MODEL_PATHS = update_info[
                        "choices"
                    ]  # Critical: keep core_logic in sync
                if (
                    "value" in update_info
                    and update_info["value"] in selector_ui_update["choices"]
                ):
                    selector_ui_update["value"] = update_info["value"]
                    core_logic.DEFAULT_MODEL_PATH = update_info[
                        "value"
                    ]  # Sync core_logic default
                else:  # If value absent or invalid, refresh from core_logic's potentially updated default
                    _, current_default_model = core_logic.update_global_model_paths()
                    selector_ui_update["value"] = current_default_model

            elif message_type == "finished":
                success_flag = payload[0]
                final_progress_value = (
                    1.0
                    if success_flag
                    else (getattr(progress, "current_progress", 0) or 0)
                )
                progress(
                    final_progress_value, desc=status_message
                )  # status_message should be set by "status" before "finished"

                # Crucially, refresh paths from core_logic which internally calls model_downloader.get_model_paths()
                refreshed_paths, new_default_path = (
                    core_logic.update_global_model_paths()
                )
                selector_ui_update["choices"] = refreshed_paths
                if success_flag:
                    # Try to select the newly downloaded model
                    # Assume model_name_to_download is the folder name
                    expected_new_model_path = os.path.join(
                        core_logic.MODEL_ROOT_DIR, model_name_to_download
                    )
                    if expected_new_model_path in refreshed_paths:
                        selector_ui_update["value"] = expected_new_model_path
                    else:
                        selector_ui_update["value"] = (
                            new_default_path  # Fallback to current default
                        )
                else:
                    selector_ui_update["value"] = (
                        new_default_path  # On failure, use current default
                    )

                yield status_message, gr.update(**selector_ui_update)
                break  # Exit loop on finished

            # Periodically refresh the model list from core_logic in case of external changes (less critical)
            if loop_count % 20 == 0:  # Every ~1 second if timeout is 0.05s
                refreshed_paths, new_default_path = (
                    core_logic.update_global_model_paths()
                )
                selector_ui_update["choices"] = refreshed_paths
                # Don't change value unless necessary to avoid interrupting user selection during download
                if selector_ui_update["value"] not in refreshed_paths:
                    selector_ui_update["value"] = new_default_path

            yield status_message, gr.update(**selector_ui_update)

        except queue.Empty:
            if not thread.is_alive() and update_q.empty():
                break
            yield status_message, gr.update(**selector_ui_update)  # Keep UI responsive
            time.sleep(0.05)

    # Final state yield after loop completion
    final_prog_val = getattr(progress, "current_progress", 0) or 0
    # Ensure status_message reflects the final outcome
    # The last 'status' message from the queue before 'finished' should be the most accurate.
    if "success" in status_message.lower() and final_prog_val < 1.0:
        final_prog_val = 1.0
    elif "error" in status_message.lower() and final_prog_val == 1.0:
        final_prog_val = 0  # If error but 100%, show 0
    progress(final_prog_val, desc=status_message)

    # One last refresh and update of the selector state from core_logic
    refreshed_paths_final, final_default_value = core_logic.update_global_model_paths()
    final_ui_update = {"choices": refreshed_paths_final, "value": final_default_value}

    # If download was successful, try to ensure the downloaded model is selected
    # This check needs to be after the final status_message is determined
    if (
        message_type == "finished" and success_flag
    ):  # Check if loop broke due to 'finished' and it was a success
        expected_dl_model_path = os.path.join(
            core_logic.MODEL_ROOT_DIR, model_name_to_download
        )
        if expected_dl_model_path in refreshed_paths_final:
            final_ui_update["value"] = expected_dl_model_path

    yield status_message, gr.update(**final_ui_update)


# Placeholder for APP_UI_STRUCTURE. This will be defined in the UI build part.
APP_UI_STRUCTURE = {}


# --- API Server Control Callbacks ---
def handle_start_api_server_ui(api_host, api_port_str):
    try:
        api_port = int(api_port_str)
        if not (1024 <= api_port <= 65535):  # Standard port range check
            raise ValueError("Port number out of valid range (1024-65535)")
    except ValueError as e:
        return core_logic.get_text(
            "error_invalid_port", port=api_port_str, error=str(e)
        )

    status_message = start_api_server(host=api_host, port=api_port)
    return status_message


def handle_stop_api_server_ui():
    status_message = stop_api_server()
    return status_message


# --- Main Gradio App Setup (Full UI Build) ---
if __name__ == "__main__":
    if core_logic.DEFAULT_MODEL_PATH:
        print(
            f"FastVLM UI: Attempting to load initial model '{core_logic.DEFAULT_MODEL_PATH}'"
        )
        core_logic.load_model_globally(
            core_logic.DEFAULT_MODEL_PATH, core_logic.DEFAULT_MODEL_BASE
        )
    else:
        available_models, _ = core_logic.update_global_model_paths()
        if not model_downloader.MODEL_DOWNLOAD_LINKS and not available_models:
            error_key = "error_no_model_found_or_downloadable"
            print(f"[FATAL ERROR] {core_logic.get_text(error_key)}")
            # Consider exiting if this is critical: raise RuntimeError(core_logic.get_text(error_key))

    init_model_loaded = core_logic.is_model_loaded()
    init_btn_txt_key = (
        "unload_model_button" if init_model_loaded else "load_model_button"
    )
    init_btn_var = "stop" if init_model_loaded else "secondary"
    init_submit_interactive = init_model_loaded

    with gr.Blocks(theme="NoCrypt/miku") as app_ui:
        # Language Dropdown (place it visibly, perhaps at the top)
        with gr.Row():
            lang_choices = (
                [
                    (v.get("lang_display_name", k.upper()), k)
                    for k, v in core_logic.I18N_MESSAGES.items()
                ]
                if core_logic.I18N_MESSAGES
                else [("English", "en")]
            )
            lang_dropdown_ui = gr.Dropdown(
                choices=lang_choices,
                value=core_logic.current_language_state,
                label=core_logic.get_text("lang_label"),
                interactive=True,
            )
            APP_UI_STRUCTURE["lang_dropdown_ui"] = lang_dropdown_ui

        with gr.Tabs() as main_tabs:
            # ---- Main Interface Tab ----
            with gr.TabItem(
                label=core_logic.get_text("main_interface_tab_label")
            ) as main_interface_tab:
                APP_UI_STRUCTURE["main_tab"] = main_interface_tab
                title_md = gr.Markdown(core_logic.get_text("title"))
                APP_UI_STRUCTURE["title_md"] = title_md
                desc_md = gr.Markdown(core_logic.get_text("desc"))
                APP_UI_STRUCTURE["desc_md"] = desc_md

                with gr.Row():
                    with gr.Column(scale=1):
                        model_selector_ui = gr.Dropdown(
                            choices=core_logic.ALL_MODEL_PATHS,
                            value=core_logic.DEFAULT_MODEL_PATH,
                            label=core_logic.get_text("select_model"),
                        )
                        APP_UI_STRUCTURE["model_selector_ui"] = model_selector_ui

                        load_unload_button_ui = gr.Button(
                            value=core_logic.get_text(init_btn_txt_key),
                            variant=init_btn_var,
                        )
                        APP_UI_STRUCTURE["load_unload_button_ui"] = (
                            load_unload_button_ui
                        )

                        model_status_text_ui = gr.Textbox(
                            label=core_logic.get_text("model_load_unload_status_label"),
                            interactive=False,
                            lines=2,
                        )
                        APP_UI_STRUCTURE["model_status_text_ui"] = model_status_text_ui

                        image_input_ui = gr.Image(
                            type="pil",
                            label=core_logic.get_text("upload_image"),
                            sources=["upload", "clipboard", "webcam"],
                        )
                        APP_UI_STRUCTURE["image_input_ui"] = image_input_ui

                        prompt_input_ui = gr.Textbox(
                            label=core_logic.get_text("prompt_label"),
                            value=core_logic.get_text("default_prompt"),
                            lines=2,
                        )
                        APP_UI_STRUCTURE["prompt_input_ui"] = prompt_input_ui

                        adv_params_accordion = gr.Accordion(
                            core_logic.get_text("adv_params"), open=False
                        )
                        APP_UI_STRUCTURE["adv_params_accordion"] = adv_params_accordion
                        with adv_params_accordion:
                            temperature_ui = gr.Slider(
                                0.0,
                                2.0,
                                step=0.01,
                                value=core_logic.DEFAULT_TEMPERATURE,
                                label=core_logic.get_text("temperature"),
                            )
                            APP_UI_STRUCTURE["temperature_ui"] = temperature_ui
                            top_p_ui = gr.Slider(
                                0.0,
                                1.0,
                                step=0.01,
                                value=core_logic.DEFAULT_TOP_P,
                                label=core_logic.get_text("top_p"),
                                info=core_logic.get_text("top_p_info"),
                            )
                            APP_UI_STRUCTURE["top_p_ui"] = top_p_ui
                            num_beams_ui = gr.Slider(
                                1,
                                10,
                                step=1,
                                value=core_logic.DEFAULT_NUM_BEAMS,
                                label=core_logic.get_text("num_beams"),
                            )
                            APP_UI_STRUCTURE["num_beams_ui"] = num_beams_ui
                            conv_mode_ui = gr.Dropdown(
                                choices=sorted(list(core_logic.conv_templates.keys())),
                                value=core_logic.DEFAULT_CONV_MODE,
                                label=core_logic.get_text("conv_mode"),
                            )
                            APP_UI_STRUCTURE["conv_mode_ui"] = conv_mode_ui

                        submit_button_ui = gr.Button(
                            core_logic.get_text("submit"),
                            variant="primary",
                            interactive=init_submit_interactive,
                        )
                        APP_UI_STRUCTURE["submit_button_ui"] = submit_button_ui

                    with gr.Column(scale=1):
                        output_text_ui = gr.Textbox(
                            label=core_logic.get_text("output"),
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )
                        APP_UI_STRUCTURE["output_text_ui"] = output_text_ui

                        with gr.Group():  # Model Download Group
                            model_download_selector_ui = gr.Dropdown(
                                choices=list(
                                    model_downloader.MODEL_DOWNLOAD_LINKS.keys()
                                ),
                                label=core_logic.get_text("select_model_to_download"),
                                value=None,
                            )
                            APP_UI_STRUCTURE["model_download_selector_ui"] = (
                                model_download_selector_ui
                            )

                            download_button_ui = gr.Button(
                                core_logic.get_text("download_model_button"),
                                variant="secondary",
                            )
                            APP_UI_STRUCTURE["download_button_ui"] = download_button_ui

                            download_progress_bar_ui = gr.Progress()  # No direct label
                            # APP_UI_STRUCTURE['download_progress_bar_ui'] = download_progress_bar_ui # Not localized by label

                            download_status_ui = gr.Textbox(
                                label=core_logic.get_text("download_status_label"),
                                interactive=False,
                                lines=3,
                            )
                            APP_UI_STRUCTURE["download_status_ui"] = download_status_ui

            # ---- API Settings Tab ----
            with gr.TabItem(
                label=core_logic.get_text("api_settings_tab_label")
            ) as api_settings_tab:
                APP_UI_STRUCTURE["api_tab"] = api_settings_tab
                api_desc_md = gr.Markdown(core_logic.get_text("api_settings_desc"))
                APP_UI_STRUCTURE["api_desc_md"] = api_desc_md

                api_host_ui = gr.Textbox(
                    label=core_logic.get_text("api_host_label"), value="0.0.0.0"
                )
                APP_UI_STRUCTURE["api_host_ui"] = api_host_ui

                api_port_ui = gr.Textbox(
                    label=core_logic.get_text("api_port_label"), value="8008"
                )
                APP_UI_STRUCTURE["api_port_ui"] = api_port_ui

                start_api_button_ui = gr.Button(
                    core_logic.get_text("start_api_button"), variant="primary"
                )
                APP_UI_STRUCTURE["start_api_button_ui"] = start_api_button_ui

                stop_api_button_ui = gr.Button(
                    core_logic.get_text("stop_api_button"), variant="stop"
                )
                APP_UI_STRUCTURE["stop_api_button_ui"] = stop_api_button_ui

                api_status_text_ui = gr.Textbox(
                    label=core_logic.get_text("api_status_label"),
                    interactive=False,
                    lines=2,
                    value=core_logic.get_text("api_not_started_status"),
                )
                APP_UI_STRUCTURE["api_status_text_ui"] = api_status_text_ui

        # --- Define the order of outputs for language change handler ---
        # This must match the order of gr.update() calls in on_lang_change_handler
        # and the items in APP_UI_STRUCTURE used by it.
        ordered_ui_elements_for_lang_update = [
            APP_UI_STRUCTURE["lang_dropdown_ui"],
            APP_UI_STRUCTURE["main_tab"],
            APP_UI_STRUCTURE["title_md"],
            APP_UI_STRUCTURE["desc_md"],
            APP_UI_STRUCTURE["model_selector_ui"],
            APP_UI_STRUCTURE["load_unload_button_ui"],
            APP_UI_STRUCTURE["model_status_text_ui"],
            APP_UI_STRUCTURE["image_input_ui"],
            APP_UI_STRUCTURE["prompt_input_ui"],
            APP_UI_STRUCTURE["adv_params_accordion"],
            APP_UI_STRUCTURE["temperature_ui"],
            APP_UI_STRUCTURE["top_p_ui"],
            APP_UI_STRUCTURE["num_beams_ui"],
            APP_UI_STRUCTURE["conv_mode_ui"],
            APP_UI_STRUCTURE["submit_button_ui"],
            APP_UI_STRUCTURE["output_text_ui"],
            APP_UI_STRUCTURE["model_download_selector_ui"],
            APP_UI_STRUCTURE["download_button_ui"],
            APP_UI_STRUCTURE["download_status_ui"],
            APP_UI_STRUCTURE["api_tab"],
            APP_UI_STRUCTURE["api_desc_md"],
            APP_UI_STRUCTURE["api_host_ui"],
            APP_UI_STRUCTURE["api_port_ui"],
            APP_UI_STRUCTURE["start_api_button_ui"],
            APP_UI_STRUCTURE["stop_api_button_ui"],
            APP_UI_STRUCTURE["api_status_text_ui"],
        ]

        # --- Event Listeners ---
        lang_dropdown_ui.change(
            fn=on_lang_change_handler,
            inputs=[
                lang_dropdown_ui,
                api_host_ui,
                api_port_ui,
            ],  # api_host/port to preserve their values on lang change
            outputs=ordered_ui_elements_for_lang_update,
        )
        model_selector_ui.change(
            fn=on_model_change_handler,
            inputs=[model_selector_ui],
            outputs=[model_status_text_ui, load_unload_button_ui, submit_button_ui],
        )
        load_unload_button_ui.click(
            fn=handle_load_unload_click_handler,
            inputs=[model_selector_ui],
            outputs=[model_status_text_ui, load_unload_button_ui, submit_button_ui],
        )
        download_button_ui.click(
            fn=download_model_gradio_wrapper,
            inputs=[model_download_selector_ui],
            outputs=[download_status_ui, model_selector_ui],
        )
        submit_button_ui.click(
            fn=generate_description_ui_wrapper,
            inputs=[
                image_input_ui,
                prompt_input_ui,
                temperature_ui,
                top_p_ui,
                num_beams_ui,
                conv_mode_ui,
            ],
            outputs=output_text_ui,
            api_name="generate_description",  # Keep for Gradio client compatibility
        )

        # API Tab Listeners
        start_api_button_ui.click(
            fn=handle_start_api_server_ui,
            inputs=[api_host_ui, api_port_ui],
            outputs=[api_status_text_ui],
        )
        stop_api_button_ui.click(
            fn=handle_stop_api_server_ui, inputs=None, outputs=[api_status_text_ui]
        )

    app_ui.queue().launch(server_name="0.0.0.0", share=False)
