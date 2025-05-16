# core_logic.py
import os
import torch
from PIL import Image
import gradio as gr  # Keep for Progress in stream_wrapper, but ideally remove direct gr dependency
import json
import glob
import requests  # Used by model_downloader, implicitly
import zipfile  # Used by model_downloader, implicitly
import shutil  # Used by model_downloader, implicitly
import threading  # Used by model_downloader, implicitly
import queue  # Used by model_downloader, implicitly
import time
import asyncio  # For async stream generation
from typing import Optional, List, Dict, Any, AsyncGenerator

# LLaVA related imports
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

import model_downloader  # Direct import

# --- Global Model State ---
MODEL = None
TOKENIZER = None
IMAGE_PROCESSOR = None
CONTEXT_LEN = None
MODEL_NAME_GLOBAL = None
MODEL_CONFIG = None

# --- Internationalization (i18n) ---
LOCALES_DIR = "locales"
I18N_MESSAGES = {}
DEFAULT_LANG = "en"
current_language_state = DEFAULT_LANG  # This will be updated by Gradio UI


def load_translations():
    """Loads all JSON translation files from LOCALES_DIR."""
    global I18N_MESSAGES
    if not os.path.isdir(LOCALES_DIR):
        return
    for file_path in glob.glob(os.path.join(LOCALES_DIR, "*.json")):
        lang_code = os.path.basename(file_path)[:-5]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                I18N_MESSAGES[lang_code] = json.load(f)
        except Exception:
            pass  # Silently ignore failed loads


load_translations()  # Load at module import


def get_text(key: str, lang: Optional[str] = None, **kwargs) -> str:
    """Gets translated text with parameterization and fallback."""
    resolved_lang = lang if lang else current_language_state
    lang_translations = I18N_MESSAGES.get(resolved_lang)

    if lang_translations is None or key not in lang_translations:
        lang_translations = I18N_MESSAGES.get(DEFAULT_LANG, {})

    text_template = lang_translations.get(key)

    if text_template is None:
        return f"[{resolved_lang.upper() if resolved_lang else DEFAULT_LANG.upper()}:{key}]"
    if kwargs:
        try:
            return text_template.format(**kwargs)
        except KeyError as e:
            return f"[{resolved_lang.upper() if resolved_lang else DEFAULT_LANG.upper()}:{key} - Format Error: {e}]"
    return text_template


def set_current_language(lang_code: str):
    """Sets the global current language for get_text."""
    global current_language_state
    if lang_code in I18N_MESSAGES or lang_code == DEFAULT_LANG:
        current_language_state = lang_code
    else:  # Fallback if lang_code is invalid
        current_language_state = DEFAULT_LANG


# --- Default Paths and Parameters ---
MODEL_ROOT_DIR = "model"
ALL_MODEL_PATHS = model_downloader.get_model_paths()  # Initialize from downloader
DEFAULT_MODEL_PATH = ALL_MODEL_PATHS[0] if ALL_MODEL_PATHS else None
DEFAULT_MODEL_BASE = None
DEFAULT_CONV_MODE = "qwen_2"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 1.0
DEFAULT_NUM_BEAMS = 1
# DEFAULT_PROMPT is fetched via get_text("default_prompt", current_language_state) when needed

# --- Generation Config Handling State ---
# These need to be mutable if different models have different configs
_current_model_gen_config_original_path = None
_current_model_gen_config_backup_path = None
_renamed_current_model_gen_config = False


def _update_gen_config_paths_for_model(model_path_str: Optional[str]):
    global _current_model_gen_config_original_path, _current_model_gen_config_backup_path
    if model_path_str:
        expanded_path = os.path.expanduser(model_path_str)
        _current_model_gen_config_original_path = os.path.join(
            expanded_path, "generation_config.json"
        )
        _current_model_gen_config_backup_path = os.path.join(
            expanded_path, ".generation_config.json"
        )
    else:
        _current_model_gen_config_original_path = None
        _current_model_gen_config_backup_path = None


def _rename_generation_config_for_current_model():
    global _renamed_current_model_gen_config
    if _current_model_gen_config_original_path and os.path.exists(
        _current_model_gen_config_original_path
    ):
        try:
            os.rename(
                _current_model_gen_config_original_path,
                _current_model_gen_config_backup_path,
            )
            _renamed_current_model_gen_config = True
        except OSError:
            _renamed_current_model_gen_config = False  # Failed to rename
        except Exception:
            _renamed_current_model_gen_config = False
    else:
        _renamed_current_model_gen_config = False  # No original file to rename


def restore_generation_config_for_current_model():
    """Restores the generation_config.json if it was backed up for the current model."""
    global _renamed_current_model_gen_config
    if _renamed_current_model_gen_config:
        if (
            _current_model_gen_config_backup_path
            and os.path.exists(_current_model_gen_config_backup_path)
            and _current_model_gen_config_original_path
            and not os.path.exists(_current_model_gen_config_original_path)
        ):
            try:
                os.rename(
                    _current_model_gen_config_backup_path,
                    _current_model_gen_config_original_path,
                )
            except Exception:
                pass  # Silently ignore
        _renamed_current_model_gen_config = False  # Mark as restored or attempt failed


# --- Device Detection ---
def get_optimal_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


# --- Model Management ---
def load_model_globally(
    model_path_str: str, model_base_str: Optional[str] = None
) -> bool:
    global MODEL, TOKENIZER, IMAGE_PROCESSOR, CONTEXT_LEN, MODEL_NAME_GLOBAL, MODEL_CONFIG

    # Before loading a new model, restore config of any previously loaded model
    if MODEL is not None:  # If a model is already loaded
        restore_generation_config_for_current_model()

    actual_model_path = os.path.expanduser(model_path_str)
    _update_gen_config_paths_for_model(
        actual_model_path
    )  # Update paths for the new model
    _rename_generation_config_for_current_model()  # Attempt to rename for new model

    disable_torch_init()
    _model_name_loaded = get_model_name_from_path(actual_model_path)
    selected_device = get_optimal_device()

    try:
        _tokenizer, _model, _image_processor, _context_len = load_pretrained_model(
            actual_model_path,
            model_base_str,
            _model_name_loaded,
            device=selected_device,
        )
    except Exception as e:
        # print(f"Failed to load model {actual_model_path}: {e}") # For debugging
        restore_generation_config_for_current_model()  # Restore if load failed
        _update_gen_config_paths_for_model(None)  # Clear gen config paths
        (
            MODEL,
            TOKENIZER,
            IMAGE_PROCESSOR,
            CONTEXT_LEN,
            MODEL_NAME_GLOBAL,
            MODEL_CONFIG,
        ) = (None, None, None, None, None, None)
        return False

    MODEL = _model
    TOKENIZER = _tokenizer
    IMAGE_PROCESSOR = _image_processor
    CONTEXT_LEN = _context_len
    MODEL_NAME_GLOBAL = _model_name_loaded
    MODEL_CONFIG = _model.config

    if TOKENIZER.pad_token_id is None:
        TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
    if (
        MODEL.generation_config.pad_token_id is None
        or MODEL.generation_config.pad_token_id != TOKENIZER.pad_token_id
    ):
        MODEL.generation_config.pad_token_id = TOKENIZER.pad_token_id

    return True


def unload_model_globally():
    global MODEL, TOKENIZER, IMAGE_PROCESSOR, CONTEXT_LEN, MODEL_NAME_GLOBAL, MODEL_CONFIG
    restore_generation_config_for_current_model()  # Restore before unloading
    MODEL, TOKENIZER, IMAGE_PROCESSOR, CONTEXT_LEN, MODEL_NAME_GLOBAL, MODEL_CONFIG = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    _update_gen_config_paths_for_model(None)  # Clear gen config paths
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def is_model_loaded() -> bool:
    return MODEL is not None


def load_default_model_if_not_loaded() -> bool:
    if not is_model_loaded():
        if DEFAULT_MODEL_PATH:
            print(f"CoreLogic: Attempting to load default model: {DEFAULT_MODEL_PATH}")
            return load_model_globally(DEFAULT_MODEL_PATH, DEFAULT_MODEL_BASE)
        else:
            print("CoreLogic: No default model path set. Cannot auto-load model.")
            return False
    return True  # Already loaded


def get_current_model_name() -> Optional[str]:
    return MODEL_NAME_GLOBAL


def get_default_prompt() -> str:
    return get_text(
        "default_prompt", current_language_state
    )  # Uses localized default prompt


def get_tokenizer():
    return TOKENIZER


def get_model_config_object():  # Renamed to avoid conflict with global MODEL_CONFIG
    return MODEL_CONFIG


def update_global_model_paths():
    """Refreshes ALL_MODEL_PATHS from model_downloader."""
    global ALL_MODEL_PATHS, DEFAULT_MODEL_PATH
    ALL_MODEL_PATHS = model_downloader.get_model_paths()
    if not DEFAULT_MODEL_PATH or DEFAULT_MODEL_PATH not in ALL_MODEL_PATHS:
        DEFAULT_MODEL_PATH = ALL_MODEL_PATHS[0] if ALL_MODEL_PATHS else None
    return ALL_MODEL_PATHS, DEFAULT_MODEL_PATH


# --- Inference Logic ---
def _prepare_inference_input(
    prompt_str: str, image_input_pil: Optional[Image.Image], conv_mode_str: str
):
    """Prepares inputs for the model. Factored out from original generate_description_i18n."""
    if not all([MODEL, TOKENIZER, IMAGE_PROCESSOR, MODEL_CONFIG]):
        raise ValueError(get_text("error_model_not_init", current_language_state))

    if (
        image_input_pil is None
        and "<image>" not in prompt_str
        and DEFAULT_IMAGE_TOKEN not in prompt_str
    ):  # check if prompt implies image
        # If no image explicitly passed and no image token in prompt, it might be text-only if model supports
        # For now, we assume FastVLM primarily works with images.
        # This behavior can be adjusted based on actual model capabilities.
        pass  # Allow to proceed, model might handle text-only or error out appropriately

    current_device = MODEL.device

    # Construct full_prompt based on whether an image is present
    if image_input_pil is not None:  # Image is provided
        if MODEL_CONFIG.mm_use_im_start_end:
            image_segment = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
        else:
            image_segment = DEFAULT_IMAGE_TOKEN
        # Check if prompt_str already contains an image token placeholder; if not, prepend.
        if DEFAULT_IMAGE_TOKEN not in prompt_str and "<image>" not in prompt_str:
            full_prompt = image_segment + "\\n" + prompt_str
        else:
            # Replace generic <image> with specific token if needed, or assume prompt is structured correctly
            full_prompt = (
                prompt_str.replace("<image>", image_segment)
                if "<image>" in prompt_str
                else prompt_str
            )

    else:  # No image provided, assume text-only if prompt is structured for it or model supports it
        full_prompt = prompt_str

    if conv_mode_str not in conv_templates:
        raise ValueError(
            get_text(
                "error_invalid_conv",
                current_language_state,
                conv_mode=conv_mode_str,
                modes=list(conv_templates.keys()),
            )
        )

    conv = conv_templates[conv_mode_str].copy()
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    processed_prompt_str = conv.get_prompt()

    try:
        input_ids_tensor = (
            tokenizer_image_token(
                processed_prompt_str, TOKENIZER, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(current_device)
        )
    except Exception as e:
        raise RuntimeError(
            get_text("error_tokenize", current_language_state, err=str(e))
        )

    image_tensor_processed = None
    if image_input_pil:
        try:
            # Ensure image is RGB for processing
            rgb_image_pil = (
                image_input_pil.convert("RGB")
                if image_input_pil.mode != "RGB"
                else image_input_pil
            )
            image_tensor_processed = process_images(
                [rgb_image_pil], IMAGE_PROCESSOR, MODEL_CONFIG
            )[0]
        except Exception as e:
            raise RuntimeError(
                get_text("error_image_proc", current_language_state, err=str(e))
            )

    return (
        input_ids_tensor,
        image_tensor_processed,
        image_input_pil.size if image_input_pil else None,
        current_device,
    )


def execute_model_prediction(
    image_input_pil: Optional[Image.Image],
    prompt_str: str,
    temperature_float: float,
    top_p_float: float,
    num_beams_int: int,
    conv_mode_str: str,
    max_new_tokens: int = 512,
) -> str:
    """Executes non-streaming model prediction."""

    if not is_model_loaded():
        return get_text("error_model_not_init", current_language_state)
    if image_input_pil is None:  # VLM usually needs an image
        # Adjust if model can be text-only
        return get_text("error_no_image", current_language_state)
    if not prompt_str:
        return get_text("error_no_prompt", current_language_state)

    try:
        input_ids_tensor, image_tensor_input, image_size, device = (
            _prepare_inference_input(prompt_str, image_input_pil, conv_mode_str)
        )
    except (
        ValueError,
        RuntimeError,
    ) as e:  # Catch errors from _prepare_inference_input
        return str(e)

    with torch.inference_mode():
        try:
            actual_top_p = top_p_float if 0 < top_p_float < 1.0 else None

            # Prepare images argument for model.generate()
            images_arg = (
                image_tensor_input.unsqueeze(0).half().to(device)
                if image_tensor_input is not None
                else None
            )
            image_sizes_arg = [image_size] if image_size is not None else None

            output_ids_tensor = MODEL.generate(
                input_ids_tensor,
                images=images_arg,
                image_sizes=image_sizes_arg,
                do_sample=True if temperature_float > 0 else False,
                temperature=temperature_float,
                top_p=actual_top_p,
                num_beams=num_beams_int,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        except Exception as e:
            return get_text("error_infer", current_language_state, err=str(e))

        generated_text = TOKENIZER.batch_decode(
            output_ids_tensor, skip_special_tokens=True
        )[0].strip()
    return generated_text


async def execute_model_prediction_stream(
    image_input_pil: Optional[Image.Image],
    prompt_str: str,
    temperature_float: float,
    top_p_float: float,
    num_beams_int: int,
    conv_mode_str: str,
    max_new_tokens: int = 512,
) -> AsyncGenerator[str, None]:
    """
    Executes streaming model prediction.
    Note: This is a simplified stream for models that don't natively support async token generation.
    It generates the full response and then yields it token by token (char by char for now).
    For true token-by-token streaming from the model, MODEL.generate would need to support a callback
    or be an async generator itself, or run in a thread with a queue.
    """
    if not is_model_loaded():
        yield get_text("error_model_not_init", current_language_state)
        return
    if image_input_pil is None:  # VLM usually needs an image
        yield get_text("error_no_image", current_language_state)
        return
    if not prompt_str:
        yield get_text("error_no_prompt", current_language_state)
        return

    try:
        # This part is synchronous
        input_ids_tensor, image_tensor_input, image_size, device = (
            _prepare_inference_input(prompt_str, image_input_pil, conv_mode_str)
        )
    except (ValueError, RuntimeError) as e:
        yield str(e)
        return

    # For actual streaming, the Hugging Face `generate` method can accept a `TextStreamer`
    # or a custom streamer class. For an API, we need to adapt this.
    # Simplest approach for now: generate full text then stream it out.
    # This does NOT provide real-time tokens as they are generated by the model.

    # --- Start of synchronous block for generation ---
    try:
        actual_top_p = top_p_float if 0 < top_p_float < 1.0 else None
        images_arg = (
            image_tensor_input.unsqueeze(0).half().to(device)
            if image_tensor_input is not None
            else None
        )
        image_sizes_arg = [image_size] if image_size is not None else None

        # This is the blocking call
        output_ids_tensor = MODEL.generate(
            input_ids_tensor,
            images=images_arg,
            image_sizes=image_sizes_arg,
            do_sample=True if temperature_float > 0 else False,
            temperature=temperature_float,
            top_p=actual_top_p,
            num_beams=num_beams_int,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        generated_text = TOKENIZER.batch_decode(
            output_ids_tensor, skip_special_tokens=True
        )[0].strip()
    except Exception as e:
        yield get_text("error_infer", current_language_state, err=str(e))
        return
    # --- End of synchronous block ---

    # Stream out the generated_text (char by char for demo)
    if generated_text:
        for char_token in generated_text:
            yield char_token
            await asyncio.sleep(0.01)  # Simulate network delay / token processing time
    else:  # Handle case of empty generation
        yield ""
