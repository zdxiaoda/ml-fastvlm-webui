# app_fastvlm_ui.py

import os
import torch
from PIL import Image
import gradio as gr
import atexit
import json
import glob
import requests
import zipfile
import shutil
import threading
import queue

# LLaVA 相关导入 (从 predict.py 复制)
from llava.utils import disable_torch_init
from llava.conversation import conv_templates  # 用于对话模式选择和 predict_ui
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,  # 用于 load_model_globally
)
from llava.constants import (  # 用于 predict_ui
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

# --- 从新的下载器模块导入 ---
import model_downloader  # 直接导入模块

# --- 国际化支持 ---
LOCALES_DIR = "locales"  # 存储翻译文件的目录
I18N_MESSAGES = {}  # 存储加载的翻译
DEFAULT_LANG = "en"  # 默认语言


def load_translations():
    """从 LOCALES_DIR 加载所有 JSON 翻译文件。"""
    global I18N_MESSAGES
    if not os.path.isdir(LOCALES_DIR):
        # 如果目录不存在，则不执行任何操作，I18N_MESSAGES 将为空
        # 或者可以抛出错误，或者尝试创建目录
        return

    for file_path in glob.glob(os.path.join(LOCALES_DIR, "*.json")):
        lang_code = os.path.basename(file_path)[:-5]  # e.g., "en" from "en.json"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                I18N_MESSAGES[lang_code] = json.load(f)
        except Exception:
            # 此处可以添加错误处理，例如记录到标准错误流，但根据要求不使用 print
            pass  # 跳过无法加载的文件


load_translations()  # 程序启动时加载翻译

# 全局当前语言状态
current_language_state = DEFAULT_LANG


def get_text(key: str, lang: str, **kwargs) -> str:
    """获取翻译文本，支持参数化和回退。"""
    # 首先尝试获取指定语言的翻译
    lang_translations = I18N_MESSAGES.get(lang)

    # 如果指定语言未找到或该语言下没有对应key，尝试默认语言
    if lang_translations is None or key not in lang_translations:
        lang_translations = I18N_MESSAGES.get(
            DEFAULT_LANG, {}
        )  # 回退到默认语言的翻译, 如果默认语言也没有则为空字典

    text_template = lang_translations.get(key)

    if text_template is None:
        # 如果在任何配置中都找不到key，返回一个提示性字符串
        return f"[{lang.upper() if lang else DEFAULT_LANG.upper()}:{key}]"

    if kwargs:
        try:
            return text_template.format(**kwargs)
        except KeyError as e:
            # 如果格式化时缺少参数，返回带错误信息的提示
            return f"[{lang.upper() if lang else DEFAULT_LANG.upper()}:{key} - Format Error: {e}]"
    return text_template


# --- 全局变量：模型和配置 ---
MODEL = None
TOKENIZER = None
IMAGE_PROCESSOR = None
CONTEXT_LEN = None  # 加载后存储，但未在 predict_ui 中直接使用
MODEL_NAME_GLOBAL = None  # 存储全局加载的模型名称
MODEL_CONFIG = None  # 全局存储 model.config

# --- 默认路径和参数 (来自用户命令和 predict.py 的默认值) ---
MODEL_ROOT_DIR = "model"  # 这个常量仍然可以在这里，或者也从 model_downloader 获取

# 使用 model_downloader 中的函数初始化 ALL_MODEL_PATHS
ALL_MODEL_PATHS = model_downloader.get_model_paths()
DEFAULT_MODEL_PATH = ALL_MODEL_PATHS[0] if ALL_MODEL_PATHS else None
DEFAULT_MODEL_BASE = None  # 来自 predict.py args 的默认值
DEFAULT_CONV_MODE = "qwen_2"  # 来自 predict.py args 的默认值
DEFAULT_TEMPERATURE = 0.2  # 来自 predict.py args 的默认值
DEFAULT_TOP_P = 1.0  # Gradio 滑块的默认值设为1.0 (禁用Top P)
DEFAULT_NUM_BEAMS = 1  # 来自 predict.py args 的默认值
DEFAULT_PROMPT = "描述这张图片。"  # 改为中文并符合常见用法

# --- 处理 generation_config.json ---
# 这些路径基于上面定义的 DEFAULT_MODEL_PATH
_initial_model_path_expanded = os.path.expanduser(DEFAULT_MODEL_PATH)
_initial_gen_config_original_path = os.path.join(
    _initial_model_path_expanded, "generation_config.json"
)
_initial_gen_config_backup_path = os.path.join(
    _initial_model_path_expanded, ".generation_config.json"
)
_renamed_generation_config_globally = False  # 标记是否已执行重命名


def _restore_generation_config_on_exit():
    global _renamed_generation_config_globally
    if _renamed_generation_config_globally:
        if os.path.exists(_initial_gen_config_backup_path) and not os.path.exists(
            _initial_gen_config_original_path
        ):
            try:
                os.rename(
                    _initial_gen_config_backup_path, _initial_gen_config_original_path
                )
            except Exception:
                pass
        elif not os.path.exists(_initial_gen_config_backup_path):
            pass
        _renamed_generation_config_globally = False


# --- 设备自动检测 ---
def get_optimal_device():
    """自动检测可用的最佳 PyTorch 设备。"""
    if torch.cuda.is_available():
        # 优先使用 NVIDIA CUDA (或兼容的 AMD ROCm)
        return "cuda"
    elif torch.backends.mps.is_available():
        # 其次是 Apple Silicon (MPS)
        if not torch.backends.mps.is_built():
            # MPS 未构建，回退到 CPU
            return "cpu"
        return "mps"
    # 可以在此添加对其他设备（如 Intel XPU）的检查
    # elif hasattr(torch, "xpu") and torch.xpu.is_available():
    #     return "xpu"
    return "cpu"  # 默认回退到 CPU


# --- 模型加载函数 ---
def load_model_globally(model_path_str, model_base_str=None):
    global MODEL, TOKENIZER, IMAGE_PROCESSOR, CONTEXT_LEN, MODEL_NAME_GLOBAL, MODEL_CONFIG

    actual_model_path = os.path.expanduser(model_path_str)

    disable_torch_init()
    _model_name_loaded = get_model_name_from_path(actual_model_path)

    selected_device = get_optimal_device()
    # Consider logging the selected_device if logging is re-enabled later
    # print(f"Attempting to load model on device: {selected_device}")

    try:
        _tokenizer, _model, _image_processor, _context_len = load_pretrained_model(
            actual_model_path,
            model_base_str,
            _model_name_loaded,
            device=selected_device,
        )
    except Exception:  # Minimal error handling as per no-logging request
        MODEL = None
        TOKENIZER = None
        IMAGE_PROCESSOR = None
        CONTEXT_LEN = None
        MODEL_NAME_GLOBAL = None
        MODEL_CONFIG = None
        return

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


def unload_model_globally():
    """卸载当前加载的模型并释放资源。"""
    global MODEL, TOKENIZER, IMAGE_PROCESSOR, CONTEXT_LEN, MODEL_NAME_GLOBAL, MODEL_CONFIG
    MODEL = None
    TOKENIZER = None
    IMAGE_PROCESSOR = None
    CONTEXT_LEN = None
    MODEL_NAME_GLOBAL = None
    MODEL_CONFIG = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return get_text("model_unloaded_success", current_language_state)


# --- Gradio 的模型下载包装器 (现在使用 model_downloader) ---
def download_model_gradio_wrapper(
    model_name_to_download, progress=gr.Progress(track_tqdm=True)
):
    """Gradio 包装器，用于启动模型下载线程并处理 UI 更新。"""
    global ALL_MODEL_PATHS  # 声明全局变量以便修改

    if not model_name_to_download:
        # 如果模型列表可能已过时，可以考虑刷新
        # current_paths = model_downloader.get_model_paths()
        # if set(ALL_MODEL_PATHS) != set(current_paths):
        #     ALL_MODEL_PATHS = current_paths
        return get_text(
            "error_no_model_selected_for_download", current_language_state
        ), gr.update(choices=ALL_MODEL_PATHS)

    url = model_downloader.MODEL_DOWNLOAD_LINKS.get(model_name_to_download)
    if not url:
        # current_paths = model_downloader.get_model_paths()
        # if set(ALL_MODEL_PATHS) != set(current_paths):
        #     ALL_MODEL_PATHS = current_paths
        return get_text(
            "error_model_url_not_found",
            current_language_state,
            model_name=model_name_to_download,
        ), gr.update(choices=ALL_MODEL_PATHS)

    update_q = queue.Queue()

    thread = threading.Thread(
        target=model_downloader._perform_download_and_extraction_task,
        args=(model_name_to_download, url, update_q),
    )
    thread.start()

    status_message = get_text(
        "download_queued", current_language_state, model_name=model_name_to_download
    )
    # 初始化 model_selector_update_dict，使用当前的 ALL_MODEL_PATHS
    # 这个 ALL_MODEL_PATHS 应该在接收到 'selector_update_info' 时被更新
    model_selector_update_dict = {
        "choices": ALL_MODEL_PATHS,
        "value": DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH in ALL_MODEL_PATHS else None,
    }

    while thread.is_alive() or not update_q.empty():
        try:
            item = update_q.get(timeout=0.1 if thread.is_alive() else 0.01)
            message_type, *payload = item

            if message_type == "status":
                key, kwargs_dict = payload
                status_message = get_text(key, current_language_state, **kwargs_dict)
            elif message_type == "selector_update_info":
                (update_info,) = payload
                if "choices" in update_info:
                    ALL_MODEL_PATHS = update_info["choices"]  # 更新全局列表
                model_selector_update_dict = (
                    update_info  # 保存整个字典以供 gr.update 使用
                )
            elif message_type == "finished":
                # 'finished' 信号，确保在退出循环前产生最后一次更新
                # 最终的状态信息应该已经在 'status' 更新中设置好了
                pass  # 将在下一次迭代或循环结束时产生

            yield status_message, gr.update(**model_selector_update_dict)

        except queue.Empty:
            if not thread.is_alive() and update_q.empty():
                break  # 线程结束且队列为空，退出

    # 确保返回最终状态给 Gradio outputs
    # 此时 status_message 和 model_selector_update_dict 应该是最新的
    yield status_message, gr.update(**model_selector_update_dict)


# --- Gradio 的预测函数 (旧版，错误信息改为英文) ---
def generate_description(
    image_input_pil,
    prompt_str,
    temperature_float,
    top_p_float,
    num_beams_int,
    conv_mode_str,
):
    if not all([MODEL, TOKENIZER, IMAGE_PROCESSOR, MODEL_CONFIG]):
        return "Error: Model components not fully initialized. Please check server logs and restart."

    if image_input_pil is None:
        return "Error: Please input an image."
    if not prompt_str:
        return "Error: Please input a prompt."

    current_device = MODEL.device  # Assumes MODEL is loaded and has a device property

    if MODEL_CONFIG.mm_use_im_start_end:
        full_prompt = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + prompt_str
        )
    else:
        full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_str

    if conv_mode_str not in conv_templates:
        return f"Error: Invalid conversation mode '{conv_mode_str}'. Available: {list(conv_templates.keys())}"

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
        return f"Error: Tokenization failed - {e}"

    image_pil = image_input_pil.convert("RGB")
    try:
        image_tensor_processed = process_images(
            [image_pil], IMAGE_PROCESSOR, MODEL_CONFIG
        )[0]
    except Exception as e:
        return f"Error: Image processing failed - {e}"

    with torch.inference_mode():
        try:
            actual_top_p = (
                top_p_float
                if top_p_float is not None and 0 < top_p_float < 1.0
                else None
            )

            output_ids_tensor = MODEL.generate(
                input_ids_tensor,
                images=image_tensor_processed.unsqueeze(0).half().to(current_device),
                image_sizes=[image_pil.size],
                do_sample=True if temperature_float > 0 else False,
                temperature=temperature_float,
                top_p=actual_top_p,
                num_beams=num_beams_int,
                max_new_tokens=512,
                use_cache=True,
            )
        except Exception as e:
            return f"Error: Model inference error - {e}"

        generated_text = TOKENIZER.batch_decode(
            output_ids_tensor, skip_special_tokens=True
        )[0].strip()

    return generated_text


# --- 主要 Gradio 应用设置 ---
if __name__ == "__main__":
    if not ALL_MODEL_PATHS and not model_downloader.MODEL_DOWNLOAD_LINKS:
        # 修改: 如果本地没有模型且没有可供下载的链接，则报错
        raise RuntimeError(
            get_text("error_no_model_found_or_downloadable", current_language_state)
        )

    # 处理 generation_config.json 路径
    def update_gen_config_paths(model_path):
        global _initial_model_path_expanded, _initial_gen_config_original_path, _initial_gen_config_backup_path
        _initial_model_path_expanded = os.path.expanduser(model_path)
        _initial_gen_config_original_path = os.path.join(
            _initial_model_path_expanded, "generation_config.json"
        )
        _initial_gen_config_backup_path = os.path.join(
            _initial_model_path_expanded, ".generation_config.json"
        )

    update_gen_config_paths(DEFAULT_MODEL_PATH)

    if os.path.exists(_initial_gen_config_original_path):
        try:
            os.rename(
                _initial_gen_config_original_path, _initial_gen_config_backup_path
            )
            _renamed_generation_config_globally = True
        except OSError:
            pass
        except Exception:
            pass

    atexit.register(_restore_generation_config_on_exit)

    load_model_globally(DEFAULT_MODEL_PATH, model_base_str=DEFAULT_MODEL_BASE)

    if not MODEL:
        pass

    # --- UI 构建 ---
    with gr.Blocks(theme="gradio/miku") as app_ui:  # 应用 Miku 主题

        title_md = gr.Markdown(get_text("title", current_language_state))
        desc_md = gr.Markdown(get_text("desc", current_language_state))

        with gr.Row():
            with gr.Column(scale=1):
                # 模型下载 UI
                with gr.Group():  # 将下载相关的组件分组
                    model_download_header_md = gr.Markdown(
                        get_text("model_download_header", current_language_state)
                    )
                    model_download_selector_ui = gr.Dropdown(
                        choices=list(
                            model_downloader.MODEL_DOWNLOAD_LINKS.keys()
                        ),  # 使用导入的链接
                        label=get_text(
                            "select_model_to_download", current_language_state
                        ),
                        value=None,
                    )
                    download_button_ui = gr.Button(
                        get_text("download_model_button", current_language_state),
                        variant="secondary",  # 使用次要按钮样式
                    )
                    download_status_ui = gr.Textbox(
                        label=get_text("download_status_label", current_language_state),
                        interactive=False,
                        lines=3,  # 允许多行状态信息
                    )

                # 已有模型选择器
                model_selector_ui = gr.Dropdown(
                    choices=ALL_MODEL_PATHS,  # 这个会由下载器更新
                    value=(
                        DEFAULT_MODEL_PATH
                        if DEFAULT_MODEL_PATH in ALL_MODEL_PATHS
                        else (ALL_MODEL_PATHS[0] if ALL_MODEL_PATHS else None)
                    ),
                    label=get_text("select_model", current_language_state),
                )
                unload_button_ui = gr.Button(  # 新增卸载模型按钮
                    get_text("unload_model_button", current_language_state),
                    variant="stop",  # 使用停止/危险按钮样式
                )
                model_status_text_ui = gr.Textbox(  # 新增模型状态文本框
                    label=get_text(
                        "model_load_unload_status_label", current_language_state
                    ),
                    interactive=False,
                    lines=2,  # 状态信息通常不需要太长
                    value="",  # 初始为空
                )
                image_input_ui = gr.Image(
                    type="pil",
                    label=get_text("upload_image", current_language_state),
                    sources=["upload", "clipboard", "webcam"],
                )
                prompt_input_ui = gr.Textbox(
                    label=get_text("prompt_label", current_language_state),
                    value=get_text("default_prompt", current_language_state),
                    lines=2,
                )

                with gr.Accordion(
                    get_text("adv_params", current_language_state), open=False
                ) as adv_params_accordion:
                    temperature_ui = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.01,
                        value=DEFAULT_TEMPERATURE,
                        label=get_text("temperature", current_language_state),
                    )
                    top_p_ui = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=DEFAULT_TOP_P,
                        label=get_text("top_p", current_language_state),
                        info=get_text("top_p_info", current_language_state),
                    )
                    num_beams_ui = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=DEFAULT_NUM_BEAMS,
                        label=get_text("num_beams", current_language_state),
                    )
                    conv_mode_ui = gr.Dropdown(
                        choices=sorted(
                            list(conv_templates.keys())
                        ),  # conv_templates might not be available if LLaVA fails
                        value=DEFAULT_CONV_MODE,
                        label=get_text("conv_mode", current_language_state),
                    )

                submit_button_ui = gr.Button(
                    get_text("submit", current_language_state), variant="primary"
                )

            with gr.Column(scale=1):
                output_text_ui = gr.Textbox(
                    label=get_text("output", current_language_state),
                    lines=15,
                    interactive=False,
                    show_copy_button=True,
                )
        # 动态生成语言选项
        lang_choices = []
        if I18N_MESSAGES:  # Check if translations were loaded
            for code, translations in I18N_MESSAGES.items():
                display_name = translations.get("lang_display_name", code.upper())
                lang_choices.append((display_name, code))
        else:  # Fallback if no translations loaded
            lang_choices = [("English", "en")]

        lang_dropdown = gr.Dropdown(
            choices=lang_choices,
            value=current_language_state,
            label=get_text("lang_label", current_language_state),
        )

        # --- 语言切换回调 ---
        def on_lang_change_handler(new_lang):
            global current_language_state
            current_language_state = new_lang

            updates = [
                gr.update(label=get_text("lang_label", new_lang)),
                gr.update(value=get_text("title", new_lang)),
                gr.update(value=get_text("desc", new_lang)),
                # 模型下载UI更新 (先于模型选择，因为它们在UI中靠前)
                gr.update(value=get_text("model_download_header", new_lang)),
                gr.update(label=get_text("select_model_to_download", new_lang)),
                gr.update(value=get_text("download_model_button", new_lang)),
                gr.update(label=get_text("download_status_label", new_lang)),
                # 主要模型选择和其他组件
                gr.update(label=get_text("select_model", new_lang)),
                gr.update(
                    value=get_text("unload_model_button", new_lang)
                ),  # 更新卸载按钮文本
                gr.update(
                    label=get_text("model_load_unload_status_label", new_lang)
                ),  # 更新模型状态标签
                gr.update(label=get_text("upload_image", new_lang)),
                gr.update(
                    label=get_text("prompt_label", new_lang),
                    value=get_text("default_prompt", new_lang),
                ),
                gr.update(label=get_text("adv_params", new_lang)),
                gr.update(label=get_text("temperature", new_lang)),
                gr.update(
                    label=get_text("top_p", new_lang),
                    info=get_text("top_p_info", new_lang),
                ),
                gr.update(label=get_text("num_beams", new_lang)),
                gr.update(label=get_text("conv_mode", new_lang)),
                gr.update(value=get_text("submit", new_lang)),
                gr.update(label=get_text("output", new_lang)),
            ]
            return updates

        lang_dropdown.change(
            fn=on_lang_change_handler,
            inputs=[lang_dropdown],
            outputs=[  # Ensure all components that need text updates are listed here in correct order
                lang_dropdown,
                title_md,
                desc_md,
                # 模型下载UI组件更新
                model_download_header_md,
                model_download_selector_ui,
                download_button_ui,
                download_status_ui,
                # 主要模型选择和其他组件
                model_selector_ui,
                unload_button_ui,  # 添加卸载按钮到输出列表
                model_status_text_ui,  # 添加模型状态文本框到输出列表
                image_input_ui,
                prompt_input_ui,
                adv_params_accordion,
                temperature_ui,
                top_p_ui,
                num_beams_ui,
                conv_mode_ui,
                submit_button_ui,
                output_text_ui,
            ],
        )

        # --- 生成描述函数（国际化错误信息） ---
        def generate_description_i18n(
            image_input_pil,
            prompt_str,
            temperature_float,
            top_p_float,
            num_beams_int,
            conv_mode_str,
        ):
            lang = current_language_state  # Use the global state

            if not all([MODEL, TOKENIZER, IMAGE_PROCESSOR, MODEL_CONFIG]):
                return get_text("error_model_not_init", lang)
            if image_input_pil is None:
                return get_text("error_no_image", lang)
            if not prompt_str:
                return get_text("error_no_prompt", lang)

            current_device = MODEL.device  # Assuming MODEL is loaded

            # Check if MODEL_CONFIG is None before accessing mm_use_im_start_end
            if MODEL_CONFIG is None:  # Should be caught by the "not all" check above
                return get_text("error_model_not_init", lang)

            if MODEL_CONFIG.mm_use_im_start_end:
                full_prompt = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt_str
                )
            else:
                full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_str

            if conv_mode_str not in conv_templates:
                return get_text(
                    "error_invalid_conv",
                    lang,
                    conv_mode=conv_mode_str,
                    modes=list(conv_templates.keys()),
                )
            conv = conv_templates[conv_mode_str].copy()
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)
            processed_prompt_str = conv.get_prompt()
            try:
                input_ids_tensor = (
                    tokenizer_image_token(
                        processed_prompt_str,
                        TOKENIZER,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    )
                    .unsqueeze(0)
                    .to(current_device)
                )
            except Exception as e:
                return get_text("error_tokenize", lang, err=str(e))

            image_pil = image_input_pil.convert("RGB")
            try:
                image_tensor_processed = process_images(
                    [image_pil], IMAGE_PROCESSOR, MODEL_CONFIG
                )[0]
            except Exception as e:
                return get_text("error_image_proc", lang, err=str(e))

            with torch.inference_mode():
                try:
                    actual_top_p = (
                        top_p_float
                        if top_p_float is not None and 0 < top_p_float < 1.0
                        else None
                    )
                    output_ids_tensor = MODEL.generate(
                        input_ids_tensor,
                        images=image_tensor_processed.unsqueeze(0)
                        .half()
                        .to(current_device),
                        image_sizes=[image_pil.size],
                        do_sample=True if temperature_float > 0 else False,
                        temperature=temperature_float,
                        top_p=actual_top_p,
                        num_beams=num_beams_int,
                        max_new_tokens=512,
                        use_cache=True,
                    )
                except Exception as e:
                    return get_text("error_infer", lang, err=str(e))

                generated_text = TOKENIZER.batch_decode(
                    output_ids_tensor, skip_special_tokens=True
                )[0].strip()
            return generated_text

        # 切换模型时自动加载新模型
        def on_model_change(new_model_path):
            # 此函数在用户从下拉列表选择模型时触发
            # 或者在模型下载并自动选中后，由Gradio的 .change() 触发
            if not new_model_path or not os.path.exists(new_model_path):
                # 如果路径无效，可以显示错误或不执行任何操作
                # For now, if path is invalid, model loading will fail gracefully or use previous.
                # The dropdown should ideally only contain valid paths.
                return get_text(
                    "error_invalid_model_path_selected",
                    current_language_state,
                    path=new_model_path,
                )

            _restore_generation_config_on_exit()  # Restore previous before renaming for new one
            update_gen_config_paths(new_model_path)
            # Rename new model's generation_config if exists
            if os.path.exists(_initial_gen_config_original_path):
                try:
                    os.rename(
                        _initial_gen_config_original_path,
                        _initial_gen_config_backup_path,
                    )
                    _renamed_generation_config_globally = (
                        True  # Mark that we renamed for current model
                    )
                except OSError:
                    pass  # logging.warning(f"Could not rename new generation_config.json: {e}")
            else:
                _renamed_generation_config_globally = (
                    False  # No file to rename for this model
                )

            load_model_globally(new_model_path, model_base_str=DEFAULT_MODEL_BASE)
            lang = current_language_state
            if MODEL:  # Check if model loaded successfully
                # 清空之前的状态消息，或显示成功加载消息
                return get_text("model_switched", lang, model_path=new_model_path)
            else:
                return get_text(
                    "error_model_load_failed", lang, model_path=new_model_path
                )

        model_selector_ui.change(  # 更新: model_change_output 现在是 model_status_text_ui
            fn=on_model_change,
            inputs=[model_selector_ui],
            outputs=model_status_text_ui,  # 输出到新的状态文本框
        )

        # 模型下载按钮点击事件
        download_button_ui.click(
            fn=download_model_gradio_wrapper,
            inputs=[model_download_selector_ui],
            outputs=[download_status_ui, model_selector_ui],
        )

        submit_button_ui.click(
            fn=generate_description_i18n,
            inputs=[
                image_input_ui,
                prompt_input_ui,
                temperature_ui,
                top_p_ui,
                num_beams_ui,
                conv_mode_ui,
            ],
            outputs=output_text_ui,
            api_name="generate_description",
        )

        # 卸载模型按钮点击事件
        unload_button_ui.click(
            fn=unload_model_globally,
            inputs=[],
            outputs=model_status_text_ui,  # 输出到模型状态文本框
        )

    app_ui.launch(server_name="0.0.0.0", share=False)
