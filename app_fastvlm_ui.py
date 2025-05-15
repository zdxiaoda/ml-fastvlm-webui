# app_fastvlm_ui.py

import os
import torch
from PIL import Image
import gradio as gr
import atexit  # 用于在程序退出时执行清理操作
import traceback

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

# --- 全局变量：模型和配置 ---
MODEL = None
TOKENIZER = None
IMAGE_PROCESSOR = None
CONTEXT_LEN = None  # 加载后存储，但未在 predict_ui 中直接使用
MODEL_NAME_GLOBAL = None  # 存储全局加载的模型名称
MODEL_CONFIG = None  # 全局存储 model.config

# --- 默认路径和参数 (来自用户命令和 predict.py 的默认值) ---
# 自动遍历 model 目录下所有子文件夹作为可选模型
MODEL_ROOT_DIR = "model"


def get_all_model_dirs():
    if not os.path.exists(MODEL_ROOT_DIR):
        return []
    return [
        os.path.join(MODEL_ROOT_DIR, d)
        for d in os.listdir(MODEL_ROOT_DIR)
        if os.path.isdir(os.path.join(MODEL_ROOT_DIR, d))
    ]


ALL_MODEL_PATHS = get_all_model_dirs()
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
                print(f"退出时：已从备份恢复 {_initial_gen_config_original_path}。")
            except Exception as e:
                print(f"错误：退出时恢复 generation_config.json 失败: {e}")
        elif not os.path.exists(_initial_gen_config_backup_path):
            print(
                f"警告：退出时未找到备份配置文件 {_initial_gen_config_backup_path} 以供恢复。"
            )
        _renamed_generation_config_globally = False


# --- 模型加载函数 ---
def load_model_globally(model_path_str, model_base_str=None):
    global MODEL, TOKENIZER, IMAGE_PROCESSOR, CONTEXT_LEN, MODEL_NAME_GLOBAL, MODEL_CONFIG

    print(f"正在从路径加载模型: {model_path_str}")
    actual_model_path = os.path.expanduser(model_path_str)

    disable_torch_init()
    _model_name_loaded = get_model_name_from_path(actual_model_path)

    try:
        _tokenizer, _model, _image_processor, _context_len = load_pretrained_model(
            actual_model_path, model_base_str, _model_name_loaded, device="cuda"
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        traceback.print_exc()
        return

    MODEL = _model
    TOKENIZER = _tokenizer
    IMAGE_PROCESSOR = _image_processor
    CONTEXT_LEN = _context_len
    MODEL_NAME_GLOBAL = _model_name_loaded
    MODEL_CONFIG = _model.config

    if TOKENIZER.pad_token_id is None:
        print("警告: tokenizer.pad_token_id 为 None，将使用 tokenizer.eos_token_id。")
        TOKENIZER.pad_token_id = TOKENIZER.eos_token_id

    if (
        MODEL.generation_config.pad_token_id is None
        or MODEL.generation_config.pad_token_id != TOKENIZER.pad_token_id
    ):
        print(f"为模型设置 pad_token_id: {TOKENIZER.pad_token_id}")
        MODEL.generation_config.pad_token_id = TOKENIZER.pad_token_id

    print(f"模型 '{MODEL_NAME_GLOBAL}' 已成功加载到 CUDA 设备。")


# --- Gradio 的预测函数 ---
def generate_description(
    image_input_pil,
    prompt_str,
    temperature_float,
    top_p_float,
    num_beams_int,
    conv_mode_str,
):
    if not all([MODEL, TOKENIZER, IMAGE_PROCESSOR, MODEL_CONFIG]):
        return "错误：模型组件未完全初始化。请检查服务器日志并重启服务。"

    if image_input_pil is None:
        return "错误：请输入一张图片。"
    if not prompt_str:
        return "错误：请输入提示文本。"

    current_device = MODEL.device

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
        return f"错误：无效的对话模式 '{conv_mode_str}'。可用模式: {list(conv_templates.keys())}"

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
        print(f"分词 (Tokenization) 错误: {e}")
        traceback.print_exc()
        return f"错误: 分词失败 - {e}"

    image_pil = image_input_pil.convert("RGB")
    try:
        image_tensor_processed = process_images(
            [image_pil], IMAGE_PROCESSOR, MODEL_CONFIG
        )[0]
    except Exception as e:
        print(f"图像处理错误: {e}")
        traceback.print_exc()
        return f"错误: 图像处理失败 - {e}"

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
            print(f"模型生成错误: {e}")
            traceback.print_exc()
            return f"错误: 模型推理时发生错误 - {e}"

        generated_text = TOKENIZER.batch_decode(
            output_ids_tensor, skip_special_tokens=True
        )[0].strip()

    return generated_text


# --- 主要 Gradio 应用设置 ---
if __name__ == "__main__":
    if not ALL_MODEL_PATHS:
        raise RuntimeError("未在 model 目录下找到任何模型文件夹，请先下载模型。")

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
            print(
                f"启动时：暂时将 {_initial_gen_config_original_path} 重命名为 {_initial_gen_config_backup_path}"
            )
        except OSError as e:
            print(f"警告：启动时无法重命名全局 generation_config.json: {e}。")
        except Exception as e:
            print(f"警告：启动时重命名全局 generation_config.json 时发生未知错误: {e}")

    atexit.register(_restore_generation_config_on_exit)

    print(f"Web UI 启动中...正在加载模型，请稍候 (路径: {DEFAULT_MODEL_PATH})...")
    load_model_globally(DEFAULT_MODEL_PATH, model_base_str=DEFAULT_MODEL_BASE)

    if not MODEL:
        print("错误：模型未能成功加载。Gradio UI 将无法正常工作。请检查日志。")
        # exit(1)

    with gr.Blocks(theme=gr.themes.Soft()) as app_ui:
        gr.Markdown("<h1>图像描述 Web UI (FastVLM)</h1>")
        gr.Markdown("上传一张图片并输入提示，模型将生成相应的描述。")

        with gr.Row():
            with gr.Column(scale=1):
                # 新增模型选择下拉框
                model_selector_ui = gr.Dropdown(
                    choices=ALL_MODEL_PATHS,
                    value=DEFAULT_MODEL_PATH,
                    label="选择模型目录 (自动检测)",
                )
                image_input_ui = gr.Image(
                    type="pil", label="上传图片", sources=["upload", "clipboard"]
                )
                prompt_input_ui = gr.Textbox(
                    label="输入提示 (例如：详细描述图片内容)",
                    value=DEFAULT_PROMPT,
                    lines=2,
                )

                with gr.Accordion("高级参数设置", open=False):
                    temperature_ui = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.01,
                        value=DEFAULT_TEMPERATURE,
                        label="温度 (Temperature)",
                    )
                    top_p_ui = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=DEFAULT_TOP_P,
                        label="Top P",
                        info="设为0或1以禁用。典型值 0.7-0.95。",
                    )
                    num_beams_ui = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=DEFAULT_NUM_BEAMS,
                        label="束搜索数 (Num Beams)",
                    )
                    conv_mode_ui = gr.Dropdown(
                        choices=sorted(list(conv_templates.keys())),
                        value=DEFAULT_CONV_MODE,
                        label="对话模式 (Conversation Mode)",
                    )

                submit_button_ui = gr.Button("生成描述", variant="primary")

            with gr.Column(scale=1):
                output_text_ui = gr.Textbox(
                    label="模型输出", lines=15, interactive=False, show_copy_button=True
                )

        # 切换模型时自动加载新模型
        def on_model_change(new_model_path):
            update_gen_config_paths(new_model_path)
            load_model_globally(new_model_path, model_base_str=DEFAULT_MODEL_BASE)
            return f"已切换并加载模型：{new_model_path}"

        model_change_output = gr.Textbox(visible=False)
        model_selector_ui.change(
            fn=on_model_change,
            inputs=[model_selector_ui],
            outputs=model_change_output,
        )

        submit_button_ui.click(
            fn=generate_description,
            inputs=[
                image_input_ui,
                prompt_input_ui,
                temperature_ui,
                top_p_ui,
                num_beams_ui,
                conv_mode_ui,
            ],
            outputs=output_text_ui,
            api_name="generate_description",  # 添加 API 名称以便通过 API 访问
        )

    print("Gradio UI 配置完成。正在启动服务器...")
    app_ui.launch(server_name="0.0.0.0", share=False)
