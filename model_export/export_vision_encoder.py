#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import json
import copy
import argparse

import torch
import numpy as np
import coremltools

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path


def export(args):
    # Load model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,
                                                                           args.model_base,
                                                                           model_name,
                                                                           device="mps")

    # Save extra metadata that is not saved during LLaVA training
    # required by HF for auto-loading model and for mlx-vlm preprocessing

    # Save image processing config
    setattr(image_processor, "processor_class", "LlavaProcessor")
    output_path = os.path.join(model_path, "preprocessor_config.json")
    image_processor.to_json_file(output_path)

    # Create processor config
    processor_config = dict()
    processor_config["image_token"] = "<image>"
    processor_config["num_additional_image_tokens"] = 0
    processor_config["processor_class"] = "LlavaProcessor"
    processor_config["patch_size"] = 64
    output_path = os.path.join(model_path, "processor_config.json")
    json.dump(processor_config, open(output_path, "w"), indent=2)

    # Modify tokenizer to include <image> special token.
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    tokenizer_config = json.load(open(tokenizer_config_path, 'r'))
    token_ids = list()
    image_token_is_present = False
    for k, v in tokenizer_config['added_tokens_decoder'].items():
        token_ids.append(int(k))
        if v["content"] == "<image>":
            image_token_is_present = True
            token_ids.pop()

    # Append only if <image> token is not present
    if not image_token_is_present:
        tokenizer_config['added_tokens_decoder'][f'{max(token_ids) + 1}'] = copy.deepcopy(
            tokenizer_config['added_tokens_decoder'][f'{token_ids[0]}'])
        tokenizer_config['added_tokens_decoder'][f'{max(token_ids) + 1}']["content"] = "<image>"
        json.dump(tokenizer_config, open(tokenizer_config_path, 'w'), indent=2)

    # Modify config to contain token id for <image>
    config_path = os.path.join(model_path, "config.json")
    model_config = json.load(open(config_path, 'r'))
    model_config["image_token_index"] = max(token_ids) + 1
    json.dump(model_config, open(config_path, 'w'), indent=2)

    # Export the vision encoder to CoreML
    image_res = image_processor.to_dict()['size']['shortest_edge']
    inputs = torch.rand(1, 3, image_res, image_res)
    inputs_tensor = [
        coremltools.TensorType(
            name="images",
            shape=inputs.shape,
        )
    ]
    vision_model = model.get_vision_tower()
    vision_model = vision_model.float()
    traced_model = torch.jit.trace(vision_model, torch.Tensor(inputs))
    pt_name = "fastvithd.pt"
    traced_model.save(pt_name)

    # Export
    ml_model = coremltools.convert(
        model=pt_name,
        outputs=[coremltools.TensorType(name="image_features", dtype=np.float32)],
        inputs=inputs_tensor,
        convert_to="mlprogram",
        debug=False,
        compute_units=coremltools.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=coremltools.target.iOS16,
        compute_precision=coremltools.precision.FLOAT32
    )
    ml_model_path = os.path.join(model_path, "fastvithd.mlpackage")
    ml_model.save(ml_model_path)

    # Remove traced model
    os.remove(pt_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_2")

    args = parser.parse_args()

    export(args)
