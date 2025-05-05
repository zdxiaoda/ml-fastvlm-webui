# Model Export for inference on Apple Silicon
Disclaimer: this is not an official recommendation, just research and exploration. 

## Export Vision Encoder
We found that LLaVA trainer does not save all the states needed for auto inference, 
predominantly used in third party libraries like `mlx-vlm`. We save additional metadata
to model checkpoint directory and export the vision model using coremltools. 
Export vision encoder and patch the checkpoint using the instruction below. 
```bash
python export_vision_encoder.py --model-path /path/to/fastvlm-checkpoint
```

## Export VLM 

### Install mlx-vlm
We provide a patch to `mlx-vlm` to support inference of FastVLM.
```bash
git clone https://github.com/Blaizzy/mlx-vlm.git
cd mlx-vlm 
git checkout 1884b551bc741f26b2d54d68fa89d4e934b9a3de
git apply ../fastvlm_mlx-vlm.patch
pip install -e .
```

Export model using the following instruction.
```bash
python -m mlx_vlm.convert --hf-path  /path/to/fastvlm-checkpoint \
                          --mlx-path /path/to/exported-fastvlm \
                          --only-llm
```
To quantize the LLM, additional options can be provided as shown below.
`--q-bits` specifies bits per weight, the command below exports the LLM with 8-bit quantization. 
```bash
python -m mlx_vlm.convert --hf-path  /path/to/fastvlm-checkpoint \
                          --mlx-path /path/to/exported-fastvlm \
                          --only-llm \
                          -q \
                          --q-bits 8       # For 4-bit quantization, specify 4
```

### Generate
The exported model can be used for inference in a python environment following the instruction below.
```bash
python -m mlx_vlm.generate --model /path/to/exported-fastvlm \
                           --image /path/to/image.png \
                           --prompt "Describe the image." \ 
                           --max-tokens 256 \
                           --temp 0.0
```

## Troubleshooting
We noticed that sometimes `config.json` for the LLaVA model incorrectly sets the value for `tie_word_embeddings`.
This causes the following error during conversion, `ValueError: Received parameters not in model: language_model.lm_head.weight.`
If you encounter this error, set the value of `tie_word_embeddings` accordingly.
