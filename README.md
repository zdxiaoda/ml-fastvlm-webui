# FastVLM: Efficient Vision Encoding for Vision Language Models

This is a focused version of the FastVLM repository, configured to run a Gradio-based Web UI for image description. For full details on the original project and research, please refer to:
**[FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303). (CVPR 2025)**

### Highlights

- FastVLM introduces FastViTHD, a novel hybrid vision encoder for efficient processing of high-resolution images, leading to faster performance.
- This repository provides a Gradio Web UI (`app_fastvlm_ui.py`) for easy interaction with FastVLM models.

## Model Zoo

The following table lists available pre-trained FastVLM models. For detailed information on various evaluations, please refer to the [paper](https://www.arxiv.org/abs/2412.13303).

| Model        | Stage |                                       Pytorch Checkpoint (url)                                        |
| :----------- | :---: | :---------------------------------------------------------------------------------------------------: |
| FastVLM-0.5B |   2   | [fastvlm_0.5b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage2.zip) |
|              |   3   | [fastvlm_0.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip) |
| FastVLM-1.5B |   2   | [fastvlm_1.5b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage2.zip) |
|              |   3   | [fastvlm_1.5b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip) |
| FastVLM-7B   |   2   |   [fastvlm_7b_stage2](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage2.zip)   |
|              |   3   |   [fastvlm_7b_stage3](https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip)   |

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/zdxiaoda/ml-fastvlm-webui
    cd ml-fastvlm-webui
    ```
2.  **Create a Python environment and activate it (e.g., using Conda):**
    ```bash
    conda create -n fastvlm_ui python=3.10 -y
    conda activate fastvlm_ui
    ```
3.  **Install dependencies:**
    This project requires Python 3.10 or higher.
    ```bash
    pip install -r requirements.txt
    pip install -e .  # Installs the llava package from the local directory
    ```

## Download Pretrained Models

1.  **Choose a model** from the Model Zoo table above and download its Pytorch Checkpoint. For example, to download `FastVLM-0.5B (Stage 3)`:

    ```bash
    wget https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip
    ```

2.  **Create a directory** to store your models, for example, `model/` in the root of this project:

    ```bash
    mkdir -p model
    ```

3.  **Unzip the downloaded model** into the created directory.

    ```bash
    unzip llava-fastvithd_0.5b_stage3.zip -d model/
    # This should create a sub-directory, e.g., model/llava-fastvithd_0.5b_stage3
    ```

4.  **Configure the model path in the script:**
    Open the `app_fastvlm_ui.py` script and locate the `DEFAULT_MODEL_PATH` variable.
    Update its value to the path of the model checkpoint directory you just downloaded and unzipped.
    For example, if you placed the model in `model/llava-fastvithd_0.5b_stage3/`, you should change the line to:
    ```python
    DEFAULT_MODEL_PATH = "model/llava-fastvithd_0.5b_stage3/"
    ```
    Ensure this path is correct for the script to load the model.

## Running the Web UI

After completing the setup and model configuration:

1.  **Start the Gradio web server:**

    ```bash
    python app_fastvlm_ui.py
    ```

    The script will load the model (which may take some time) and then start the web server.

2.  **Open the UI in your browser:**
    Once the server is running, it will typically print a local URL to the console, such as `Running on local URL:  http://0.0.0.0:7860` or `http://127.0.0.1:7860`. Open this URL in your web browser.

### How to use the Web UI:

- **Upload Image:** On the left panel, click the "Upload Image" area to browse for an image file, or drag and drop an image. You can also paste an image directly from your clipboard.
- **Enter Prompt:** Below the image upload, type your question or instruction for the model in the "Enter prompt" textbox (e.g., "Describe the image in detail", "How many cats are in the picture?"). The default prompt is "Describe the image.".
- **Adjust Parameters (Optional):** If needed, expand the "Advanced Parameters" accordion menu to adjust settings like:
  - `Temperature`: Controls randomness. Lower is more deterministic.
  - `Top P`: Nucleus sampling parameter. Set to 0 or 1 to disable.
  - `Num Beams`: Number of beams for beam search. 1 means no beam search.
  - `Conversation Mode`: Selects the conversation template.
- **Generate Description:** Click the "Generate Description" button.
- **View Output:** The model's generated text will appear in the "Model Output" textbox on the right panel. You can use the "Copy" button to copy the output.

## Citation

If you found the original FastVLM work useful, please cite their paper:

```
@InProceedings{fastvlm2025,
  author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
  title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
}
```

## Acknowledgements

This codebase builds upon multiple open-source contributions. Please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details from the original project.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and [LICENSE_MODEL](LICENSE_MODEL) for the released models (if applicable to the models you download).
