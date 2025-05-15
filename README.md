# FastVLM: Efficient Vision Encoding for Vision Language Models

This is a focused version of the FastVLM repository, configured to run a Gradio-based Web UI for image description. For full details on the original project and research, please refer to:
**[FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303). (CVPR 2025)**

### Highlights

- FastVLM introduces FastViTHD, a novel hybrid vision encoder for efficient processing of high-resolution images, leading to faster performance.
- This repository provides a Gradio Web UI (`app_fastvlm_ui.py`) for easy interaction with FastVLM models, featuring:
  - **In-UI Model Downloader**: Easily download and manage pre-trained models directly within the application.
  - **Automatic Device Selection**: Optimally utilizes available hardware (CUDA, MPS, or CPU).
  - **Multi-language Support**: Interface available in multiple languages.
  - **Interactive Controls**: Adjust generation parameters like temperature, top-p, and beam search.

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
    # The following command installs the llava package from the local directory
    # and is necessary for the application to run correctly.
    pip install -e .
    ```

## Using the Web UI

1.  **Start the Gradio web server:**

    ```bash
    python app_fastvlm_ui.py
    ```

    The script will automatically detect the optimal device (CUDA, MPS, or CPU) and then start the web server. The first time you run the UI, or if no models are found in the `model/` directory, it might take a moment to populate model choices.

2.  **Open the UI in your browser:**
    Once the server is running, it will typically print a local URL to the console, such as `Running on local URL:  http://0.0.0.0:7860` or `http://127.0.0.1:7860`. Open this URL in your web browser.

    ![FastVLM Web UI Screenshot](./img/webui_example.jpeg)

### Interacting with the Web UI:

The Web UI is organized into several key areas:

- **Model Management (Top Section):**

  - **Select Model to Download:** Choose a model from the dropdown list provided in the "Model Zoo" section of this README.
  - **Download Model:** Click this button to download the selected model. Progress will be shown, and upon completion, the model will be available in the "Select Model to Load" dropdown. Models are saved to the `model/` directory.
  - **Select Model to Load:** After downloading, or if you have manually placed models in the `model/` directory, select the desired model from this dropdown.
  - **Load Model:** Click to load the selected model into memory. This may take some time. A confirmation message will appear.
  - **Unload Model:** Click to free up resources by unloading the currently active model.
  - **Language Selection (语/言/Lang):** Choose your preferred interface language from the dropdown (e.g., English, 中文).

- **Image Interaction (Main Section):**
  - **Upload Image (Left Panel):** Click the "Upload Image" area to browse for an image file, or drag and drop an image. You can also paste an image directly from your clipboard or use the webcam input if available.
  - **Enter Prompt (Left Panel):** Below the image, type your question or instruction for the model in the "Enter prompt" textbox (e.g., "Describe the image in detail", "How many cats are in the picture?"). The default prompt is "Describe this image." (or its translation).
  - **Advanced Parameters (Left Panel, Collapsible):** If needed, expand the "Advanced Parameters" accordion menu to adjust settings like:
    - `Temperature`: Controls randomness. Lower is more deterministic.
    - `Top P`: Nucleus sampling parameter. Set to 1.0 to disable.
    - `Num Beams`: Number of beams for beam search. 1 means no beam search.
    - `Conversation Mode`: Selects the conversation template suitable for the loaded model.
  - **Generate Description (Left Panel):** Click the "Generate Description" button.
  - **Model Output (Right Panel):** The model's generated text will appear in the "Model Output" textbox. You can use the "Copy" button to copy the output or "Clear" to empty the textbox.

**Note on Models:**

- The UI dynamically scans the `model/` directory for available models. If you manually add models, ensure they are in subdirectories within `model/` (e.g., `model/your_model_name/`).
- The first model in the "Select Model to Load" dropdown is often selected by default when the UI starts, but it might not be automatically loaded. Always ensure you click "Load Model" for the desired model.

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
