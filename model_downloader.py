import os
import requests
import zipfile
import shutil
import glob  # For get_model_paths

# --- Constants ---
MODEL_ROOT_DIR = "model"
MODEL_DOWNLOAD_LINKS = {
    "FastVLM-0.5B (Stage 2)": "https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage2.zip",
    "FastVLM-0.5B (Stage 3)": "https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip",
    "FastVLM-1.5B (Stage 2)": "https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage2.zip",
    "FastVLM-1.5B (Stage 3)": "https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip",
    "FastVLM-7B (Stage 2)": "https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage2.zip",
    "FastVLM-7B (Stage 3)": "https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_7b_stage3.zip",
}


# --- Helper Functions ---
def get_model_paths():
    """Gets the list of all downloaded model paths."""
    if not os.path.exists(MODEL_ROOT_DIR):
        return []
    return sorted(
        [
            os.path.join(MODEL_ROOT_DIR, d)
            for d in os.listdir(MODEL_ROOT_DIR)
            if os.path.isdir(os.path.join(MODEL_ROOT_DIR, d))
        ]
    )


# --- Core Download Logic (meant to be run in a thread) ---
def _perform_download_and_extraction_task(model_name_to_download, url, update_queue):
    """
    Performs model download and extraction. Sends updates via update_queue.
    Queue item format:
    - ('status', 'message_key', {kwargs})
    - ('selector_update_info', {'choices': new_choices_list, 'value': new_value_path_or_None})
    - ('finished', success_boolean) # Indicates task completion
    """
    try:
        update_queue.put(
            ("status", "download_starting", {"model_name": model_name_to_download})
        )

        if not os.path.exists(MODEL_ROOT_DIR):
            os.makedirs(MODEL_ROOT_DIR)
            update_queue.put(("status", "created_model_dir", {"dir": MODEL_ROOT_DIR}))

        file_name = url.split("/")[-1]
        zip_download_path = os.path.join(
            MODEL_ROOT_DIR, file_name
        )  # Path for the zip file

        # Download
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded_size = 0
            # Create a temporary file for downloading to avoid issues with partially downloaded zips
            # if the process is interrupted or if the same filename (model name) is downloaded again.
            temp_zip_path = zip_download_path + ".tmp"

            with open(temp_zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress = (
                        (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    )
                    update_queue.put(
                        (
                            "status",
                            "download_progress",
                            {
                                "progress": progress,
                                "model_name": model_name_to_download,
                            },
                        )
                    )

            # Rename temporary file to final zip path after successful download
            os.rename(temp_zip_path, zip_download_path)

        update_queue.put(
            (
                "status",
                "download_complete",
                {"model_name": model_name_to_download, "path": zip_download_path},
            )
        )

        # Unzip
        update_queue.put(("status", "unzip_starting", {"file": file_name}))
        extracted_folders = []
        with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
            # Determine the root directory name(s) within the zip
            # This assumes the zip file contains a single top-level directory for the model
            namelist = zip_ref.namelist()
            if namelist:
                # Often the first item is the directory itself, or a file within it
                # e.g. ['model_abc/', 'model_abc/config.json', ...]
                # We want to find 'model_abc'
                for member_name in namelist:
                    if "/" in member_name:
                        top_folder = member_name.split("/")[0]
                        if (
                            top_folder not in extracted_folders
                            and not top_folder.startswith(".")
                        ):
                            extracted_folders.append(top_folder)
                    # If there are files at the root of the zip, this logic might need adjustment
                    # For now, we assume a common structure like 'model_name/...'

            zip_ref.extractall(MODEL_ROOT_DIR)

        update_queue.put(("status", "unzip_complete", {"dir": MODEL_ROOT_DIR}))

        # 清理下载的 zip 文件
        os.remove(zip_download_path)
        update_queue.put(("status", "zip_removed", {"file": file_name}))

        current_model_paths = get_model_paths()
        newly_added_model_path = None

        if extracted_folders:
            # Try to determine the path of the newly extracted model.
            # This assumes the zip extracts into a folder with a name related to 'extracted_folders[0]'
            # It's possible the folder name in the zip is different from the model_name_to_download.
            # We check against current_model_paths which are actual directory names.
            for folder_name in extracted_folders:
                potential_new_path = os.path.join(MODEL_ROOT_DIR, folder_name)
                if (
                    potential_new_path in current_model_paths
                ):  # Check if this extracted folder is now a listed model path
                    newly_added_model_path = potential_new_path
                    break

        update_queue.put(
            (
                "selector_update_info",
                {"choices": current_model_paths, "value": newly_added_model_path},
            )
        )

        if newly_added_model_path:
            update_queue.put(
                (
                    "status",
                    "model_download_success_with_path",
                    {
                        "model_name": model_name_to_download,
                        "path": newly_added_model_path,
                    },
                )
            )
        else:
            # This case can happen if the extracted folder name is unexpected or if get_model_paths doesn't pick it up immediately.
            update_queue.put(
                (
                    "status",
                    "model_download_success_no_path",
                    {"model_name": model_name_to_download},
                )
            )

        update_queue.put(("finished", True))

    except Exception as e:
        update_queue.put(
            (
                "status",
                "download_error",
                {"model_name": model_name_to_download, "error": str(e)},
            )
        )
        # Attempt to clean up temporary download file if it exists
        if "temp_zip_path" in locals() and os.path.exists(temp_zip_path):
            try:
                os.remove(temp_zip_path)
            except OSError:
                pass  # Ignore if removal fails
        # Attempt to clean up zip file if it exists and extraction failed mid-way
        if "zip_download_path" in locals() and os.path.exists(zip_download_path):
            try:
                os.remove(zip_download_path)
            except OSError:
                pass

        current_model_paths_on_error = get_model_paths()
        update_queue.put(
            (
                "selector_update_info",
                {"choices": current_model_paths_on_error, "value": None},
            )
        )
        update_queue.put(("finished", False))
