import os
import requests
import zipfile
import shutil
import glob  # For get_model_paths
import threading
import time  # For retry delay
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

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


def download_chunk(
    url,
    start,
    end,
    temp_file_path,
    chunk_id,
    progress_queue,
    model_name,
    total_chunks,
    max_retries=3,
    retry_delay=5,
):
    """Downloads a chunk of the file and reports progress."""
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, headers=headers, stream=True, timeout=60
            )  # Increased timeout
            response.raise_for_status()
            part_file_path = f"{temp_file_path}.part{chunk_id}"
            with open(part_file_path, "wb") as f:
                for chunk_content in response.iter_content(chunk_size=8192):
                    if chunk_content:
                        f.write(chunk_content)
            # Report progress after chunk is downloaded
            progress_queue.put(
                ("download_progress_percent", (chunk_id + 1) / total_chunks, model_name)
            )
            return True
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                progress_queue.put(
                    (
                        "status",
                        "download_retry",
                        {
                            "model_name": model_name,
                            "chunk_id": chunk_id,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "error": str(e),
                        },
                    )
                )
                time.sleep(retry_delay)
            else:
                error_msg = f"Chunk {chunk_id} download failed after {max_retries} attempts: {str(e)}"
                if hasattr(e.response, "status_code"):
                    error_msg += f"\nHTTP Status Code: {e.response.status_code}"
                if hasattr(e.response, "text"):
                    error_msg += f"\nServer Response: {e.response.text[:200]}"
                progress_queue.put(
                    (
                        "status",
                        "download_error",
                        {"model_name": model_name, "error": error_msg},
                    )
                )
                return False
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                progress_queue.put(
                    (
                        "status",
                        "download_retry",
                        {
                            "model_name": model_name,
                            "chunk_id": chunk_id,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "error": str(e),
                        },
                    )
                )
                time.sleep(retry_delay)
            else:
                error_msg = f"Unknown error in download_chunk {chunk_id} after {max_retries} attempts: {str(e)}"
                progress_queue.put(
                    (
                        "status",
                        "download_error",
                        {"model_name": model_name, "error": error_msg},
                    )
                )
                return False
    return False  # Should be unreachable if logic is correct, but as a fallback


# --- Core Download Logic (meant to be run in a thread) ---
def _perform_download_and_extraction_task(model_name_to_download, url, update_queue):
    """Optimized download and extraction task with detailed progress reporting."""
    temp_zip_path = ""
    num_chunks = 4  # Define num_chunks for consistent use
    try:
        update_queue.put(
            ("status", "download_starting", {"model_name": model_name_to_download})
        )

        if not os.path.exists(MODEL_ROOT_DIR):
            os.makedirs(MODEL_ROOT_DIR)
            update_queue.put(("status", "created_model_dir", {"dir": MODEL_ROOT_DIR}))

        file_name = url.split("/")[-1]
        zip_download_path = os.path.join(MODEL_ROOT_DIR, file_name)
        temp_zip_path = zip_download_path + ".tmp"  # Define here for cleanup in finally

        try:
            response = requests.head(url, timeout=10)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            if total_size == 0:
                raise Exception(
                    "Failed to get file size: content-length is 0 or missing."
                )
        except requests.exceptions.RequestException as e:
            error_msg = (
                f"Failed to get file info for {model_name_to_download}: {str(e)}"
            )
            if hasattr(e.response, "status_code"):
                error_msg += f"\nHTTP Code: {e.response.status_code}"
            if hasattr(e.response, "text"):
                error_msg += f"\nServer Response: {e.response.text[:200]}"
            update_queue.put(
                (
                    "status",
                    "download_error",
                    {"model_name": model_name_to_download, "error": error_msg},
                )
            )
            update_queue.put(("finished", False))
            return

        chunk_ranges = []
        bytes_per_chunk = total_size // num_chunks
        for i in range(num_chunks):
            start = i * bytes_per_chunk
            end = start + bytes_per_chunk - 1
            if i == num_chunks - 1:
                end = total_size - 1  # Ensure last chunk covers the remainder
            chunk_ranges.append((start, end))

        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = [
                executor.submit(
                    download_chunk,
                    url,
                    s,
                    e,
                    temp_zip_path,
                    i,
                    update_queue,
                    model_name_to_download,
                    num_chunks,
                )
                for i, (s, e) in enumerate(chunk_ranges)
            ]

            all_successful = True
            for future in futures:
                if not future.result():  # download_chunk now returns bool
                    all_successful = False
                    # Error has been put on the queue by download_chunk
                    break  # Stop processing further if one chunk fails

            if not all_successful:
                # No need to raise an exception here if error already sent
                # Ensure 'finished' is sent if not already by an error path in download_chunk
                update_queue.put(
                    ("finished", False)
                )  # Could be redundant if error path in chunk sent it
                return

        # If all chunks downloaded successfully, proceed to combine
        with open(temp_zip_path, "wb") as outfile:
            for i in range(num_chunks):
                part_file = f"{temp_zip_path}.part{i}"
                with open(part_file, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
                os.remove(part_file)

        if os.path.exists(zip_download_path):
            os.remove(
                zip_download_path
            )  # Remove if exists from a previous failed attempt
        shutil.move(temp_zip_path, zip_download_path)
        temp_zip_path = ""  # Clear to prevent deletion in finally if successful

        update_queue.put(
            (
                "status",
                "unzip_starting",
                {"file": file_name, "model_name": model_name_to_download},
            )
        )
        extracted_folders = []
        with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
            namelist = zip_ref.namelist()
            if namelist:
                for member_name in namelist:
                    if "/" in member_name:
                        top_folder = member_name.split("/")[0]
                        if (
                            top_folder not in extracted_folders
                            and not top_folder.startswith((".", "__MACOSX"))
                        ):
                            extracted_folders.append(top_folder)
                zip_ref.extractall(MODEL_ROOT_DIR)

        main_extracted_folder_name = (
            extracted_folders[0]
            if extracted_folders
            else model_name_to_download.split(" ")[0]
        )

        update_queue.put(
            (
                "status",
                "unzip_complete",
                {
                    "dir": os.path.join(MODEL_ROOT_DIR, main_extracted_folder_name),
                    "model_name": model_name_to_download,
                },
            )
        )

        os.remove(zip_download_path)
        update_queue.put(
            (
                "status",
                "zip_removed",
                {"file": file_name, "model_name": model_name_to_download},
            )
        )

        current_model_paths = get_model_paths()
        newly_added_model_path = None
        if extracted_folders:
            for folder_name in extracted_folders:
                potential_new_path = os.path.join(MODEL_ROOT_DIR, folder_name)
                if potential_new_path in current_model_paths:
                    newly_added_model_path = potential_new_path
                    break
        # If not found by exact match, try to guess based on model_name_to_download (more robust)
        if not newly_added_model_path:
            # Attempt to find a folder that starts with the model name (e.g. FastVLM-0.5B)
            simplified_model_name = (
                model_name_to_download.split(" ")[0].replace("(", "").replace(")", "")
            )
            for p in current_model_paths:
                if os.path.basename(p).startswith(simplified_model_name):
                    newly_added_model_path = p
                    break

        update_queue.put(
            (
                "selector_update_info",
                {
                    "choices": current_model_paths,
                    "value": newly_added_model_path,
                },
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
            update_queue.put(
                (
                    "status",
                    "model_download_success_no_path",
                    {"model_name": model_name_to_download},
                )
            )

        update_queue.put(("finished", True))

    except Exception as e:
        # Ensure specific error messages from download_chunk are prioritized if they were sent
        # This generic catch is for other unexpected errors in the _perform_download_and_extraction_task itself.
        error_msg = f"Error in download task for {model_name_to_download}: {str(e)}"
        update_queue.put(
            (
                "status",
                "download_error",
                {"model_name": model_name_to_download, "error": error_msg},
            )
        )
        update_queue.put(("finished", False))
    finally:
        # Cleanup temporary files
        if temp_zip_path and os.path.exists(temp_zip_path):
            try:
                os.remove(temp_zip_path)
            except OSError:
                pass  # Error already logged or not critical for cleanup message
        for i in range(num_chunks):
            part_file = f"{zip_download_path + '.tmp'}.part{i}"  # temp_zip_path might be empty if move succeeded
            if os.path.exists(part_file):
                try:
                    os.remove(part_file)
                except OSError:
                    pass
