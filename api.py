# api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import json
import time
import logging
from typing import List, Optional, Union, Dict, Any, AsyncGenerator
from PIL import Image  # For type hinting
import base64  # For image parsing
import io  # For image parsing

# --- Import Core Logic ---
import core_logic

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for OpenAI Compatibility ---


class APIBaseModel(BaseModel):
    class Config:
        # For Pydantic V2, use model_config = {"extra": "ignore"}
        # For Pydantic V1, use extra = "ignore"
        extra = "ignore"  # Pydantic v1 style
        # model_config = {"extra": "ignore"} # Pydantic v2 style


class MessageContentItem(APIBaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = (
        None  # e.g. {"url": "data:image/jpeg;base64,..."}
    )


class Message(APIBaseModel):
    role: str
    content: Union[str, List[MessageContentItem]]


class ChatCompletionRequest(APIBaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 512
    # Add other OpenAI parameters if needed, e.g., presence_penalty, frequency_penalty


class ChoiceDelta(APIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamingChoice(APIBaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None  # OpenAI includes this, optional for us


class StreamingChatCompletionResponse(APIBaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamingChoice]
    system_fingerprint: Optional[str] = None  # OpenAI includes this


class ChatMessage(APIBaseModel):
    role: str
    content: str


class Choice(APIBaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[Any] = None


class Usage(APIBaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(APIBaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


app = FastAPI()


# --- Helper Functions ---
def _parse_messages(messages: List[Message]) -> tuple[str, Optional[Image.Image]]:
    user_prompt = ""
    image_input_pil = None

    for msg in reversed(messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                user_prompt = msg.content
            elif isinstance(msg.content, list):
                for item in msg.content:
                    if item.type == "text" and item.text:
                        user_prompt = item.text
                    elif (
                        item.type == "image_url"
                        and item.image_url
                        and item.image_url.url
                    ):
                        image_url_data = item.image_url.url
                        if image_url_data.startswith("data:image"):
                            try:
                                header, encoded = image_url_data.split(",", 1)
                                image_data = base64.b64decode(encoded)
                                image_input_pil = Image.open(io.BytesIO(image_data))
                            except Exception as e:
                                logger.error(f"Invalid base64 image data: {e}")
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Invalid base64 image data: {e}",
                                )
                        else:
                            logger.warning(
                                f"Image URL not in data URI format is not yet supported: {image_url_data}"
                            )
                            raise HTTPException(
                                status_code=400,
                                detail="Image URL not in data URI format is not yet supported.",
                            )
            if user_prompt or image_input_pil:
                break

    if not user_prompt and image_input_pil:
        user_prompt = (
            core_logic.get_default_prompt()
        )  # Use default prompt from core_logic

    return user_prompt, image_input_pil


def _count_tokens(text: str) -> int:
    tokenizer = core_logic.get_tokenizer()
    if tokenizer and text:
        return len(tokenizer.encode(text))
    return 0


# --- API Endpoint ---
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, http_request: Request):
    request_id = f"chatcmpl-{time.time_ns()}"
    created_time = int(time.time())

    if not core_logic.is_model_loaded():
        logger.info("API: Model not loaded. Attempting to auto-load.")
        if not core_logic.load_default_model_if_not_loaded():
            logger.error("API: Model not loaded and failed to auto-load.")
            raise HTTPException(
                status_code=503,
                detail="Model is not currently loaded and failed to auto-load.",
            )
        logger.info("API: Model auto-loaded successfully.")

    # Use the model name reported by core_logic, or the one from request if preferred
    # For now, always use the one from core_logic as it reflects the actual loaded model.
    model_name_to_report = core_logic.get_current_model_name() or request.model

    try:
        user_prompt, image_input_pil = _parse_messages(request.messages)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing messages: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error parsing messages: {str(e)}")

    if not user_prompt:
        raise HTTPException(
            status_code=400, detail="No text prompt found or generated from messages."
        )

    # FastVLM is a vision model, an image is generally expected.
    if image_input_pil is None:
        raise HTTPException(
            status_code=400,
            detail="No image provided in messages. This is a vision model.",
        )

    temperature = (
        request.temperature
        if request.temperature is not None
        else core_logic.DEFAULT_TEMPERATURE
    )
    top_p = request.top_p if request.top_p is not None else core_logic.DEFAULT_TOP_P
    max_tokens = request.max_tokens or 512  # Use request max_tokens or a default
    # num_beams and conv_mode are not typically part of basic OpenAI API, using defaults from core_logic
    num_beams = core_logic.DEFAULT_NUM_BEAMS
    conv_mode = core_logic.DEFAULT_CONV_MODE

    if request.stream:

        async def stream_generator():
            full_response_content = ""
            try:
                async for token in core_logic.execute_model_prediction_stream(
                    image_input_pil=image_input_pil,
                    prompt_str=user_prompt,
                    temperature_float=temperature,
                    top_p_float=top_p,
                    num_beams_int=num_beams,
                    conv_mode_str=conv_mode,
                    max_new_tokens=max_tokens,
                ):
                    if await http_request.is_disconnected():
                        logger.info("Client disconnected, stopping stream.")
                        break

                    # Check if the token is an error message from core_logic stream start
                    # This is a convention based on how execute_model_prediction_stream yields errors first
                    if (
                        token.startswith("[") and ":error_" in token.lower()
                    ):  # Heuristic for error string
                        logger.error(f"Error from model stream: {token}")
                        # Send error in stream (custom, not strictly OpenAI format for stream errors)
                        error_chunk = StreamingChatCompletionResponse(
                            id=request_id,
                            created=created_time,
                            model=model_name_to_report,
                            choices=[
                                StreamingChoice(
                                    index=0,
                                    delta=ChoiceDelta(content=f"Error: {token}"),
                                    finish_reason="error",
                                )
                            ],
                        )
                        yield f"data: {error_chunk.model_dump_json(exclude_none=True)}\n\n"  # Ensure correct termination and newlines
                        break  # Stop stream after error

                    full_response_content += token
                    chunk = StreamingChatCompletionResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name_to_report,
                        choices=[
                            StreamingChoice(index=0, delta=ChoiceDelta(content=token))
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                if not await http_request.is_disconnected():
                    final_chunk = StreamingChatCompletionResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name_to_report,
                        choices=[
                            StreamingChoice(
                                index=0, delta=ChoiceDelta(), finish_reason="stop"
                            )
                        ],  # or "length" if applicable
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                    yield f"data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error during streaming inference: {e}", exc_info=True)
                error_content = json.dumps(
                    {
                        "error": {
                            "message": f"Server error during streaming: {str(e)}",
                            "type": "server_error",
                        }
                    }
                )
                yield f"data: {error_content}\n\n"

            # Token counting after stream for logging (optional)
            # prompt_tokens = _count_tokens(user_prompt)
            # completion_tokens = _count_tokens(full_response_content)
            # logger.info(f"Streamed usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:  # Non-streaming
        try:
            generated_text = core_logic.execute_model_prediction(
                image_input_pil=image_input_pil,
                prompt_str=user_prompt,
                temperature_float=temperature,
                top_p_float=top_p,
                num_beams_int=num_beams,
                conv_mode_str=conv_mode,
                max_new_tokens=max_tokens,
            )

            # Check if generated_text indicates an error from core_logic
            if generated_text.startswith("[") and ":error_" in generated_text.lower():
                logger.error(f"Error from model prediction: {generated_text}")
                raise HTTPException(status_code=500, detail=generated_text)

            prompt_tokens = _count_tokens(user_prompt)
            completion_tokens = _count_tokens(generated_text)
            total_tokens = prompt_tokens + completion_tokens
            usage_stats = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            response = ChatCompletionResponse(
                id=request_id,
                created=created_time,
                model=model_name_to_report,
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(
                            role="assistant", content=generated_text.strip()
                        ),
                        finish_reason="stop",  # or "length" if max_tokens caused truncation
                    )
                ],
                usage=usage_stats,
            )
            return response
        except (
            HTTPException
        ):  # Re-raise if it's already an HTTPException (e.g. from error check)
            raise
        except Exception as e:
            logger.error(f"Error during non-streaming inference: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )


# --- Server management ---
_api_server_instance = None
_api_server_thread = None


def start_api_server(host: str = "0.0.0.0", port: int = 8008) -> str:
    global _api_server_instance, _api_server_thread

    if (
        _api_server_instance
        and hasattr(_api_server_instance, "started")
        and _api_server_instance.started
    ) or (_api_server_thread and _api_server_thread.is_alive()):
        logger.info(f"API server is likely already running on port {port}.")
        return f"API server may already be running on http://{host}:{port}"

    # Model loading is now handled by the endpoint itself if not loaded,
    # or by the startup_event if run standalone.
    # However, a pre-check here can be useful for immediate feedback if called from Gradio.
    if not core_logic.is_model_loaded():
        logger.info(
            "start_api_server: Model not loaded. Will be loaded on first request or by Uvicorn startup."
        )

    import threading

    config = uvicorn.Config(
        "api:app", host=host, port=port, log_level="info", loop="asyncio"
    )  # Use string for app
    _api_server_instance = uvicorn.Server(config)

    _api_server_thread = threading.Thread(target=_api_server_instance.run, daemon=True)
    _api_server_thread.start()

    # It might take a moment for the server to be fully up.
    # Checking _api_server_instance.started might not be immediately true.
    time.sleep(1)  # Give it a second to initialize
    status = "starting"
    try:  # Uvicorn 0.29 removed server.started, check thread for liveness
        if _api_server_thread.is_alive():
            status = "running (thread is alive)"
        else:
            status = "failed to start (thread not alive)"
    except Exception:
        status = "state unknown"

    logger.info(f"API server {status} on http://{host}:{port}")
    return f"API server {status} on http://{host}:{port}. Check logs for confirmation."


def stop_api_server() -> str:
    global _api_server_instance, _api_server_thread
    if _api_server_instance and hasattr(_api_server_instance, "should_exit"):
        _api_server_instance.should_exit = True
        if _api_server_thread:
            _api_server_thread.join(timeout=5.0)
            if _api_server_thread.is_alive():
                logger.warning(
                    "API server thread did not exit cleanly after 5 seconds. Consider force_exit if available or manual process kill."
                )
                # For newer uvicorn, direct shutdown might be instance.shutdown()
                # but needs to be awaited if called from async context.
                # Since this is sync, setting should_exit is the primary way.
            else:
                logger.info("API server thread exited.")
        else:
            logger.info("API server instance found but no thread to join.")

        _api_server_instance = None
        _api_server_thread = None
        logger.info("API server stop requested and instance cleared.")
        return "API server stopped."
    elif _api_server_thread and _api_server_thread.is_alive():
        logger.warning(
            "API server thread is alive, but no server instance to control. Stop may be incomplete."
        )
        # This state is unusual, might mean server was started externally or instance was lost
        _api_server_thread = None  # Clear stale thread ref
        return "API server thread was alive but instance was missing. Attempted to clear thread."
    else:
        logger.info("API server is not running or instance not found.")
        _api_server_instance = None
        _api_server_thread = None
        return "API server is not running or already stopped."


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup: Initializing core logic...")
    # Ensure core_logic (especially model) is initialized if API runs standalone
    if not core_logic.is_model_loaded():
        logger.info(
            "API Startup: Model not loaded. Attempting to load default model..."
        )
        if core_logic.load_default_model_if_not_loaded():  # This is a synchronous call
            logger.info(
                f"API Startup: Model '{core_logic.get_current_model_name()}' loaded successfully."
            )
        else:
            logger.error(
                "API Startup: Failed to load default model. API may not function correctly."
            )
    else:
        current_model = core_logic.get_current_model_name()
        logger.info(f"API Startup: Model '{current_model}' is already loaded.")


if __name__ == "__main__":
    port = 8008
    logger.info(f"Attempting to run API server independently on port {port}...")
    # uvicorn.run expects the app as a string "module:app_variable_name"
    # The startup_event in the app itself will handle model loading.
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
