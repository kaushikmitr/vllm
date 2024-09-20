import asyncio
import base64
import importlib
import inspect
import re
from contextlib import asynccontextmanager
from http import HTTPStatus
import struct
from typing import Optional, Set

import json
import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest,
                                              EmbeddingResponse, ErrorResponse,
                                              LoadLoraAdapterRequest,
                                              TokenizeRequest,
                                              TokenizeResponse,
                                              UnloadLoraAdapterRequest)
from vllm.entrypoints.openai.rpc.client import AsyncEngineRPCClient
from vllm.entrypoints.openai.rpc.server import run_rpc_server
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.version import __version__ as VLLM_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_embedding: OpenAIServingEmbedding

logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


app = fastapi.FastAPI(lifespan=lifespan)

def dict_to_orca(data_dict):
    orca_data = bytearray()
    orca_data.extend(b'ORCA')  # Magic number
    orca_data.extend(struct.pack('I', 1))  # Version number
    orca_data.extend(struct.pack('I', len(data_dict)))  # Number of key-value pairs

    for key, value in data_dict.items():
        key_bytes = key.encode('utf-8')
        orca_data.extend(struct.pack('I', len(key_bytes)))  # Length of key
        orca_data.extend(key_bytes)  # Key
        orca_data.extend(struct.pack('i', value))  # Value (assuming 4-byte integer)

    return orca_data

def int_to_orca(value):
    orca_data = bytearray()
    orca_data.extend(b'ORCA')  # Magic number
    orca_data.extend(struct.pack('I', 1))  # Version number
    orca_data.extend(struct.pack('i', value))  # Single integer value
    return orca_data

    return orca_data
def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    generator = await openai_serving_completion.create_tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, TokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@app.post("/detokenize")
async def detokenize(request: DetokenizeRequest):
    generator = await openai_serving_completion.create_detokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, DetokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    custom_headers = {}
    if hasattr(generator, 'usage') and generator.usage:
        if generator.usage.registered_lora_adapters is not None:
            custom_headers["registered_lora_adapters"] = json.dumps(generator.usage.registered_lora_adapters)
            orca_registered_lora_adapters = dict_to_orca(generator.usage.registered_lora_adapters)
            custom_headers["orca_registered_lora_adapters"] = base64.b64encode(orca_registered_lora_adapters).decode('utf-8')
            
        if generator.usage.active_lora_adapters is not None:
            custom_headers["active_lora_adapters"] = json.dumps(generator.usage.active_lora_adapters)
            orca_active_lora_adapters = dict_to_orca(generator.usage.active_lora_adapters)
            custom_headers["orca_active_lora_adapters"] = base64.b64encode(orca_active_lora_adapters).decode('utf-8')
            
        if generator.usage.pending_queue_size is not None:
            custom_headers["pending_queue_size"] = str(generator.usage.pending_queue_size)
            orca_pending_queue_size = int_to_orca(generator.usage.pending_queue_size)
            custom_headers["orca_pending_queue_size"] = base64.b64encode(orca_pending_queue_size).decode('utf-8')
        custom_headers["total_tokens"] = str(generator.usage.total_tokens)
        custom_headers["model"] = request.model
            
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code, headers=custom_headers)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump(), headers=custom_headers)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    custom_headers = {}
    if hasattr(generator, 'usage') and generator.usage:
        if generator.usage.registered_lora_adapters is not None:
            custom_headers["registered_lora_adapters"] = json.dumps(generator.usage.registered_lora_adapters)
            orca_registered_lora_adapters = dict_to_orca(generator.usage.registered_lora_adapters)
            custom_headers["orca_registered_lora_adapters"] = base64.b64encode(orca_registered_lora_adapters).decode('utf-8')
            
        if generator.usage.active_lora_adapters is not None:
            custom_headers["active_lora_adapters"] = json.dumps(generator.usage.active_lora_adapters)
            orca_active_lora_adapters = dict_to_orca(generator.usage.active_lora_adapters)
            custom_headers["orca_active_lora_adapters"] = base64.b64encode(orca_active_lora_adapters).decode('utf-8')
            
        if generator.usage.pending_queue_size is not None:
            custom_headers["pending_queue_size"] = str(generator.usage.pending_queue_size)
            orca_pending_queue_size = int_to_orca(generator.usage.pending_queue_size)
            custom_headers["orca_pending_queue_size"] = base64.b64encode(orca_pending_queue_size).decode('utf-8')
        custom_headers["total_tokens"] = str(generator.usage.total_tokens)
        custom_headers["model"] = request.model
                
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code, headers=custom_headers)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump(), headers=custom_headers)


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await openai_serving_embedding.create_embedding(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!")

    @router.post("/start_profile")
    async def start_profile():
        logger.info("Starting profiler...")
        await async_engine_client.start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile():
        logger.info("Stopping profiler...")
        await async_engine_client.stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
    logger.warning(
        "Lora dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!")

    @router.post("/v1/load_lora_adapter")
    async def load_lora_adapter(request: LoadLoraAdapterRequest):
        response = await openai_serving_chat.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        response = await openai_serving_completion.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)

    @router.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(request: UnloadLoraAdapterRequest):
        response = await openai_serving_chat.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        response = await openai_serving_completion.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)


def build_app(args: Namespace) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    engine_args = AsyncEngineArgs.from_cli_args(args)

    # Enforce pixel values as image input type for vision language models
    # when serving with API server
    if engine_args.image_input_type is not None and \
        engine_args.image_input_type.upper() != "PIXEL_VALUES":
        raise ValueError(
            f"Invalid image_input_type: {engine_args.image_input_type}. "
            "Only --image-input-type 'pixel_values' is supported for serving "
            "vision language models with the vLLM API server.")

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)

    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats

    state.openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        base_model_paths,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser)
    state.openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )
    state.openai_serving_embedding = OpenAIServingEmbedding(
        engine_client,
        model_config,
        base_model_paths,
        request_logger=request_logger,
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    temp_socket.bind(("", args.port))

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        init_app_state(engine_client, model_config, app.state, args)

        temp_socket.close()

        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    uvloop.run(run_server(args))
