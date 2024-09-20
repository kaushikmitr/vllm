import time
from typing import AsyncIterator, List, Optional, Tuple

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (EmbeddingRequest,
                                              EmbeddingResponse,
<<<<<<< HEAD
                                              EmbeddingResponseData, UsageInfo)
from vllm.entrypoints.openai.serving_completion import parse_prompt_format
from vllm.entrypoints.openai.serving_engine import OpenAIServing
=======
                                              EmbeddingResponseData,
                                              ErrorResponse, UsageInfo)
from vllm.entrypoints.openai.serving_engine import BaseModelPath, OpenAIServing
>>>>>>> 260d40b5 ([Core] Support Lora lineage and base model metadata management (#6315))
from vllm.logger import init_logger
from vllm.outputs import EmbeddingRequestOutput
from vllm.utils import merge_async_iterators, random_uuid

logger = init_logger(__name__)

TypeTokenIDs = List[int]


def request_output_to_embedding_response(
    final_res_batch: List[EmbeddingRequestOutput],
    request_id: str,
    created_time: int,
    model_name: str,
) -> EmbeddingResponse:
    data: List[EmbeddingResponseData] = []
    num_prompt_tokens = 0
    for idx, final_res in enumerate(final_res_batch):
        assert final_res is not None
        prompt_token_ids = final_res.prompt_token_ids

        embedding_data = EmbeddingResponseData(
            index=idx, embedding=final_res.outputs.embedding)
        data.append(embedding_data)

        num_prompt_tokens += len(prompt_token_ids)

    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        total_tokens=num_prompt_tokens,
    )

    return EmbeddingResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        data=data,
        usage=usage,
    )


class OpenAIServingEmbedding(OpenAIServing):

<<<<<<< HEAD
    def __init__(self, engine: AsyncLLMEngine, model_config: ModelConfig,
                 served_model_names: List[str]):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=None)
        self._check_embedding_mode(model_config.embedding_mode)
=======
    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        base_model_paths: List[BaseModelPath],
        *,
        request_logger: Optional[RequestLogger],
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         base_model_paths=base_model_paths,
                         lora_modules=None,
                         prompt_adapters=None,
                         request_logger=request_logger)
        self._enabled = self._check_embedding_mode(model_config.embedding_mode)
>>>>>>> 260d40b5 ([Core] Support Lora lineage and base model metadata management (#6315))

    async def create_embedding(self, request: EmbeddingRequest,
                               raw_request: Request):
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # Return error for unsupported features.
        if request.encoding_format == "base64":
            return self.create_error_response(
                "base64 encoding is not currently supported")
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = request.model
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.monotonic())

        # Schedule the request and get the result generator.
        generators = []
        try:
            prompt_is_tokens, prompts = parse_prompt_format(request.input)
            pooling_params = request.to_pooling_params()

            for i, prompt in enumerate(prompts):
                if prompt_is_tokens:
                    prompt_formats = self._validate_prompt_and_tokenize(
                        request, prompt_ids=prompt)
                else:
                    prompt_formats = self._validate_prompt_and_tokenize(
                        request, prompt=prompt)

                prompt_ids, prompt_text = prompt_formats

                generator = self.engine.encode(
                    {
                        "prompt": prompt_text,
                        "prompt_token_ids": prompt_ids
                    },
                    pooling_params,
                    f"{request_id}-{i}",
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator: AsyncIterator[Tuple[
            int, EmbeddingRequestOutput]] = merge_async_iterators(*generators)

        # Non-streaming response
        final_res_batch: List[Optional[EmbeddingRequestOutput]]
        final_res_batch = [None] * len(prompts)
        try:
            async for i, res in result_generator:
                if await raw_request.is_disconnected():
                    # Abort the request if the client disconnects.
                    await self.engine.abort(f"{request_id}-{i}")
                    # TODO: Use a vllm-specific Validation Error
                    return self.create_error_response("Client disconnected")
                final_res_batch[i] = res
            response = request_output_to_embedding_response(
                final_res_batch, request_id, created_time, model_name)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response

    def _check_embedding_mode(self, embedding_mode: bool):
        if not embedding_mode:
            logger.warning(
                "embedding_mode is False. Embedding API will not work.")
        else:
            logger.info("Activating the server engine with embedding enabled.")
