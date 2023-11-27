from typing import Any, Dict, List, Optional, Mapping, Type
import logging
import json
import requests

from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import root_validator, Field
from langchain.utils import (
    get_pydantic_field_names,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatMessage,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.messages import (
    AIMessageChunk,
    BaseMessageChunk,
    ChatMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
)
from wasm_chat import WasmChat, Metadata

logger = logging.getLogger(__name__)

DEFAULT_API_SERVER_BASE = "http://localhost:10889"


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        return AIMessage(content=_dict.get("content", "") or "")
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


class ChatWasm(BaseChatModel):
    """WasmEdge locally runs large language models."""

    wasm_chat: Optional[WasmChat] = None
    """WasmChat instance"""
    request_timeout: int = 60
    """request timeout for chat http requests"""
    llama_api_server_base: str = Field(default=DEFAULT_API_SERVER_BASE)
    """Llama-api-server custom endpoints"""

    model = "llama-2-7b"
    """model name, default is `llama-2-7b`."""
    temperature: float = 0.3
    """What sampling temperature to use."""
    top_k: int = 5
    """What search sampling control to use."""
    top_p: float = 0.85
    """Whether to use search enhance, default is False."""
    streaming: bool = False
    """Whether to stream the results or not."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Baichuan API."""
        normal_params = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        return {**normal_params, **self.model_kwargs}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        res = self._chat(messages, **kwargs)
        return res

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        # parameters = {**self._default_params, **kwargs}

        # model = parameters.pop("model")
        # headers = parameters.pop("headers", {})

        # payload = {
        #     "model": model,
        #     "messages": [_convert_message_to_dict(m) for m in messages],
        # }

        # url = f"{self.llama_api_server_base}/v1/chat/completions"
        # res = requests.post(
        #     url=url,
        #     timeout=self.request_timeout,
        #     headers={
        #         "accept": "application/json",
        #         "Content-Type": "application/json",
        #         **headers,
        #     },
        #     data=json.dumps(payload),
        # )

        # init wasm environment
        if self.wasm_chat is None:
            model_file = "/Volumes/Dev/secondstate/me/pyo3/wasm-chat/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf"
            model_alias = "default"
            wasm_file = "/Volumes/Dev/secondstate/me/wasm-llm/target/wasm32-wasi/release/inference.wasm"
            dir_mapping = ".:."

            # create WasmChat instance
            self.wasm_chat = WasmChat(model_file, model_alias, wasm_file, dir_mapping)

            metadata = Metadata()
            print(f"log_enable: {metadata.log_enable}")
            print(f"reverse_prompt: {metadata.reverse_prompt}")

            # init inference context
            self.wasm_chat.init_inference_context(model_alias, metadata)

        prompt = """<|im_start|>system
            Answer as concisely as possible.<|im_end|>
            <|im_start|>user
            What is the capital of France?<|im_end|>
            <|im_start|>assistant"""

        # run inference
        assistant_message = self.wasm_chat.infer(prompt)
        print(f"[Answer] {assistant_message}")

        token_usage = {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }

        # create ChatResult
        generations = [ChatGeneration(message=AIMessage(content=assistant_message))]
        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    # def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
    #     message = _convert_dict_to_message(response.get("choices")[0].get("message"))
    #     generations = [ChatGeneration(message=message)]

    #     token_usage = response["usage"]
    #     llm_output = {"token_usage": token_usage, "model": self.model}
    #     return ChatResult(generations=generations, llm_output=llm_output)

    def _create_chat_result(
        self, message: str, token_usage: Mapping[str, Any]
    ) -> ChatResult:
        generations = [ChatGeneration(message=AIMessage(content=message))]

        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "wasmedge-chat"
