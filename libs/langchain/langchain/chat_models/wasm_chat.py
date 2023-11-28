from typing import Any, Dict, List, Optional, Mapping
from pathlib import Path
import logging
import json

from langchain.chat_models.base import BaseChatModel
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
from wasm_chat import WasmChat, Metadata, PromptTemplateType

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
    model_file: Optional[str] = None
    """Path to gguf model file."""
    wasm_file: Optional[str] = None
    """Path to wasm file."""
    prompt_template: Optional[PromptTemplateType] = None
    """Prompt template to use for generating prompts."""
    model: Optional[str] = None
    """Name of gguf model."""

    ctx_size: int = 4096
    """Size of the prompt context, default is 4096."""
    n_predict: int = 1024
    """Number of tokens to predict, default is 1024."""
    n_gpu_layers: int = 100
    """Number of layers to run on GPU, default is 100."""
    batch_size: int = 4096
    """Batch size for prompt processing, default is 4096."""
    reverse_prompt: Optional[str] = None
    """Halt generation at PROMPT, return control. Default is None."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def _default_metadata(self) -> Dict[str, Any]:
        """Get the default parameters for calling Baichuan API."""

        normal_params = {
            "ctx_size": self.ctx_size,
            "n_predict": self.n_predict,
            "n_gpu_layers": self.n_gpu_layers,
            "batch_size": self.batch_size,
            "reverse_prompt": self.reverse_prompt,
        }

        return {**normal_params}

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
        # init wasm environment
        if self.wasm_chat is None:
            # set the 'model' field
            model_file = Path(self.model_file).resolve()
            self.model = model_file.stem

            # set metadata
            parameters = {**self._default_metadata, **kwargs}
            metadata = Metadata(**parameters)

            # create WasmChat instance
            self.wasm_chat = WasmChat(
                self.model_file,
                self.wasm_file,
                self.prompt_template,
            )

            # init inference context
            self.wasm_chat.init_inference_context(metadata)

        payload = {
            "model": self.model,
            "messages": [_convert_message_to_dict(m) for m in messages],
        }
        data = json.dumps(payload)

        # generate prompt string
        prompt = self.wasm_chat.generate_prompt_str(data)

        # run inference
        ai_message = self.wasm_chat.infer(prompt)

        token_usage = {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }

        # create ChatResult
        generations = [ChatGeneration(message=AIMessage(content=ai_message))]
        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_chat_result(
        self, message: str, token_usage: Mapping[str, Any]
    ) -> ChatResult:
        generations = [ChatGeneration(message=AIMessage(content=message))]

        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "wasmedge-chat"
