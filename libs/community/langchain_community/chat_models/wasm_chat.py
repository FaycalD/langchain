import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import root_validator
from langchain_core.utils import get_pydantic_field_names
from wasm_chat import Metadata, PromptTemplateType, WasmChat

logger = logging.getLogger(__name__)


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


class WasmChatService(BaseChatModel):
    """Chat with LLMs via `llama-api-server`

    For the information about `llama-api-server`, visit https://github.com/second-state/llama-utils
    """

    request_timeout: int = 60
    """request timeout for chat http requests"""
    service_url: Optional[str] = None
    """URL of WasmChat service"""
    model: str = "NA"
    """model name, default is `NA`."""

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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        res = self._chat(messages, **kwargs)

        if res.status_code != 200:
            raise ValueError(f"Error code: {res.status_code}, reason: {res.reason}")

        response = res.json()

        return self._create_chat_result(response)

    def _chat(self, messages: List[BaseMessage], **kwargs: Any) -> requests.Response:
        if self.service_url is None:
            res = requests.models.Response()
            res.status_code = 503
            res.reason = "The IP address or port of the chat service is incorrect."
            return res

        service_url = f"{self.service_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [_convert_message_to_dict(m) for m in messages],
        }

        res = requests.post(
            url=service_url,
            timeout=self.request_timeout,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
        )

        return res

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        message = _convert_dict_to_message(response["choices"][0].get("message"))
        generations = [ChatGeneration(message=message)]

        token_usage = response["usage"]
        llm_output = {"token_usage": token_usage, "model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "wasm-chat"


class WasmChatLocal(BaseChatModel):
    """Chat with LLMs locally"""

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
            if self.wasm_file is None:
                self.wasm_chat = WasmChat(
                    self.model_file,
                    self.prompt_template,
                )
            else:
                self.wasm_chat = WasmChat(
                    self.model_file,
                    self.prompt_template,
                    self.wasm_file,
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
