from langchain.chat_models.wasm_chat import (
    WasmChatLocal,
    PromptTemplateType,
    WasmChatService,
)
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage

import pytest
import os


def test_chat_wasm() -> None:
    model_file = "/Volumes/Store/models/gguf/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf"

    chat = WasmChatLocal(
        model_file=model_file,
        prompt_template=PromptTemplateType.ChatML,
    )
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]
    chat_result = chat(messages)
    assert isinstance(chat_result, AIMessage)
    assert isinstance(chat_result.content, str)
    assert "Paris" in chat_result.content


def test_chat_wasm_with_wasm_file() -> None:
    model_file = "/Volumes/Store/models/gguf/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf"
    home_dir = os.getenv("HOME")
    wasm_file = f"{home_dir}/.wasmedge/wasm/wasm_infer.wasm"

    chat = WasmChatLocal(
        model_file=model_file,
        prompt_template=PromptTemplateType.ChatML,
        wasm_file=wasm_file,
    )
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]
    chat_result = chat(messages)
    assert isinstance(chat_result, AIMessage)
    assert isinstance(chat_result.content, str)
    assert "Paris" in chat_result.content


def test_chat_wasm_with_reverse_prompt() -> None:
    model_file = "/Volumes/Store/models/gguf/mistrallite.Q5_K_M.gguf"

    chat = WasmChatLocal(
        model_file=model_file,
        prompt_template=PromptTemplateType.MistralLite,
        reverse_prompt="</s>",
    )
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]
    chat_result = chat(messages)
    print(f"chat_result: {chat_result}")
    assert isinstance(chat_result, AIMessage)
    assert isinstance(chat_result.content, str)
    assert "Paris" in chat_result.content


@pytest.mark.enable_socket
def test_chat_wasmedge() -> None:
    chat = WasmChatService(service_ip_addr="50.112.58.64", service_port="8080")
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]
    response = chat(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "Paris" in response.content


@pytest.mark.enable_socket
def test_chat_service_default_url() -> None:
    chat = WasmChatService()
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]

    with pytest.raises(ValueError, match="Error code: 502, reason: Bad Gateway"):
        response = chat(messages)
