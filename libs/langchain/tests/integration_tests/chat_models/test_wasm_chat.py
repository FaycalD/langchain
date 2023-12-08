"""
The tests in this file requires the installation of [WasmEdge Runtime](https://github.com/WasmEdge/WasmEdge)
as the dependecy `wasm_chat` is run on WasmEdge Runtime.

To install WasmEdge Runtime and the required `wasm-infer.wasm`, run the following
commands:

```bash
curl -sSf https://raw.githubusercontent.com/second-state/wasm-llm/main/deploy.sh | bash
```

To uninstall WasmEdge Runtime and `wasm-infer.wasm`, run the following commands:

```bash
curl -sSf https://raw.githubusercontent.com/second-state/wasm-llm/main/deploy.sh | \
    bash -s -- uninstall
```

"""

import os
from urllib.request import urlretrieve

import pytest

from langchain.chat_models.wasm_chat import (
    ChatWasmLocal,
    ChatWasmService,
    PromptTemplateType,
)
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage


def get_assets(url: str) -> str:
    """Download model or api-server wasm file."""
    local_filename = url.split("/")[-1]

    def download_progress(count: int, block_size: int, total_size: int) -> None:
        percent = count * block_size * 100 // total_size
        print(f"\rDownloading {local_filename}: {percent}%", end="")

    if not os.path.exists(local_filename):
        urlretrieve(url, local_filename, download_progress)

    return local_filename


def remove_downloaded(file: str) -> None:
    """Remove the downloaded assets."""
    if os.path.exists(file):
        os.remove(file)


@pytest.mark.enable_socket
def test_chat_wasm() -> None:
    model_file = get_assets(
        "https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf"
    )

    chat = ChatWasmLocal(
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

    remove_downloaded(model_file)


@pytest.mark.enable_socket
def test_chat_wasm_with_wasm_file() -> None:
    model_file = get_assets(
        "https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf"
    )
    home_dir = os.getenv("HOME")
    wasm_file = f"{home_dir}/.wasmedge/wasm/wasm_infer.wasm"

    chat = ChatWasmLocal(
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

    remove_downloaded(model_file)


@pytest.mark.enable_socket
def test_chat_wasm_with_reverse_prompt() -> None:
    model_file = get_assets(
        "https://huggingface.co/second-state/MistralLite-7B-GGUF/resolve/main/mistrallite.Q5_K_M.gguf"
    )

    chat = ChatWasmLocal(
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

    remove_downloaded(model_file)


@pytest.mark.enable_socket
def test_chat_wasm_service() -> None:
    """This test requires the port 8080 is not occupied."""

    service_url = "https://14cb-50-112-58-64.ngrok-free.app"
    chat = ChatWasmService(service_url=service_url)
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]
    response = chat(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "Paris" in response.content
