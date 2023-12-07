"""
The tests in this file requires the installation of [WasmEdge Runtime](https://github.com/WasmEdge/WasmEdge)
as the dependecy `wasm_chat` is run on WasmEdge Runtime.

To install WasmEdge Runtime and the required `wasm-infer.wasm`, run the following commands:

```bash
curl -sSf https://raw.githubusercontent.com/second-state/wasm-llm/main/deploy.sh | bash
```

To uninstall WasmEdge Runtime and `wasm-infer.wasm`, run the following commands:

```bash
curl -sSf https://raw.githubusercontent.com/second-state/wasm-llm/main/deploy.sh | bash -s -- uninstall
```

"""

from langchain.chat_models.wasm_chat import (
    ChatWasmLocal,
    PromptTemplateType,
    ChatWasmService,
)
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from urllib.request import urlretrieve
import pytest
import os
import subprocess


def get_assets(url: str) -> str:
    """Download model or api-server wasm file."""
    local_filename = url.split("/")[-1]

    def download_progress(count, block_size, total_size):
        percent = count * block_size * 100 // total_size
        print(f"\rDownloading {local_filename}: {percent}%", end="")

    if not os.path.exists(local_filename):
        urlretrieve(url, local_filename, download_progress)

    return local_filename


def remove_downloaded(file: str):
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
def test_chat_wasmedge() -> None:
    """This test requires the port 8080 is avaiable."""
    model_file = get_assets(
        "https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf"
    )
    api_server_wasm_file = get_assets(
        "https://github.com/second-state/llama-utils/raw/main/api-server/llama-api-server.wasm"
    )

    command = [
        "wasmedge",
        "--dir",
        ".:.",
        "--nn-preload",
        f"default:GGML:AUTO:{model_file}",
        "llama-api-server.wasm",
        "-p",
        "chatml",
    ]
    server_process = subprocess.Popen(command)

    chat = ChatWasmService(service_ip_addr="0.0.0.0", service_port="8080")
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]
    response = chat(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "Paris" in response.content

    remove_downloaded(model_file)
    remove_downloaded(api_server_wasm_file)

    server_process.terminate()


@pytest.mark.enable_socket
def test_chat_service_default_url() -> None:
    """This test requires the port 10889 is not avaiable."""
    chat = ChatWasmService(service_ip_addr="0.0.0.0", service_port="10889")
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]

    with pytest.raises(ValueError, match="Error code: 502, reason: Bad Gateway"):
        response = chat(messages)
