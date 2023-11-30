from langchain.chat_models.wasm_chat import ChatWasm, PromptTemplateType
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage


def test_chat_wasm() -> None:
    model_file = (
        "/home/ubuntu/workspace/wasm-llm/wasm-chat/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf"
    )

    chat = ChatWasm(
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
    model_file = (
        "/home/ubuntu/workspace/wasm-llm/wasm-chat/tinyllama-1.1b-chat-v0.3.Q5_K_M.gguf"
    )
    wasm_file = "/home/ubuntu/.wasmedge/wasm/wasm_infer.wasm"

    chat = ChatWasm(
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
    model_file = "/home/ubuntu/workspace/models/second-state/MistralLite-7B-GGUF/mistrallite.Q5_K_M.gguf"

    chat = ChatWasm(
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
