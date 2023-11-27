from langchain.chat_models.wasm_chat import ChatWasm
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from urllib.request import urlretrieve
from typing import Optional

import pytest
import os
import logging
import subprocess
import signal
import shutil
import time


def test_chat_wasmedge(self) -> None:
    chat = ChatWasm()
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]
    response = chat(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "Paris" in response.content
