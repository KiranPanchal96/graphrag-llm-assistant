import os
import sys
import time
from typing import Any

import pytest
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler


@pytest.mark.integration
def test_langfuse_trace_created() -> None:
    """Integration test to verify Langfuse traces are created."""
    # Load environment
    load_dotenv()
    LANGFUSE_PUBLIC_API_KEY = os.getenv("LANGFUSE_PUBLIC_API_KEY")
    LANGFUSE_SECRET_API_KEY = os.getenv("LANGFUSE_SECRET_API_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Basic environment checks
    assert LANGFUSE_PUBLIC_API_KEY, "Missing LANGFUSE_PUBLIC_API_KEY"
    assert LANGFUSE_SECRET_API_KEY, "Missing LANGFUSE_SECRET_API_KEY"
    assert OPENAI_API_KEY, "Missing OPENAI_API_KEY"

    # Langfuse setup
    Langfuse(
        public_key=LANGFUSE_PUBLIC_API_KEY,
        secret_key=LANGFUSE_SECRET_API_KEY,
        host=LANGFUSE_HOST,
    )
    langfuse_handler = CallbackHandler()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0
    )

    # Send prompt to LLM (traced via Langfuse)
    prompt = "Say hello from automated Langfuse pytest."
    result: Any = llm.invoke(prompt, config={"callbacks": [langfuse_handler]})
    assert hasattr(result, "content"), "No content in OpenAI response"
    assert "hello" in result.content.lower()

    # Wait for trace to sync to Langfuse
    time.sleep(4)

    # Check for recent trace via Langfuse API
    url = f"{LANGFUSE_HOST}/api/public/traces?limit=5"
    resp = requests.get(url, auth=(LANGFUSE_PUBLIC_API_KEY, LANGFUSE_SECRET_API_KEY))
    assert (
        resp.status_code == 200
    ), f"Langfuse API returned {resp.status_code}: {resp.text}"
    traces = resp.json().get("data", [])
    print("Langfuse traces:", traces)
    assert traces, f"No traces found in Langfuse. Raw data: {str(traces)}"

    # Optionally, check if our prompt or similar is found in recent traces
    found = any(
        prompt[:10].lower()
        in (str(trace.get("name", "")).lower() + str(trace.get("input", "")).lower())
        for trace in traces
    )
    assert (
        found
    ), f"Prompt snippet '{prompt[:10]}' not found in recent Langfuse traces: {str(traces)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
