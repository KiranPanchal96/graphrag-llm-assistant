import os
import sys
from langchain.prompts import PromptTemplate
from langfuse import Langfuse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.prompts.prompt_templates import EXAMPLE_PROMPT, SUFFIX_PROMPT

PROMPT_SOURCE = os.getenv("PROMPT_SOURCE", "manual").lower() # manual or langfuse

def load_prompt_from_langfuse(langfuse: Langfuse, prompt_name: str, fallback: str) -> str:
    try:
        prompt_obj = langfuse.get_prompt(name=prompt_name)
        return prompt_obj.get_langchain_prompt()
    except Exception as e:
        print(f"Could not fetch '{prompt_name}' prompt from Langfuse, using static default.", e)
        return fallback

def get_example_prompt(langfuse: Langfuse) -> PromptTemplate:
    if PROMPT_SOURCE == "manual":
        print("Using manual/static example prompt.")
        return PromptTemplate.from_template(EXAMPLE_PROMPT)
    else:
        example_prompt_string = load_prompt_from_langfuse(
            langfuse, "main_rag_example", EXAMPLE_PROMPT
        )
        return PromptTemplate.from_template(example_prompt_string)

def get_suffix_prompt(langfuse: Langfuse) -> str:
    if PROMPT_SOURCE == "manual":
        print("Using manual/static suffix prompt.")
        return SUFFIX_PROMPT
    else:
        return load_prompt_from_langfuse(
            langfuse, "main_rag_suffix", SUFFIX_PROMPT
        )
