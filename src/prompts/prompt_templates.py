# src/prompts/prompt_templates.py

"""
Prompt templates and prompt management for Life Strategy RAG.
All default prompt strings and helpers are defined here for import and re-use.
"""

# --- MAIN FEW-SHOT RAG PROMPT EXAMPLES ---

EXAMPLE_PROMPT = """
Example:
Context:
{context}

Question:
{question}

Answer:
{answer}
"""

SUFFIX_PROMPT = """
Context:
{context}

Question:
{question}

Let's think step by step:
1. Identify the key elements in the context.
2. Reason through what is most relevant to the question.
3. Provide a clear and accurate answer.

Answer:
"""

# --- EVALUATION PROMPTS (for correctness, etc.) ---

CORRECTNESS_EVAL_PROMPT = """
Evaluate the correctness of the generation on a continuous scale from 0 to 1.
A generation can be considered correct (Score: 1) if it includes all the key facts from the ground truth
and if every fact presented in the generation is factually supported by the ground truth or common sense.

Input:
Query: {{query}}
Generation: {{generation}}
Ground truth: {{ground_truth}}

Think step by step.
"""
