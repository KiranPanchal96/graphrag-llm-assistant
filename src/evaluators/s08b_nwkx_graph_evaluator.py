"""
s08b_nwkx_graph_evaluator.py
Evaluates the Graph-RAG pipeline using LLM-based QAEvalChain-style grading,
with optional BLEU, ROUGE-L, and BERTScore metrics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import nltk
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

from src.inference.s07b_nwkx_graph_pipeline import run_graph_rag_pipeline

# Optional metrics
USE_BLEU = False
USE_ROUGE = False
USE_BERTSCORE = False

if USE_BLEU or USE_ROUGE:
    nltk.download("punkt")


def load_eval_data(path: str = "data/eval/questions.json") -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_bleu(reference: str, prediction: str) -> float:
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    reference_tokens = [word_tokenize(reference.lower())]
    prediction_tokens = word_tokenize(prediction.lower())
    smoothie = SmoothingFunction().method4
    return float(
        sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)
    )


def compute_rouge_l(reference: str, prediction: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return float(score["rougeL"].fmeasure)


def compute_bertscore(reference: str, prediction: str) -> float:
    import bert_score

    P, R, F1 = bert_score.score(
        [prediction],
        [reference],
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
        device="cpu",
    )
    return float(F1[0].item())


def run_qa_eval(
    examples: list[dict[str, str]], predictions: list[dict[str, str]]
) -> list[dict[str, str]]:
    print("\n🧠 Running LLM-based evaluation (QAEvalChain-style)...")

    eval_prompt = PromptTemplate.from_template(
        """
You are evaluating the quality of an answer based on a reference answer.

Question: {query}
Reference Answer: {ground_truth}
Predicted Answer: {result}

Grade the answer:
- EXCELLENT: fully correct and complete
- ACCEPTABLE: partially correct or vague but somewhat relevant
- NEEDS IMPROVEMENT: incorrect or misses key elements

Return your response in this format:
Grade: <EXCELLENT/ACCEPTABLE/NEEDS IMPROVEMENT>
Rationale: <your reasoning>
"""
    )

    llm = ChatOpenAI(temperature=0)
    eval_chain = LLMChain(llm=llm, prompt=eval_prompt)

    graded: list[dict[str, str]] = []
    for example, prediction in zip(examples, predictions):
        inputs: dict[str, str] = {**example, **prediction}
        result = eval_chain.run(inputs)
        try:
            lines = result.strip().split("\n")
            grade = lines[0].replace("Grade:", "").strip()
            rationale = (
                lines[1].replace("Rationale:", "").strip() if len(lines) > 1 else ""
            )
        except (IndexError, AttributeError, ValueError):
            grade = "UNKNOWN"
            rationale = result.strip()
        graded.append({"grade": grade, "rationale": rationale})
    return graded


def evaluate_graph_rag(data: list[dict[str, str]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    examples: list[dict[str, str]] = []
    predictions: list[dict[str, str]] = []

    for item in data:
        question = item["question"]
        reference = item["reference_answer"]

        print(f"\n❓ Q: {question}")
        pipeline_result: dict[str, Any] = run_graph_rag_pipeline(question)

        # Extract answer text safely
        predicted_answer: str = ""
        if isinstance(pipeline_result, dict):
            predicted_answer = str(pipeline_result.get("result", ""))
        else:
            predicted_answer = str(pipeline_result)  # fallback

        print(f"🤖 A: {predicted_answer}")
        print(f"✅ Ref: {reference}")

        result: dict[str, Any] = {
            "question": question,
            "reference_answer": reference,
            "predicted_answer": predicted_answer,
            "raw_pipeline_output": pipeline_result,  # keep full output for debugging
        }

        examples.append({"query": question, "ground_truth": reference})
        predictions.append({"result": predicted_answer})

        if USE_BLEU:
            bleu = compute_bleu(reference, predicted_answer)
            print(f"📏 BLEU: {bleu:.3f}")
            result["bleu_score"] = bleu

        if USE_ROUGE:
            rouge = compute_rouge_l(reference, predicted_answer)
            print(f"📏 ROUGE-L: {rouge:.3f}")
            result["rouge_l_score"] = rouge

        if USE_BERTSCORE:
            bert = compute_bertscore(reference, predicted_answer)
            print(f"📏 BERTScore: {bert:.3f}")
            result["bertscore"] = bert

        results.append(result)

    llm_grades = run_qa_eval(examples, predictions)
    for result, eval_result in zip(results, llm_grades):
        result["llm_grade"] = eval_result["grade"]
        result["llm_rationale"] = eval_result["rationale"]

    return results


def save_results(
    results: list[dict[str, Any]], output_dir: str = "outputs/eval_results"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"graph_eval_results_{timestamp}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {path}")


if __name__ == "__main__":
    eval_data = load_eval_data()
    results = evaluate_graph_rag(eval_data)
    save_results(results)
