"""
s08c_neo4j_graph_evaluator.py
Evaluates the Neo4j Graph-RAG pipeline using LLM-based QAEvalChain-style grading,
with optional BLEU, ROUGE-L, and BERTScore metrics.
"""

import os
import sys
import json
import nltk
from datetime import datetime

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.inference.s07c_neo4j_graph_pipeline import run_graph_rag_pipeline

# Optional metrics
USE_BLEU = False
USE_ROUGE = False
USE_BERTSCORE = False

if USE_BLEU or USE_ROUGE:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    from rouge_score import rouge_scorer
    nltk.download("punkt")

if USE_BERTSCORE:
    import bert_score


def load_eval_data(path="data/eval/questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_bleu(reference, prediction):
    reference_tokens = [word_tokenize(reference.lower())]
    prediction_tokens = word_tokenize(prediction.lower())
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)


def compute_rouge_l(reference, prediction):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return score["rougeL"].fmeasure


def compute_bertscore(reference, prediction):
    P, R, F1 = bert_score.score(
        [prediction], [reference],
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
        device="cpu"
    )
    return F1[0].item()


def extract_neo4j_answer(output):
    """Extracts the answer string from Neo4j pipeline output."""
    if isinstance(output, dict):
        # Most common: 'result' or 'answer' key
        return output.get("result") or output.get("answer") or str(output)
    return str(output)


def run_qa_eval(examples, predictions):
    print("\n🧠 Running LLM-based evaluation (QAEvalChain-style)...")

    eval_prompt = PromptTemplate.from_template("""
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
""")

    llm = ChatOpenAI(temperature=0)
    eval_chain = LLMChain(llm=llm, prompt=eval_prompt)

    graded = []
    for example, prediction in zip(examples, predictions):
        inputs = {**example, **prediction}
        result = eval_chain.run(inputs)
        try:
            lines = result.strip().split("\n")
            grade = lines[0].replace("Grade:", "").strip()
            rationale = lines[1].replace("Rationale:", "").strip() if len(lines) > 1 else ""
        except Exception:
            grade = "UNKNOWN"
            rationale = result.strip()
        graded.append({"grade": grade, "rationale": rationale})
    return graded


def evaluate_graph_rag(data):
    results = []
    examples = []
    predictions = []

    for item in data:
        question = item["question"]
        reference = item["reference_answer"]

        print(f"\n❓ Q: {question}")
        output = run_graph_rag_pipeline(question)
        predicted_answer = extract_neo4j_answer(output)
        print(f"🤖 A: {predicted_answer}")
        print(f"✅ Ref: {reference}")

        result = {
            "question": question,
            "reference_answer": reference,
            "predicted_answer": predicted_answer
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


def save_results(results, output_dir="outputs/eval_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"neo4j_graph_eval_results_{timestamp}.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {path}")


if __name__ == "__main__":
    eval_data = load_eval_data()
    results = evaluate_graph_rag(eval_data)
    save_results(results)
