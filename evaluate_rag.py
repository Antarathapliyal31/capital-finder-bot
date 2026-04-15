"""Evaluate RAG pipeline using RAGAS metrics and send results to Langfuse."""

import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langfuse import Langfuse
from rag_chain import create_chain_with_contexts

load_dotenv()

# Sample evaluation questions and ground truths
EVAL_QUESTIONS = [
    "What is the capital of Japan?",
    "What is the capital of France?",
    "What is the capital of Brazil?",
    "What is the capital of Australia?",
    "What is the capital of India?",
    "What is the capital of Germany?",
    "What is the capital of South Africa?",
    "What is the capital of Turkey?",
    "What is the capital of Mongolia?",
    "What is the capital of Iceland?",
]

GROUND_TRUTHS = [
    "Tokyo is the capital of Japan. Tokyo has been the capital since 1868.",
    "Paris is the capital of France. Paris is known as the City of Light.",
    "Brasilia is the capital of Brazil. It was purpose-built and became the capital in 1960.",
    "Canberra is the capital of Australia. It was selected as a compromise between Sydney and Melbourne.",
    "New Delhi is the capital of India. It was inaugurated as the capital in 1931.",
    "Berlin is the capital of Germany. Berlin became the capital of reunified Germany in 1990.",
    "South Africa has three capitals: Pretoria (executive), Cape Town (legislative), and Bloemfontein (judicial).",
    "Ankara is the capital of Turkey. Ankara replaced Istanbul as the capital in 1923.",
    "Ulaanbaatar is the capital of Mongolia. It is the coldest capital city in the world.",
    "Reykjavik is the capital of Iceland. It is the northernmost capital of a sovereign state.",
]


def generate_responses():
    """Run the RAG chain on eval questions and collect answers + contexts."""
    chain = create_chain_with_contexts()
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Generating responses for evaluation...")
    for i, question in enumerate(EVAL_QUESTIONS):
        print(f"  [{i+1}/{len(EVAL_QUESTIONS)}] {question}")
        result = chain.invoke(question)
        questions.append(result["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(GROUND_TRUTHS[i])

    return questions, answers, contexts, ground_truths


def run_ragas_evaluation(questions, answers, contexts, ground_truths):
    """Run RAGAS evaluation for faithfulness and answer relevancy."""
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS evaluation (faithfulness & answer_relevancy)...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
    )
    return results


def send_to_langfuse(questions, answers, contexts, ragas_results):
    """Send RAGAS evaluation scores to Langfuse."""
    langfuse = Langfuse()

    df = ragas_results.to_pandas()

    print("\nSending results to Langfuse...")
    for i, row in df.iterrows():
        trace = langfuse.trace(
            name=f"rag-eval: {questions[i]}",
            input={"question": questions[i]},
            output={"answer": answers[i]},
            tags=["ragas-eval"],
            metadata={
                "contexts": contexts[i],
                "eval_type": "ragas",
            },
        )

        trace.score(
            name="faithfulness",
            value=row["faithfulness"],
            comment="RAGAS faithfulness score - measures factual consistency of answer with retrieved context",
        )
        trace.score(
            name="answer_relevancy",
            value=row["answer_relevancy"],
            comment="RAGAS answer relevancy score - measures how relevant the answer is to the question",
        )

        print(f"  Q: {questions[i]}")
        print(f"    Faithfulness: {row['faithfulness']:.4f}")
        print(f"    Answer Relevancy: {row['answer_relevancy']:.4f}")

    langfuse.flush()
    print("\nResults sent to Langfuse successfully!")
    return df


def main():
    questions, answers, contexts, ground_truths = generate_responses()
    ragas_results = run_ragas_evaluation(questions, answers, contexts, ground_truths)

    print("\n--- Aggregate RAGAS Scores ---")
    print(f"  Faithfulness:      {ragas_results['faithfulness']:.4f}")
    print(f"  Answer Relevancy:  {ragas_results['answer_relevancy']:.4f}")

    send_to_langfuse(questions, answers, contexts, ragas_results)


if __name__ == "__main__":
    main()
