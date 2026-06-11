from app.pipeline.rag_pipeline import RAGPipeline

from bert_score import score
from rouge_score import rouge_scorer
import pandas as pd

# -------------------------
# LOAD PIPELINE
# -------------------------

pipeline = RAGPipeline()

pipeline.load_pdf_from_path(
    "data/uploads/Attention_is_all_you_need.pdf"
)

# -------------------------
# EVALUATION DATASET
# -------------------------

eval_data = [
    {
        "question": "What is self attention?",
        "ground_truth":
        "Self-attention relates different positions of a sequence to compute a representation."
    },

    {
        "question": "Why is multi-head attention used?",
        "ground_truth":
        "Multi-head attention allows the model to attend to information from different representation subspaces."
    },

    {
        "question": "What is positional encoding?",
        "ground_truth":
        "Positional encoding injects information about token positions into the model."
    },

    {
        "question": "What is the Transformer architecture?",
        "ground_truth":
        "The Transformer is a sequence transduction model based entirely on attention mechanisms."
    },

    {
        "question": "What are queries, keys and values?",
        "ground_truth":
        "Queries, keys and values are used in attention to compute weighted outputs."
    }
]

# -------------------------
# STORE RESULTS
# -------------------------

predictions = []
references = []

rouge_scores = []

scorer = rouge_scorer.RougeScorer(
    ['rougeL'],
    use_stemmer=True
)

# -------------------------
# RUN EVALUATION
# -------------------------

for item in eval_data:

    print(f"\nEvaluating: {item['question']}")

    result = pipeline.run(
        item["question"]
    )

    generated_answer = result["answer"]

    ground_truth = item["ground_truth"]

    predictions.append(
        generated_answer
    )

    references.append(
        ground_truth
    )

    rouge_result = scorer.score(
        ground_truth,
        generated_answer
    )

    rouge_scores.append(
        rouge_result["rougeL"].fmeasure
    )

# -------------------------
# BERT SCORE
# -------------------------

P, R, F1 = score(
    predictions,
    references,
    lang="en",
    verbose=True
)

# -------------------------
# RESULTS
# -------------------------

print("\n")
print("=" * 60)
print("RAG EVALUATION RESULTS")
print("=" * 60)

print(f"\nAverage BERT Precision : {P.mean().item():.4f}")
print(f"Average BERT Recall    : {R.mean().item():.4f}")
print(f"Average BERT F1        : {F1.mean().item():.4f}")

print(f"\nAverage ROUGE-L        : {sum(rouge_scores)/len(rouge_scores):.4f}")

# -------------------------
# SAVE RESULTS
# -------------------------

results_df = pd.DataFrame({
    "Question": [x["question"] for x in eval_data],
    "Ground Truth": references,
    "Generated Answer": predictions,
    "ROUGE-L": rouge_scores
})

results_df.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nSaved results to evaluation_results.csv")