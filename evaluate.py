# from app.pipeline.rag_pipeline import RAGPipeline
# from ragas import evaluate
# from datasets import Dataset
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall
# )

# pipeline = RAGPipeline()

# # Load PDF first
# # pipeline.load_documents(...)
# pipeline.load_pdf_from_path(
#     "data/uploads/Attention_is_all_you_need.pdf"
# )

# # -------------------------
# # EVALUATION DATASET
# # -------------------------

# eval_data = [
#     {
#         "question": "What is self attention?",
#         "ground_truth":
#         "Self-attention relates different positions of a sequence to compute a representation."
#     },

#     {
#         "question": "Why is multi head attention used?",
#         "ground_truth":
#         "Multi-head attention allows the model to attend to information from different representation subspaces."
#     },

#     {
#         "question": "What is positional encoding?",
#         "ground_truth":
#         "Positional encoding injects information about token positions into the model."
#     },

#     {
#         "question": "What is the Transformer architecture?",
#         "ground_truth":
#         "The Transformer is a sequence transduction model based entirely on attention mechanisms."
#     },

#     {
#         "question": "What are queries keys and values?",
#         "ground_truth":
#         "Attention uses queries, keys and values to calculate weighted outputs."
#     }
# ]

# # -------------------------
# # BUILD RAGAS DATASET
# # -------------------------

# questions = []
# answers = []
# contexts = []
# ground_truths = []

# for item in eval_data:

#     print(
#         f"Evaluating: {item['question']}"
#     )

#     result = pipeline.run(
#         item["question"]
#     )

#     questions.append(
#         item["question"]
#     )

#     answers.append(
#         result["answer"]
#     )

#     contexts.append(
#         result["contexts"]
#     )

#     ground_truths.append(
#         item["ground_truth"]
#     )

# dataset = Dataset.from_dict(
#     {
#         "question": questions,
#         "answer": answers,
#         "contexts": contexts,
#         "ground_truth": ground_truths
#     }
# )

# # -------------------------
# # RUN RAGAS
# # -------------------------

# results = evaluate(
#     dataset,
#     metrics=[
#         faithfulness,
#         answer_relevancy,
#         context_precision,
#         context_recall
#     ]
# )

# # -------------------------
# # PRINT RESULTS
# # -------------------------

# print("\n")
# print("=" * 50)
# print("RAG EVALUATION RESULTS")
# print("=" * 50)

# print(results)

# print("\n")
# print("Faithfulness      :", results["faithfulness"])
# print("Answer Relevancy  :", results["answer_relevancy"])
# print("Context Precision :", results["context_precision"])
# print("Context Recall    :", results["context_recall"])


from app.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

pipeline.load_pdf_from_path(
    "data/uploads/Attention_is_all_you_need.pdf"
)

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
        "Positional encoding provides information about token positions."
    }
]

for item in eval_data:

    result = pipeline.run(
        item["question"]
    )

    print("\n" + "="*60)
    print("QUESTION:")
    print(item["question"])

    print("\nGROUND TRUTH:")
    print(item["ground_truth"])

    print("\nRAG ANSWER:")
    print(result["answer"])

    print("\nCONTEXTS RETRIEVED:")
    print(len(result["contexts"]))