from app.pipeline.indexing_pipeline import run_indexing_pipeline

from app.retrieval.faiss_retriever import FAISSRetriever
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.multi_query import MultiQueryRetriever, MultiQueryGenerator

from app.llm.groq_llm import GroqLLM

from app.reranker.cross_encoder import CrossEncoderReranker
from app.postprocessing.context_processor import ContextProcessor

from app.generation.answer_generator import AnswerGenerator
from app.generation.confidence import ConfidenceScorer



class RAGPipeline:
    def __init__(self, pdf_path):
        # Load once (IMPORTANT)
        self.vectorstore, self.chunks = run_indexing_pipeline(pdf_path)

        self.faiss_retriever = FAISSRetriever(self.vectorstore, k=3)
        self.bm25_retriever = BM25Retriever(self.chunks, k=3)
        self.hybrid_retriever = HybridRetriever(
            self.faiss_retriever,
            self.bm25_retriever
        )

        self.llm = GroqLLM()
        self.query_generator = MultiQueryGenerator(self.llm)
        self.multi_query_retriever = MultiQueryRetriever(
            self.hybrid_retriever,
            self.query_generator
        )

        self.reranker = CrossEncoderReranker()
        self.processor = ContextProcessor(self.chunks)
        self.generator = AnswerGenerator(self.llm)
        self.scorer = ConfidenceScorer()

    def run(self, query: str):
        # Retrieval
        retrieved_docs = self.multi_query_retriever.retrieve(query)

        # Reranking
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=5)

        # Processing
        final_docs = self.processor.process(reranked_docs)

        # Limit context
        final_docs = final_docs[:5]

        # Generate answer
        answer = self.generator.generate(query, final_docs)

        # Score
        confidence = self.scorer.score(answer, final_docs)

        return answer, confidence