from langchain_core.prompts import PromptTemplate
import re

class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a highly accurate AI assistant.

Answer the question using ONLY the provided context.

IMPORTANT RULES:
1. You MUST use citations in the format [Context X]
2. DO NOT use any other citation format like [1], [2], etc.
3. Every key statement MUST include a citation
4. If answer is not found, say: "I don't have enough information."
5. Use at least 2 different context sources if available

Context:
{context}

Question:
{question}

Answer (with STRICT [Context X] citations):
"""
        )

    def generate(self, query: str, context_docs):
        context_parts = []

        for i, doc in enumerate(context_docs):
            context_parts.append(f"[Context {i+1}]\n{doc.page_content}")

        context_text = "\n\n".join(context_parts)

        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )

        response = self.llm.invoke(prompt)
        answer = response.strip()
        
        answer = re.sub(r"\[\d+(,\s*\d+)*\]", "", answer)

        return answer