class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, chat_history, query):
        history_text = ""

        for turn in chat_history[-3:]:  # last 3 turns only
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        prompt = f"""
You are an AI assistant.

Convert the user's question into a standalone question using chat history.

Chat History:
{history_text}

Current Question:
{query}

Rewritten Standalone Question:
"""

        response = self.llm.invoke(prompt)

        return response.strip()