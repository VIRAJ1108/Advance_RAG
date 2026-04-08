class ConfidenceScorer:
    def score(self, answer: str, context_docs):
        if "i don't have enough information" in answer.lower():
            return 0.0

        answer_length = len(answer.split())

        # simple heuristic (we'll improve later)
        if answer_length < 5:
            return 0.2
        elif answer_length < 20:
            return 0.5
        else:
            return 0.8