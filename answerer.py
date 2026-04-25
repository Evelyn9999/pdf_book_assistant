def build_short_answer(results: list[dict]) -> str:
    if not results:
        return "No relevant answer was found."

    best_text = results[0]["text"].strip()

    # Very simple first version:
    # take the first 300 characters of the best passage
    return best_text[:300] + ("..." if len(best_text) > 300 else "")