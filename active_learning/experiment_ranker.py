from .information_gain import compute_information_gain

def rank_candidates(candidates, model):
    scores = {}
    for c in candidates:
        scores[c] = compute_information_gain(None, None) # Placeholder args
    return sorted(scores, key=scores.get, reverse=True)
