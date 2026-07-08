from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def semantic_similarity(predicted, ground_truth):

    emb1 = model.encode(predicted, convert_to_tensor=True)
    emb2 = model.encode(ground_truth, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()

    return round(score, 4)