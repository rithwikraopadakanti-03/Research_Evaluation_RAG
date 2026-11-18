def reciprocal_rank_fusion(result_lists, k=60, top_k=5):
    scores, objects = {}, {}
    for results in result_lists:
        for rank, item in enumerate(results, 1):
            cid = item["chunk"].chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            objects[cid] = item["chunk"]
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{"chunk": objects[cid], "score": scores[cid]} for cid in ranked]
