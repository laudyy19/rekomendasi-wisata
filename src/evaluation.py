def precision_at_k(
    recommended,
    relevant,
    k=5
):

    recommended = recommended[:k]

    hit = len(
        set(recommended)
        &
        set(relevant)
    )

    return hit / k