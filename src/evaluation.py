def precision_at_k(
    recommendations,
    actual_category
):

    relevant = (
        recommendations["Category"]
        == actual_category
    ).sum()

    return relevant / len(recommendations)