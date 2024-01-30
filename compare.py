def compare(a, b):
    """Compare two arrays entry by entry.
    The array that has the largest entry first is considered to be larger.
    
    arguments:
    a -- first array
    b -- second array
    
    return: Is the first array greater, smaller or equal as the second array.
    """
    for value_a, value_b in zip(a, b):
        if value_a > value_b:
            return "greater"
        if value_b > value_a:
            return "smaller"
    return "equal"