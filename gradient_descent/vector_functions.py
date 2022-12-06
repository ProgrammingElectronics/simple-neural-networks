def elementwise_multiplication(vec_a, vec_b):

    assert(len(vec_a) == len(vec_b))
    return [x[0]*x[1] for x in zip(vec_a, vec_b)]


def elementwise_addition(vec_a, vec_b):

    assert(len(vec_a) == len(vec_b))
    return [sum(x) for x in zip(vec_a, vec_b)]

def vector_sum(vec):

    return sum(vec)

def vector_average(vec):

    return vector_sum(vec) / len(vec)

def dot_product(vec_a, vec_b):

    return vector_sum(elementwise_multiplication(vec_a, vec_b))
