import unittest


# Elementwise Multiplication
def elementwise_multiplication(vec_a, vec_b):

    assert(len(vec_a) == len(vec_b))

    output = 0

    for i in range(len(vec_a)):
        output += vec_a[i] * vec_b[i]

    return output



class TestVectorFunctions(unittest.TestCase):

    def test_elementwise_multiplication(self):
        vec_a = [0.1, 0.2, 0]
        vec_b = [8.5, 0.65, 1.2]
        result = elementwise_multiplication(vec_a, vec_b)
        self.assertAlmostEqual(result, 0.98)

if __name__ == '__main__':
    unittest.main()