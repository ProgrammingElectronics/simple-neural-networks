import unittest


# Elementwise Multiplication
# Takes 2 cevtors of teh same length and returns a single vector of their Hadamard product
def elementwise_multiplication(vec_a, vec_b):

    assert(len(vec_a) == len(vec_b))

    output = []

    for i in range(len(vec_a)):
        output.append(vec_a[i] * vec_b[i])

    return output


class TestVectorFunctions(unittest.TestCase):

    def test_elementwise_multiplication(self):
        vec_a = [0.1, 0.2, 0]
        vec_b = [8.5, 0.65, 1.2]
        result = elementwise_multiplication(vec_a, vec_b)
        self.assertListAlmostEqual(result, [0.85, 0.13, 0], 5)
          

    def assertListAlmostEqual(self, list1, list2, tolerance):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a,b,tolerance)


if __name__ == '__main__':
    unittest.main()