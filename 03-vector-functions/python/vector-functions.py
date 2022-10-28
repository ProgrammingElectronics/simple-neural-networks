import unittest


################################################
# Vector Functions
################################################

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

##########################################
# Test Data
##########################################

# Don't change these arrays, it will break your tests, as the expected values defined in each test are based on these
test_vec_a = [0.1, 0.2, 0]
test_vec_b = [8.5, 0.65, 1.2]

class TestVectorFunctions(unittest.TestCase):

    def test_elementwise_multiplication(self):
        expected = [0.85, 0.13, 0]
        under_test = elementwise_multiplication(test_vec_a, test_vec_b)
        self.assertListAlmostEqual(under_test, expected)
          
    def assertListAlmostEqual(self, list1, list2, tolerance = 5):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a,b,tolerance)

    def test_elementwise_addition(self):
        expected = [8.6, 0.85, 1.2]
        under_test = elementwise_addition(test_vec_a, test_vec_b)
        self.assertListAlmostEqual(under_test, expected)

    def test_vector_sum(self):
        expected = 0.3
        under_test = vector_sum(test_vec_a)
        self.assertAlmostEqual(under_test, expected)

    def test_vector_average(self):
        expected = 0.1
        under_test = vector_average(test_vec_a)
        self.assertAlmostEqual(under_test, expected)

    def test_dot_product(self):
        expected = 0.98
        under_test = dot_product(test_vec_a, test_vec_b)
        self.assertAlmostEqual(under_test, expected)

if __name__ == '__main__':
    unittest.main()