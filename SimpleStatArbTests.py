import unittest
from SimpleStatArb import compute_rolling_mean_online, compute_rolling_std_dev_online, compute_rolling_covariance_online
from math import sqrt

class TestComputeRollingMeanOnline(unittest.TestCase):
    def test_compute_rolling_mean_online(self):
        # Test with a simple case: prev_mean = 2, n = 3, curr_entries = 4, new_price = 5, old_price = 1
        result = compute_rolling_mean_online(2, 3, 4, 5, 1)
        self.assertEqual(result, 2 + 4/3)

        # Test with a case that includes negative numbers
        result = compute_rolling_mean_online(-0.5, 2, 3, 3, -1)
        self.assertEqual(result, 3/2)

        # Test with a case where old_price is None
        result = compute_rolling_mean_online(2, 3, 3, 5)
        self.assertEqual(result, 4 / 3 + 5 / 3)

        # Test with a case where n is 1
        result = compute_rolling_mean_online(1, 1, 2, 2, 1)
        self.assertEqual(result, 2)
        
        # Test with a case where n is 1 and old_price is None
        result = compute_rolling_mean_online(1, 1, 1, 2)
        self.assertEqual(result, 2)
        
        vector = [1, 2, 3, 4, 5]
        result = compute_rolling_mean_online(0, 2, 1, vector[0], None)
        self.assertEqual(result, 1)
        
        result = compute_rolling_mean_online(1, 2, 2, vector[1])
        self.assertEqual(result, 1.5)
        
        result = compute_rolling_mean_online(1.5, 2, 3, vector[2], vector[0])
        self.assertEqual(result, 2.5)
        
        with self.assertRaises(ValueError):
            compute_rolling_mean_online(1.5, 2, 3, vector[2])


class TestComputeRollingStdDevOnline(unittest.TestCase):
    def test_compute_rolling_std_dev_online(self):
        vector = [1, 2, 3, -4, 5]
        
        result = compute_rolling_std_dev_online(0, 1, 0, 3, 1, vector[0], None)
        self.assertEqual(result, 0)
        
        result = compute_rolling_std_dev_online(0, 1.5, 1, 3, 2, vector[1])
        self.assertAlmostEqual(result, sqrt(0.5))
        
        result = compute_rolling_std_dev_online(sqrt(0.5), 2, 1.5, 3, 3, vector[2])
        self.assertAlmostEqual(result, 1)

        with self.assertRaises(ValueError):
            compute_rolling_std_dev_online(1, 1/3, 2, 3, 4, vector[3])
        
        result = compute_rolling_std_dev_online(1, 1/3, 2, 3, 4, vector[3], vector[0])
        self.assertAlmostEqual(result, sqrt((29 - 1/3)/2))
        
        result = compute_rolling_std_dev_online(sqrt((29 - 1/3)/2), 4/3, 1/3, 3, 4, vector[4], vector[1])
        self.assertAlmostEqual(result, sqrt((50 - 16 / 3) / 2))
        
        
class TestComputeRollingCovarianceOnline(unittest.TestCase):
    def test_compute_rolling_covariance_online(self):
        vector1 = [1, 2, 3, 4, -5]
        vector2 = [2, 3, 4, -5, -6]
        
        result = compute_rolling_covariance_online(0, 0, 0, 3, 1, vector1[0], vector2[0], None)
        self.assertEqual(result, 0)
        
        result = compute_rolling_covariance_online(0, 1, 2, 3, 2, vector1[1], vector2[1])
        self.assertAlmostEqual(result, 0.5)
        
        result = compute_rolling_covariance_online(0.5, 1.5, 2.5, 3, 3, vector1[2], vector2[2])
        self.assertAlmostEqual(result, 1)
        
        with self.assertRaises(ValueError):
            compute_rolling_covariance_online(1, 1/3, 2, 3, 4, vector1[3], vector2[3])
        
        result = compute_rolling_covariance_online(1, 2, 3, 3, 4, vector1[3], vector2[3], vector1[0], vector2[0])
        self.assertAlmostEqual(result, -4)
        
        result = compute_rolling_covariance_online(-4, 3, 2/3, 3, 4, vector1[4], vector2[4], vector1[1], vector2[1])
        self.assertAlmostEqual(result, (22 + 14 / 3) / 2)
        
if __name__ == '__main__':
    unittest.main()