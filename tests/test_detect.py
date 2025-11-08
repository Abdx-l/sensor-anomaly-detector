import unittest
import numpy as np
from detect import detect, forest_scores, svg


class DetectorTests(unittest.TestCase):
    def test_injected_spike_and_flatline(self):
        rng = np.random.default_rng(3)
        values = rng.normal(10,.12,300)
        values[80] = 25
        values[180:215] = 10.0
        result = detect(values,window=20,seed=11)
        self.assertTrue(result["flags"][80])
        self.assertTrue(result["flat"][210])
        self.assertTrue(any(e["start"] <= 80 <= e["end"] for e in result["events"]))
        self.assertEqual(len(result["scores"]),len(values))
        self.assertIn("<svg",svg(values,result))

    def test_reproducible_scores_and_input_validation(self):
        features = np.arange(60,dtype=float).reshape(20,3)
        np.testing.assert_array_equal(forest_scores(features,seed=9),forest_scores(features,seed=9))
        with self.assertRaises(ValueError): detect(np.array([1.,2.]),window=20)


if __name__ == "__main__": unittest.main()

