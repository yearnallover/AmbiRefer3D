import unittest

from scripts.plot_topk_from_json import validate


def example():
    return {"k": [1, 3], "series": [{"name": "Direct",
            "true_target_at_k": [20, 40], "query_matching_share_at_k": [80, 60]}]}


class PlotInputTests(unittest.TestCase):
    def test_minimal_input(self):
        validate(example())

    def test_wrong_length(self):
        data = example()
        data["series"][0]["true_target_at_k"] = [20]
        with self.assertRaisesRegex(ValueError, "one value per K"):
            validate(data)

    def test_invalid_percentages(self):
        for value in (float("nan"), float("inf"), -1, 101, "20"):
            data = example()
            data["series"][0]["true_target_at_k"][0] = value
            with self.assertRaisesRegex(ValueError, "finite percentages"):
                validate(data)

    def test_duplicate_or_unsorted_k(self):
        for ks in ([1, 1], [3, 1], [0, 3]):
            data = example()
            data["k"] = ks
            with self.assertRaises(ValueError):
                validate(data)

    def test_duplicate_names(self):
        data = example()
        data["series"].append(dict(data["series"][0]))
        with self.assertRaisesRegex(ValueError, "unique name"):
            validate(data)


if __name__ == "__main__":
    unittest.main()
