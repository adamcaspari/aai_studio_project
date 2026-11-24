import unittest

from phase12_core import plan_metrics, sample_project, validate_project


class Phase12CoreTests(unittest.TestCase):
    def test_sample_project_and_budget_preview(self):
        project = sample_project()
        self.assertEqual(len(project.generation_alternatives), 5)
        metrics = plan_metrics(project, {"Solar": 100})
        self.assertEqual(metrics["generation"], 100)
        self.assertEqual(metrics["cost"], 90_000_000)
        self.assertEqual(metrics["co2"], 0)
        self.assertEqual(metrics["land"], 2)

    def test_reject_storage_as_generation(self):
        raw = sample_project().to_dict()
        raw["generation_alternatives"][0]["name"] = "Battery storage"
        with self.assertRaisesRegex(ValueError, "储能"):
            validate_project(raw)

    def test_reject_unaffordable_project(self):
        raw = sample_project().to_dict()
        raw["budget"] = 1
        with self.assertRaisesRegex(ValueError, "没有可行"):
            validate_project(raw)

    def test_reject_negative_allocation(self):
        with self.assertRaisesRegex(ValueError, "非负"):
            plan_metrics(sample_project(), {"Wind": -1})


if __name__ == "__main__":
    unittest.main()
