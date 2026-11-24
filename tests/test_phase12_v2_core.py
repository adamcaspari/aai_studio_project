import unittest

from phase12_v2_core import (
    check_comprehension, default_draft, plan_errors, plan_metrics,
    publish_checks, transport_draft, validate_project,
)


class TeacherAuthoredV2Tests(unittest.TestCase):
    def test_example_passes_both_publish_checks(self):
        project = validate_project(default_draft())
        ok, problems, costs = publish_checks(project)
        self.assertTrue(ok)
        self.assertEqual(problems, [])
        self.assertEqual(costs, (80_000_000, 88_000_000))

    def test_post_event_can_block_publication(self):
        draft = default_draft()
        draft["budget"] = 84_000_000
        ok, problems, _ = publish_checks(validate_project(draft))
        self.assertFalse(ok)
        self.assertEqual(len(problems), 1)
        self.assertIn("Post-event", problems[0])

    def test_teacher_cap_is_checked_in_plan_and_publication(self):
        draft = default_draft()
        draft["extra_constraints"] = [{"metric": "co2", "value": 0, "label": "Zero CO2"}]
        project = validate_project(draft)
        self.assertFalse(publish_checks(project)[0])
        metrics = plan_metrics(project, {"Natural Gas": 100})
        self.assertTrue(any("Zero CO2" in issue for issue in plan_errors(project, metrics)))

    def test_comprehension_gate(self):
        project = validate_project(default_draft())
        self.assertEqual(check_comprehension(project, {0: 0, 1: 1}), [1])
        self.assertEqual(check_comprehension(project, {0: 1, 1: 1}), [])

    def test_incomplete_rubric_cannot_publish(self):
        draft = default_draft()
        draft["rubric"][0]["levels"][2] = ""
        with self.assertRaisesRegex(ValueError, "rubric criterion"):
            validate_project(draft)

    def test_transport_example_publish_and_capacity_limits(self):
        project = validate_project(transport_draft())
        self.assertEqual(project.unit_label, "riders/hour")
        ok, problems, costs = publish_checks(project)
        self.assertTrue(ok)
        self.assertEqual(problems, [])
        self.assertEqual(costs, (2_300_000, 2_670_000))
        allocations = {
            "Safer Walking Routes": 160, "Protected Bike Lanes": 200,
            "Electric Shuttle": 200, "Park-and-Ride Bus": 40, "Diesel Shuttle": 0,
        }
        metrics = plan_metrics(project, allocations)
        self.assertEqual(metrics["generation"], 600)
        self.assertEqual(metrics["cost"], 2_500_000)
        self.assertEqual(plan_errors(project, metrics, allocations=allocations), [])
        allocations["Safer Walking Routes"] = 180
        metrics = plan_metrics(project, allocations)
        self.assertTrue(any("Safer Walking Routes exceeds" in issue
                            for issue in plan_errors(project, metrics, allocations=allocations)))


if __name__ == "__main__":
    unittest.main()
