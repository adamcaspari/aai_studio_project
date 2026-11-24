"""Check that every shipped classroom case can be published and played."""
import json
import unittest
from pathlib import Path

from phase12_v2_core import (
    check_comprehension, plan_errors, plan_metrics, publish_checks, validate_project,
)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DELIVERABLES = EXAMPLES.parent / "deliverables"
SAMPLE_PLANS = {
    "riverside_water_supply.json": {
        "Treatment Plant Expansion": 400, "Groundwater Wells": 300,
        "Modular Filtration": 200, "Solar-Assisted Treatment": 0,
        "Desalination Pilot": 0,
    },
    "waste_free_campus.json": {
        "Food Scraps Composting": 350, "Anaerobic Digestion": 0,
        "Paper Recycling": 400, "Container Recycling": 250,
        "Textile Reuse": 0,
    },
    "meals_for_a_growing_school.json": {
        "Central Kitchen": 220, "Satellite Kitchen": 0,
        "Electric Food Truck": 200, "Community Kitchen Partnership": 180,
        "Gas Food Truck": 200,
    },
}


class ShippedExamplesTests(unittest.TestCase):
    def test_each_case_can_publish_and_accept_a_student_plan(self):
        self.assertEqual({path.name for path in EXAMPLES.glob("*.json")}, set(SAMPLE_PLANS))
        for filename, allocations in SAMPLE_PLANS.items():
            with self.subTest(filename=filename):
                draft = json.loads((EXAMPLES / filename).read_text())
                project = validate_project(draft)
                ok, problems, costs = publish_checks(project)
                self.assertTrue(ok, problems)
                self.assertLessEqual(costs[0], project.budget)
                self.assertLessEqual(costs[1], project.budget)
                metrics = plan_metrics(project, allocations)
                self.assertEqual(plan_errors(project, metrics, allocations=allocations), [])
                answers = {i: q.correct_option for i, q in enumerate(project.comprehension_questions)}
                self.assertEqual(check_comprehension(project, answers), [])

    def test_prepublished_bundles_are_importable(self):
        bundles = {
            "riverside_water_supply.json": "ecopbl_v2_riverside_scenario_v2.json",
            "waste_free_campus.json": "ecopbl_v2_waste_free_campus_published_v1.json",
            "meals_for_a_growing_school.json": "ecopbl_v2_meals_for_a_growing_school_published_v1.json",
        }
        for draft_file, bundle_file in bundles.items():
            with self.subTest(bundle=bundle_file):
                draft = json.loads((EXAMPLES / draft_file).read_text())
                bundle = json.loads((DELIVERABLES / bundle_file).read_text())
                self.assertEqual(bundle["format"], "ecopbl-phase12-v2")
                self.assertIs(bundle["approved"], True)
                project = validate_project(bundle["project"])
                self.assertEqual(project.scenario_name, draft["scenario_name"])
                self.assertTrue(publish_checks(project)[0])


if __name__ == "__main__":
    unittest.main()
