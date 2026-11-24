"""Agent generation and shared publication behavior."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase12_v2_core import generate_publishable_project, transport_draft, validate_project
from phase12_v2_store import latest, publish


def brief() -> dict:
    draft = transport_draft()
    return {
        "scenario_name": "Teacher's Campus Mobility Case",
        "school_subject": "Environmental Science",
        "grade": "Grades 6-8",
        "unit_label": "riders/hour",
        "budget": 2_800_000,
        "required_generation": 600,
        "option_names": [source["name"] for source in draft["sources"]],
        "instructions": "Consider access and emissions.",
    }


class AgentPublicationTests(unittest.TestCase):
    def test_agent_pins_teacher_facts_and_checks_solvability(self):
        response = transport_draft()
        response["scenario_name"] = "A different title"
        response["unit_label"] = "MW"
        response["budget"] = 10
        response["required_generation"] = 10
        response["sources"][0]["name"] = "A different option"
        with patch("phase12_v2_core.call_huggingface", return_value=json.dumps(response)) as model:
            project = generate_publishable_project(brief(), "test-token", "test-model")
        self.assertEqual(model.call_count, 1)
        self.assertEqual(project.scenario_name, "Teacher's Campus Mobility Case")
        self.assertEqual(project.unit_label, "riders/hour")
        self.assertEqual(project.budget, 2_800_000)
        self.assertEqual(project.required_generation, 600)
        self.assertEqual(project.generation_alternatives[0].name, "Safer Walking Routes")

    def test_agent_retries_invalid_output_before_publication(self):
        valid = json.dumps(transport_draft())
        with patch("phase12_v2_core.call_huggingface", side_effect=["{}", valid]) as model:
            project = generate_publishable_project(brief(), "test-token", "test-model")
        self.assertEqual(model.call_count, 2)
        self.assertEqual(project.required_generation, 600)

    def test_no_token_cannot_publish(self):
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            generate_publishable_project(brief(), "", "test-model")

    def test_shared_store_versions_and_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.sqlite3"
            self.assertIsNone(latest(path))
            first = publish(validate_project(transport_draft()), source="Teacher", path=path)
            second = publish(validate_project(transport_draft()), source="AI agent", path=path)
            self.assertEqual((first["version"], second["version"]), ("v1", "v2"))
            self.assertEqual(latest(path)["id"], second["id"])
            self.assertEqual(latest(path)["source"], "AI agent")

    def test_teacher_agent_form_publishes_to_a_new_browser_session(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.sqlite3"
            app_file = str(Path(__file__).resolve().parents[1] / "ecopbl_phase12_v2.py")
            with patch.dict(os.environ, {"ECOPBL_STORE_PATH": str(path)}, clear=False), \
                    patch("phase12_v2_core.call_huggingface",
                          return_value=json.dumps(transport_draft())) as model:
                app = AppTest.from_file(app_file, default_timeout=20).run()
                app.text_input(key="agent_title").set_value("Campus Mobility From AI")
                app.text_input(key="agent_unit").set_value("riders/hour")
                app.number_input(key="agent_target").set_value(600.0)
                app.number_input(key="agent_budget").set_value(2_800_000.0)
                app.text_area(key="agent_options").set_value(
                    "\n".join(source["name"] for source in transport_draft()["sources"])
                )
                app.text_input(key="agent_token_0").set_value("test-token")
                next(button for button in app.button if button.label == "Generate & Publish").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(model.call_count, 1)
                self.assertEqual(app.session_state["published"]["project"].scenario_name,
                                 "Campus Mobility From AI")
                self.assertEqual(app.session_state["published"]["version"], "v1")
                self.assertNotIn("agent_token_0", app.session_state)
                fresh = AppTest.from_file(app_file, default_timeout=20).run()
                self.assertEqual(fresh.session_state["published"]["project"].scenario_name,
                                 "Campus Mobility From AI")


if __name__ == "__main__":
    unittest.main()
