# EcoPBL Studio · Teacher-authored v2

This English Streamlit app follows the teacher-authored `EcoPBL_Colab_UI_v2.ipynb` structure for Phases 1 and 2.

## What it does

- **Phase 1 — Teacher design and publish:** Choose a ready-to-run example, import a scenario JSON, start a blank design, or use **AI agent · Generate & Publish**. The agent form takes a title, subject, grade, capacity unit, target, budget, candidate option names, and optional context. AI fills the remaining activity fields. Python validates the complete project and solves the initial and post-event allocation problems before publishing it. The full editor remains available for manual changes and republishing.
- **Phase 2 — Student inquiry and planning:** Read the mission and fact cards; pass teacher-authored comprehension questions; ask experts; allocate capacity in the scenario's unit; see the Python-calculated preview for all configured constraints; submit two evidence-based reasons. Download a JSON record tagged with the published scenario version.

Phases 3 and 4 are represented in the workflow but are not implemented in this app. A Phase 2 submission is marked `awaiting_phase3_deterministic_check`.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements_phase12_v2.txt
.venv/bin/streamlit run ecopbl_phase12_v2.py
```

Open the local URL displayed by Streamlit, usually `http://localhost:8501`.

AI generation and expert responses use the Hugging Face Chat Completions API. Enter a token with Inference Providers permission in the teacher agent form, or set `HF_TOKEN` as an environment variable or in `.streamlit/secrets.toml`. A token entered in the form stays in the current Streamlit session while drafting and is cleared after successful publication or switching to Student; it is not saved in SQLite or included in scenario exports. The model can be changed in the form; `HF_MODEL` sets its default and otherwise defaults to `Qwen/Qwen2.5-72B-Instruct`. See the [Hugging Face chat-completion documentation](https://huggingface.co/docs/inference-providers/tasks/chat-completion) for token permissions and available models.

```bash
export HF_TOKEN='your-token'
.venv/bin/streamlit run ecopbl_phase12_v2.py
```

The examples, manual authoring, publish checks, and planning preview work without a token. **Generate & Publish requires a token.** Without one, expert responses are visibly labeled as offline guides.

## AI agent publication

The agent must return the same structured scenario contract as the editor. The teacher's title, subject, grade, capacity unit, target, budget, and option names are pinned in code; the model cannot change them. A failed schema or feasibility check triggers one repair attempt, then stops without publishing if still invalid. A successful project is saved as a new version and becomes the live Student activity. The agent changes scenario data, not Streamlit source code.

Published versions are stored in the local SQLite database `data/ecopbl_v2.sqlite3`; a new browser session on the same server loads the latest version automatically. Set `ECOPBL_STORE_PATH` to use another location. Student plans and expert conversations remain browser-session data until downloaded. This local prototype has no teacher authentication and exposes one shared live activity, so add authentication and role controls before hosting it for a school.

## Ready-to-run examples

In the Teacher view, select a case under **Choose a ready-to-run example**, click **Load selected example**, review the editable fields, and click **Publish to students**. Then click **Open student view**. The three additional cases are also standalone draft JSON files that can be loaded with **Import a scenario JSON into the editor**:

| Case | Capacity target | Budget | Draft file |
| --- | ---: | ---: | --- |
| Riverside Water Supply | 900 m3/day | $3.9 million | `examples/riverside_water_supply.json` |
| Waste-Free Campus | 1,000 kg/day | $1.35 million | `examples/waste_free_campus.json` |
| Meals for a Growing School | 800 meals/day | $2 million | `examples/meals_for_a_growing_school.json` |

The energy and transport cases remain in the selector. Each case has comprehension questions, expert roles, a rubric, an event that raises demand by 10%, and a feasible initial and post-event allocation. All numbers are invented classroom parameters. The water case assumes every option meets drinking-water quality standards; the school-meals case assumes every option meets nutrition and food-safety standards. Those standards are not checked by this numeric model.

For a direct Student-view demonstration, import the prepublished sample bundles in `deliverables/ecopbl_v2_riverside_scenario_v2.json`, `deliverables/ecopbl_v2_waste_free_campus_published_v1.json`, or `deliverables/ecopbl_v2_meals_for_a_growing_school_published_v1.json`. These bundles are classroom demos; a teacher should review the content before using it with students.

## Teacher-to-student handoff

After publishing, students who open the same server see the latest live activity. Teachers can also download the published scenario JSON, and students on another deployment can import that file in the Student view. Download student plans to retain their work. The `approved` field in an imported JSON file is a prototype handoff marker, not authentication.

The data contract retains the notebook's internal energy-oriented field names (`cost_per_megawatt`, `co2_per_megawatt`, `land_per_megawatt`, `required_generation`). The teacher sets the unit shown throughout the interface; the transport example uses `riders/hour`. These are simplified additive-capacity classroom models, and all example figures are illustrative.
