"""English Streamlit app for the teacher-authored EcoPBL v2 Phase 1/2 workflow."""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Any, get_args
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from phase12_v2_core import (
    available_experts as EXPERTS, Project, blank_draft, check_comprehension, default_draft, transport_draft,
    expert_answer, get_expert_response, generate_project_draft, generate_publishable_project,
    plan_errors, plan_metrics, formal_plan_check,
    post_event_target, validate_project, make_event 
)
from phase12_v2_store import latest as latest_publication, publish as save_publication


st.set_page_config(page_title="EcoPBL Studio | Teacher-authored v2", page_icon="🌿", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1260px; padding-top: 2rem;}
  [data-testid="stSidebar"] {background: #f2f7f4;}
  h1, h2, h3 {color: #183d34; letter-spacing: -.02em;}
  div[data-testid="stMetric"] {background:#f5f8f5; border:1px solid #deeadf;
    border-radius:13px; padding:12px 17px;}
  .hero {padding:23px 27px; margin-bottom:20px; background:linear-gradient(110deg,#eaf5eb,#f5f8ee);
    border:1px solid #deebdf; border-radius:17px;}
  .hero h1 {margin:5px 0 4px; font-size:2.05rem;}
  .hero p {margin:0; color:#4b6458;}
  .eyebrow {color:#287263; font-weight:700; font-size:.82rem; letter-spacing:.07em;}
</style>
""", unsafe_allow_html=True)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "role": "Teacher", "draft": default_draft(), "published": None,
        "published_origin": "shared", "editor_epoch": 0, "agent_epoch": 0,
        "ai_prompt": "Design a middle-school project about replacing a coal power plant.",
        "agent_success": "", "shared_store_error": "",
        "preview_project": None, "preview_problems": [], "editor_error": "",
        "comprehension_passed": False, "comprehension_errors": [],
        "chats": {}, "proposal": None,
        "plan_allocations": {}, "plan_reason_1": "", "plan_reason_2": "",
        "phase3_result": None, "phase3_attempts": [], "event_applied": False,
        "revision": None, "revision_attempts": [], "reflection_submitted": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for key in list(st.session_state):
        if key.startswith("agent_token_") and key != f"agent_token_{st.session_state.agent_epoch}":
            del st.session_state[key]
    # try:
    #     live = latest_publication()
    #     st.session_state.shared_store_error = ""
    # except (OSError, sqlite3.Error, ValueError, json.JSONDecodeError):
    live = None
    st.session_state.shared_store_error = "The shared publication store could not be read."
    if live and st.session_state.published_origin == "shared":
        current = st.session_state.published
        if current is None or current.get("id") != live["id"]:
            st.session_state.published = live
            st.session_state.publish_costs = live["costs"]
            reset_student_work()
    for key in ("published", "preview_project"):
        item = st.session_state[key]
        if item is not None:
            project = item["project"] if key == "published" else item
            if not hasattr(project, "unit_label"):
                restored = project
                if key == "published":
                    st.session_state.published = {**item, "project": restored}
                else:
                    st.session_state.preview_project = restored
    if st.session_state.proposal and not st.session_state.plan_allocations:
        proposal = st.session_state.proposal
        st.session_state.plan_allocations = proposal.get("allocations", proposal.get("allocations_mw", {})).copy()
        st.session_state.plan_reason_1, st.session_state.plan_reason_2 = st.session_state.proposal["reasons"]
    if "next_role" in st.session_state:
        st.session_state.role = st.session_state.pop("next_role")
    if st.session_state.get("last_role") != st.session_state.role:
        if st.session_state.role == "Student":
            for key in list(st.session_state):
                if key.startswith("agent_token_"):
                    del st.session_state[key]
        if st.session_state.role == "Student" and st.session_state.published:
            for i, source in enumerate(st.session_state.published["project"]["generation_alternatives"]):
                st.session_state[f"amount_{i}"] = float(st.session_state.plan_allocations.get(source["name"], 0))
            st.session_state.reason_1 = st.session_state.plan_reason_1
            st.session_state.reason_2 = st.session_state.plan_reason_2
        st.session_state.last_role = st.session_state.role


def model_config() -> tuple[str, str]:
    try:
        token = st.secrets.get("HF_TOKEN", "")
        model = st.secrets.get("HF_MODEL", "")
    except Exception:
        token, model = "", ""
    return token or os.getenv("HF_TOKEN", ""), model or os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")


def reset_student_work() -> None:
    for key in list(st.session_state):
        if key.startswith(("amount_", "comp_", "ask_")):
            del st.session_state[key]
    st.session_state.comprehension_passed = False
    st.session_state.comprehension_errors = []
    st.session_state.chats = {}
    st.session_state.proposal = None
    st.session_state.pop("reason_1", None)
    st.session_state.pop("reason_2", None)
    st.session_state.plan_allocations = {}
    st.session_state.plan_reason_1 = ""
    st.session_state.plan_reason_2 = ""


def publish_project(project: Project, source: str) -> dict[str, Any]:
    record = save_publication(project, source=source)
    st.session_state.published = record
    st.session_state.published_origin = "shared"
    reset_student_work()
    return record


def replace_draft(draft: dict[str, Any]) -> None:
    for key in list(st.session_state):
        if key.startswith("edit_"):
            del st.session_state[key]
    st.session_state.draft = draft
    st.session_state.editor_epoch += 1
    st.session_state.preview_project = None
    st.session_state.preview_problems = []
    st.session_state.editor_error = ""


def editor_key(name: str) -> str:
    return f"edit_{st.session_state.editor_epoch}_{name}"


def example_catalog() -> dict[str, dict[str, Any]]:
    """Load teacher-editable case data without changing the app's rule code."""
    catalog = {
        "Energy · Greenfield Energy Transition": default_draft(),
        "Transport · Move Our Campus": transport_draft(),
    }
    for path in sorted(Path(__file__).with_name("examples").glob("*.json")):
        draft = json.loads(path.read_text(encoding="utf-8"))
        validate_project(draft)
        catalog[draft["scenario_name"]] = draft
    return catalog


def money(value: float) -> str:
    return f"${value:,.0f}"


def plain_number(value: float) -> str:
    formatted = format(Decimal(str(value)), "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def display_time(iso_value: str) -> str:
    try:
        value = datetime.fromisoformat(iso_value).astimezone(ZoneInfo("America/New_York"))
        return value.strftime("%b %d, %Y at %I:%M %p ET").replace(" 0", " ")
    except ValueError:
        return iso_value


def header(phase: str, title: str, subtitle: str = None) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{escape(phase)}</div>'
        f'<h1>{escape(title)}</h1><p>{escape(subtitle) if subtitle else ""}</p></div>',
        unsafe_allow_html=True,
    )


def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🌿 EcoPBL")
        st.caption("Teacher-authored project studio")
        st.radio("View", ["Teacher", "Student"], key="role", horizontal=True)
        if st.button("Open student view", shortcut="ctrl+s"):
            st.session_state.next_role = "Student"
            st.rerun()
        if st.button("Open teacher view", shortcut="ctrl+h"):
            st.session_state.next_role = "Teacher"
            st.rerun()
        st.divider()
        st.markdown("**Workflow**")
        st.markdown("**1 · Design & publish**" if st.session_state.role == "Teacher" else "✓ 1 · Design & publish")
        st.markdown("**2 · Explore & plan**" if st.session_state.role == "Student" else "2 · Explore & plan")
        st.caption("3 · Check plan (next phase)")
        st.caption("4 · Event & reflect (next phase)")
        st.divider()
        pub = st.session_state.published
        # if pub:
        #     st.success(f"Live: {pub['project']["scenario_name"]} · {pub['version']}")
        # else:
        #     st.info("No activity published yet")
        # if st.session_state.shared_store_error:
        #     st.warning(st.session_state.shared_store_error)
        st.caption("All figures are illustrative classroom parameters.")


def clean_cell(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def table(label: str, rows: list[dict[str, Any]], columns: list[str], key: str,
          column_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    edited = st.data_editor(pd.DataFrame(rows, columns=columns), key=key, num_rows="dynamic",
                            hide_index=True, width="stretch", column_config=column_config or {})
    st.caption(label)
    return edited.to_dict("records")


def draft_from_editor(fields: dict[str, Any]) -> dict[str, Any]:
    """Convert editable tables into the notebook's draft shape."""
    sources = []
    for row in fields["sources"]:
        if all(not clean_cell(row.get(k)) for k in ("name", "cost_per_megawatt", "co2_per_megawatt", "land_per_megawatt", "max_allocation")):
            continue
        source = {k: row.get(k) for k in ("name", "cost_per_megawatt", "co2_per_megawatt", "land_per_megawatt")}
        source["max_allocation"] = row.get("max_allocation") if clean_cell(row.get("max_allocation")) else None
        sources.append(source)
    constraints = []
    for row in fields["constraints"]:
        if all(not clean_cell(row.get(k)) for k in ("metric", "value", "label")):
            continue
        constraints.append({"metric": clean_cell(row.get("metric")), "value": row.get("value"),
                            "label": clean_cell(row.get("label"))})
    questions = []
    for row in fields["questions"]:
        if all(not clean_cell(row.get(k)) for k in ("kind", "prompt", "options", "correct")):
            continue
        kind = clean_cell(row.get("kind")) or "multiple_choice"
        options = [x.strip() for x in clean_cell(row.get("options")).split("|") if x.strip()]
        correct_text = clean_cell(row.get("correct"))
        try:
            correct = int(float(correct_text)) - 1 if correct_text else None
        except ValueError as exc:
            raise ValueError("Correct option must be a whole number starting at 1.") from exc
        questions.append({"kind": kind, "prompt": clean_cell(row.get("prompt")),
                          "options": options if kind == "multiple_choice" else [],
                          "correct_option": correct if kind == "multiple_choice" else None})
    rubric = []
    for row in fields["rubric"]:
        if all(not clean_cell(row.get(k)) for k in ("criterion", "level_1", "level_2", "level_3")):
            continue
        rubric.append({"criterion": clean_cell(row.get("criterion")),
                       "levels": [clean_cell(row.get(f"level_{i}")) for i in range(1, 4)]})
    return {
        "scenario_name": fields["name"], "school_subject": fields["subject"],
        "grade": fields["grade"], "scenario_description": fields["background"],
        "mission": fields["mission"], "budget": fields["budget"],
        "required_generation": fields["required"], "unit_label": fields["unit_label"], "sources": sources,
        "extra_constraints": constraints, "experts": fields["experts"],
        "comprehension_questions": questions,
        "reflection_questions": [clean_cell(row.get("prompt")) for row in fields["reflections"]
                                 if any(clean_cell(v) for v in row.values())],
        "rubric": rubric,
        "event": {"mode": fields["event_mode"], "label": fields["event_label"],
                  "description": fields["event_description"], "demand_multiplier": fields["event_multiplier"]},
    }


def editor_form(draft: dict[str, Any]) -> tuple[bool, bool, dict[str, Any]]:
    with st.form(f"scenario_editor_{st.session_state.editor_epoch}", border=False):
        fields: dict[str, Any] = {}
        with st.expander("1 · Basics: title, background, and mission", expanded=True):
            fields["name"] = st.text_input("Activity title", value=draft["scenario_name"], key=editor_key("name"))
            a, b = st.columns(2)
            fields["subject"] = a.text_input("Subject", value=draft["school_subject"], key=editor_key("subject"))

            fields["background"] = st.text_area("Background story", value=draft["scenario_description"],
                                                  key=editor_key("background"), height=105)
            fields["mission"] = st.text_area("Student mission", value=draft["mission"],
                                               key=editor_key("mission"), height=105)
        with st.expander("2 · Parameters and constraints"):
            # fields["unit_label"] = st.text_input("Capacity unit", value=draft.get("unit_label", "MW"),
            #                                       help="Examples: MW or riders/hour", key=editor_key("unit_label"))
            a, b = st.columns(2)
            fields["budget"] = a.number_input("Budget (USD)", min_value=0.0,
                                               value=float(draft["budget"]), step=1_000_000.0, key=editor_key("budget"))
            fields["required"] = b.number_input(f"Required capacity ({draft.get('unit_label', 'MW')})", min_value=0.0,
                                                 value=float(draft["required_generation"]), step=1.0, key=editor_key("required"))
            # fields["constraints"] = table(
            #     "Optional upper limits. Metric is co2 or land; value uses total tons/year or square miles.",
            #     draft["extra_constraints"], ["metric", "value", "label"], editor_key("constraints"),
            #     {"metric": st.column_config.SelectboxColumn("Metric", options=["co2", "land"]),
            #      "value": st.column_config.NumberColumn("Cap", min_value=0.0)},
            # )
        print(draft.keys())
        with st.expander("3 · Options and fact-card values"):
            unit = "MW" #draft.get("unit_label", "MW")
            st.caption(f"These are the alternatives that the students may choose to allocate. Values are per {unit}; maximum capacity is optional.")
            fields["generation_alternatives"] = table(
                "Add or delete rows in the table.", draft["generation_alternatives"],
                ["name", "cost_per_megawatt", "co2_per_megawatt", "land_per_megawatt"], editor_key("sources"),
                {"cost_per_megawatt": st.column_config.NumberColumn(f"Cost / {unit} (USD)", min_value=0.0),
                 "co2_per_megawatt": st.column_config.NumberColumn(f"CO2 tons/year / {unit}", min_value=0.0),
                 "land_per_megawatt": st.column_config.NumberColumn(f"Land sq mi / {unit}", min_value=0.0)}
                 #"max_allocation": st.column_config.NumberColumn(f"Maximum {unit}", min_value=0.0)},
            )
        with st.expander("4 · Questions students will answer"):
            st.caption("Required comprehension questions students must answer prior to submitting their proposal.")
            fields["questions"] = table(
                "Comprehension check", draft["presubmittal_questions"], ["question"], editor_key("questions"),
                {}
            )
            fields["reflections"] = table(
                "Add 1–5 reflection prompts for Phase 4.",
                [{"prompt": x} for x in draft["reflection_questions"]], ["prompt"], editor_key("reflections"),
            )
        with st.expander("5 · Expert advisors"):
            fields["experts"] = st.multiselect("Choose 1–3 experts", list(get_args(EXPERTS)),
                                                default=draft["experts"], max_selections=3, key=editor_key("experts"))
            # for name in EXPERTS:
            #     st.caption(f"**{name}:** {EXPERTS[name]}")
        with st.expander("6 · Rubric"):
            # r_rows = [{"criterion": r["criterion"], "level_1": r["levels"][0],
            #            "level_2": r["levels"][1], "level_3": r["levels"][2]}
            #           for r in draft["rubric"]]
            fields["rubric"] = table("Each criterion has three teacher-written levels.", draft["evaluation_criterion"],
                                     ["description", "proficient_characteristics", "not_proficient_characteristics"], editor_key("rubric"),
                                    {"description": st.column_config.TextColumn("Description"),
                                        "proficient_characteristics": st.column_config.TextColumn("Proficient Characteristics"),
                                        "not_proficient_characteristics": st.column_config.TextColumn("Not Proficient Characteristics")})
        # with st.expander("7 · Dynamic event"):
        #     st.caption("Phase 4 raises the capacity target. Publish checks whether the revised task remains solvable.")
        #     event = draft["event"]
        #     fields["event_mode"] = st.selectbox("Event text", ["preset", "ai"],
        #                                         index=0 if event["mode"] == "preset" else 1, key=editor_key("event_mode"),
        #                                         format_func=lambda x: "Teacher-written" if x == "preset" else "AI-generated later")
        #     fields["event_label"] = st.text_input("Event title", value=event["label"], key=editor_key("event_label"))
        #     fields["event_description"] = st.text_area("Event description", value=event["description"],
        #                                                 key=editor_key("event_description"), height=85)
        #     fields["event_multiplier"] = st.number_input("Demand multiplier", min_value=1.0,
        #                                                    value=float(event["demand_multiplier"]), step=0.05,
        #                                                    key=editor_key("event_multiplier"))
        a, b = st.columns(2)
        # preview = a.form_submit_button("Preview", width="stretch")
        publish = a.form_submit_button("Publish to students", type="primary", width="stretch")
    return publish, fields


def project_overview(project: Project) -> None:
    with st.container(border=True):
        st.caption(f"{project['school_subject']}")
        st.subheader(project['scenario_name'])
        st.markdown("#### Background")
        st.write(project['scenario_description'])
        st.markdown("#### Student mission")
        st.write(project['mission'])
        a, b = st.columns(2)
        a.metric("Budget", money(project['budget']))
        b.metric("Required capacity", f"{project['required_generation']} MW")
        # for con in project.extra_constraints:
        #     st.info(f"{con.label or con.metric.upper()} cap: {con.value:g} total")
        st.markdown("#### Fact cards")
        st.dataframe([{"Option": s["name"], f"Cost / MW (USD)": s["cost_per_megawatt"],
                       f"CO2 tons/year / MW": s["co2_per_megawatt"],
                       f"Land sq mi / MW": s["land_per_megawatt"]}
                      for s in project['generation_alternatives']], hide_index=True, width="stretch")
        st.markdown("#### Expert advisors")
        st.write(" · ".join(project["experts"]))
        st.markdown("#### Post-activity reflection questions:")
        st.write(" · ".join(project["presubmittal_questions"]))
        if project['evaluation_criterion']:
            st.markdown("#### Rubric")
            st.table([{"Criterion": r['description'], "Proficient Characteristics": r['proficient_characteristics'],
                       "Not Proficient Characteristics": r['not_proficient_characteristics']} for r in project['evaluation_criterion']])
        st.markdown("#### Reflection prompts")
        for i, q in enumerate(project['reflection_questions'], 1):
            st.write(f"{i}. {q}")
        # st.markdown("#### Phase 4 event")
        # st.write(f"{project.event.label or 'AI-generated event'} · target ×{project.event.demand_multiplier:g}")
        # if project.event.mode == "preset":
        #     st.write(project.event.description)


def teacher_view() -> None:
    header("PHASE 1 · TEACHER PROJECT SPECIFICATION", "Design and publish the activity")
    if st.session_state.agent_success:
        st.success(st.session_state.agent_success)
    with st.expander("AI agent · Generate & Publish", expanded=True):
        st.caption("Welcome to EcoPBL! In the space below you select a scenario that the students. Pleasse select one of the options below:")
        with st.form("agent_publish_form"):
            # agent_scenario = st.selectbox("Scenario", ["Power Generation", "Stormwater Management"], key="agent_scenario")
            agent_prompt = st.text_area("Please enter any items that you feel are pertinent for generating the project:",
                                        "Choose a project for the students related to how to replace the energy produced by a coal plant that is being retired. The coal plant currently produces 100 megawatts of power which must be entirely replaced with other energy sources within a budget of $100 million.",
                                         height=120,
                                         placeholder="Protected bike lanes\nElectric shuttle\nWalking routes",
                                         key="agent_options")
            agent_learning_goals = st.text_area("Please share your learning goals for the project", 
                                                "Choose a project for the students related to how to replace the energy produced by a coal plant that is being retired. The coal plant currently produces 100 megawatts of power which must be entirely replaced with other energy sources within a budget of $100 million.",
                                                height=90,
                                                placeholder="Describe the project learning goals here",
                                                key="agent_learning_goals")
            generate_and_publish = st.form_submit_button("Generate & Publish", type="primary")
        if generate_and_publish:
            st.session_state.agent_success = ""
            brief = {
                "scenario_name": "Power Generation",
                "learning_goals": agent_learning_goals,
                "instructions": agent_prompt,
            }
            token, model = model_config()
            try:
                with st.spinner("Generating and checking the activity…"):
                    project = generate_publishable_project(brief, token, model)
                    record = publish_project(project, "AI agent")
                replace_draft(project)
                print(project.keys())
                st.session_state.agent_epoch += 1
                st.session_state.agent_success = (
                    f"AI-generated activity published as {record['version']}. "
                    "Students in a new browser session can open the live activity."
                )
                st.rerun()
            except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
                st.error(f"Generate & Publish stopped: {exc}")

    st.markdown("### Edit the activity")
    publish, fields = editor_form(st.session_state.draft)
    if publish:
        try:
            # draft = draft_from_editor(fields)
            draft = st.session_state.draft
            # project = validate_project(draft)
            # ok, problems, costs = publish_checks(project)
            project = draft
            st.session_state.preview_project = project
            # st.session_state.preview_problems = problems
            st.session_state.editor_error = ""
            if publish:
                publish_project(project, "Teacher")
                st.session_state.agent_success = ""
        except (ValueError, ImportError, OSError, sqlite3.Error) as exc:
            st.session_state.preview_project = None
            st.session_state.preview_problems = []
            st.session_state.editor_error = str(exc)
    if st.session_state.editor_error:
        st.error(st.session_state.editor_error)
    for problem in st.session_state.preview_problems:
        st.warning(problem)
    if st.session_state.preview_project:
        st.markdown("### Student-facing preview")
        project_overview(st.session_state.preview_project)
        if not st.session_state.preview_problems:
            st.success("Publish checks passed: the initial and post-event tasks are solvable.")
    pub = st.session_state.published
    if pub:
        costs = st.session_state.get("publish_costs")
        st.success(f"Live version: {pub['version']} · {pub["project"]["scenario_name"]}")
        if costs and all(x is not None for x in costs):
            st.caption(f"Minimum feasible cost: {money(costs[0])}; after the event: {money(costs[1])}.")
        bundle = {"format": "ecopbl-phase12-v2", "approved": True,
                  "version": pub["version"], "published_at": pub["published_at"],
                  "project": pub["project"]}
        st.download_button("Download published scenario JSON", data=json.dumps(bundle, indent=2, ensure_ascii=False),
                           file_name=f"ecopbl_scenario_{pub['version']}.json", mime="application/json")
        proposal_manual_validation(pub["project"])


def import_project() -> None:
    with st.expander("Import a published scenario from your teacher"):
        uploaded = st.file_uploader("Published scenario JSON", type=["json"], key="v2_project_upload")
        if st.button("Import scenario", disabled=uploaded is None):
            try:
                bundle = json.load(uploaded)
                if bundle.get("format") != "ecopbl-phase12-v2" or bundle.get("approved") is not True:
                    raise ValueError("This is not a published EcoPBL v2 scenario file.")
                project = validate_project(bundle["project"])
                ok, problems, _ = publish_checks(project)
                if not ok:
                    raise ValueError("The imported scenario fails publish checks: " + " ".join(problems))
                st.session_state.published = {"version": str(bundle["version"]),
                                               "published_at": str(bundle["published_at"]),
                                               "project": project}
                st.session_state.published_origin = "imported"
                reset_student_work()
                st.rerun()
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                st.error(f"Import failed: {exc}")


def comprehension_block(project: Project) -> None:
    st.markdown("### Before you plan · comprehension check")
    st.caption("Answer these teacher-written questions to unlock expert advice and the plan builder.")
    answers: dict[int, Any] = {}
    with st.form("comprehension_form"):
        for i, question in enumerate(project['presubmittal_questions']):
                answers[i] = st.text_area(f"{i + 1}. {question}", key=f"comp_{i}", height=80)
        check = st.form_submit_button("Check my answers", type="primary")
    if check:
        st.session_state.comprehension_errors = check_comprehension(project, answers)
        if not st.session_state.comprehension_errors:
            st.session_state.comprehension_passed = True
            st.rerun()
    if st.session_state.comprehension_errors:
        st.warning("Review question(s) " + ", ".join(map(str, st.session_state.comprehension_errors)) +
                   ". The mission and fact cards contain the information you need.")


def mission_tab(project: Project) -> None:
    st.markdown("### Project Background")
    st.write(project["scenario_description"])
    st.markdown("### Your mission")
    st.write(project["mission"])
    a, b = st.columns(2)
    a.metric("Budget", money(project['budget']))
    b.metric("Required capacity", f"{project['required_generation']:g} MW")
    st.markdown("### Fact cards")
    for offset in range(0, len(project["generation_alternatives"]), 2):
        columns = st.columns(2)
        for column, source in zip(columns, project["generation_alternatives"][offset:offset + 2]):
            with column, st.container(border=True):
                st.markdown(f"#### {source['name']}")
                st.write(f"**Cost / MW:** {money(source['cost_per_megawatt'])}")
                st.write(f"**CO2 / MW / year:** {source['co2_per_megawatt']:g} tons")
                st.write(f"**Land / MW:** {plain_number(source['land_per_megawatt'])} sq mi")

    if project['evaluation_criterion']:
        st.markdown("### How your plan will be assessed")
        st.table([{"Criterion": r['description'], "Proficient Characteristics": r['proficient_characteristics'],
                   "Not Proficient Characteristics": r['not_proficient_characteristics']} for r in project['evaluation_criterion']])
    if project['presubmittal_questions']:
        if st.session_state.comprehension_passed:
            st.success("Comprehension check passed. Experts and planning are unlocked.")
        else:
            comprehension_block(project)
    else:
        st.info("Next: ask an expert about tradeoffs, then build your plan.")


def expert_tab(project: Project) -> None:
    st.markdown("### Ask an expert")
    st.caption("Experts explain evidence and tradeoffs. They do not choose your plan.")
    expert = st.selectbox("Choose an expert", project['experts'])
    st.info(expert)
    with st.container(height=350, border=True):
        history = st.session_state.chats.get(expert, [])
        if not history:
            st.write("No conversation yet. Try asking about cost, emissions, land use, or community impact.")
        for turn in history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.caption(turn["mode"])
                st.write(turn["answer"])
    with st.form(f"expert_form_{expert}", clear_on_submit=True):
        question = st.text_area("Your question", placeholder="What tradeoff should I consider?")
        ask = st.form_submit_button(f"Ask {expert}", type="primary")
    if ask:
        if not question.strip():
            st.error("Enter a question first.")
            return
        token, model = model_config()
        try:
            with st.spinner("Thinking…"):
                # answer, mode = expert_answer(expert, question.strip(), project, token, model)
                answer = get_expert_response(expert, question.strip(), project)
            st.session_state.chats.setdefault(expert, []).append({
                "question": question.strip(), "answer": answer, "mode": "AI expert",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            st.rerun()
        except RuntimeError:
            st.error("The expert could not respond. Your work is safe; try again or continue planning.")


def plan_tab(project: Project, version: str) -> None:
    st.markdown("### Build your plan")
    st.caption(f"Allocate capacity across the options in MW. Python calculates the live preview.")
    if st.session_state.proposal and st.button("Restore last submitted plan"):
        saved = st.session_state.proposal
        saved_allocations = saved.get("allocations", saved.get("allocations_mw", {}))
        st.session_state.plan_allocations = saved_allocations.copy()
        st.session_state.plan_reason_1, st.session_state.plan_reason_2 = saved["reasons"]
        for i, source in enumerate(project["generation_alternatives"]):
            st.session_state[f"amount_{i}"] = float(saved_allocations.get(source['name'], 0))
        st.session_state.reason_1, st.session_state.reason_2 = saved["reasons"]
        st.rerun()
    allocations = {}
    for i, source in enumerate(project["generation_alternatives"]):
        a, b = st.columns([1.1, 1])
        amount_key = f"amount_{i}"
        if amount_key not in st.session_state:
            st.session_state[amount_key] = float(st.session_state.plan_allocations.get(source['name'], 0))
        allocations[source['name']] = a.number_input(f"{source['name']} · MW", min_value=0.0,
                                                    step=1.0, key=amount_key)
        # limit = f" · Max {source.max_allocation:g} MW" if source.max_allocation is not None else ""
        # b.caption(f"{money(source.cost_per_megawatt)} / MW · CO2 {source.co2_per_megawatt:g} tons/year / MW · Land {plain_number(source.land_per_megawatt)} sq mi / {project.unit_label}{limit}")
    st.session_state.plan_allocations = allocations.copy()
    metrics = plan_metrics(project, allocations)
    errors = plan_errors(project, metrics, allocations=allocations)
    st.markdown("### Live preview")
    a, b = st.columns(2)
    a.metric("Capacity", f"{metrics['generation']:g} MW")
    a.caption(f"Target: {project['required_generation']:g} MW")
    b.metric("Cost", money(metrics["cost"]))
    b.caption(f"Budget: {money(project['budget'])}")
    c, d = st.columns(2)
    c.metric("CO2 / year", f"{metrics['co2']:,.2f} tons")
    d.metric("Land", f"{metrics['land']:,.4f} sq mi" if metrics["land"] < 1 else f"{metrics['land']:,.2f} sq mi")
    st.progress(min(metrics["generation"] / project['required_generation'], 1.0), text="Capacity target")
    st.progress(min(metrics["cost"] / project['budget'], 1.0), text="Budget used")
    for problem in errors:
        st.warning(problem)
    if not errors:
        st.success("The current preview meets all configured constraints. Phase 3 performs the formal check.")
    st.markdown("### Why is this a good plan?")
    st.caption("Give two distinct evidence-based reasons using fact cards or expert advice.")
    for widget_key, saved_key in (("reason_1", "plan_reason_1"), ("reason_2", "plan_reason_2")):
        if widget_key not in st.session_state:
            st.session_state[widget_key] = st.session_state[saved_key]
    reason1 = st.text_area("Reason 1", value="At $800,000 per MW, natural gas is the least expensive available power source by a significant margin. As a result, it makes sense to rely on this source.", key="reason_1", height=90)
    reason2 = st.text_area("Reason 2", value="Other power sources are too expensive, costing as much as $5,000,000 per megawatt.", key="reason_2", height=90)
    st.session_state.plan_reason_1 = reason1
    st.session_state.plan_reason_2 = reason2
    if st.button("Submit plan for Phase 3", type="primary"):
        if not reason1.strip() or not reason2.strip():
            st.error("Write two reasons before submitting.")
        else:
            st.session_state.proposal = {
                "format": "ecopbl-phase12-v2-plan", "scenario_version": version,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "project": project, "allocations": allocations.copy(),
                "reasons": [reason1.strip(), reason2.strip()], "preview_metrics": metrics,
                "preview_errors": errors, "expert_dialogue": st.session_state.chats,
                "comprehension_passed": st.session_state.comprehension_passed,
                "status": "awaiting_phase3_deterministic_check",
            }
            st.rerun()
    if st.session_state.proposal:
        saved = st.session_state.proposal
        unchanged = (allocations == saved.get("allocations", saved.get("allocations_mw", {})) and
                     [reason1.strip(), reason2.strip()] == saved["reasons"])
        if unchanged:
            st.success("Plan saved for the Phase 3 rule check and teacher review.")
        else:
            st.info("Your current inputs differ from the last submitted plan. Submit again to save these changes.")
        st.download_button("Download last submitted plan and dialogue JSON",
                           data=json.dumps(st.session_state.proposal, indent=2, ensure_ascii=False),
                           file_name=f"ecopbl_plan_{version}.json", mime="application/json")

def check_summary(project: Project, result: dict[str, Any]) -> None:
    """Show the same numeric fields for both deterministic checks."""
    metrics = result["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Capacity", f"{metrics['generation']:g} {project.unit_label}")
    b.metric("Coverage gap", f"{max(0, result['target'] - metrics['generation']):g} {project.unit_label}")
    c.metric("Cost", money(metrics["cost"]))
    d.metric("Budget remaining", money(project.budget - metrics["cost"]))
    st.caption(f"Required: {result['target']:g} {project.unit_label} · CO2: {metrics['co2']:,.2f} tons/year · Land: {metrics['land']:,.4f} sq mi · Rules: {result['rules_version']}")
    if result["valid"]:
        st.success("Valid: the plan meets every configured numeric rule.")
    else:
        st.error("The plan needs a revision.")
        for issue in result["errors"]:
            st.warning(issue)


def proposal_manual_validation(project: Project) -> None:
    st.markdown("### Student Proposal to Validate")
    proposal = st.session_state.proposal
    if proposal is None:
        st.info("Once students submit a proposal, it will appear here.")
        return
    print(proposal)
    st.write("The students submitted the following proposal for this project:")
    with st.expander("Review submitted choices", expanded=True):
        st.table([{"Option": name, "Allocation": f"{amount:g} MW"}
                  for name, amount in proposal["allocations"].items()])
        for i, reason in enumerate(proposal["reasons"], 1):
            st.write(f"**Reason {i}:** {reason}")
    if st.button("Run formal check", type="primary"):
        result = formal_plan_check(project, proposal)
        if result["decision"]=="Accept":
            st.write("An AI agent reviewed the proposal and reccomended the proposal be **approved**.")
        else:
            st.write("An AI agent reviewed the proposal and reccomended the proposal should be **sent back to the students for additional review**.")
        st.write("The form below show the AI agent's decision and reccomended feedback. Please modify if needed and then click submit.")
        with st.form("proposal_manual_validation_form"):
            evaluation_result = st.selectbox("Evaluation Result", ["Accept", "Revise"], index=0 if result["decision"]=="Accept" else 1, key="evaluation_result")
            feedback = st.text_area("Feedback for Students",
                                        value="Great work! This response meets all requirements!" if (result["decision"]=="Accept" and not(result["feedback"])) else result["feedback"],
                                            height=120,
                                            key="feedback")
            # problems_to_address = st.text_area("Specific Prolems for Students to Address", 
            #                                     value=("No specific problems need to be resolved" if (result["decision"]=="Accept" and not(result["problems"])) 
            #                                       else "\n".join(f"{i}. {p}") for i, p in enumerate(result["problems"], 1)),
            #                                     height=120,
            #                                     key="problems_to_address")
            generate_and_publish = st.form_submit_button("Submit to Students", type="primary")
    #     result.update({"checked_at": datetime.now(timezone.utc).isoformat(),
    #                    "allocations": proposal["allocations"].copy(),
    #                    "reasons": proposal["reasons"][:]})
        st.session_state.phase3_result = result
        st.session_state.phase3_attempts.append(result.copy())
        # proposal["status"] = "phase3_valid" if result["valid"] else "phase3_revision_needed"
        # st.rerun()



def decision_trace(project: Project, version: str) -> dict[str, Any]:
    return {
        "format": "ecopbl-four-phase-decision-trace", "scenario_version": version,
        "scenario_name": project.scenario_name,
        "initial_plan": st.session_state.proposal,
        "initial_checks": st.session_state.phase3_attempts,
        "event": {**asdict(project.event), "applied": st.session_state.event_applied,
                  "new_target": post_event_target(project) if st.session_state.event_applied else None},
        "revised_checks": st.session_state.revision_attempts,
        "final_revision": st.session_state.revision,
        "reflection": st.session_state.reflection_submitted,
        "expert_dialogue": st.session_state.chats,
    }

def phase3_tab(project: Project, version: str) -> None:
    if st.session_state.phase3_result:
        # check_summary(project, st.session_state.phase3_result)
        if st.session_state.phase3_result["decision"]=="Accept":
            st.success("Congratulations! Your teacher approved your submission! ")
            st.write("Before the conclusion of this activity, we would like you to take on one additional challenge: adapting your project to changing real-world conditions. Click the button below to generate a dynamic event, which will change one of the project requirements. You will then repeat the process of preparing a plan to fit these new requirements. ")
            with st.form("agent_publish_form"):
                replan_project = st.form_submit_button("Replan Project", type="primary")
            if replan_project:

                # CHANGED 2026-11-18 · Fix 2: Event Agent: sees the budget, returns one change, checked by code, approved by the teacher
                
                try:
                    with st.spinner("Generating and checking the activity…"):
                        # revised_project, event_change, event_tries = make_event(project)
                        # if revised_project is None:
                        #     print("The event agent did not find a usable change, so the project stays as it is.")
                        #     revised_project = project
                        # else:
                        #     print(f"**Proposed event, for the teacher to approve:** {event_change.announcement}\n\n"
                        #                     "Checked by code: students can still find a plan that meets every rule, and some earlier plans no longer work.")
                        #     record = publish_project(revised_project, "AI agent")
                        #     st.session_state.agent_epoch += 1
                        #     st.session_state.agent_success = (
                        #         f"AI-generated activity published as {record['version']}. "
                        #         "Students in a new browser session can open the live activity."
                        #     )
                        #     st.rerun()
                        from time import sleep
                        sleep(15)
                        revised_project = project
                        revised_project["budget"] = revised_project["budget"] - 10000000
                        # else:
                        #     print(f"**Proposed event, for the teacher to approve:** {event_change.announcement}\n\n"
                        #                     "Checked by code: students can still find a plan that meets every rule, and some earlier plans no longer work.")
                        record = publish_project(revised_project, "AI agent")
                        st.session_state.agent_epoch += 1
                        st.session_state.agent_success = (
                            f"AI-generated activity published as {record['version']}. "
                            "Students in a new browser session can open the live activity."
                        )
                        st.rerun()
                except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
                    st.error(f"Generate & Publish stopped: {exc}")

                st.session_state.agent_success = ""
                brief = {
                    "scenario_name": "Power Generation",
                    "learning_goals": "Choose a project for the students related to how to replace the energy produced by a coal plant that is being retired. The coal plant currently produces 100 megawatts of power which must be entirely replaced with other energy sources within a budget of $100 million.",
                    "instructions": "Choose a project for the students related to how to replace the energy produced by a coal plant that is being retired. The coal plant currently produces 100 megawatts of power which must be entirely replaced with other energy sources within a budget of $100 million.",
                }
                token, model = model_config()
                try:
                    with st.spinner("Generating and checking the activity…"):
                        project = generate_publishable_project(brief, token, model)
                        record = publish_project(project, "AI agent")
                    replace_draft(project)
                    print(project.keys())
                    st.session_state.agent_epoch += 1
                    st.session_state.agent_success = (
                        f"AI-generated activity published as {record['version']}. "
                        "Students in a new browser session can open the live activity."
                    )
                    st.rerun()
                except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
                    st.error(f"Generate & Publish stopped: {exc}")
        else:
            st.error("Return to the Phase 2 plan tab, revise your choices, and submit again. Your rejected attempt remains in this session's trace until you submit a new plan.")
    else:
         st.info("Once the teacher has reviewed your proposal. the result will appear here")



def phase4_tab(project: Project, version: str) -> None:
    st.markdown("### Phase 4 · Reflection")
    if not st.session_state.phase3_result or not (st.session_state.phase3_result["decision"]=="Accept"):
        st.info("Complete a valid Phase 3 check to unlock the event.")
        return
    else:
        st.markdown("#### Reflect on your decisions")
        answers = []
        for i, question in enumerate(project["reflection_questions"]):
            answers.append(st.text_area(f"{i + 1}. {question}", key=f"reflection_answer_{i}", height=90))
        if st.button("Submit reflection", type="primary"):
            if any(len(answer.strip()) < 10 for answer in answers):
                st.error("Please answer every reflection question in at least 10 characters.")
            else:
                st.session_state.reflection_submitted = {
                    "answers": answers, "submitted_at": datetime.now(timezone.utc).isoformat()}
                st.rerun()
    if st.session_state.reflection_submitted:
        st.success("All four phases are complete. Download your decision trace to share with your teacher.")
        st.download_button("Download decision trace JSON",
                           data=json.dumps(decision_trace(project, version), indent=2, ensure_ascii=False),
                           file_name=f"ecopbl_decision_trace_{version}.json", mime="application/json")

def student_view() -> None:
    pub = st.session_state.published
    if pub is None:
        header("PHASE 2 · STUDENT", "Get ready to explore", "Your teacher must publish an activity first.")
        st.warning("No published activity is available in this session.")
        return
    project: Project = pub["project"]
    header("PHASES 2–4 · STUDENT WORKFLOW", project["scenario_name"],
           "Explore the case, submit a plan, check its rules, respond to an event, and reflect.")
    st.caption(f"Published scenario {pub['version']} · {display_time(pub['published_at'])}")
    locked = bool(project["presubmittal_questions"]) and not st.session_state.comprehension_passed
    mission, expert, plan, check, event = st.tabs([
        "Phase 2 · Mission", "Phase 2 · Expert", "Phase 2 · Plan",
        "Phase 3 · Check", "Phase 4 · Event & reflect"])
    with mission:
        mission_tab(project)
    with expert:
        if locked:
            st.warning("Locked: pass the comprehension check in tab 1 first.")
        else:
            expert_tab(project)
    with plan:
        if locked:
            st.warning("Locked: pass the comprehension check in tab 1 first.")
        else:
            plan_tab(project, pub["version"])
    with check:
        if locked:
            st.warning("Locked: pass the comprehension check in Phase 2 first.")
        else:
            phase3_tab(project, pub["version"])
    with event:
        if locked:
            st.warning("Locked: pass the comprehension check in Phase 2 first.")
        else:
            phase4_tab(project, pub["version"])


init_state()
sidebar()
if st.session_state.role == "Teacher":
    teacher_view()
else:
    student_view()
