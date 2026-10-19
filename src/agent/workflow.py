"""Workflow state machine for the analyst conversation.

Stages only advance when their guard is satisfied, so neither the LLM nor the guided agent can skip a
mandatory step. Going *back* to an earlier stage is allowed (e.g. a new ambiguity is discovered).
"""
from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    UPLOAD = "UPLOAD"
    PROFILE = "PROFILE"
    OBJECTIVE = "OBJECTIVE"
    CLARIFICATION = "CLARIFICATION"
    CLEANING_REVIEW = "CLEANING_REVIEW"
    CLEANING_APPROVAL = "CLEANING_APPROVAL"
    DASHBOARD_PLANNING = "DASHBOARD_PLANNING"
    PLAN_VALIDATION = "PLAN_VALIDATION"
    DASHBOARD_GENERATION = "DASHBOARD_GENERATION"
    READY = "READY"


ORDER = list(Stage)

# Friendly names and the short progress steps shown to non-technical users.
STEP_LABELS = [
    ("Upload", {Stage.UPLOAD}),
    ("Understand data", {Stage.PROFILE}),
    ("Your goal", {Stage.OBJECTIVE}),
    ("Questions", {Stage.CLARIFICATION}),
    ("Clean-up", {Stage.CLEANING_REVIEW, Stage.CLEANING_APPROVAL}),
    ("Build dashboard", {Stage.DASHBOARD_PLANNING, Stage.PLAN_VALIDATION, Stage.DASHBOARD_GENERATION}),
    ("Explore", {Stage.READY}),
]


class WorkflowError(Exception):
    """A requested transition is not allowed in the current state."""


def index(stage: Stage | str) -> int:
    return ORDER.index(Stage(stage))


def step_index(stage: Stage | str) -> int:
    s = Stage(stage)
    return next(i for i, (_, stages) in enumerate(STEP_LABELS) if s in stages)


def unmet_requirement(memory, target: Stage) -> str | None:
    """Why ``target`` can't be entered yet (None when every earlier guard is satisfied)."""
    t = index(target)
    if t > index(Stage.PROFILE) and not memory.profiled:
        return "The dataset hasn't been profiled yet."
    if t > index(Stage.OBJECTIVE) and not memory.objective:
        return "The user's business objective hasn't been recorded yet."
    if t > index(Stage.CLARIFICATION):
        if memory.pending_question:
            return "A clarification question is still waiting for the user's answer."
        open_cols = memory.unresolved_columns()
        if open_cols:
            return f"These columns still need an interpretation: {', '.join(open_cols[:5])}."
    if t > index(Stage.CLEANING_REVIEW) and not memory.cleaning_reviewed:
        return "Data-quality issues haven't been reviewed with the user yet."
    if t > index(Stage.CLEANING_APPROVAL):
        undecided = [i["id"] for i in memory.cleaning_issues if i.get("requires_approval") and i.get("status") == "pending"]
        if undecided:
            return f"{len(undecided)} cleaning proposal(s) still need the user's approve/reject decision."
        if memory.cleaning_issues and not memory.cleaning_applied:
            return "Approved cleaning decisions haven't been applied yet."
    if t > index(Stage.DASHBOARD_PLANNING) and not memory.plan:
        return "No dashboard plan has been created yet."
    if t > index(Stage.PLAN_VALIDATION) and not memory.plan_valid:
        return "The dashboard plan hasn't passed validation."
    if t > index(Stage.DASHBOARD_GENERATION) and not memory.dashboard_id:
        return "The dashboard hasn't been generated yet."
    return None


def sync(memory) -> Stage:
    """Advance to the furthest stage whose guards are satisfied (never beyond the next blocked step)."""
    current = Stage(memory.stage)
    target = current
    for stage in ORDER[index(current) + 1:]:
        if unmet_requirement(memory, stage) is None:
            target = stage
        else:
            break
    memory.stage = target.value
    return target


def go_back(memory, target: Stage | str, reason: str) -> Stage:
    """Revisit an earlier stage. Later decisions stay recorded, but downstream steps must be re-confirmed."""
    t = Stage(target)
    if index(t) >= index(memory.stage):
        raise WorkflowError(f"Can only move back to an earlier stage; currently at {memory.stage}.")
    if index(t) < index(Stage.OBJECTIVE):
        raise WorkflowError("The upload and profiling stages can't be revisited; upload the file again instead.")
    memory.stage = t.value
    if index(t) <= index(Stage.CLEANING_APPROVAL):
        memory.plan, memory.plan_valid = None, False
    if index(t) <= index(Stage.CLEANING_REVIEW):
        memory.cleaning_reviewed = False
    if index(t) <= index(Stage.OBJECTIVE):
        memory.objective = ""
    memory.log_event("stage_back", {"to": t.value, "reason": reason[:300]})
    return t


def tool_min_stage() -> dict[str, Stage]:
    """Earliest stage at which each tool may be used."""
    return {
        "inspect_dataset": Stage.PROFILE,
        "get_column_profile": Stage.PROFILE,
        "get_sample_records": Stage.PROFILE,
        "identify_ambiguous_columns": Stage.PROFILE,
        "ask_user_clarification": Stage.PROFILE,
        "set_business_objective": Stage.OBJECTIVE,
        "save_column_interpretation": Stage.OBJECTIVE,
        "go_to_stage": Stage.CLARIFICATION,
        "propose_cleaning_operations": Stage.CLEANING_REVIEW,
        "preview_cleaning_operations": Stage.CLEANING_REVIEW,
        "request_cleaning_approval": Stage.CLEANING_REVIEW,
        "apply_approved_transformations": Stage.CLEANING_APPROVAL,
        "suggest_dashboard_plan": Stage.DASHBOARD_PLANNING,
        "create_dashboard_plan": Stage.DASHBOARD_PLANNING,
        "validate_dashboard_plan": Stage.DASHBOARD_PLANNING,
        "execute_dashboard_plan": Stage.PLAN_VALIDATION,
        "answer_data_question": Stage.READY,
        "update_dashboard": Stage.READY,
    }


def tools_for(stage: Stage | str) -> list[str]:
    s = index(stage)
    return [name for name, minimum in tool_min_stage().items() if index(minimum) <= s]
