"""Task Fulfillment and Recovery Quality judges for IHBench.

Prompts and judge logic copied from eval-runner:
  eval-runner/custom/inspectai/interruption_handling_benchmark/judge.py

Uses AsyncOpenAI with structured output (Pydantic) and tenacity retry.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class TaskFulfillmentJudgment(BaseModel):
    rationale_a: str
    rationale_b: str
    comparison_rationale: str
    choice: Literal["A", "B"]


class CriterionAssessment(BaseModel):
    criterion: str
    met: bool
    rationale: str


class RecoveryQualityJudgment(BaseModel):
    assessments: list[CriterionAssessment]
    overall_rationale: str
    verdict: Literal["pass", "fail"]


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

TF_SYSTEM = """\
You are a strict judge for spoken dialogue systems. You evaluate how well an assistant \
fulfills its task after being interrupted mid-utterance.

You will be given:
- A conversation history where the assistant was interrupted
- A single evaluation criterion defining what the assistant should accomplish next
- Two candidate assistant responses (A and B)

Your job: determine which response better satisfies the evaluation criterion.

Transcript format:
- If an assistant message ends with <INTERRUPTED />, it was cut off at that exact \
point. The rest was never delivered.
- A user message wrapped in <INTERRUPTION>...</INTERRUPTION> is what the user said \
while interrupting.

Judgment rules:
- Judge ONLY based on the evaluation criterion. Not naturalness, grammar, tone, length, \
or recovery style.
- If neither response satisfies the criterion, pick the one that comes closer.
- "More detail", "more comprehensive", "more specific", or "longer" are NOT valid \
reasons to prefer one response. Only prefer a response if it more directly and \
concretely satisfies the evaluation criterion.
- You MUST pick a winner. You MUST name a concrete way the loser fails or falls short \
of the evaluation criterion.

Output format (strict):
Return ONE JSON object and NOTHING ELSE, matching exactly:

class TaskFulfillmentJudgment(BaseModel):
  rationale_a: str
  rationale_b: str
  comparison_rationale: str
  choice: Literal["A", "B"]

- rationale_a: Independent assessment of how well option A satisfies the evaluation \
criterion. Do not reference option B.
- rationale_b: Independent assessment of how well option B satisfies the evaluation \
criterion. Do not reference option A.
- comparison_rationale: Direct comparison. State the concrete way the loser fails \
the evaluation criterion.
- choice: "A" or "B".\
"""

TF_USER = """\
======= Conversation history =======
{conversation_history}

======= Evaluation criterion =======
{criterion}

======= Candidate responses =======

****** Option A ******
{option_a}

****** Option B ******
{option_b}

Judge which response better satisfies the evaluation criterion.
Return JSON only.\
"""

RQ_SYSTEM = """\
You are a strict pass/fail judge for spoken dialogue systems. You evaluate whether \
an assistant's response correctly recovers from a user interruption.

You will be given:
- A conversation history where the assistant was interrupted
- A list of criteria that the response MUST satisfy for a PASS
- The assistant's response to evaluate

Transcript format:
- If an assistant message ends with <INTERRUPTED />, it was cut off at that exact \
point. The rest was never delivered.
- A user message wrapped in <INTERRUPTION>...</INTERRUPTION> is what the user said \
while interrupting.

Judgment rules:
- Assess each criterion independently. For each, determine if the response satisfies \
it (true/false) with a brief rationale.
- Be strict: "close enough" is not a pass. The criterion must be clearly satisfied.
- Do not introduce your own criteria beyond what is provided. Judge only on what is \
listed.
- Do not penalize the response for behaviors not covered by the criteria.
- The overall verdict MUST be mechanically consistent with the individual assessments:
  - PASS: ALL criteria are met (every assessment has met = true)
  - FAIL: ANY criterion is not met (any assessment has met = false)
  Do NOT override this rule for any reason.

Output format (strict):
Return ONE JSON object and NOTHING ELSE, matching exactly:

class CriterionAssessment(BaseModel):
  criterion: str    # Copy the criterion text exactly as provided
  met: bool
  rationale: str

class RecoveryQualityJudgment(BaseModel):
  assessments: list[CriterionAssessment]   # One per criterion, same order as provided
  overall_rationale: str
  verdict: Literal["pass", "fail"]\
"""

RQ_USER = """\
======= Conversation history =======
{conversation_history}

======= Recovery quality criteria (ALL must be satisfied for PASS) =======
{criteria}

======= Response to evaluate =======
{response}

Assess the response against each criterion. Return JSON only.\
"""


# ---------------------------------------------------------------------------
# History formatting for judge prompts
# ---------------------------------------------------------------------------


def format_history(conversation_history: list[dict], up_to_idx: int) -> str:
    """Format history[:up_to_idx+1] with <INTERRUPTED /> / <INTERRUPTION> markup."""
    window = conversation_history[:up_to_idx + 1]
    n = len(window)
    lines = []
    for i, m in enumerate(window):
        role = m.get("role", "")
        content = m.get("content", "")
        next_msg = window[i + 1] if i + 1 < n else None

        is_about_to_be_interrupted = (
            role == "assistant"
            and next_msg is not None
            and next_msg.get("interruption_spec")
        )
        if is_about_to_be_interrupted:
            truncated = next_msg["interruption_spec"].get(
                "truncated_assistant_utterance", content
            )
            lines.append(f"assistant: {truncated} <INTERRUPTED />")
        elif role == "user" and m.get("interruption_spec"):
            lines.append(f"user: <INTERRUPTION>{content}</INTERRUPTION>")
        else:
            lines.append(f"{role}: {content}")

    return "\n\n".join(lines) if lines else "(empty)"


# ---------------------------------------------------------------------------
# Judge API call helpers
# ---------------------------------------------------------------------------


def _is_retryable(exc: Exception) -> bool:
    return exc.__class__.__name__ in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
    }


async def call_tf_judge(
    client: AsyncOpenAI,
    model_name: str,
    conversation_history: list[dict],
    msg_idx: int,
    criterion: str,
    prediction: str,
    gt_response: str,
    sample_id: str,
    judge_temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> tuple[TaskFulfillmentJudgment | None, str]:
    """Run the Task Fulfillment comparative judge.

    Returns (judgment, response_letter) where response_letter is which
    position ("A" or "B") the model's prediction was placed in.
    """
    hist_str = format_history(conversation_history, msg_idx)

    # Randomize A/B position to avoid positional bias
    rng = random.Random(sample_id)
    options = [("GT", gt_response), ("RESPONSE", prediction)]
    rng.shuffle(options)
    (a_label, option_a), (_, option_b) = options
    response_letter = "A" if a_label == "RESPONSE" else "B"

    user_prompt = TF_USER.format(
        conversation_history=hist_str,
        criterion=criterion,
        option_a=option_a,
        option_b=option_b,
    )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
        before_sleep=lambda s: _log.warning(
            "[TF judge] retry %d/5: %s",
            s.attempt_number,
            s.outcome.exception().__class__.__name__,
        ),
    )
    async def _call() -> TaskFulfillmentJudgment:
        kwargs: dict = {
            "model": model_name,
            "response_format": TaskFulfillmentJudgment,
            "messages": [
                {"role": "system", "content": TF_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        }
        if judge_temperature is not None:
            kwargs["temperature"] = judge_temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        resp = await client.beta.chat.completions.parse(**kwargs)
        return resp.choices[0].message.parsed

    judgment = await _call()
    return judgment, response_letter


async def call_rq_judge(
    client: AsyncOpenAI,
    model_name: str,
    conversation_history: list[dict],
    msg_idx: int,
    criteria: list[str],
    response: str,
    judge_temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> RecoveryQualityJudgment | None:
    """Run the Recovery Quality absolute judge."""
    hist_str = format_history(conversation_history, msg_idx)
    criteria_str = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, 1))

    user_prompt = RQ_USER.format(
        conversation_history=hist_str,
        criteria=criteria_str,
        response=response,
    )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
        before_sleep=lambda s: _log.warning(
            "[RQ judge] retry %d/5: %s",
            s.attempt_number,
            s.outcome.exception().__class__.__name__,
        ),
    )
    async def _call() -> RecoveryQualityJudgment:
        kwargs: dict = {
            "model": model_name,
            "response_format": RecoveryQualityJudgment,
            "messages": [
                {"role": "system", "content": RQ_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        }
        if judge_temperature is not None:
            kwargs["temperature"] = judge_temperature
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        resp = await client.beta.chat.completions.parse(**kwargs)
        return resp.choices[0].message.parsed

    return await _call()


# ---------------------------------------------------------------------------
# Synchronous convenience wrappers
# ---------------------------------------------------------------------------


def evaluate_sample(
    client: AsyncOpenAI,
    model_name: str,
    conversation_history: list[dict],
    msg_idx: int,
    rubric: dict,
    prediction: str,
    baseline_response: str,
    sample_id: str,
    interruption_type: str,
    turn_depth: int,
    judge_temperature: float | None = None,
) -> dict:
    """Evaluate a single sample on both axes. Returns a result dict.

    Can be called from sync code; internally runs the async judges.
    """

    async def _run():
        criterion = rubric["task_fulfillment"]["criterion"]
        criteria = rubric["recovery_quality"]["criteria"]

        tf_judgment, resp_letter = await call_tf_judge(
            client, model_name, conversation_history, msg_idx,
            criterion, prediction, baseline_response, sample_id, judge_temperature,
        )
        rq_judgment = await call_rq_judge(
            client, model_name, conversation_history, msg_idx,
            criteria, prediction, judge_temperature,
        )

        result = {
            "sample_id": sample_id,
            "interruption_type": interruption_type,
            "turn_depth": turn_depth,
        }

        if tf_judgment:
            result["tf_win_score"] = int(tf_judgment.choice == resp_letter)
            result["tf_choice"] = tf_judgment.choice
            result["tf_model_position"] = resp_letter
            result["tf_rationale_a"] = tf_judgment.rationale_a
            result["tf_rationale_b"] = tf_judgment.rationale_b
            result["tf_comparison_rationale"] = tf_judgment.comparison_rationale

        if rq_judgment:
            n_criteria = len(rq_judgment.assessments)
            n_met = sum(a.met for a in rq_judgment.assessments)
            result["rq_pass"] = rq_judgment.verdict == "pass"
            result["rq_score"] = n_met / n_criteria if n_criteria > 0 else 0.0
            result["rq_verdict"] = rq_judgment.verdict
            result["rq_assessments"] = [a.model_dump() for a in rq_judgment.assessments]
            result["rq_overall_rationale"] = rq_judgment.overall_rationale

        return result

    return asyncio.run(_run())
