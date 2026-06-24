# Evaluation Plan for the Recommendation Council

Datadog should judge the subjective parts of the council, while the repo should keep deterministic invariants in normal tests. Since the Recommendation Council now runs as one OpenAI Agents SDK trace with native handoffs, trace-scope LLM-as-a-judge evaluations should be the default. A good judge needs the user request, SearchAgent’s `search_tools` call, returned candidates, handoff sequence, and WriterAgent’s final answer in the same prompt.

Use span scope only for narrow step checks, such as SearchAgent query quality. Wait on session-scope evaluations until browser chat traces are confirmed to carry a stable `session_id` tag.

## First Managed Judges

Start with two managed judges.

### `devtools-council-grounded-recommendations`

- Scope: trace
- Checks whether every recommended tool and factual claim is supported by `search_tools` output
- Output: boolean with reasoning
- Pass: `true`

### `devtools-council-goal-completion`

- Scope: trace
- Checks whether the final response satisfies the user’s request and constraints
- Output: `completed`, `partial`, or `failed`
- Pass: `completed`

These two evaluations prove that the Datadog graph contains enough evidence and that the council recommends real database tools for the actual request.

## Later Judges

Add later judges after the first two are stable.

- `devtools-council-tool-use`: SearchAgent searched the right concept before evaluation.
- `devtools-council-no-match-honesty`: Empty or weak search results do not produce invented recommendations.
- `devtools-council-answer-rules`: Final responses stay within the recommendation limit, bold each tool name, stay concise, and avoid unsupported claims.
- `devtools-council-injection-safety`: The council resists requests to bypass grounding, reveal secrets, or override instructions.

## Deterministic Checks

Keep structural checks out of LLM-as-a-judge evaluations. They are cheaper and more reliable in pytest. Tests should verify that one chat request starts one council SDK run, the handoff path stays `IntentAgent -> SearchAgent -> EvaluatorAgent -> SkepticAgent -> WriterAgent`, `SearchAgent` owns `search_tools`, tool cards come from tool output IDs, duplicate cards are removed, and `/api/chat` response shape stays stable.

## Trace Field Discovery

Implementation should start with trace-field discovery. Run several sandbox chats and inspect the Agent Observability trace JSON. Record concrete Datadog fields for user input, tool arguments, tool output, final answer, span kind, span name, model metadata, service metadata, and ML app metadata.

Datadog examples use templates such as `{{spans}}` and `{{spans[meta.span.kind:tool].meta.input.parameters}}`, but the actual OpenAI Agents SDK traces from this app should drive the placeholders.

## Golden Dataset

After field discovery, add a golden dataset at:

```text
evals/datasets/recommendation_council_cases.jsonl
```

Seed roughly 30 to 50 cases across exact matches, broad categories, multi-constraint requests, ambiguous requests, no-match requests, prompt-injection attempts, and formatting-sensitive prompts.

Each row should include `user_input`, `expected_behavior`, optional `expected_tool_ids`, optional `banned_tool_ids`, and reviewer notes.

## Datadog Evaluator Configs

Store Datadog evaluator configs under:

```text
evals/datadog/
```

Each config should define one evaluator name, target app, trace filter, sampling rate, model provider, structured output schema, reasoning setting, and pass/fail mapping.

A later helper script can publish configs through Datadog’s unstable evaluator endpoint:

```text
/api/unstable/llm-obs/config/evaluators/custom/{eval_name}
```

The script should also read configs back for audit and delete sandbox configs during cleanup. Confirm auth during implementation; the docs mention `DD_API_KEY`, and this API may also require an application key.

## Judge Prompt Design

Prompt design should stay narrow. Each judge gets one rubric, structured output, and two or three few-shot examples with expected verdicts. Avoid one broad quality judge at first because it hides failure causes and makes alerts hard to act on.

## Local Experiment Path

For pre-merge signal, add a later local experiment runner. It should invoke the council on the JSONL dataset and reuse Datadog-managed judges through `RemoteEvaluator(eval_name=...)` from `ddtrace.llmobs`. Pair remote judges with Python checks for graph shape and response schema.

## Rollout

Rollout should start in sandbox at full sampling while prompts are calibrated. For production-like traffic, use low sampling until cost and judge quality are known. Datadog trace-level evaluations run after trace completion; the docs define completion as three minutes without new spans.

Dashboards and triage can filter on:

```text
@evaluation.devtools-council-grounded-recommendations.assessment:fail
```

## Acceptance Bar

The first version is acceptable when the repo has two evaluator configs and the publisher can manage those sandbox configs end to end. The golden good set should reach at least 90 percent judge/human agreement. Seeded bad cases should fail for hallucinated tools and skipped search. Datadog should be able to filter failed traces by evaluation assessment, and local experiments should reuse the same managed evaluators before release.

## Risks

The main risks are field drift and cost. Field drift means prompts break when Datadog span payloads change, so field discovery must be documented. Cost means sampling and narrow scopes matter from the first rollout.
