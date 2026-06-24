# Prompt Optimization Ideas

Datadog Prompt Optimization can propose prompt changes after running an app task against a dataset and scoring the outputs. This repo is a good fit because the Recommendation Council has clear prompts, repeatable local data, and failure modes that can be scored.

The first optimization target should be narrow. Optimizing all five council prompts at once would make results hard to interpret. Start with `devtools-council-writer` or `devtools-council-search`.

## Scenario: Optimize WriterAgent for Grounded Answers

Use a dataset of recommendation requests where expected behavior is known. The task function should run the council with a candidate writer prompt and return the final answer plus the tool rows collected from `search_tools`.

Record-level evaluators can check whether the final answer stays under five recommendations, uses tool names from search output, avoids unsupported claims, and gives an honest no-match response. A summary evaluator can combine those checks into one score. Label failures with clear names such as `UNSUPPORTED TOOL`, `MISSED STRONG MATCH`, `TOO VERBOSE`, and `WEAK NO_MATCH`.

Example inputs should include "Find observability tools for Flask," "Recommend LLM evaluation tools," "I need a database migration tool for Postgres," and "Give me a tool that proves my startup will pass SOC 2 tomorrow." The last case should reward honest refusal or weak-match language.

## Scenario: Optimize SearchAgent Query Formation

`SearchAgent` decides the query passed to `search_tools`. A contributor could optimize its prompt against expected search concepts rather than final prose.

For each record, expected output can include a short target concept such as `observability flask`, `llm evaluation`, or `sqlite browser`. The evaluator should compare the actual tool-call argument with the expected concept and then inspect whether the returned rows were plausible.

This scenario is cheaper than full-answer optimization because it scores one step in the graph. It is also useful when final answers are poor because retrieval started with the wrong phrase.

## Scenario: Train, Validation, and Test Split

The Prompt Optimization docs recommend dataset splitting when enough records exist. Build at least 60 records before enabling a split. Training examples can guide prompt improvement, validation can pick the best prompt, and the test set can reveal overfitting.

A useful split would keep prompt-injection records and no-match records represented in every subset. If the best validation prompt becomes too cautious and rejects normal requests, the test score should expose that failure.

## Scenario: Human Review of Proposed Prompts

Prompt Optimization should not auto-promote a prompt to `production`. A safer exercise is writing the best candidate prompt to a review file, linking the Datadog experiment, and requiring a human to compare sample traces before any managed prompt label moves.

The review should quote the old prompt and candidate prompt. It should also list changed rubric behavior, score movement, and any traces where the new prompt looked worse.

## Boundaries

Keep optimization costs visible. Run fewer records while debugging the task function, then use the larger dataset only when evaluators are stable. Do not optimize against a single broad quality score. This app needs separate checks for grounding, no-match honesty, response shape, and latency or token cost.
