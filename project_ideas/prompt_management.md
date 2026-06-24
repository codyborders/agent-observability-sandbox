# Prompt Management Ideas

This repo already has a preview path for Datadog Prompt Management. `ai_classifier.py` and `chatbot.py` call `LLMObs.get_prompt(..., label="production", fallback=...)`, then use local templates when Datadog cannot return a managed prompt. Future work can turn that preview path into a teaching feature.

Prompt Management should help users edit prompts outside the codebase while keeping a safe fallback in git. The app has clear candidates because the classifier uses three prompt IDs and the Recommendation Council uses five prompt IDs.

## Scenario: Managed Prompt Bootstrapper

Build a script that reads the local fallback prompts and publishes them to Datadog Prompt Management under the existing IDs. It should label the first version `production` only after a dry run prints every prompt ID, template length, and target environment.

Realistic prompt IDs are already listed in `README.md`. The council prompts include `devtools-council-intent`, `devtools-council-search`, `devtools-council-evaluator`, `devtools-council-skeptic`, and `devtools-council-writer`. Classifier prompts include `devtools-binary-classifier`, `devtools-batch-classifier`, and `devtools-category-classifier`.

A good version would also read prompts back and compare them with the local fallback strings. The user should see exactly which prompts differ before any label change happens.

## Scenario: Environment-Specific Labels

Teach users how labels affect behavior by adding support for `PROMPT_LABEL`. The default can remain `production`, but a contributor could run the sandbox with `PROMPT_LABEL=experiment-writer-v2` to test only one managed prompt change.

A realistic exercise is changing `WriterAgent` to include shorter bullets or a stronger no-match message. The same chat requests should then show which traces used the experimental label and whether the final answers changed.

## Scenario: Fallback Visibility

The current fallback path catches prompt-fetch failures and keeps the app running. That is good for local teaching, but users need to know when the fallback path is active.

A contributor could add a small health-check field or admin endpoint showing whether each prompt came from Datadog or from the bundled fallback. Example cases should include missing Datadog keys, an unknown prompt ID, and a successful fetch with the `production` label.

## Scenario: Prompt Drift Report

Build a script that compares managed prompts against local fallbacks and writes a markdown report. The report should show prompt ID, managed version, managed label, local hash, managed hash, and whether the template text differs.

This would help users learn why prompt registries need review habits. A prompt may work in Datadog but drift away from the code path that tests use. The report should make that drift visible before experiments or evals are trusted.

## Boundaries

Do not remove local fallback prompts. The sandbox should still boot without Datadog Prompt Management access. Do not publish secrets inside prompt templates. Managed prompts should contain instructions and placeholders, never `.env` values or scraped private data.
