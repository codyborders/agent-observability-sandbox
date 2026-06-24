# Experiments Ideas

Datadog Agent Observability Experiments would let this repo compare changes to the Recommendation Council before a user treats them as improvements. The current app already has useful axes for comparison: five-agent council versus single-agent chat, `gpt-5-nano` versus an override model, wider versus narrower database context, and different handoff prompts.

Datadog's Experiments product is built for versioned datasets, experiment runs, and result comparison. In this sandbox, that maps best to controlled chat requests where the expected behavior is known well enough to score.

## Scenario: Council Versus Single-Agent Chat

A contributor could run the same dataset through `CHATBOT_WORKFLOW=council` and `CHATBOT_WORKFLOW=simple`. Good records would include requests such as "I need an open-source observability tool for a Python API," "Find tools for evaluating LLM apps," and "Show me devtools for hosted Postgres migrations."

The experiment should compare groundedness, response usefulness, latency, token usage, and tool-call behavior. The council should usually win on grounded answers and trace detail. The simple path may win on latency and cost. That tradeoff matters because the sandbox teaches people how to read traces and costs alongside answer quality.

## Scenario: Model Swap for the Council

A second experiment could keep the council prompts fixed and compare `CHATBOT_COUNCIL_MODEL` values. Start with the current default, then test a stronger model for only `WriterAgent` or the whole council.

Realistic records should stress different failure modes. One input could ask for "developer tools for SQLite full-text search with Python examples." Another could ask for "a tool like Datadog but cheaper for a solo project." A third could ask for "anything that helps with OpenAI Agents SDK tracing," where the database may have weak matches. The result view should show whether a more expensive model improves fit enough to justify the added cost.

## Scenario: Search Context Size

`CHATBOT_MAX_TOOLS` controls how many database rows enter the conversation. An experiment could compare the current value with one smaller context window and one larger context window on the same dataset.

Expected outcomes are not obvious. More rows may help broad requests such as "CI tools for monorepos," but hurt narrow ones when weak candidates distract the writer. The experiment should track recommendation precision and whether duplicate or off-topic cards reach the final answer.

## Scenario: Handoff Prompt Changes

The council relies on stable handoffs from `IntentAgent` through `WriterAgent`. A contributor could change one prompt at a time and run an experiment against the same dataset version.

Example changes include telling `SearchAgent` to run a broader query when the first search returns no rows, asking `SkepticAgent` to keep only tools with direct evidence, or requiring `WriterAgent` to cite the exact search phrase that found each recommendation. The experiment should show whether the prompt change improves one failure mode without damaging other request types.

## Useful Output

A good experiment report should link to the Datadog experiment, name the dataset version, record the app configuration, and summarize where the change helped or hurt. Put that report in a future `docs/experiments/` file or pull request description.
