# Prompt Tracking Ideas

Datadog Prompt Tracking links prompt templates and versions to LLM spans. The app already annotates some paths with prompt metadata, but the council path depends mostly on OpenAI Agents SDK auto-instrumentation. Future work can make prompt usage easier to compare across traces.

Tracking is useful here because small prompt edits can change handoff order, search terms, final answer length, and cost. One changed sentence can alter the whole graph. A user should be able to filter traces by prompt ID or version and then inspect the exact requests affected by a prompt change.

## Scenario: Prompt Metadata for Council Agents

Add stable prompt metadata for each council agent span. The prompt ID should match the existing names, and the version should come from managed prompts when available. Fallback prompts can use a local version such as `fallback-1`.

A realistic test chat is "Recommend tools for testing a Flask API and monitoring it in Datadog." The resulting trace should show which prompt version was used by `IntentAgent`, `SearchAgent`, `EvaluatorAgent`, `SkepticAgent`, and `WriterAgent`. Datadog Trace Explorer should be able to filter requests that used a specific writer prompt version.

## Scenario: Prompt Variables for Search Context

Prompt Tracking supports variables. For this app, useful variables include the raw user request, sanitized search query, number of tool rows returned, and whether the final answer used fallback response logic.

A contributor could start with `WriterAgent` because it is closest to the final answer. The tracked prompt should include variables for approved tool names and the user's request. If the response is poor, a user can open the prompt side panel and see whether the prompt lacked good context or the model ignored it.

## Scenario: RAG Context Hints

The docs support `rag_context_variables` and `rag_query_variables` for tracked prompts. The council is not classic RAG, but `search_tools` output acts like retrieval context.

An experiment could mark the user request as the query variable and the returned tool summaries as context variables. This may make Datadog's RAG-oriented evaluations more useful for groundedness checks. The scenario should include one exact match, one broad category request, and one no-match request.

## Scenario: Prompt Version Regression View

After prompt tracking is attached, run the same dataset with two versions of `devtools-council-writer`. One version can be terse; another can include stronger evidence language.

The useful view is not only the final score. The contributor should compare latency, input tokens, output tokens, cost, no-match honesty, and failed groundedness evaluations by prompt version. That gives users a concrete reason to keep prompt IDs stable.

## Boundaries

Do not log full raw database rows as prompt variables if they contain unnecessary text. Prefer the fields already shown to users. Good prompt variables include tool ID, name, source, URL, and a trimmed description. Keep variable names stable because Datadog filters and dashboards will depend on them.
