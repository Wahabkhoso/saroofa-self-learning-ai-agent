# Saroofa — Self-Learning AI Agent

Saroofa is a portfolio-grade AI agent web application that plans, uses
tools, evaluates its own results, and improves over time via persistent
memory — not model fine-tuning. It combines a deterministic, rule-based
planner with an LLM-driven answer-writing layer, so its behavior stays
explainable and testable while still producing natural, conversational
answers.

This README documents the full system as it stands, including every
feature built, the design reasoning behind each one, and the bugs found
and fixed during development.

---

## Table of Contents

- [What makes this different from "just calling an LLM"](#what-makes-this-different-from-just-calling-an-llm)
- [Tech stack](#tech-stack)
- [Running locally](#running-locally)
- [Project structure](#project-structure)
- [Core pipeline](#core-pipeline)
- [Features](#features)
  - [Chat sessions ("New Chat")](#chat-sessions-new-chat)
  - [Attachments: photos vs. documents](#attachments-photos-vs-documents)
  - [Agent Activity panel](#agent-activity-panel)
  - [Memory Dashboard & Task History](#memory-dashboard--task-history)
  - [Voice dictation](#voice-dictation)
  - [Feedback (thumbs up/down)](#feedback-thumbs-updown)
- [Memory model](#memory-model)
- [LLM provider layer](#llm-provider-layer)
- [Bugs found and fixed during development](#bugs-found-and-fixed-during-development)
- [Known limitations](#known-limitations)
- [Manual test checklist](#manual-test-checklist)

---

## What makes this different from "just calling an LLM"

- **Deterministic tool planning.** Which tool runs for a given task is
  decided by regex/keyword rules in `app/agent/planner.py`, not left to
  the LLM to choose freely. This keeps tool-selection behavior
  predictable and testable without burning API calls on every planning
  decision, and avoids a whole class of failure where the model
  "decides" to call a tool with malformed or hallucinated arguments.
- **Deterministic failure checks.** The evaluator's pass/fail logic (did
  the planned tools actually succeed? did they return output at all?)
  is checked in code first, in `app/agent/evaluator.py`, and this
  verdict can never be overridden by the LLM's own opinion of its work.
  The LLM is only allowed to adjust SUCCESS vs. PARTIAL_SUCCESS and the
  written explanation — never turn a deterministic failure into a
  success.
- **Persistent, evaluated memory.** Every completed task is evaluated,
  distilled into a reusable "experience," and stored in SQL (the
  authoritative record) and ChromaDB (for semantic retrieval). Future
  similar tasks retrieve and apply past lessons — this is the
  "self-learning" part of the project, and it is retrieval-based, not
  fine-tuning. The system prompt and UI are explicit about this
  distinction wherever it matters.
- **Full execution trace.** Every task run is recorded step by step —
  task analysis, memory retrieval, planning, tool execution, evaluation,
  learning — with per-step status and timing, visible live in the UI's
  "Agent Activity" panel.

---

## Tech stack

- **Backend:** Python 3.14, FastAPI, SQLAlchemy, SQLite
- **Vector memory:** ChromaDB (semantic search over past experiences and
  the knowledge base)
- **LLM:** Google Gemini (via the `google-genai` SDK) as the default
  provider, behind a pluggable interface (`app/llm/base.py`) that also
  supports Anthropic and OpenAI
- **Frontend:** a single self-contained `frontend/index.html` — vanilla
  JS, no framework, no build step
- **Testing:** pytest

---

## Running locally

### First-time setup

1. Create a virtual environment and activate it:

python -m venv venv
venv\Scripts\Activate.ps1 (PowerShell)

2. Install dependencies:

pip install -r requirements.txt

3. Copy `.env.example` to `.env` and fill in at least one LLM API key.
   `GEMINI_API_KEY` is the default provider — generate one at
   https://aistudio.google.com/apikey. Make sure `MODEL_NAME` is set to
   a currently-available model (see
   [Known limitations](#known-limitations) — this bit the project once
   already during development).

### Every time after that

Double-click **`start.bat`** in the project folder. This activates the
virtual environment, opens the browser to `http://127.0.0.1:8000`, and
starts the server — no need to open VS Code or type any commands.

If you'd rather run it manually from a terminal:

venv\Scripts\Activate.ps1
uvicorn app.main:app --reload


---

## Project structure

app/
agent/
agent.py orchestrator — runs the full pipeline per task
planner.py TaskAnalyzer (intent classification) + Planner (rule-based tool routing)
evaluator.py deterministic + LLM evaluation of a completed task
learning.py distills a completed task into a stored Experience
state.py Plan/PlanStep/ToolCallRecord/TraceRecorder dataclasses, AgentLimits
api/routes/
chat.py POST /api/chat — the main conversation endpoint
tasks.py GET/DELETE /api/tasks, /api/tasks/sessions — history & session management
knowledge.py upload / extract / list / delete documents
memory.py GET /api/memory — stored experiences
stats.py GET /api/stats — aggregate dashboard numbers
feedback.py POST /api/feedback — thumbs up/down
database/
models.py SQLAlchemy ORM models: Task, Experience, Feedback, Document
repositories.py thin data-access layer, one class per table
database.py engine/session setup, init_db()
knowledge/
loader.py extracts text from .pdf/.docx/.txt/.md/.csv
chunker.py splits text into chunks for embedding
embeddings.py offline hashing vectorizer (default) or a real embedding API
retriever.py ChromaDB-backed ingest + semantic search (RAG)
llm/
base.py LLMProvider interface, LLMMessage/LLMImage/LLMResponse
gemini_provider.py Gemini implementation (default), with retry/backoff for transient errors
anthropic_provider.py, openai_provider.py alternative providers behind the same interface
memory/
short_term.py in-process, per-session conversation memory (resets on restart)
long_term.py persistent, evaluated experience memory (SQL + ChromaDB)
tools/
base.py Tool / ToolResult / ToolRegistry
calculator.py AST-based safe arithmetic (no eval())
python_tool.py sandboxed-ish Python execution (subprocess isolation + timeout)
file_tool.py read-only inspection of uploaded files
knowledge_tool.py RAG search over the persistent knowledge base
schemas/ Pydantic request/response models
core/ config, logging
main.py FastAPI app setup, router registration, static frontend mount
frontend/
index.html the entire web UI — single file, no build step
tests/
pytest test suite
start.bat double-click to activate venv + launch the server + open the browser


---

## Core pipeline

Task
-> Task Analyzer classify intent: calculation / CSV-data / file-read / knowledge-query / general
-> Memory Retrieval semantic search over past experiences (ChromaDB), top-3 by similarity
-> Planner rule-based: intent -> concrete tool step(s), or no tool at all
-> Tool Execution calculator / python_executor / file_inspector / knowledge_search
-> Evaluator deterministic pass (never overridden) + optional LLM qualitative pass
-> retry loop (bounded) only re-enters on a deterministic FAILURE
-> Final Answer LLM writes the natural-language reply from tool results + memory + any attachment
-> Learning result distilled into a stored Experience for future retrieval


Every step above is timestamped by a `TraceRecorder` and shown live in
the "Agent Activity" panel — this is the single clearest way to *show*,
not just claim, that the agent goes through a real multi-step process
rather than being a thin wrapper around one LLM call.

Two narrow special cases bypass the normal planner, each for a specific
reason:

- **Images** skip tool planning entirely, because none of the four
  tools understand images — the agent goes straight to a vision-capable
  LLM call.
- **Greetings and queries that match no specific intent** (calculation,
  CSV, file, or an explicit knowledge-base reference) skip tool
  execution too and go straight to the LLM. This was *not* the original
  behavior — see [Bugs found and fixed](#bugs-found-and-fixed-during-development).

---

## Features

### Chat sessions ("New Chat")

- A **"New Chat"** button starts a fresh conversation with its own
  `session_id` (a client-generated UUID, persisted in the browser's
  `localStorage` so refreshing the page doesn't lose your place).
- The sidebar's **"Recent chats"** list shows every past session, titled
  by its first message, with a relative timestamp ("5h ago"). Clicking
  one loads that conversation's full history back into the main panel.
- Each chat can be **deleted** individually (hover to reveal a trash
  icon, with a confirmation prompt). Deleting a chat removes its `Task`
  rows and any `Feedback`/`Experience` rows tied to them — handled
  explicitly in `TaskRepository.delete_session()`, since there is no
  database-level cascade-delete configured and SQLite's foreign-key
  enforcement is off by default in this project.
- **Design choice:** there is no separate `sessions` table. Each `Task`
  row simply carries a `session_id` column; the sidebar's list is
  derived with a `GROUP BY session_id` query. This was a deliberate
  simplicity trade-off — no session-level metadata is needed beyond what
  can already be derived from its tasks.
- **Memory Dashboard and system-wide stats stay global** across all
  sessions on purpose — cross-session learning is the actual
  "self-learning" claim of the project, so an experience learned in one
  chat can inform the answer in a completely different chat. Task
  History defaults to showing everything too, with a **"this chat
  only"** toggle for when a scoped view is more useful.

### Attachments: photos vs. documents

The paperclip button opens an explicit choice — **Photo** or
**Document** — rather than silently routing by file type (the original
behavior, which was confusing: picking a PDF always meant "add to
knowledge base" with no way to say otherwise).

- **Photo** → attached directly to that one message as base64 image
  data and answered by a vision-capable LLM call.
- **Document** (`.txt`/`.md`/`.csv`/`.pdf`/`.docx`) → its text is
  extracted via a dedicated endpoint, **`POST /api/knowledge/extract`**,
  which reads the file's content **without saving it to disk or adding
  it to the persistent knowledge base**. That extracted text is injected
  into the prompt for that one message only (as an `ATTACHED DOCUMENT`
  context block) — so you can attach a PDF and immediately ask "what's
  in this?" without a separate upload step first.
  - A checkbox, **"Also save this document to the knowledge base,"**
    lets you additionally make it permanently searchable via the
    existing `POST /api/knowledge/upload` pipeline (chunked, embedded,
    stored in ChromaDB). Left unchecked, the document is one-off context
    only and is not retained anywhere after that message.
  - Extracted text is capped at 8,000 characters
    (`MAX_EXTRACT_CHARS` in `app/api/routes/knowledge.py`) to keep the
    prompt a reasonable size; very long documents should be saved to the
    knowledge base instead, where the RAG pipeline chunks and searches
    the full text.

### Agent Activity panel

Shows the pipeline running step by step in real time (task analysis,
memory retrieval, planning, tool execution, evaluation, final answer,
learning), each with a status icon and duration. A **collapse/expand
toggle** (the chevron next to the panel title) lets you hide it for a
cleaner day-to-day chat view; the collapsed/expanded state is saved in
`localStorage` and persists across refreshes. Useful to keep expanded
specifically when demoing the project, since it's the clearest visible
proof that a real multi-step agent pipeline is running.

### Memory Dashboard & Task History

- **Memory Dashboard** — aggregate stats (total/successful/failed
  experiences, average confidence) plus a scrollable list of individual
  stored experiences, each showing its lesson, confidence score, and how
  many times it's been reused in retrieval.
- **Task History** — every task ever run, in a sortable table (status,
  score, timestamp), with the "this chat only" scope toggle described
  above.

### Voice dictation

Uses the browser's native Web Speech API (`SpeechRecognition`) to
transcribe speech into the message box live, with interim results shown
as you speak. Falls back gracefully (button disabled with an
explanatory tooltip) in browsers that don't support it.

### Feedback (thumbs up/down)

Each agent response has "Helpful" / "Not helpful" buttons that call
`POST /api/feedback`, which adjusts the underlying experience's
`usefulness_score` — this score is one of the factors (alongside
similarity, recency, and confidence) in a deterministic formula that
ranks which past experiences get retrieved for future similar tasks.

---

## Memory model

- **Short-term (session) memory** — `app/memory/short_term.py`. Plain
  in-process Python objects, one per `session_id`, holding the last few
  turns so the agent can resolve references like "give it to me" within
  a conversation. Intentionally **not** persisted — it resets if the
  server restarts. This is a different thing from the "New Chat"
  session history described above, which lives in the database and does
  persist across restarts.
- **Long-term memory** — the `Experience` table (SQL, authoritative)
  kept in sync with ChromaDB (semantic search). Retrieval ranking
  combines similarity + usefulness (adjusted by feedback) + recency +
  confidence via a deterministic, inspectable formula — not an opaque
  black box.

---

## LLM provider layer

`app/llm/` defines a provider interface (`LLMProvider`); Gemini is the
default implementation, with Anthropic and OpenAI as alternative
implementations behind the same interface — swapping providers is a
one-line config change, not a code rewrite.

Gemini calls retry automatically (bounded, 2 attempts with backoff) for
transient `503 UNAVAILABLE` / `RESOURCE_EXHAUSTED` "high demand" errors
specifically — not for real configuration errors like an invalid key or
a deprecated model name, which fail immediately. When the LLM call fails
after retries, the agent falls back to a clean, honest status message
("the AI service is temporarily unavailable, please try again") rather
than surfacing raw tool output or a stack trace.

---

## Bugs found and fixed during development

Documented here deliberately — these are good interview talking points,
since they show real debugging rather than just feature-building.

1. **Unmatched queries were always routed to `knowledge_search`.** The
   planner's fallback for "no specific intent matched" unconditionally
   called the knowledge-base search tool — even for a plain "hi" or an
   unrelated coding request. This surfaced irrelevant document chunks
   (e.g. an uploaded networking quiz) as if they were part of the
   answer to "write me a Python password generator." **Fix:** unmatched
   queries now go straight to the LLM with no tool call; only an
   explicit knowledge-base reference (e.g. "according to the document I
   uploaded") still routes to `knowledge_search`.

2. **LLM failures silently dumped raw tool output to the user.** When
   the final-answer LLM call failed, the agent fell back to printing raw
   tool results directly — which, combined with bug #1, meant a failed
   call could show an unrelated wall of JSON-like text with no
   indication anything had gone wrong. **Fix:** an explicit
   `llm_failed` path in `agent.py`'s `_template_answer()` now returns a
   clean, honest fallback message instead ("Sorry, I couldn't generate a
   response just now — the AI service is temporarily unavailable.").

3. **Deprecated model name caused every LLM call to fail.** Diagnosed
   from the server log: Gemini was returning `404 NOT_FOUND` with
   `"This model ... is no longer available to new users"` for the
   originally configured model. Google's own error message named the
   replacement model, which was then set in `.env`'s `MODEL_NAME`. This
   is a good example of reading the *actual* error text rather than
   guessing — an initial hypothesis (invalid API key) was ruled out
   because a `404` with that specific message, not a `401`, pointed
   clearly at model availability instead.

4. **Document uploads auto-routed to the knowledge base with no
   alternative.** Originally, any non-image file attached in the chat
   bar silently went straight into the permanent knowledge base via
   `/api/knowledge/upload` — there was no way to attach a document for
   one-off context without permanently ingesting it. **Fix:** a new
   `/api/knowledge/extract` endpoint reads a file's text without saving
   it; the frontend's explicit "Photo vs. Document" attach menu plus an
   opt-in "also save to knowledge base" checkbox now cover both use
   cases cleanly.

---

## Known limitations

- **Gemini free-tier reliability.** Even with a valid key and a
  currently-supported model name, free-tier calls can fail
  intermittently under load. Retry/backoff and the graceful fallback
  message reduce the *visible* impact but don't eliminate the
  underlying intermittency — if this recurs during a demo, it's expected
  free-tier behavior, not a regression.
- **Short-term (session) memory is in-process only and never explicitly
  evicted.** Over a long server uptime with many sessions, this grows
  unbounded in memory. Fine at demo/portfolio scale; would need an
  eviction policy (LRU, or clear-on-new-chat) for production use.
- **`TaskRepository.list_sessions()` is N+1** — one query for session
  metadata, plus one additional query per session to fetch its title.
  Fine at demo scale; a single subquery (or a dedicated `sessions`
  table) would be needed to scale to many sessions per user.
- **No database-level cascade-delete.** Deleting a chat session deletes
  its `Task`, `Feedback`, and `Experience` rows explicitly in
  application code, not via `ON DELETE CASCADE`. This works correctly
  given this project's usage pattern, but relies on deletion always
  going through `TaskRepository.delete_session()` rather than any ad-hoc
  row deletion elsewhere.
- **Google sign-in in the UI is a visual placeholder only** — clicking
  it shows an alert; no real OAuth flow is wired up. Deliberately
  deferred.
- **Document attachment truncates at 8,000 characters** of extracted
  text per message. Longer documents are better served by also saving
  to the knowledge base, where the full text is chunked and searched
  rather than truncated.

---

## Manual test checklist

Use this before any demo or submission.

**Core chat**
- [ ] Plain-text message gets a coherent answer (never raw tool output)
- [ ] A greeting ("hi") is answered directly, no tool call
- [ ] A calculation routes to the calculator tool with a correct result
- [ ] A CSV question routes to `python_executor` with a correct statistic
- [ ] An unrelated general question (e.g. a coding request) is answered
      directly by the LLM, with no irrelevant knowledge-base content

**Sessions**
- [ ] "New Chat" clears the view and starts a fresh session
- [ ] A new chat appears in the sidebar after its first message
- [ ] Switching between chats shows only that chat's own history
- [ ] Deleting a chat removes it from the sidebar (with confirmation)

**Attachments**
- [ ] Attach menu shows separate "Photo" and "Document" options
- [ ] A photo attaches to the message and is answered by the vision model
- [ ] A document's content is usable in that message without a separate
      upload step
- [ ] "Also save to knowledge base" checkbox correctly makes it (or
      doesn't make it, if unchecked) permanently searchable later

**Agent Activity**
- [ ] Collapse/expand toggle works and persists across a refresh

**Failure handling**
- [ ] A failed LLM call shows a clean "temporarily unavailable" message,
      never raw tool output or an unhandled exception