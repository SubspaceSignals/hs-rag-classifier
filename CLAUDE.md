# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

A local RAG classifier that suggests UK commodity codes (at least 8 digits) from free-text product descriptions, with calibrated confidence and a rationale citing the tariff text. Pipeline: profile (optional), retrieve, two-stage decide (6-digit, then 8-digit), validate against the Trade Tariff API, abstain or escalate.

The full plan, decisions log and session log live in the project handoff one level up, outside this repo:
`../Commodity Code AI Foundry Project Handoff.md`. Read it, then its decisions log and latest session log entry, before suggesting anything. It is private: never copy its contents into this repo, and never move it inside the repo.

## Hard constraints

- **Zero spend.** Everything runs on the local RTX 5070 Ti (16GB VRAM). No paid services, no paid APIs (Claude API included), no Azure account.
- **Models:** Qwen3.5-9B (4-bit GGUF) via llama.cpp server for inference; QLoRA on the same model for fine-tuning. One heavy GPU job at a time.
- **No company or personal data** anywhere in the repo or pipeline. No employer or supplier data.
- **No secrets in git.** The only key (Trade Tariff API) lives in an uncommitted `.env`.
- **Datasets are cited and downloaded from source at run time, never re-hosted.** OGL attribution stays in the README.
- **Test sets are frozen before any model work.** Never tune on the test split. Claude-written text never goes into a test set.
- **Trade Tariff search references are a test set only; never index them for retrieval.**
- Output is advisory, never a legal ruling.

## Working conventions

- State the overall plan first, then do one step at a time; check requirements before any long task.
- Never delete files without explicit permission; say so whenever anything is deleted.
- Each session ends with one line in the handoff's session log: what was done, what comes next.

## Environment

- Python 3.12 managed with uv. Run `uv sync`, then `uv run pytest`.
- The project folder sits on a Windows (NTFS) drive. Keep the virtual environment, model files and data on the Linux drive: set `UV_PROJECT_ENVIRONMENT` outside the repo on the host (the dev container does this already), and keep models under `~/.cache/huggingface` or `~/models`.
