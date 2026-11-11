# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

A local RAG classifier that suggests UK commodity codes (at least 8 digits) from free-text product descriptions, with a confidence score and a rationale citing the tariff text. Pipeline: profile (optional), retrieve, two-stage decide (6-digit, then 8-digit), validate against the Trade Tariff API, then send low-confidence, disagreeing or out-of-distribution items to human review (there is no escalation to a larger model).

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
- Never open, edit or revert the project handoff file (it lives outside this repo and is maintained in the planning chat). At the end of each task, put a suggested one-line session-log entry (date: what was done; Next: ...) in your final report instead.
- Run Python with PYTHONDONTWRITEBYTECODE=1.
- Never push without the user's say for that commit; show `git show --stat` first.
- The project is in its write-up phase: no new arms, no new test-set runs, no new infrastructure unless a prompt asks for it.
- Public text (README, docs): every figure is checked against its source before commit. A difference between two arms is credited only to what differs between them (the fine-tune's effect is a4 minus a3, not a4 minus a1). The 80% auto-accept figure is the operating-point rule on ruling-style input; the service adds the out-of-distribution flag. CI has no evaluation gate and no image build.

## Environment

- Python 3.12 managed with uv. Run `uv sync`, then `uv run pytest`.
- A4 QLoRA fine-tuning uses the `finetune` group (PEFT, bitsandbytes, accelerate at exact pins; no Unsloth or TRL, see `pyproject.toml`) together with `ml`: `uv sync --group llm --group ml --group finetune`. Name every group you use, or `uv sync` removes the others.
- Decider server: `~/llama.cpp/build/bin/llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL -ngl 99 -c 16384 --port 8080 --jinja --cache-ram 0 -np 1`. Keep `--cache-ram 0`: the default 8 GB host-RAM prompt cache grows about 250 MB per item and gets llama-server OOM-killed on this 14 GiB machine. Keep `-np 1`: A2 output depended on llama-server slot and prompt-cache state; `-np 1` plus `cache_prompt=false` (`CACHE_PROMPT` in `pipeline/llm.py`) gives bit-identical runs. `-np 1` serves one request at a time; the Docker Compose service runs the same frozen command plus --offline and --lora. The `server` block in `configs/a1_pipeline.json` records the server the locked runs used; leave it, because `configs/a2_pipeline.json` pins that file by SHA-256.
- The project folder sits on a Windows (NTFS) drive. Keep the virtual environment, model files and data on the Linux drive: set `UV_PROJECT_ENVIRONMENT` outside the repo on the host (the dev container does this already), and keep models under `~/.cache/huggingface` or `~/models`.
