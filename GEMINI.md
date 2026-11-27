# GEMINI.md — Marhaba ClipMind Skill

Follow `SKILL.md` in this directory for universal AI video clipping (Workflow A: Single URL / Local File; Workflow B: Creator / Channel Discovery).

- Always check user intent first: run `python clipmind.py probe <URL> --json` and ask at most 2 questions ONLY when instructions are missing.
- Keep execution CPU-first and 8 GB RAM safe (`--whisper-model tiny|base|small`, sequential loading).
- Always inspect generated `check_<clip_id>.jpg` strips before reporting completion.
- Report verified output files, `POSTING.md`, and `manifest.json` honestly without promising virality.
