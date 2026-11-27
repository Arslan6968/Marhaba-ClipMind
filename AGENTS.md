# AGENTS.md — Marhaba ClipMind Skill

This folder contains the **Marhaba ClipMind** universal AI video clipping skill (`SKILL.md`).
Read `SKILL.md` in full before executing commands.

## Key Rules
1. Run `python clipmind.py doctor` on first use to verify FFmpeg (`libass`), `yt-dlp`, Whisper models, and RAM.
2. Never use or ask for paid cloud AI APIs (OpenAI, Gemini, Claude, Groq). All processing is local-first.
3. Never auto-download entire channels or bypass DRM, age-gates, paywalls, or platform restrictions.
4. For `--brain agent`, read `STORY_TASK.md`, inspect the contact sheet `.jpg` files visually, write `.story.answer.json`, and re-run the command.
5. After rendering, inspect `check_<clip_id>.jpg` visually and use `python clipmind.py remake` if hook text, framing, or preset needs adjustment.
