# Marhaba ClipMind v4 — Extended CLI & Preset Reference

Read this file only when you need detailed flag behavior, `--brain agent` JSON schemas, or preset specs.

## 1. Full CLI Flags
- `doctor`: Checks FFmpeg (`libass` + RTL), `yt-dlp`, JS runtime (`node`/`deno`), RAM, Whisper models (`tiny`/`base`/`small`), YuNet, and fonts.
- `probe <url_or_path> [--intent "..."] [--json] [--compact]`
- `find <creator_or_url> [--topic "..."] [--within 24h|48h|week|month|all] [--sort hot|views|latest] [--count N] [--include-shorts] [--all] [--json] [--compact]`
- `trend <creator_or_topic> [--topic "..."] [--channel] [--count N] [--json] [--compact]`
- `clip <url_or_path>`:
  - `--intent "<goal>"`, `--topic "<topic>"`, `--speaker "<name>"`, `--section "1:00:00-2:30:00"`
  - `--preset clean|dynamic|podcast|gaming|educational|social_media`
  - `--aspect 9:16|16:9|1:1|original`, `--duration short|medium|long|any`, `--max-clips N`
  - `--caption-lang original|roman_urdu|none`, `--language en|ur`, `--whisper-model tiny|base|small`
  - `--brain auto|agent|local|rules|none`
  - `--sfx`, `--no-normalize`, `--fast` (hardware encoder probe), `--out <dir>`, `--compact`, `--json`, `--full-json`
- `remake <out_dir> [--clip 1 --clip 2] [--preset ...] [--aspect ...] [--caption-lang ...] [--hook "..."] [--title "..."] [--sfx] [--compact] [--json]`
- `prune --days 7`

## 2. Six Presets
- `clean`: Steady framing, fade-in/out, minimal captions (business, keynotes, docs).
- `dynamic`: Loudness-triggered 1.14x punch-ins + gold karaoke pop (vlogs, commentary).
- `podcast`: Cosine-eased gliding speaker crop + emerald captions (podcasts, interviews).
- `gaming`: Canny+YuNet stacked facecam + gameplay + punch-ins (gaming streams).
- `educational`: Rounded vector Hook Card + steady clarity + key-stat popups (lectures, tutorials).
- `social_media`: 5:4 dark canvas + sticky Hook Card + coral/white karaoke + punch-ins (general reels).

## 3. `--brain agent` Batch Schema
When `clip ... --brain agent` pauses with exit code `3`, read `STORY_TASK.json` + contact sheets and write `STORY_ANSWERS.json`:
```json
{
  "clip_1": {
    "verdict": "keep|fix|drop",
    "start": 10.5,
    "end": 42.0,
    "title": "Grounded Title",
    "hook": "Punchy *accent* hook",
    "reason": "1 sentence"
  }
}
```
Then re-run the exact same `clip` command.
