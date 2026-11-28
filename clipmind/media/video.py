"""Video scene cut detection, sampled frame analysis, story contact sheets, and check strips."""
from pathlib import Path
import subprocess
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from ..config import get_settings


def detect_scene_cuts(video_path: Path) -> List[float]:
    """Detects visual scene transitions using PySceneDetect with FFmpeg fallback."""
    try:
        from scenedetect import detect, ContentDetector
        scene_list = detect(str(video_path), ContentDetector(threshold=27.0))
        cuts = [round(scene[0].get_seconds(), 2) for scene in scene_list if scene[0].get_seconds() > 0.5]
        return cuts
    except Exception:
        settings = get_settings()
        cmd = [
            settings.ffmpeg_bin,
            "-nostdin", "-i", str(video_path),
            "-filter_complex", "select='gt(scene,0.3)',metadata=print:file=-",
            "-f", "null", "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        cuts = []
        for line in res.stdout.splitlines():
            if "pts_time:" in line:
                try:
                    val = float(line.split("pts_time:")[1].strip())
                    cuts.append(round(val, 2))
                except Exception:
                    pass
        return cuts


def extract_sampled_frames(video_path: Path, work_dir: Path, step_sec: float = 4.0, max_dim: int = 320) -> List[Dict[str, Any]]:
    """Extracts low-res sampled frames and computes sharpness, brightness, and inter-frame motion."""
    settings = get_settings()
    work_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(work_dir.glob("frame_*.jpg"))
    if not existing:
        out_pattern = str(work_dir / "frame_%04d.jpg")
        cmd = [
            settings.ffmpeg_bin,
            "-y", "-v", "error", "-nostdin",
            "-i", str(video_path),
            "-vf", f"fps=1/{step_sec},scale='min({max_dim},iw)':-2",
            "-q:v", "5",
            out_pattern,
        ]
        subprocess.run(cmd)
        existing = sorted(work_dir.glob("frame_*.jpg"))

    frames = []
    prev_gray = None
    for idx, fpath in enumerate(existing):
        t = round(idx * step_sec, 2)
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(np.mean(gray))
        motion = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))) if prev_gray is not None else 0.0
        prev_gray = gray
        frames.append({
            "t": t,
            "path": str(fpath),
            "sharp": round(sharp, 2),
            "bright": round(bright, 2),
            "motion": round(motion, 2),
        })

    return frames


def create_check_strip(video_path: Path, start: float, hook_time: float, end: float, out_path: Path) -> Optional[Path]:
    """Generates a 3-frame contact strip (Start, Hook/Event, End) of a video interval."""
    settings = get_settings()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / f"tmp_strip_{int(start * 100)}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    times = [
        ("START", max(0.0, start)),
        ("HOOK", hook_time if start <= hook_time <= end else (start + end) / 2.0),
        ("END", max(start + 0.3, end - 0.3)),
    ]

    frame_imgs = []
    for label, t in times:
        frame_file = temp_dir / f"{label}.jpg"
        cmd = [
            settings.ffmpeg_bin,
            "-y", "-v", "error", "-nostdin",
            "-ss", f"{t:.2f}",
            "-i", str(video_path),
            "-vframes", "1",
            "-vf", "scale=360:-2",
            "-q:v", "3",
            str(frame_file),
        ]
        subprocess.run(cmd)
        if frame_file.is_file():
            img = cv2.imread(str(frame_file))
            if img is not None:
                cv2.putText(img, f"{label} ({t:.1f}s)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
                cv2.putText(img, f"{label} ({t:.1f}s)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                frame_imgs.append(img)

    for f in temp_dir.glob("*.jpg"):
        f.unlink(missing_ok=True)
    try:
        temp_dir.rmdir()
    except Exception:
        pass

    if not frame_imgs:
        return None

    # Normalize heights for horizontal stack
    min_h = min(im.shape[0] for im in frame_imgs)
    norm_imgs = [im[:min_h, :] for im in frame_imgs]
    strip = np.hstack(norm_imgs) if len(norm_imgs) > 1 else norm_imgs[0]
    cv2.imwrite(str(out_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return out_path


def create_rendered_check_strip(rendered_mp4: Path, duration: float, out_path: Path) -> Optional[Path]:
    """Generates a 3-frame inspection strip (12%, 50%, 85%) from the final rendered video (with hook & captions)."""
    return create_check_strip(
        rendered_mp4,
        start=max(0.1, duration * 0.12),
        hook_time=max(0.2, duration * 0.50),
        end=max(0.4, duration * 0.85),
        out_path=out_path,
    )


def create_composite_check_strip(strip_paths: List[Path], out_path: Path, max_width: int = 1200) -> Optional[Path]:
    """Combines multiple clip check strips vertically into a single `check_all.jpg` for 1-read agent verification."""
    imgs = []
    for p in strip_paths:
        if not p or not Path(p).is_file():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            im = cv2.resize(im, (max_width, max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
            h, w = im.shape[:2]
        label = Path(p).stem.replace("check_", "")
        cv2.rectangle(im, (6, max(4, h - 38)), (150, h - 6), (15, 23, 42), -1)
        cv2.putText(im, label, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 190, 50), 2, cv2.LINE_AA)
        imgs.append(im)

    if not imgs:
        return None
    target_w = min(im.shape[1] for im in imgs)
    pad = np.full((6, target_w, 3), 20, dtype=np.uint8)
    stacked_parts = []
    for idx, im in enumerate(imgs):
        if im.shape[1] != target_w:
            nh = max(2, int(im.shape[0] * (target_w / float(im.shape[1]))))
            im = cv2.resize(im, (target_w, nh), interpolation=cv2.INTER_AREA)
        if idx > 0:
            stacked_parts.append(pad)
        stacked_parts.append(im)
    composite = np.vstack(stacked_parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return out_path


def create_story_sheets(video_path: Path, t_start: float, t_end: float, dest_dir: Path,
                        every_sec: float = 2.0, cols: int = 4, rows: int = 4, thumb_w: int = 320) -> List[Path]:
    """Creates timestamped 4x4 contact sheet grid(s) for story verification / agent inspection (no Pillow needed)."""
    settings = get_settings()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.glob("_sf_*.jpg"):
        old.unlink(missing_ok=True)

    dur = max(1.0, t_end - t_start)
    cmd = [
        settings.ffmpeg_bin,
        "-y", "-v", "error", "-nostdin",
        "-ss", f"{max(0.0, t_start):.2f}",
        "-t", f"{dur:.2f}",
        "-i", str(video_path),
        "-vf", f"fps=1/{every_sec},scale={thumb_w}:-2",
        "-q:v", "4",
        str(dest_dir / "_sf_%03d.jpg"),
    ]
    subprocess.run(cmd, timeout=300)

    raw_frames = sorted(dest_dir.glob("_sf_*.jpg"))
    if not raw_frames:
        return []

    imgs = []
    for i, f in enumerate(raw_frames):
        im = cv2.imread(str(f))
        if im is not None:
            t_label = f"{t_start + i * every_sec:.0f}s"
            cv2.rectangle(im, (4, 4), (92, 34), (0, 0, 0), -1)
            cv2.putText(im, t_label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
            imgs.append(im)
        f.unlink(missing_ok=True)

    if not imgs:
        return []

    th, tw = imgs[0].shape[:2]
    per_sheet = cols * rows
    pad = 6
    out_sheets: List[Path] = []

    for s_idx in range(0, len(imgs), per_sheet):
        batch = imgs[s_idx:s_idx + per_sheet]
        n_rows = (len(batch) + cols - 1) // cols
        canvas_w = cols * tw + (cols - 1) * pad
        canvas_h = n_rows * th + (n_rows - 1) * pad
        canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
        for i, im in enumerate(batch):
            r, c = divmod(i, cols)
            y0 = r * (th + pad)
            x0 = c * (tw + pad)
            ih, iw = im.shape[:2]
            canvas[y0:y0 + min(th, ih), x0:x0 + min(tw, iw)] = im[:min(th, ih), :min(tw, iw)]
        sheet_path = dest_dir / f"story_sheet_{s_idx // per_sheet + 1}.jpg"
        cv2.imwrite(str(sheet_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
        out_sheets.append(sheet_path)

    return out_sheets
