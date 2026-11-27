"""Max-Level Face Detection, Edge-Aware Facecam Discovery, Smooth Speaker Tracking, and Motion Profiling.
Uses OpenCV YuNet ONNX with Haar Cascade fallback and FFmpeg fast frame sampling.
"""
from pathlib import Path
import subprocess
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from ..config import get_settings


def _even(v: float) -> int:
    iv = int(v)
    return max(2, iv - iv % 2)


class FaceTracker:
    def __init__(self, model_path: Optional[str] = None):
        settings = get_settings()
        resolved = Path(model_path) if model_path else settings.resolve_yunet_model()
        self.model_path = str(resolved) if resolved else ""
        self.detector = None
        self.use_haar = False

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)

        if self.model_path and Path(self.model_path).is_file():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.65,
                    nms_threshold=0.3,
                    top_k=20,
                )
            except Exception:
                self.use_haar = True
        else:
            self.use_haar = True

    def detect_faces_px(self, image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Returns [(x, y, w, h, score)] in image pixels."""
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []

        results: List[Tuple[float, float, float, float, float]] = []
        if self.detector is not None and not self.use_haar:
            try:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(np.ascontiguousarray(image))
                if faces is not None:
                    for f in faces:
                        fx, fy, fw, fh = [float(v) for v in f[:4]]
                        score = float(f[14])
                        if fw > 6 and fh > 6:
                            results.append((max(0.0, fx), max(0.0, fy), min(float(w), fw), min(float(h), fh), score))
                return results
            except Exception:
                pass

        if self.haar is not None:
            gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(16, 16))
            for fx, fy, fw, fh in faces:
                results.append((float(fx), float(fy), float(fw), float(fh), 0.75))
        return results

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, float]]:
        """Detects faces in an image and returns normalized bounding boxes [0, 1]."""
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []
        px_faces = self.detect_faces_px(image)
        return [
            {
                "x": round(fx / w, 4),
                "y": round(fy / h, 4),
                "w": round(fw / w, 4),
                "h": round(fh / h, 4),
                "score": round(sc, 2),
            }
            for fx, fy, fw, fh, sc in px_faces
        ]

    def analyze_frame_faces(self, frame_paths: List[str]) -> Dict[str, Any]:
        """Quick whole-video frame survey to check general face presence and static corner cam."""
        if not frame_paths:
            return {"has_faces": False, "face_count": 0, "is_facecam": False, "cam_box": None, "center": None}

        detected = []
        for path in frame_paths:
            img = cv2.imread(path)
            if img is not None:
                detected.extend(self.detect_faces(img))

        if not detected:
            return {"has_faces": False, "face_count": 0, "is_facecam": False, "cam_box": None, "center": None}

        corner_faces = [
            f for f in detected
            if (f["x"] < 0.35 or f["x"] > 0.65) and (f["y"] < 0.38 or f["y"] > 0.62) and f["w"] <= 0.30
        ]
        is_facecam = len(corner_faces) >= len(detected) * 0.6 and len(detected) >= 3

        avg_x = sum(f["x"] for f in detected) / len(detected)
        avg_y = sum(f["y"] for f in detected) / len(detected)
        avg_w = sum(f["w"] for f in detected) / len(detected)
        avg_h = sum(f["h"] for f in detected) / len(detected)

        return {
            "has_faces": True,
            "face_count": len(detected),
            "is_facecam": is_facecam,
            "center": (round(avg_x + avg_w / 2.0, 3), round(avg_y + avg_h / 2.0, 3)),
            "cam_box": [round(avg_x, 3), round(avg_y, 3), round(avg_w, 3), round(avg_h, 3)],
        }


def sample_clip_frames(video_path: Path, start: float, length: float, src_w: int, src_h: int,
                       sample_fps: float = 2.0, sample_w: int = 640, max_samples: int = 90) -> Tuple[np.ndarray, float]:
    """Samples BGR frames cleanly with FFmpeg rawvideo pipe (consistent across Windows/macOS/Linux)."""
    if src_w <= 0 or src_h <= 0 or length <= 0.2:
        return np.zeros((0, 360, sample_w, 3), dtype=np.uint8), 1.0
    settings = get_settings()
    h = _even(src_h * sample_w / src_w)
    cmd = [
        settings.ffmpeg_bin, "-nostdin", "-v", "error",
        "-ss", f"{max(0.0, start):.3f}", "-t", f"{length:.3f}",
        "-i", str(video_path),
        "-an", "-vf", f"fps={sample_fps},scale={sample_w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    frame_bytes = sample_w * h * 3
    n = len(raw) // frame_bytes
    if not n:
        return np.zeros((0, h, sample_w, 3), dtype=np.uint8), src_w / sample_w
    frames = np.frombuffer(raw[:n * frame_bytes], dtype=np.uint8).reshape(n, h, sample_w, 3)
    if n > max_samples:
        idx = np.linspace(0, n - 1, max_samples).astype(int)
        frames = frames[idx]
    return frames, src_w / sample_w


def _cluster_face_tracks(dets: List[Tuple[int, float, float, float, float]], frames: np.ndarray) -> List[Dict[str, Any]]:
    """Groups per-frame detections into persistent spatial tracks with liveness & jitter metrics."""
    if not dets or len(frames) == 0:
        return []
    n, h, w = len(frames), frames.shape[1], frames.shape[2]
    groups = []
    for d in sorted(dets):
        cx, cy = d[1] + d[3] / 2.0, d[2] + d[4] / 2.0
        for g in groups:
            if abs(cx - g["cx"]) < 0.07 * w and abs(cy - g["cy"]) < 0.10 * h:
                g["items"].append(d)
                k = len(g["items"])
                g["cx"] += (cx - g["cx"]) / k
                g["cy"] += (cy - g["cy"]) / k
                break
        else:
            groups.append({"cx": cx, "cy": cy, "items": [d]})

    gray = frames.mean(axis=3)
    out = []
    for g in groups:
        idx = sorted({d[0] for d in g["items"]})
        box = np.median(np.array([d[1:] for d in g["items"]]), axis=0)
        x, y, bw, bh = [int(v) for v in box]
        patch = gray[:, max(0, y):y + max(2, bh), max(0, x):x + max(2, bw)]
        motion = float(np.abs(np.diff(patch, axis=0)).mean()) if len(patch) > 1 else 0.0
        centres = np.array([[d[1] + d[3] / 2.0, d[2] + d[4] / 2.0] for d in g["items"]])
        jitter = float(centres.std(axis=0).max() / w) if len(centres) > 1 else 0.0
        out.append({
            "box": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            "presence": len(idx) / n,
            "size": float(box[2]) / w,
            "alive": motion > 1.0,
            "items": g["items"],
            "jitter": jitter,
            "cx_norm": float((box[0] + box[2] / 2.0) / w),
            "cy_norm": float((box[1] + box[3] / 2.0) / h),
        })
    return sorted(out, key=lambda t: -t["presence"])


def _detect_cam_overlay_box(frames: np.ndarray, face_box: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float, float, float], str]:
    """Finds webcam overlay borders via persistent Canny edges around the face, or falls back to proportional box."""
    n, H, W = frames.shape[:3]
    fx, fy, fw, fh = face_box
    persist = np.zeros((H, W), dtype=np.float32)
    for f in frames:
        persist += cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 60, 150) > 0
    persist /= max(1, n)
    persist = cv2.dilate(persist, np.ones((3, 3), dtype=np.uint8))

    xs = slice(int(max(0, fx - fw)), int(min(W, fx + 2 * fw)))
    ys = slice(int(max(0, fy - 0.5 * fh)), int(min(H, fy + 2 * fh)))
    rows = persist[:, xs].mean(axis=1)
    cols = persist[ys, :].mean(axis=0)

    def strongest(scores, a, b, edge):
        ai, bi = int(max(0, a)), int(min(len(scores), b))
        if bi - ai < 2:
            return edge
        seg = scores[ai:bi]
        k = int(np.argmax(seg))
        return ai + k if seg[k] >= 0.58 else edge

    top = strongest(rows, fy - 4.0 * fh, fy - 0.3 * fh, 0 if fy - 4.0 * fh <= 0 else None)
    bottom = strongest(rows, fy + 1.4 * fh, fy + 5.0 * fh, H if fy + 5.0 * fh >= H else None)
    left = strongest(cols, fx - 6.0 * fw, fx - 0.3 * fw, 0 if fx - 6.0 * fw <= 0 else None)
    right = strongest(cols, fx + 1.3 * fw, fx + 7.0 * fw, W if fx + 7.0 * fw >= W else None)

    if None not in (top, bottom, left, right):
        bw, bh = right - left, bottom - top
        if 2.0 * fw <= bw <= 10 * fw and 1.8 * fh <= bh <= 7 * fh and bw <= 0.65 * W and bh <= 0.75 * H and 0.6 <= bw / bh <= 2.6:
            return (float(left), float(top), float(bw), float(bh)), "overlay edges"

    bw = min(3.8 * fw, 0.55 * W)
    bh = min(bw * 0.75, 0.65 * H)
    x0 = min(max(fx + fw / 2.0 - bw / 2.0, 0.0), W - bw)
    y0 = min(max(fy + fh / 2.0 - 0.42 * bh, 0.0), H - bh)
    return (float(x0), float(y0), float(bw), float(bh)), "face expansion"


def compute_motion_profile(frames: np.ndarray, exclude_box_px: Optional[Tuple[float, float, float, float]] = None,
                           target_ar: float = 9 / 16) -> Tuple[float, float]:
    """Returns (center_x_norm, concentration_0_to_1) indicating where visual action concentrates."""
    if len(frames) < 3:
        return 0.5, 0.0
    H, W = frames.shape[1:3]
    src_ar = W / max(1, H)
    gray = frames.mean(axis=3).astype(np.float32)
    diff = np.abs(np.diff(gray, axis=0))
    if exclude_box_px:
        ex, ey, ew, eh = [int(v) for v in exclude_box_px]
        diff[:, max(0, ey):min(H, ey + eh), max(0, ex):min(W, ex + ew)] = 0.0
    cols = diff.sum(axis=(0, 1))
    total = float(cols.sum())
    if total < 1e-3:
        return 0.5, 0.0
    cx = float(np.clip((cols * np.arange(W)).sum() / total / W, 0.0, 1.0))
    half = max(1, int(W * min(1.0, target_ar / src_ar) / 2))
    lo, hi = max(0, int(cx * W) - half), min(W, int(cx * W) + half)
    return round(cx, 3), round(float(cols[lo:hi].sum() / total), 3)


def analyze_clip_visuals(video_path: Path, start: float, length: float, src_w: int, src_h: int) -> Dict[str, Any]:
    """Deep per-clip visual analysis: distinguishes corner facecam, talking head, multi-speaker, or action motion."""
    frames, scale = sample_clip_frames(video_path, start, length, src_w, src_h)
    if len(frames) < 3:
        return {
            "mode": "fit_blur",
            "cam_box": None,
            "face_center": None,
            "speaker_track_norm": None,
            "motion_center": 0.5,
            "motion_concentration": 0.0,
            "is_static": False,
        }

    tracker = FaceTracker()
    dets = []
    for i, f in enumerate(frames):
        for fx, fy, fw, fh, _ in tracker.detect_faces_px(f):
            dets.append((i, fx, fy, fw, fh))

    tracks = _cluster_face_tracks(dets, frames)
    live = [t for t in tracks if t["alive"] and t["presence"] >= 0.30]

    # Sample frame dimensions
    sH, sW = frames.shape[1:3]

    # Separate corner webcams vs larger speakers
    edge_cams = [
        t for t in live
        if t["size"] <= 0.28 and t["jitter"] < 0.025 and t["presence"] >= 0.50
        and (t["cx_norm"] < 0.35 or t["cx_norm"] > 0.65)
        and (t["cy_norm"] < 0.42 or t["cy_norm"] > 0.58)
    ]
    speakers = [t for t in live if t not in edge_cams and t["presence"] >= 0.45 and t["size"] >= 0.06]

    cam_box_norm = None
    cam_box_px = None
    face_center = None
    speaker_track_norm = None
    mode = "fit_blur"

    if edge_cams:
        cam_px, _ = _detect_cam_overlay_box(frames, edge_cams[0]["box"])
        cam_box_px = cam_px
        cam_box_norm = [
            round(cam_px[0] / sW, 4),
            round(cam_px[1] / sH, 4),
            round(cam_px[2] / sW, 4),
            round(cam_px[3] / sH, 4),
        ]
        face_center = (round(edge_cams[0]["cx_norm"], 3), round(edge_cams[0]["cy_norm"], 3))
        mode = "facecam_stacked" if src_w > src_h else "facecam_overlay"
    elif len(speakers) == 1:
        sp = speakers[0]
        face_center = (round(sp["cx_norm"], 3), round(sp["cy_norm"], 3))
        # Build normalized per-sample-frame center_x series for smooth gliding crop
        series: List[Tuple[float, float]] = []
        n_f = len(frames)
        for idx, fx, fy, fw, fh in sp["items"]:
            t_rel = (idx / max(1, n_f - 1)) * length
            cx_n = (fx + fw / 2.0) / sW
            series.append((round(t_rel, 2), round(cx_n, 4)))
        speaker_track_norm = series
        mode = "subject_crop"
    elif len(speakers) >= 2:
        # Multi-speaker scene (e.g. podcast two-shot): keep full width via fit_blur so neither speaker is cropped off
        mode = "fit_blur"
        face_center = (0.5, 0.4)

    motion_cx, motion_conc = compute_motion_profile(frames, exclude_box_px=cam_box_px)
    mean_motion = float(np.abs(np.diff(frames.mean(axis=3), axis=0)).mean())

    if mode == "fit_blur" and motion_conc >= 0.78 and mean_motion >= 3.0:
        # Action tightly concentrated (e.g. mobile/centered action): can use concentrated crop
        face_center = (motion_cx, 0.5)

    return {
        "mode": mode,
        "cam_box": cam_box_norm,
        "face_center": face_center,
        "speaker_track_norm": speaker_track_norm,
        "speaker_count": len(speakers),
        "motion_center": motion_cx,
        "motion_concentration": motion_conc,
        "mean_motion": round(mean_motion, 2),
        "is_static": mean_motion < 1.2,
    }
