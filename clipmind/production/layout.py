"""Max-Level Smart Crop, Gliding Speaker Tracking, Multi-Aspect Layout & Visual Effects Engine."""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


def _even(v: float) -> int:
    iv = int(v)
    return max(2, iv - iv % 2)


def _escape_filter_path(p: str) -> str:
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _avoid_cam_crop(sw: int, sh: int, tw: int, th: int, cx: float, cam_box: Optional[List[float]]) -> Tuple[int, int, int, int]:
    target = tw / th
    if sw / sh > target:
        ch, cw = sh, sh * target
    else:
        cw, ch = sw, sw / target
    cw, ch = _even(cw), _even(ch)
    x = cx * sw - cw / 2.0
    if cam_box and cw < sw:
        c0, c1 = cam_box[0] * sw, (cam_box[0] + cam_box[2]) * sw
        if x < c1 and x + cw > c0:
            if (c0 + c1) / 2.0 < sw / 2.0 and sw - c1 >= cw:
                x = max(x, c1 + 8)
            elif (c0 + c1) / 2.0 >= sw / 2.0 and c0 >= cw:
                x = min(x, c0 - cw - 8)
    x_i = int(min(max(0.0, x), max(0.0, sw - cw)))
    y_i = int(max(0.0, (sh - ch) / 2.0))
    return cw, ch, x_i, y_i


def build_gliding_crop_expr(track_series: List[Tuple[float, float]], sw: int, cw: int, clip_duration: float) -> str:
    """Builds a smooth cosine-eased FFmpeg crop `x` expression tracking a moving speaker."""
    limit = max(0, sw - cw)
    if limit <= 4 or not track_series or len(track_series) < 2:
        cx = track_series[0][1] if track_series else 0.5
        return f"{int(min(max(0.0, cx * sw - cw / 2.0), limit))}"

    # Smooth trajectory over 1s steps
    n_steps = max(2, min(45, int(clip_duration)))
    t_grid = np.linspace(0.0, clip_duration, n_steps)
    raw_t = np.array([p[0] for p in track_series], dtype=np.float32)
    raw_x = np.array([p[1] for p in track_series], dtype=np.float32)
    interp_cx = np.interp(t_grid, raw_t, raw_x)
    # 3-tap moving smoothing
    k = min(5, len(interp_cx))
    smooth_cx = np.convolve(np.pad(interp_cx, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:len(interp_cx)]
    xs = np.clip(smooth_cx * sw - cw / 2.0, 0.0, float(limit))

    keys = [(0.0, float(xs[0]))]
    for t_val, x_val in zip(t_grid[1:], xs[1:]):
        if abs(x_val - keys[-1][1]) > max(8.0, 0.035 * sw):
            keys.append((float(t_val), float(x_val)))
    keys = keys[:32]
    if len(keys) == 1:
        return f"{keys[0][1]:.1f}"

    expr = f"{keys[-1][1]:.1f}"
    for (t0, a), (t1, b) in reversed(list(zip(keys, keys[1:]))):
        span = max(0.05, t1 - t0)
        expr = f"if(lt(t,{t1:.2f}),{a:.1f}+({b - a:.1f})*(1-cos(PI*min(1,(t-{t0:.2f})/{span:.2f})))/2,{expr})"
    return expr


def build_ffmpeg_filtergraph(
    layout_mode: str,
    src_width: int,
    src_height: int,
    target_width: int = 1080,
    target_height: int = 1920,
    cam_box: Optional[List[float]] = None,
    face_center: Optional[Tuple[float, float]] = None,
    speaker_track_norm: Optional[List[Tuple[float, float]]] = None,
    clip_duration: float = 30.0,
    ass_path: Optional[str] = None,
    fonts_dir: Optional[Path] = None,
    punch_windows: Optional[List[Tuple[float, float]]] = None,
    fade: bool = False,
    aspect_ratio: str = "9:16",
    progress_bar: bool = True,
    progress_color: str = "0x0BB9F5",
) -> Tuple[str, str]:
    """Constructs a complete FFmpeg filter_complex graph ending in output pad `[vout]`.
    Returns (filter_complex_string, output_pad_label).
    """
    sw = max(2, src_width)
    sh = max(2, src_height)
    tw = _even(target_width)
    th = _even(target_height)

    chains: List[str] = []

    if aspect_ratio == "original" or layout_mode == "original":
        ew, eh = _even(sw), _even(sh)
        chains.append(f"[0:v]scale={ew}:{eh}:flags=lanczos,setsar=1[base]")
        tw, th = ew, eh
    elif aspect_ratio == "16:9":
        if abs((sw / sh) - (16 / 9)) < 0.08:
            chains.append(f"[0:v]scale={tw}:{th}:flags=lanczos,setsar=1[base]")
        else:
            chains.append(
                f"[0:v]split=2[bg_s][fg_s];"
                f"[bg_s]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},boxblur=25:3,eq=brightness=-0.08[bg];"
                f"[fg_s]scale=-2:{th}:flags=lanczos[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"
            )
    elif aspect_ratio == "1:1":
        if layout_mode == "subject_crop" and (face_center or speaker_track_norm):
            cw, ch, cx, cy = _avoid_cam_crop(sw, sh, tw, th, face_center[0] if face_center else 0.5, None)
            chains.append(f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={tw}:{th}:flags=lanczos,setsar=1[base]")
        else:
            chains.append(
                f"[0:v]split=2[bg_s][fg_s];"
                f"[bg_s]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},boxblur=25:3,eq=brightness=-0.08[bg];"
                f"[fg_s]scale={tw}:{th}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"
            )
    elif layout_mode == "facecam_stacked" and cam_box:
        fx, fy, fw, fh = int(cam_box[0] * sw), int(cam_box[1] * sh), max(64, int(cam_box[2] * sw)), max(64, int(cam_box[3] * sh))
        fx = min(max(0, fx), max(0, sw - fw))
        fy = min(max(0, fy), max(0, sh - fh))
        ht = _even(min(tw * fh / fw, th * 0.36))
        hb = th - ht
        cw, ch, gx, gy = _avoid_cam_crop(sw, sh, tw, hb, face_center[0] if face_center else 0.5, cam_box)
        chains.append(
            f"[0:v]split=2[cam][game];"
            f"[cam]crop={fw}:{fh}:{fx}:{fy},scale={tw}:{ht}:force_original_aspect_ratio=increase,crop={tw}:{ht},setsar=1[top];"
            f"[game]crop={cw}:{ch}:{gx}:{gy},scale={tw}:{hb}:flags=lanczos,setsar=1[bot];"
            f"[top][bot]vstack=inputs=2[base]"
        )
    elif layout_mode == "facecam_overlay" and cam_box:
        fx, fy, fw, fh = int(cam_box[0] * sw), int(cam_box[1] * sh), max(64, int(cam_box[2] * sw)), max(64, int(cam_box[3] * sh))
        camw = _even(tw * 0.40)
        camh = _even(camw * fh / fw)
        chains.append(
            f"[0:v]split=2[main_s][cam_s];"
            f"[main_s]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},setsar=1[m_v];"
            f"[cam_s]crop={fw}:{fh}:{fx}:{fy},scale={camw}:{camh},setsar=1[c_v];"
            f"[m_v][c_v]overlay=(W-w)/2:70[base]"
        )
    elif layout_mode == "hook_card_5_4":
        # Clean dark canvas with 5:4 centered action window leaving clean top band for Hook Card
        box_h = _even(tw * 0.80)  # 1080x864
        box_y = _even((th - box_h) // 2 + 50)
        cw, ch, cx, cy = _avoid_cam_crop(sw, sh, tw, box_h, face_center[0] if face_center else 0.5, cam_box)
        chains.append(
            f"color=c=0x0B0F17:s={tw}x{th}:r=30[canvas];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={tw}:{box_h}:flags=lanczos,setsar=1[box];"
            f"[canvas][box]overlay=0:{box_y}:shortest=1[base]"
        )
    elif layout_mode == "subject_crop" and (speaker_track_norm or face_center):
        target_ar = tw / th
        cw = _even(min(sw, sh * target_ar))
        ch = _even(min(sh, cw / target_ar))
        cy = _even(max(0.0, (sh - ch) / 2.0))
        if speaker_track_norm and len(speaker_track_norm) >= 2:
            x_expr = build_gliding_crop_expr(speaker_track_norm, sw, cw, clip_duration)
            chains.append(f"[0:v]crop={cw}:{ch}:x='{x_expr}':y={cy},scale={tw}:{th}:flags=lanczos,setsar=1[base]")
        else:
            fc_x = face_center[0] if face_center else 0.5
            crop_x = int(min(max(0.0, fc_x * sw - cw / 2.0), max(0, sw - cw)))
            chains.append(f"[0:v]crop={cw}:{ch}:{crop_x}:{cy},scale={tw}:{th}:flags=lanczos,setsar=1[base]")
    else:
        # Default: fit_blur
        chains.append(
            f"[0:v]split=2[bg_src][fg_src];"
            f"[bg_src]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},boxblur=25:3,eq=brightness=-0.08[bg];"
            f"[fg_src]scale={tw}:-2:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"
        )

    cur = "base"

    # Optional loudness-triggered 1.14x punch-in zoom windows
    if punch_windows:
        expr = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in punch_windows[:3])
        zw, zh = _even(tw * 1.14), _even(th * 1.14)
        chains.append(
            f"[{cur}]split=2[p_a][p_b];"
            f"[p_b]scale={zw}:{zh}:flags=lanczos,crop={tw}:{th}[p_z];"
            f"[p_a][p_z]overlay=0:0:enable='{expr}'[punched]"
        )
        cur = "punched"

    # Optional clean fade-in / fade-out
    if fade and clip_duration > 2.0:
        fade_out_st = max(0.5, clip_duration - 0.45)
        chains.append(f"[{cur}]fade=t=in:st=0:d=0.35,fade=t=out:st={fade_out_st:.2f}:d=0.45[faded]")
        cur = "faded"

    # Animated top retention progress bar (0-CPU color slide overlay)
    if progress_bar and clip_duration > 1.0:
        bar_h = max(4, _even(th * 0.004))
        dur_safe = max(1.0, clip_duration)
        chains.append(
            f"color=c={progress_color}:s={tw}x{bar_h}:r=30:d={dur_safe + 0.5:.2f}[pbar];"
            f"[{cur}][pbar]overlay=x='-W+W*min(1,t/{dur_safe:.2f})':y=0:shortest=1[progressed]"
        )
        cur = "progressed"

    # Burn ASS subtitles with bundled fontsdir (guarantees Noto Nastaliq Urdu & Inter render everywhere)
    if ass_path:
        ass_esc = _escape_filter_path(ass_path)
        sub_arg = f"subtitles='{ass_esc}'"
        if fonts_dir and fonts_dir.is_dir():
            fonts_esc = _escape_filter_path(str(fonts_dir.resolve()))
            sub_arg += f":fontsdir='{fonts_esc}'"
        chains.append(f"[{cur}]{sub_arg},format=yuv420p[vout]")
    else:
        chains.append(f"[{cur}]format=yuv420p[vout]")

    return ";".join(chains), "[vout]"
