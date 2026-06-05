import numpy as np
from collections import deque
from typing import Callable, Optional
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.common.team import TeamClassifier
from app.services.offside import (
    get_attacking_direction,
    get_offside_line_x,
    check_offside,
    draw_offside_lines_on_radar,
)
from app.core.config import (
    PITCH_MODEL_PATH, PLAYER_MODEL_PATH,
    BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID,
    DETECTION_CONF, NMS_THRESHOLD, KP_CONF_THRESHOLD,
    STRIDE, BALL_TRAIL_LEN, HOMOGRAPHY_SMOOTH,
    RADAR_SCALE, RADAR_PADDING,
)
import torch
import cv2
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Models ────────────────────────────────────────────────────────────────────
pitch_detector = YOLO(str(PITCH_MODEL_PATH)).to(device)
player_detector = YOLO(str(PLAYER_MODEL_PATH)).to(device)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = SoccerPitchConfiguration()

# ── Annotators ────────────────────────────────────────────────────────────────
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    radius=8
)
edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#00BFFF'),
    thickness=2,
    edges=CONFIG.edges
)
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20, height=17
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_goalkeepers_team_id(
        players: sv.Detections,
        goalkeepers: sv.Detections
) -> np.ndarray:
    try:
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        team0_xy = players_xy[players.class_id == 0]
        team1_xy = players_xy[players.class_id == 1]

        if len(team0_xy) == 0 or len(team1_xy) == 0:
            return np.zeros(len(goalkeepers), dtype=int)

        team_0_centroid = team0_xy.mean(axis=0)
        team_1_centroid = team1_xy.mean(axis=0)

        goalkeepers_team_id = []
        for gk_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
            dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
        return np.array(goalkeepers_team_id)
    except Exception:
        return np.zeros(len(goalkeepers), dtype=int)


def safe_draw(result, fallback):
    return result if result is not None else fallback


def safe_transform(transformer, points):
    if transformer is None:
        return None
    if points is None or len(points) == 0:
        return None
    try:
        result = transformer.transform_points(points=points)
        if np.any(np.abs(result) > 1e6):
            return None
        return result
    except Exception:
        return None


def safe_predict_teams(team_classifier, players_detections, frame):
    if team_classifier is None or len(players_detections) == 0:
        return np.zeros(len(players_detections), dtype=int)
    try:
        crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        if len(crops) == 0:
            return np.zeros(len(players_detections), dtype=int)
        return team_classifier.predict(crops)
    except Exception:
        return np.zeros(len(players_detections), dtype=int)


def build_transformer(frame, homography_M):
    try:
        pitch_result = pitch_detector(frame, conf=DETECTION_CONF, device=device)[0]
        key_points = sv.KeyPoints.from_ultralytics(pitch_result)

        if key_points.confidence is None or key_points.xy is None:
            return None, None

        kp_filter = key_points.confidence[0] > KP_CONF_THRESHOLD
        frame_reference_points = key_points.xy[0][kp_filter]
        pitch_reference_points = np.array(CONFIG.vertices)[kp_filter]

        if len(frame_reference_points) < 4:
            return None, None

        transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )
        homography_M.append(transformer.m)
        transformer.m = np.mean(np.array(homography_M), axis=0)

        overlay_transformer = ViewTransformer(
            source=pitch_reference_points,
            target=frame_reference_points
        )
        return transformer, overlay_transformer

    except Exception as e:
        print(f"⚠️  Transformer build failed: {e}")
        return None, None


def safe_draw_offside_line(annotated_frame, offside_line_x,
                           overlay_transformer, offside_detected):
    if offside_line_x is None or overlay_transformer is None:
        return annotated_frame
    try:
        line_points_pitch = np.array([
            [offside_line_x, 0],
            [offside_line_x, CONFIG.width]
        ], dtype=np.float32)

        line_points_frame = safe_transform(overlay_transformer, line_points_pitch)
        if line_points_frame is None:
            return annotated_frame

        pt1 = tuple(line_points_frame[0].astype(int))
        pt2 = tuple(line_points_frame[1].astype(int))
        color = (0, 0, 255) if offside_detected else (0, 255, 0)
        cv2.line(annotated_frame, pt1, pt2, color, thickness=3)
    except Exception:
        pass
    return annotated_frame


# ── Phase 1: collect crops and fit team classifier ────────────────────────────
def collect_crops_and_fit_classifier(source_path: str):
    print("Collecting player crops for team classification...")
    try:
        frame_generator = sv.get_video_frames_generator(
            source_path=source_path, stride=STRIDE)

        crops = []
        for frame in tqdm(frame_generator, desc='collecting crops'):
            try:
                result = player_detector(frame, conf=DETECTION_CONF, device=device)[0]
                detections = sv.Detections.from_ultralytics(result)
                players_detections = detections[detections.class_id == PLAYER_ID]
                players_crops = [sv.crop_image(frame, xyxy)
                                 for xyxy in players_detections.xyxy]
                crops += players_crops
            except Exception:
                continue

        if len(crops) < 2:
            print("⚠️  Not enough player crops — team classifier disabled.")
            return None

        team_classifier = TeamClassifier(device=device)
        team_classifier.fit(crops)
        print("Team classifier ready.")
        return team_classifier

    except Exception as e:
        print(f"⚠️  Team classifier failed: {e}")
        return None


# ── Phase 2: full pipeline ────────────────────────────────────────────────────
def run_pipeline(
        source_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int], None]] = None
) -> dict:
    """
    Runs detection + offside analysis pipeline.

    Args:
        source_path       : input video path
        output_path       : where to write annotated video
        progress_callback : optional fn(int) called with 0-100 progress

    Returns:
        {
            "verdict"     : "OFFSIDE" | "ONSIDE" | "NO_FOOTBALL_DETECTED",
            "output_path" : output_path
        }
    """
    team_classifier = collect_crops_and_fit_classifier(source_path)

    tracker = sv.ByteTrack()
    tracker.reset()

    ball_trail = deque(maxlen=BALL_TRAIL_LEN)
    homography_M = deque(maxlen=HOMOGRAPHY_SMOOTH)
    last_had_transformer = False
    offside_ever_detected = False
    football_detected = False

    try:
        _sample_radar = draw_pitch(
            CONFIG,
            background_color=sv.Color.from_hex('#1a7a1a'),
            line_color=sv.Color.from_hex('#ffffff'),
            padding=RADAR_PADDING,
            scale=RADAR_SCALE
        )
        RADAR_H, RADAR_W = _sample_radar.shape[:2]
    except Exception:
        RADAR_H, RADAR_W = 495, 820
    print(f"Radar size: {RADAR_W}x{RADAR_H}px")

    try:
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)
    except Exception as e:
        print(f"❌  Could not open video: {e}")
        return {"verdict": "ERROR", "output_path": output_path}

    total_frames = video_info.total_frames or 1
    frame_index = 0

    # ── Use cv2.VideoWriter with browser-compatible codec ──────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for browser playback
    out = cv2.VideoWriter(str(output_path), fourcc, video_info.fps,
                          (video_info.width, video_info.height))

    if not out.isOpened():
        print(f"⚠️  avc1 failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, video_info.fps,
                              (video_info.width, video_info.height))

    frames_written = 0

    for frame in tqdm(frame_generator, total=total_frames, desc='processing'):
        try:
            annotated_frame = frame.copy()

            # ── Progress callback ─────────────────────────────────────────────
            frame_index += 1
            if progress_callback is not None:
                pct = int((frame_index / total_frames) * 95) + 5  # 5→100
                progress_callback(pct)

            # ── Pitch detection ───────────────────────────────────────────────
            transformer, overlay_transformer = build_transformer(
                frame, homography_M)

            if transformer is not None and not last_had_transformer:
                print("⚠️  Pitch reacquired after gap.")
            last_had_transformer = transformer is not None

            if transformer is not None and overlay_transformer is not None:
                try:
                    pitch_all_points = np.array(CONFIG.vertices)
                    frame_all_points = safe_transform(
                        overlay_transformer, pitch_all_points)
                    if frame_all_points is not None:
                        frame_all_kp = sv.KeyPoints(
                            xy=frame_all_points[np.newaxis, ...])
                        annotated_frame = edge_annotator.annotate(
                            scene=annotated_frame, key_points=frame_all_kp)
                except Exception:
                    pass

            # ── Player detection ──────────────────────────────────────────────
            try:
                player_result = player_detector(
                    frame, conf=DETECTION_CONF, device=device)[0]
                detections = sv.Detections.from_ultralytics(player_result)
            except Exception:
                detections = sv.Detections.empty()

            # ── Ball ──────────────────────────────────────────────────────────
            ball_detections = detections[detections.class_id == BALL_ID]
            if len(ball_detections) > 0:
                try:
                    ball_detections.xyxy = sv.pad_boxes(
                        xyxy=ball_detections.xyxy, px=10)
                except Exception:
                    pass

            # ── Non-ball → NMS → tracker ──────────────────────────────────────
            all_detections = detections[detections.class_id != BALL_ID]
            if len(all_detections) > 0:
                try:
                    all_detections = all_detections.with_nms(
                        threshold=NMS_THRESHOLD, class_agnostic=True)
                    all_detections = tracker.update_with_detections(
                        detections=all_detections)
                except Exception:
                    all_detections = sv.Detections.empty()
            else:
                try:
                    all_detections = tracker.update_with_detections(
                        detections=all_detections)
                except Exception:
                    pass

            # ── Split by class ────────────────────────────────────────────────
            if len(all_detections) > 0:
                goalkeepers_detections = all_detections[
                    all_detections.class_id == GOALKEEPER_ID]
                players_detections = all_detections[
                    all_detections.class_id == PLAYER_ID]
                referees_detections = all_detections[
                    all_detections.class_id == REFEREE_ID]
            else:
                goalkeepers_detections = sv.Detections.empty()
                players_detections = sv.Detections.empty()
                referees_detections = sv.Detections.empty()

            if len(players_detections) > 0:
                football_detected = True

            # ── Team classification ───────────────────────────────────────────
            players_detections.class_id = safe_predict_teams(
                team_classifier, players_detections, frame)

            if len(goalkeepers_detections) > 0 and len(players_detections) > 0:
                goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                    players_detections, goalkeepers_detections)
            elif len(goalkeepers_detections) > 0:
                goalkeepers_detections.class_id = np.zeros(
                    len(goalkeepers_detections), dtype=int)

            referees_detections.class_id = np.full(len(referees_detections), 2)

            # ── Merge & annotate ──────────────────────────────────────────────
            try:
                merged_detections = sv.Detections.merge([
                    players_detections,
                    goalkeepers_detections,
                    referees_detections
                ])
                merged_detections.class_id = merged_detections.class_id.astype(int)
                labels = [
                    f"#{tid}" if tid is not None else "#?"
                    for tid in merged_detections.tracker_id
                ]
                annotated_frame = ellipse_annotator.annotate(
                    scene=annotated_frame, detections=merged_detections)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=merged_detections,
                    labels=labels)
            except Exception:
                pass

            if len(ball_detections) > 0:
                try:
                    annotated_frame = triangle_annotator.annotate(
                        scene=annotated_frame, detections=ball_detections)
                except Exception:
                    pass

            # ── Offside logic ─────────────────────────────────────────────────
            direction_info = get_attacking_direction(
                goalkeepers_detections, transformer, CONFIG)
            offside_line_x = get_offside_line_x(
                players_detections, goalkeepers_detections,
                transformer, direction_info, CONFIG)
            offside_mask = check_offside(
                players_detections, transformer,
                direction_info, offside_line_x)

            offside_detected = bool(offside_mask.any())
            if offside_detected:
                offside_ever_detected = True

            # ── Offside line on frame ─────────────────────────────────────────
            annotated_frame = safe_draw_offside_line(
                annotated_frame, offside_line_x,
                overlay_transformer, offside_detected)

            # ── Radar ─────────────────────────────────────────────────────────
            blank = np.zeros((RADAR_H, RADAR_W, 3), dtype=np.uint8)
            try:
                radar_frame = safe_draw(draw_pitch(
                    CONFIG,
                    background_color=sv.Color.from_hex('#1a7a1a'),
                    line_color=sv.Color.from_hex('#ffffff'),
                    padding=RADAR_PADDING,
                    scale=RADAR_SCALE
                ), blank)
            except Exception:
                radar_frame = blank

            if transformer is not None:

                if len(players_detections) > 0:
                    try:
                        players_xy = players_detections.get_anchors_coordinates(
                            sv.Position.BOTTOM_CENTER)
                        players_pitch_xy = safe_transform(transformer, players_xy)
                        if players_pitch_xy is not None:
                            team0_mask = players_detections.class_id == 0
                            if team0_mask.any():
                                radar_frame = safe_draw(draw_points_on_pitch(
                                    CONFIG, xy=players_pitch_xy[team0_mask],
                                    face_color=sv.Color.from_hex('#00BFFF'),
                                    edge_color=sv.Color.from_hex('#ffffff'),
                                    radius=8, padding=RADAR_PADDING,
                                    scale=RADAR_SCALE, pitch=radar_frame
                                ), radar_frame)
                            team1_mask = players_detections.class_id == 1
                            if team1_mask.any():
                                radar_frame = safe_draw(draw_points_on_pitch(
                                    CONFIG, xy=players_pitch_xy[team1_mask],
                                    face_color=sv.Color.from_hex('#FF1493'),
                                    edge_color=sv.Color.from_hex('#ffffff'),
                                    radius=8, padding=RADAR_PADDING,
                                    scale=RADAR_SCALE, pitch=radar_frame
                                ), radar_frame)
                    except Exception:
                        pass

                if len(goalkeepers_detections) > 0:
                    try:
                        gk_xy = goalkeepers_detections.get_anchors_coordinates(
                            sv.Position.BOTTOM_CENTER)
                        gk_pitch_xy = safe_transform(transformer, gk_xy)
                        if gk_pitch_xy is not None:
                            for pt, cid in zip(gk_pitch_xy,
                                               goalkeepers_detections.class_id):
                                col = (sv.Color.from_hex('#00BFFF')
                                       if cid == 0
                                       else sv.Color.from_hex('#FF1493'))
                                radar_frame = safe_draw(draw_points_on_pitch(
                                    CONFIG, xy=pt[np.newaxis],
                                    face_color=col,
                                    edge_color=sv.Color.from_hex('#000000'),
                                    radius=10, padding=RADAR_PADDING,
                                    scale=RADAR_SCALE, pitch=radar_frame
                                ), radar_frame)
                    except Exception:
                        pass

                if len(referees_detections) > 0:
                    try:
                        ref_xy = referees_detections.get_anchors_coordinates(
                            sv.Position.BOTTOM_CENTER)
                        ref_pitch_xy = safe_transform(transformer, ref_xy)
                        if ref_pitch_xy is not None:
                            radar_frame = safe_draw(draw_points_on_pitch(
                                CONFIG, xy=ref_pitch_xy,
                                face_color=sv.Color.from_hex('#FFD700'),
                                edge_color=sv.Color.from_hex('#000000'),
                                radius=8, padding=RADAR_PADDING,
                                scale=RADAR_SCALE, pitch=radar_frame
                            ), radar_frame)
                    except Exception:
                        pass

                if len(ball_detections) > 0:
                    try:
                        ball_xy = ball_detections.get_anchors_coordinates(
                            sv.Position.CENTER)
                        ball_pitch_xy = safe_transform(transformer, ball_xy)
                        if ball_pitch_xy is not None:
                            if ball_pitch_xy.shape[0] == 1:
                                ball_trail.append(ball_pitch_xy.flatten())
                            radar_frame = safe_draw(draw_points_on_pitch(
                                CONFIG, xy=ball_pitch_xy,
                                face_color=sv.Color.from_hex('#ffffff'),
                                edge_color=sv.Color.from_hex('#000000'),
                                radius=6, padding=RADAR_PADDING,
                                scale=RADAR_SCALE, pitch=radar_frame
                            ), radar_frame)
                        else:
                            ball_trail.append(np.empty((0,), dtype=np.float32))
                    except Exception:
                        ball_trail.append(np.empty((0,), dtype=np.float32))
                else:
                    ball_trail.append(np.empty((0,), dtype=np.float32))

                if len(ball_trail) > 1:
                    try:
                        radar_frame = safe_draw(draw_paths_on_pitch(
                            config=CONFIG,
                            paths=[list(ball_trail)],
                            color=sv.Color.from_hex('#ffffff'),
                            pitch=radar_frame,
                            padding=RADAR_PADDING,
                            scale=RADAR_SCALE
                        ), radar_frame)
                    except Exception:
                        pass

            # ── Offside players on radar ──────────────────────────────────────
            if transformer is not None and offside_mask.any():
                try:
                    offside_xy = players_detections[offside_mask] \
                        .get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    offside_pitch_xy = safe_transform(transformer, offside_xy)
                    if offside_pitch_xy is not None:
                        radar_frame = safe_draw(draw_points_on_pitch(
                            CONFIG, xy=offside_pitch_xy,
                            face_color=sv.Color.from_hex('#FFD700'),
                            edge_color=sv.Color.from_hex('#000000'),
                            radius=10, padding=RADAR_PADDING,
                            scale=RADAR_SCALE, pitch=radar_frame
                        ), radar_frame)
                except Exception:
                    pass

            try:
                radar_frame = draw_offside_lines_on_radar(
                    radar_frame=radar_frame,
                    offside_line_x=offside_line_x,
                    config=CONFIG,
                    padding=RADAR_PADDING,
                    scale=RADAR_SCALE,
                    offside_detected=offside_detected,
                )
            except Exception:
                pass

            # ── Blend radar bottom-right ──────────────────────────────────────
            try:
                h, w = annotated_frame.shape[:2]
                radar_h, radar_w = radar_frame.shape[:2]
                scale_factor = min((w * 0.35) / radar_w, (h * 0.35) / radar_h)
                radar_resized = cv2.resize(
                    radar_frame,
                    (int(radar_w * scale_factor), int(radar_h * scale_factor)),
                    interpolation=cv2.INTER_LANCZOS4
                )
                rh, rw = radar_resized.shape[:2]
                y1, y2 = h - rh - 10, h - 10
                x1, x2 = w - rw - 10, w - 10
                if y1 >= 0 and x1 >= 0 and y2 <= h and x2 <= w:
                    cv2.rectangle(annotated_frame,
                                  (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                                  color=(255, 255, 255), thickness=2)
                    annotated_frame[y1:y2, x1:x2] = radar_resized
            except Exception:
                pass

            # ── Verdict overlay ───────────────────────────────────────────────
            verdict_now = "OFFSIDE" if offside_detected else "ONSIDE"
            verdict_color = (0, 0, 255) if offside_detected else (0, 255, 0)
            cv2.putText(annotated_frame, verdict_now,
                        (22, 62), cv2.FONT_HERSHEY_DUPLEX, 2.0,
                        (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(annotated_frame, verdict_now,
                        (20, 60), cv2.FONT_HERSHEY_DUPLEX, 2.0,
                        verdict_color, 3, cv2.LINE_AA)

            # ── Write frame using cv2.VideoWriter ─────────────────────────────
            out.write(annotated_frame)
            frames_written += 1

        except Exception as e:
            print(f"⚠️  Frame failed entirely, writing original: {e}")
            try:
                out.write(frame)
                frames_written += 1
            except Exception:
                pass

    # Release the video writer
    out.release()

    # Verify the output
    if frames_written > 0:
        file_size = Path(output_path).stat().st_size
        print(f"✅ Video saved: {output_path}")
        print(f"   Frames written: {frames_written}/{total_frames}")
        print(f"   File size: {file_size / 1024 / 1024:.2f} MB")
    else:
        print(f"❌ No frames were written to {output_path}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    if not football_detected:
        verdict = "NO_FOOTBALL_DETECTED"
    else:
        verdict = "OFFSIDE" if offside_ever_detected else "ONSIDE"

    return {
        "verdict": verdict,
        "output_path": output_path,
    }