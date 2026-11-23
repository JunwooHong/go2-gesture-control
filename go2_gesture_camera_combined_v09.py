"""Unified MediaPipe static gestures and dynamic path control for Go2 simulation (v0.9).

v0.3 LARGE UI: Webcam -> MediaPipe index fingertip landmarks -> 2-D trajectory
      -> directional sequence -> proposed robot commands.

Static commands go to 127.0.0.1:51682 when the camera is in IDLE mode.
Dynamic path commands go to 127.0.0.1:51683 only after confirmation.
Show FOUR fingers (index, middle, ring and pinky; thumb folded) three times:
    1st FOUR -> start recording (R)
    2nd FOUR -> finish and save path (F)
    3rd FOUR -> confirm / execute path (C)
Lower your hand or switch away from FOUR for 0.65s between triggers.
Keyboard R/F/C remain as backup; X stops/cancels a mission.
The path executor has exclusive control during an active mission.
Simulation ONLY: do not connect to a physical robot.

Keys (optional):
    R       start/restart a recording
    F/Enter finish; recognize the recorded path and save JSON
    C       confirm mission
    Chinese SIX (thumb + pinky) cancels ANY dynamic phase and restores static controls
    X       clear/abort; return to static mode immediately
    Q/Esc   quit

On startup, IDLE mode recognizes:
    Open_Palm  -> STAND
    Closed_Fist -> CROUCH
    Thumb_Up -> FORWARD
    Thumb_Down -> BACKWARD
    Victory -> TURN_RIGHT
    Three_Fingers (thumb + index + middle; ring/pinky folded) -> TURN_LEFT
    Four_Fingers  (index + middle + ring + pinky; thumb folded) -> R/F/C sequence
    Chinese_Six   (thumb + pinky up; index/middle/ring folded) -> ABORT and exit dynamic mode

During recording, confirmation, and path execution: static movement is disabled.
At any dynamic stage (RECORDING / READY / DONE / SENT), hold Chinese SIX to
abort the mission immediately and return to IDLE static mode. This also works
after the robot has finished its path. Open Palm only means STAND in IDLE.
Keyboard X remains an immediate abort.

Usage (Windows PowerShell):
    py go2_gesture_camera_combined_v09.py
    py go2_gesture_camera_combined_v09.py --self-test
    py go2_gesture_camera_combined_v09.py --camera 1
    py go2_gesture_camera_combined_v09.py --backend dshow

The script expects gesture_recognizer.task next to this file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import socket
from pathlib import Path
import time
from typing import Sequence

# MediaPipe standard gesture categories, not a retrained custom model.
STATIC_COMMANDS = {
    "Open_Palm": "STAND",
    "Closed_Fist": "CROUCH",
    "Thumb_Up": "FORWARD",
    "Thumb_Down": "BACKWARD",
    "Victory": "TURN_RIGHT",
    "Three_Fingers": "TURN_LEFT",
}
STATIC_PORT = 51682
STATIC_CONFIDENCE = 0.55
STATIC_STABLE_FRAMES = 5
STATIC_RELEASE_SECONDS = 0.50
STATIC_SEND_INTERVAL = 0.10

# v0.6: rule-based 3-finger pose recognition, no extra ML model required.
# 21 MediaPipe normalized landmarks: thumb=2,3,4; index=5,6,8;
# middle=9,10,12; ring=13,14,16; pinky=17,18,20.
# This is a rule, not an ML confidence estimate. Debouncing still uses 5 frames.
def recognize_thumb_index_middle(landmarks):
    """True when thumb, index, middle are extended; ring/pinky folded.

    Uses joint geometry and ratios that do not depend on whether the hand
    appears on the left or right of the flipped video. The user should face
    the palm toward the webcam with fingers separated and visible.
    """
    if landmarks is None or len(landmarks) != 21:
        return False

    def dist(i, j):
        a, b = landmarks[i], landmarks[j]
        return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))

    def angle(i, j, k):
        a, b, c = landmarks[i], landmarks[j], landmarks[k]
        ux, uy = float(a.x) - float(b.x), float(a.y) - float(b.y)
        vx, vy = float(c.x) - float(b.x), float(c.y) - float(b.y)
        size = math.hypot(ux, uy) * math.hypot(vx, vy)
        if size < 1e-8:
            return 0.0
        cosine = max(-1.0, min(1.0, (ux*vx + uy*vy) / size))
        return math.degrees(math.acos(cosine))

    palm = dist(5, 17)
    if palm < 0.035:  # Hand too small/landmarks too close for robust classification.
        return False

    def finger_extended(mcp, pip, tip):
        length = max(dist(mcp, pip), 1e-6)
        return (dist(mcp, tip) / length >= 1.47 and
                angle(mcp, pip, tip) >= 148.0)

    def finger_folded(mcp, pip, tip):
        length = max(dist(mcp, pip), 1e-6)
        return (dist(mcp, tip) / length <= 1.32 or
                angle(mcp, pip, tip) <= 132.0)

    index_up = finger_extended(5, 6, 8)
    middle_up = finger_extended(9, 10, 12)
    ring_down = finger_folded(13, 14, 16)
    pinky_down = finger_folded(17, 18, 20)
    # Thumb tip must lie beyond thumb IP and away from index base.
    thumb_up = (angle(2, 3, 4) >= 145.0 and
                dist(2, 4) >= 1.45 * max(dist(2, 3), 1e-6) and
                dist(4, 5) >= dist(3, 5) + .10 * palm)
    return thumb_up and index_up and middle_up and ring_down and pinky_down


# v0.9: Chinese "six" / shaka = thumb + pinky extended; the other three folded.
# This is intentionally NOT a MediaPipe pretrained label; we test 21 hand landmarks.
def recognize_chinese_six(landmarks):
    """Landmark-only rule for thumb+pinky extended, index/middle/ring folded.

    Best viewed with palm facing the webcam and the two extended fingers
    separated. This is a geometric heuristic, NOT a trained confidence score.
    """
    if landmarks is None or len(landmarks) != 21:
        return False

    def dist(i, j):
        a, b = landmarks[i], landmarks[j]
        return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))

    def angle(i, j, k):
        a, b, c = landmarks[i], landmarks[j], landmarks[k]
        ux, uy = float(a.x) - float(b.x), float(a.y) - float(b.y)
        vx, vy = float(c.x) - float(b.x), float(c.y) - float(b.y)
        d = math.hypot(ux, uy) * math.hypot(vx, vy)
        if d < 1e-8:
            return 0.0
        return math.degrees(math.acos(max(-1., min(1., (ux * vx + uy * vy) / d))))

    palm = dist(5, 17)
    if palm < .035:
        return False

    def folded(mcp, pip, tip):
        base = max(dist(mcp, pip), 1e-6)
        return (dist(mcp, tip) <= 1.35 * base or
                angle(mcp, pip, tip) <= 135.)

    def extended(mcp, pip, tip):
        base = max(dist(mcp, pip), 1e-6)
        return (dist(mcp, tip) >= 1.42 * base and
                angle(mcp, pip, tip) >= 143.)

    thumb_extended = (angle(2, 3, 4) >= 145. and
                      dist(2, 4) >= 1.42 * max(dist(2, 3), 1e-6) and
                      dist(4, 5) >= dist(3, 5) + .08 * palm)
    pinky_extended = extended(17, 18, 20)
    three_folded = all(folded(*joints) for joints in
                       ((5, 6, 8), (9, 10, 12), (13, 14, 16)))
    return thumb_extended and pinky_extended and three_folded


# A held SIX only causes one abort. The user must release it to rearm.
SIX_HOLD_FRAMES = 7
SIX_HOLD_SECONDS = .22
SIX_RELEASE_SECONDS = .60


class SixInterruptTrigger:
    """One-shot, sustained-pose trigger. Independent of the model's label."""
    def __init__(self, hold_frames=SIX_HOLD_FRAMES,
                 hold_seconds=SIX_HOLD_SECONDS,
                 release_seconds=SIX_RELEASE_SECONDS):
        self.hold_frames = hold_frames
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.armed = True
        self.frames = 0
        self.started_at = None
        self.released_since = None

    def update(self, is_six, now):
        if is_six:
            self.released_since = None
            if not self.armed:
                return False
            if self.started_at is None:
                self.started_at = now
            self.frames += 1
            if (self.frames >= self.hold_frames and
                    now - self.started_at >= self.hold_seconds):
                self.armed = False
                self.frames = 0
                self.started_at = None
                return True
            return False
        self.frames = 0
        self.started_at = None
        if not self.armed:
            if self.released_since is None:
                self.released_since = now
            elif now - self.released_since >= self.release_seconds:
                self.armed = True
        return False


# Retained from v0.7: FOUR means index+middle+ring+pinky extended, thumb folded.
# This is deliberately distinct from Open_Palm (all five fingers) and
# Three_Fingers (thumb+index+middle only). Rule-based, not a trained model.
def recognize_four_fingers(landmarks):
    if landmarks is None or len(landmarks) != 21:
        return False

    def dist(i, j):
        a, b = landmarks[i], landmarks[j]
        return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))

    def angle(i, j, k):
        a, b, c = landmarks[i], landmarks[j], landmarks[k]
        ax, ay = float(a.x) - float(b.x), float(a.y) - float(b.y)
        bx, by = float(c.x) - float(b.x), float(c.y) - float(b.y)
        den = math.hypot(ax, ay) * math.hypot(bx, by)
        if den < 1e-8:
            return 0.0
        return math.degrees(math.acos(max(-1., min(1., (ax * bx + ay * by) / den))))

    palm = dist(5, 17)
    if palm < .035:
        return False

    def extended(mcp, pip, tip):
        return (dist(mcp, tip) >= 1.45 * max(dist(mcp, pip), 1e-6)
                and angle(mcp, pip, tip) >= 146.)

    fingers_up = all(extended(*triplet) for triplet in
                     ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)))
    # Thumb tip inside the palm, close to index/middle base. This deliberately
    # rejects a typical full Open Palm whose thumb tip points away from the palm.
    thumb_tucked = (dist(4, 5) <= .86 * palm and
                    dist(4, 9) <= 1.12 * palm)
    return fingers_up and thumb_tucked


FOUR_HOLD_FRAMES = 7       # Continuous frames required to accept a pose.
FOUR_RELEASE_SECONDS = .65  # Hand must leave the pose before another press.

class FourFingerTrigger:
    """Debounced one-shot edge. A held FOUR never advances twice.

    Requires an uninterrupted, visible not-FOUR interval before rearming.
    An absent hand is allowed as a release; momentary classification flickers
    shorter than FOUR_RELEASE_SECONDS cannot re-arm the gesture.
    """
    def __init__(self, hold_frames=FOUR_HOLD_FRAMES,
                 release_seconds=FOUR_RELEASE_SECONDS):
        self.hold_frames = hold_frames
        self.release_seconds = release_seconds
        self.armed = True
        self.count = 0
        self.released_since = None

    def update(self, is_four, now):
        if is_four:
            self.released_since = None
            if not self.armed:
                return False
            self.count += 1
            if self.count >= self.hold_frames:
                self.armed = False
                self.count = 0
                return True
            return False
        self.count = 0
        if not self.armed:
            if self.released_since is None:
                self.released_since = now
            elif now - self.released_since >= self.release_seconds:
                self.armed = True
        return False


class StaticGestureDebouncer:
    """Consecutive-frame confirmation and no-hand fallback to STAND."""
    def __init__(self, threshold=STATIC_CONFIDENCE,
                 frames=STATIC_STABLE_FRAMES, release=STATIC_RELEASE_SECONDS):
        self.threshold = threshold
        self.frames = frames
        self.release = release
        self.candidate = None
        self.count = 0
        self.command = 'STAND'
        self.last_confirmed = -1e10

    def update(self, label, score, now, enabled=True):
        if not enabled:
            self.candidate = None
            self.count = 0
            self.command = 'STAND'
            return self.command
        cmd = STATIC_COMMANDS.get(label) if score >= self.threshold else None
        if cmd is None:
            self.candidate = None
            self.count = 0
        elif cmd == self.candidate:
            self.count += 1
        else:
            self.candidate = cmd
            self.count = 1
        if cmd is not None and self.count >= self.frames:
            self.command = cmd
            self.last_confirmed = now
        if now - self.last_confirmed > self.release:
            self.command = 'STAND'
        return self.command


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "gesture_recognizer.task"
RESULT_DIR = SCRIPT_DIR / "experiments" / "dynamic_gestures"

# Tunable experimental hyperparameters. Coordinates are normalized [0, 1].
SMOOTH_WINDOW = 5
RDP_EPSILON = 0.025       # Removes small turns/jitter in a trajectory.
MIN_STROKE = 0.105        # Minimum stroke along a normalized image axis.
AXIS_DOMINANCE = 1.3     # Avoid interpreting diagonal lines as cardinal moves.
MIN_SAMPLES = 12
SAMPLE_INTERVAL = 0.025  # Seconds between stored samples (<=40 per second).
MAX_RECORD_SECONDS = 15.0
MAX_POINTS = 400
MIN_POINT_DELTA = 0.003

# Camera image is mirrored before landmark detection. Increasing X = user's right.
# Image UP (decreasing Y) is *defined* as robot FORWARD (not physical depth).
ROBOT_MAPPING = {"RIGHT": "MOVE_RIGHT", "LEFT": "MOVE_LEFT",
                 "UP": "FORWARD", "DOWN": "BACKWARD"}


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def smooth_points(points: Sequence[tuple[float, float]], radius: int = SMOOTH_WINDOW // 2):
    """Center-aligned moving average for landmark jitter (not for robot motion)."""
    if not points:
        return []
    smoothed = []
    for i in range(len(points)):
        chunk = points[max(0, i - radius): min(len(points), i + radius + 1)]
        smoothed.append((sum(p[0] for p in chunk) / len(chunk),
                         sum(p[1] for p in chunk) / len(chunk)))
    return smoothed


def segment_distance(point, start, end) -> float:
    """Perpendicular distance to the finite line segment from start to end."""
    vx, vy = end[0] - start[0], end[1] - start[1]
    denom = vx * vx + vy * vy
    if denom < 1e-12:
        return distance(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * vx +
                            (point[1] - start[1]) * vy) / denom))
    return distance(point, (start[0] + t * vx, start[1] + t * vy))


def simplify_path(points, tolerance=RDP_EPSILON):
    """Ramer-Douglas-Peucker polyline simplification, iterative implementation."""
    if len(points) <= 2:
        return list(points)
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        begin, end = pending.pop()
        if end <= begin + 1:
            continue
        max_error = -1.0
        argmax = -1
        for i in range(begin + 1, end):
            err = segment_distance(points[i], points[begin], points[end])
            if err > max_error:
                max_error, argmax = err, i
        if max_error > tolerance:
            keep.add(argmax)
            pending.extend(((begin, argmax), (argmax, end)))
    return [points[i] for i in sorted(keep)]


def direction_of_segment(a, b, min_stroke=MIN_STROKE):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) >= min_stroke and abs(dx) >= AXIS_DOMINANCE * abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    if abs(dy) >= min_stroke and abs(dy) >= AXIS_DOMINANCE * abs(dx):
        return "UP" if dy < 0 else "DOWN"
    return None


def decode_path(points, *, min_stroke=MIN_STROKE):
    """Return the ordered cardinal directions from a fingertip trajectory.

    A sharp corner produces two segments, e.g. RIGHT then UP. Very short strokes,
    diagonal strokes, and consecutive duplicates are filtered out.
    """
    if len(points) < MIN_SAMPLES:
        return [], []
    path = simplify_path(smooth_points(points))
    directions = []
    for a, b in zip(path[:-1], path[1:]):
        direction = direction_of_segment(a, b, min_stroke=min_stroke)
        if direction and (not directions or directions[-1] != direction):
            directions.append(direction)
    return directions[:6], path


def save_recording(points, directions, simplified, duration, detected_gestures):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("gesture_%Y%m%d_%H%M%S_%f.json")
    file_path = RESULT_DIR / name
    result = {
        "version": "0.9-dynamic-path",
        "coordinate_system": "mirrored_camera_image_normalized_xy",
        "image_up_maps_to_robot": "FORWARD",
        "segments": directions,
        "robot_command_preview": [ROBOT_MAPPING[d] for d in directions],
        "note": "Commands execute only after explicit FOUR gesture or C confirmation; MuJoCo simulation only.",
        "duration_seconds": round(duration, 3),
        "samples": len(points),
        "parameters": {
            "smoothing_window": SMOOTH_WINDOW,
            "rdp_epsilon": RDP_EPSILON,
            "min_stroke": MIN_STROKE,
            "axis_dominance": AXIS_DOMINANCE,
        },
        "sampled_gesture_classes": detected_gestures[-120:],
        "trajectory_normalized": [[round(x, 5), round(y, 5)] for x, y in points],
        "simplified_normalized": [[round(x, 5), round(y, 5)] for x, y in simplified],
    }
    file_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path


def self_test():
    """Offline validation, independent of OpenCV/MediaPipe/webcam."""
    def linear(a, b, count=35):
        return [(a[0] + (b[0] - a[0]) * i / (count - 1),
                 a[1] + (b[1] - a[1]) * i / (count - 1))
                for i in range(count)]

    p1 = linear((.2, .7), (.7, .7)) + linear((.7, .7), (.7, .2))
    p2 = linear((.7, .2), (.7, .75)) + linear((.7, .75), (.25, .75))
    p3 = [(.4 + .002 * math.sin(i / 3), .5 + .002 * math.cos(i / 3))
          for i in range(65)]
    p4 = linear((.15, .2), (.75, .8))
    tests = [
        ("RIGHT, UP", p1, ["RIGHT", "UP"]),
        ("DOWN, LEFT", p2, ["DOWN", "LEFT"]),
        ("Stationary jitter", p3, []),
        ("Diagonal excluded", p4, []),
    ]
    for name, trajectory, expected in tests:
        got, _ = decode_path(trajectory)
        print(f"{name}: {got}")
        if got != expected:
            raise AssertionError(f"{name}: expected {expected}, got {got}")
    print("PASS: 4 offline trajectory tests")



# Display-only settings. Do not change the webcam capture resolution: on some
# Windows MSMF cameras, cap.set(CAP_PROP_FRAME_*) hangs or blocks indefinitely.
PATH_PORT = 51683
UI_WIDTH = 1280
UI_HEIGHT = 760
CAM_X, CAM_Y, CAM_W, CAM_H = 18, 69, 825, 615
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 859, 69, 403, 615


def draw_dashboard(cv2, frame, mode, gesture, confidence, tip, samples,
                   finger_directions, robot_commands, saved_path=""):
    """Draw a legible 1280x760 dashboard without resizing the camera device.

    The camera frame stays in its own aspect ratio with letterboxing. The
    recognition algorithm uses the ORIGINAL frame's normalized landmarks.
    """
    import numpy as np

    def rect(image, x1, y1, x2, y2, color, radius_unused=None):
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

    def text(image, value, x, y, *, scale=.75, color=(235, 242, 246), thick=2):
        cv2.putText(image, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    def wrap(text_str, start_x, start_y, max_width, *, scale=.66,
             color=(238, 241, 244), line_height=33, max_lines=3):
        """Fit English HUD text to panel width using real OpenCV font metrics."""
        words = text_str.replace(' -> ', '  >  ').split(' ')
        words = [w for w in words if w]
        lines = []
        current = ''
        for word in words:
            candidate = word if not current else current + ' ' + word
            if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > max_width and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        for i, line in enumerate(lines[:max_lines]):
            text(canvas, line, start_x, start_y + i * line_height,
                 scale=scale, color=color, thick=2)
        return min(len(lines), max_lines)

    canvas = np.full((UI_HEIGHT, UI_WIDTH, 3), (24, 31, 39), dtype=np.uint8)
    white = (242, 246, 250)
    muted = (164, 177, 190)
    green = (106, 229, 137)
    cyan = (244, 211, 83)
    gold = (60, 224, 255)

    rect(canvas, 0, 0, UI_WIDTH, 56, (17, 22, 30))
    text(canvas, 'GO2  /  STATIC + DYNAMIC GESTURES', 20, 39,
         scale=.93, color=white, thick=2)
    text(canvas, 'SIM ONLY | FOUR = NEXT | SIX = STOP', 904, 36,
         scale=.49, color=(114, 218, 247), thick=1)

    # Draw the finger overlay on the original frame, then scale for the screen.
    video = frame.copy()
    sh, sw = video.shape[:2]
    def pixel(p):
        return (int(round(p[0] * (sw - 1))), int(round(p[1] * (sh - 1))))
    for a, b in zip(samples[:-1], samples[1:]):
        cv2.line(video, pixel(a), pixel(b), (85, 238, 114), 5, cv2.LINE_AA)
    if samples:
        cv2.circle(video, pixel(samples[0]), 9, (255, 160, 75), -1, cv2.LINE_AA)
    if tip is not None:
        cv2.circle(video, pixel(tip), 15, (0, 0, 10), 4, cv2.LINE_AA)
        cv2.circle(video, pixel(tip), 10, gold, -1, cv2.LINE_AA)

    rect(canvas, CAM_X-2, CAM_Y-2, CAM_X+CAM_W+2, CAM_Y+CAM_H+2, (72, 87, 105))
    rect(canvas, CAM_X, CAM_Y, CAM_X+CAM_W, CAM_Y+CAM_H, (12, 17, 22))
    ratio = min(CAM_W / sw, CAM_H / sh)
    nw = max(1, int(sw * ratio))
    nh = max(1, int(sh * ratio))
    interpolation = cv2.INTER_LINEAR if ratio > 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(video, (nw, nh), interpolation=interpolation)
    dx = CAM_X + (CAM_W - nw) // 2
    dy = CAM_Y + (CAM_H - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = scaled
    # A high-contrast tracking status tag inside the camera area.
    rect(canvas, CAM_X+11, CAM_Y+11, CAM_X+340, CAM_Y+56, (17, 26, 33))
    text(canvas, 'TRACKING: INDEX FINGER TIP  #8', CAM_X+23, CAM_Y+41,
         scale=.64, color=gold, thick=2)

    # Separate dark panel so the HUD is readable even near a bright window.
    rect(canvas, PANEL_X, PANEL_Y, PANEL_X+PANEL_W, PANEL_Y+PANEL_H, (27, 37, 48))
    x = PANEL_X + 21
    content_w = PANEL_W - 44
    text(canvas, 'RECOGNITION STATUS', x, 106, scale=.63, color=muted, thick=2)
    active_color = green if mode == 'RECORDING' else (255, 208, 109) if mode == 'DONE' else white
    text(canvas, mode, x, 149, scale=1.15, color=active_color, thick=3)
    cv2.line(canvas, (x, 164), (PANEL_X+PANEL_W-21, 164), (80, 93, 108), 1)
    text(canvas, 'Hand gesture:', x, 198, scale=.62, color=muted, thick=2)
    text(canvas, gesture[:20], x, 230, scale=.79, color=white, thick=2)
    confidence_text = ('Custom rule: MATCH      Samples: %d' % len(samples)
                       if gesture in ('Three_Fingers', 'Four_Fingers', 'Chinese_Six') else
                       'Confidence: %.2f      Samples: %d' % (confidence, len(samples)))
    text(canvas, confidence_text, x, 262, scale=.57, color=cyan, thick=2)

    text(canvas, 'FINGER PATH', x, 311, scale=.70, color=gold, thick=2)
    finger_str = ' -> '.join(finger_directions) if finger_directions else '(draw a path to begin)'
    count = wrap(finger_str, x, 344, content_w, scale=.73,
                 color=green if finger_directions else muted, line_height=33, max_lines=2)

    text(canvas, 'ROBOT COMMAND PREVIEW', x, 418, scale=.67, color=gold, thick=2)
    robot_str = ' -> '.join(robot_commands) if robot_commands else '(no command yet)'
    wrap(robot_str, x, 450, content_w, scale=.68, color=white, line_height=32,
         max_lines=2)

    cv2.line(canvas, (x, 498), (PANEL_X+PANEL_W-21, 498), (80, 93, 108), 1)
    text(canvas, 'FOUR FINGERS = NEXT STEP', x, 528, scale=.58, color=gold, thick=2)
    text(canvas, '1 START  >  2 FINISH  >  3 RUN', x, 558, scale=.53)
    text(canvas, 'SIX (THUMB + PINKY) = ABORT', x, 589, scale=.49)
    text(canvas, 'Release between FOUR gestures', x, 620, scale=.54)
    if saved_path:
        text(canvas, 'Saved JSON successfully', x, 659,
             scale=.53, color=green, thick=2)

    rect(canvas, 0, 699, UI_WIDTH, UI_HEIGHT, (17, 22, 30))
    if mode == 'RECORDING':
        footer = 'RECORDING: Draw path; FOUR to finish; SIX to abort.'
    elif mode == 'DONE':
        footer = 'READY: Release hand; FOUR to EXECUTE; SIX to abort.'
    elif mode == 'READY':
        footer = 'TIME LIMIT: FOUR to finish; SIX to abort.'
    elif mode == 'SENT':
        footer = 'PATH SENT: Hold SIX to ABORT and return to static. X also stops.'
    else:
        footer = 'IDLE: Show FOUR fingers to START. Six static gestures still work.'
    text(canvas, footer, 20, 738, scale=.75, color=white, thick=2)
    return canvas


def test_three_fingers():
    """Synthetic landmark fixtures. These are unit tests, NOT webcam accuracy tests."""
    from types import SimpleNamespace
    xy = {
        0:(.50,.94), 1:(.39,.80), 2:(.32,.73), 3:(.23,.67), 4:(.14,.60),
        5:(.46,.73), 6:(.44,.55), 7:(.43,.45), 8:(.42,.34),
        9:(.54,.73), 10:(.54,.51), 11:(.54,.39), 12:(.54,.28),
        13:(.62,.75), 14:(.66,.67), 15:(.64,.79), 16:(.59,.80),
        17:(.71,.78), 18:(.76,.72), 19:(.74,.84), 20:(.69,.85),
    }
    def hand(coords):
        return [SimpleNamespace(x=coords[i][0], y=coords[i][1]) for i in range(21)]
    triple = hand(xy)
    assert recognize_thumb_index_middle(triple), 'three-finger positive example failed'
    assert not recognize_thumb_index_middle(None)
    assert not recognize_thumb_index_middle(triple[:5])
    # Victory: same index and middle, but thumb is tucked in.
    victory = dict(xy)
    victory[4] = (.35,.78)
    assert not recognize_thumb_index_middle(hand(victory)), 'Victory misdetected as triple'
    # Open palm: ring and pinky are extended.
    open_palm = dict(xy)
    open_palm.update({14:(.65,.54),15:(.66,.44),16:(.67,.35),
                      18:(.77,.61),19:(.78,.51),20:(.79,.43)})
    assert not recognize_thumb_index_middle(hand(open_palm)), 'Open palm misdetected as triple'
    # Fist: index and middle folded.
    fist = dict(xy)
    fist.update({8:(.48,.79),12:(.56,.80)})
    assert not recognize_thumb_index_middle(hand(fist)), 'Fist misdetected as triple'
    print('PASS: three-finger landmark rule vs Victory, Open Palm and Fist fixtures')



def test_four_fingers():
    """Offline geometry tests + stateful rearm tests, not real camera accuracy."""
    from types import SimpleNamespace
    # An anatomical palm-up synthetic hand, with a tucked thumb.
    xy = {
        0:(.50,.94), 1:(.39,.80), 2:(.32,.73), 3:(.34,.69), 4:(.43,.76),
        5:(.46,.73), 6:(.44,.55), 7:(.43,.45), 8:(.42,.34),
        9:(.54,.73), 10:(.54,.51), 11:(.54,.39), 12:(.54,.28),
        13:(.62,.75),14:(.65,.53), 15:(.66,.42), 16:(.67,.34),
        17:(.71,.78),18:(.77,.60),19:(.79,.49),20:(.79,.43),
    }
    def hand(coords):
        return [SimpleNamespace(x=coords[i][0], y=coords[i][1]) for i in range(21)]
    assert recognize_four_fingers(hand(xy)), 'FOUR detection failed'
    assert not recognize_four_fingers(None)
    assert not recognize_thumb_index_middle(hand(xy)), 'FOUR accidentally matched THREE'
    open_palm = dict(xy); open_palm[4] = (.14,.60)
    assert not recognize_four_fingers(hand(open_palm)), 'Open Palm confused with FOUR'
    triple = dict(xy)
    triple.update({4:(.14,.60), 14:(.66,.67),15:(.64,.79),16:(.59,.80),
                   18:(.76,.72),19:(.74,.84),20:(.69,.85)})
    assert not recognize_four_fingers(hand(triple)), 'THREE confused with FOUR'
    victory = dict(xy)
    victory.update({14:(.66,.67),15:(.64,.79),16:(.59,.80),
                    18:(.76,.72),19:(.74,.84),20:(.69,.85)})
    assert not recognize_four_fingers(hand(victory)), 'Victory confused with FOUR'

    gate = FourFingerTrigger(hold_frames=5, release_seconds=.6)
    events = [gate.update(True, i * .04) for i in range(40)]
    assert sum(events) == 1, 'Held FOUR triggered multiple times'
    assert gate.update(False, 1.61) is False
    assert gate.update(True, 1.75) is False, 'Short flicker should not rearm'
    assert gate.update(False, 1.9) is False
    assert gate.update(False, 2.55) is False
    events = [gate.update(True, 2.6 + i*.04) for i in range(5)]
    assert sum(events) == 1, 'Re-armed FOUR not triggered'
    assert not gate.update(True, 3.0), 'Repeated hold triggered twice'
    print('PASS: FOUR hand landmark geometry; held gesture = one event; release to rearm')



def test_static():
    deb = StaticGestureDebouncer()
    assert deb.update('Thumb_Up', .85, 0.0) == 'STAND'
    for i in range(1, 5):
        val = deb.update('Thumb_Up', .85, i * .03)
    assert val == 'FORWARD', val
    assert deb.update('None', 0.0, .3) == 'FORWARD'
    assert deb.update('None', 0.0, 1.0) == 'STAND'
    for i in range(5):
        val = deb.update('Closed_Fist', .9, 1.2 + i * .03)
    assert val == 'CROUCH'
    assert deb.update('Closed_Fist', .9, 1.4, enabled=False) == 'STAND'
    for label, wanted in STATIC_COMMANDS.items():
        dd = StaticGestureDebouncer()
        for i in range(5):
            got = dd.update(label, .95, 2.0 + i * .02)
        assert got == wanted, (label, got)
    test_three_fingers()
    print('PASS: six static commands, 5-frame debounce, timeout and mode gating')


def test_chinese_six():
    """Synthetic fixtures: positive shaka and negatives; not webcam validation."""
    from types import SimpleNamespace
    xy = {
        0:(.50,.94), 1:(.39,.80), 2:(.32,.73), 3:(.23,.67), 4:(.14,.60),
        5:(.46,.73), 6:(.44,.55), 7:(.43,.66), 8:(.48,.79),
        9:(.54,.73), 10:(.54,.51), 11:(.55,.66), 12:(.56,.80),
        13:(.62,.75), 14:(.66,.67), 15:(.64,.79), 16:(.59,.80),
        17:(.71,.78), 18:(.77,.60), 19:(.79,.49), 20:(.79,.43),
    }
    def hand(coords):
        return [SimpleNamespace(x=coords[i][0], y=coords[i][1])
                for i in range(21)]
    shaka = hand(xy)
    assert recognize_chinese_six(shaka), 'SIX positive fixture not recognized'
    assert not recognize_chinese_six(None)
    assert not recognize_chinese_six(shaka[:8])
    assert not recognize_four_fingers(shaka), 'SIX confused with FOUR'
    assert not recognize_thumb_index_middle(shaka), 'SIX confused with THREE'
    # Folded pinky: thumb up (not SIX).
    thumb_only = dict(xy)
    thumb_only.update({18:(.76,.72), 19:(.74,.84), 20:(.69,.85)})
    assert not recognize_chinese_six(hand(thumb_only)), 'THUMB UP confused with SIX'
    # Straight index and middle: I Love You / triple, not SIX.
    ily = dict(xy)
    ily.update({8:(.42,.34), 12:(.54,.28)})
    assert not recognize_chinese_six(hand(ily)), 'ILoveYou confused with SIX'
    # Extended all four fingers: Open Palm, not SIX.
    palm = dict(xy)
    palm.update({8:(.42,.34), 12:(.54,.28), 14:(.65,.53), 16:(.67,.34)})
    assert not recognize_chinese_six(hand(palm)), 'Open Palm confused with SIX'
    # Thumb tucked: FOUR, not SIX.
    four = dict(palm)
    four.update({4:(.43,.76)})
    assert not recognize_chinese_six(hand(four)), 'FOUR confused with SIX'
    gate = SixInterruptTrigger(hold_frames=7, hold_seconds=.22,
                               release_seconds=.6)
    events = [gate.update(True, i*.04) for i in range(100)]
    assert sum(events) == 1, 'held SIX must only interrupt once'
    gate.update(False, 4.04)
    assert not gate.update(True, 4.18), 'short release incorrectly rearmed'
    gate.update(False, 4.25)
    gate.update(False, 4.9)
    events = [gate.update(True, 5 + i*.04) for i in range(8)]
    assert sum(events) == 1, 'SIX should rearm after release'
    print('PASS: SIX geometry vs other poses; sustained hold; no repeat; release/rearm')


def test_six_mode_gating():
    for phase in ('RECORDING', 'READY', 'DONE', 'SENT'):
        assert should_interrupt_path(phase, True), phase
        assert not should_interrupt_path(phase, False), phase
    assert not should_interrupt_path('IDLE', True)
    print('PASS: SIX interrupts all dynamic modes, not regular static mode')


def test_abort_protocol():
    class FakeSocket:
        def __init__(self):
            self.sent=[]
        def sendto(self, payload, address):
            self.sent.append((payload,address))
    path=FakeSocket(); static=FakeSocket()
    send_abort_and_stand(path, ('127.0.0.1', 51683),
                         static, ('127.0.0.1', 51682))
    assert (b'ABORT',('127.0.0.1',51683)) in path.sent
    assert (b'STAND',('127.0.0.1',51682)) in static.sent
    print('PASS: SIX abort sends PATH ABORT + STATIC STAND')


def should_interrupt_path(mode, six_triggered):
    return bool(six_triggered and mode in ('RECORDING','READY','DONE','SENT'))


def send_abort_and_stand(path_sock, path_address, static_sock, static_address):
    """Existing v0.6 simulator protocol; no changes to locomotion model."""
    path_sock.sendto(b'ABORT', path_address)
    static_sock.sendto(b'STAND', static_address)


def main():
    parser = argparse.ArgumentParser(description="Index-finger dynamic gesture path recorder")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("msmf", "dshow"), default="msmf")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic offline tests")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        test_static()
        test_four_fingers()
        test_chinese_six()
        test_six_mode_gating()
        test_abort_protocol()
        return

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}\n"
            "Copy gesture_recognizer.task into the same folder as go2_gesture_camera_combined_v09.py."
        )

    # Import here so --self-test works even without the camera dependencies.
    import cv2
    import mediapipe as mp

    backend = cv2.CAP_MSMF if args.backend == "msmf" else cv2.CAP_DSHOW
    print(f"[CAMERA] Opening camera {args.camera} with {args.backend} backend...", flush=True)
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open camera {args.camera} using {args.backend}. "
            "Close the Windows Camera app or try --camera 1 / --backend dshow."
        )
    # Do NOT set CAP_PROP_FRAME_WIDTH or CAP_PROP_FRAME_HEIGHT on startup.
    # Some Windows MSMF webcam drivers hang when OpenCV requests a new resolution.
    # The tracker stores normalized landmarks, so native webcam resolution is fine.
    print("[CAMERA] Opened. Using camera default resolution.", flush=True)

    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )

    path_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    static_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = ('127.0.0.1', PATH_PORT)
    static_destination = ('127.0.0.1', STATIC_PORT)
    debouncer = StaticGestureDebouncer()
    four_trigger = FourFingerTrigger()
    six_trigger = SixInterruptTrigger()
    sent_at = -1e10
    static_last_sent = -1e10
    static_last_display = None
    hb_sent = 0.0
    started_at = time.perf_counter()
    last_timestamp_ms = -1
    mode = "IDLE"
    samples = []
    classes = []
    finished_directions = []
    simplified = []
    start_record_time = None
    last_sample_time = 0.0
    last_saved = ""
    hand_lost_since = None
    name = "Go2 Gesture Control - STATIC + DYNAMIC v0.9"
    print('[INFO] v0.9: six static gestures + four-finger path recording / confirmation.')
    print('[STATIC] Palm=STAND | Fist=CROUCH | Thumb Up=FORWARD | Thumb Down=BACKWARD')
    print('[STATIC] Victory=TURN_RIGHT | Thumb+Index+Middle=TURN_LEFT (landmark rule)')
    print('[FOUR] 1st=R start | 2nd=F finish | 3rd=C confirm (release hand between).')
    print('[SIX] Hold thumb+pinky with other fingers folded to STOP a path and return to STATIC at any stage.')
    print('[KEYS] R/F/C still work; X ABORT | Q quit')
    print("[TIP] Show one raised INDEX finger; move RIGHT then UP; keep the finger visible.")

    def begin_recording():
        nonlocal mode, last_saved, samples, classes, finished_directions
        nonlocal simplified, start_record_time, last_sample_time, hand_lost_since
        nonlocal sent_at
        # Cancel an earlier mission before preparing a new path.
        path_sock.sendto(b'ABORT', destination)
        mode = 'RECORDING'
        sent_at = -1e10
        last_saved = ''
        samples = []
        classes = []
        finished_directions = []
        simplified = []
        start_record_time = time.perf_counter()
        last_sample_time = 0.0
        hand_lost_since = None
        print('[FOUR 1/3] RECORDING started. Draw with one index finger.', flush=True)

    def finish_recording():
        nonlocal mode, last_saved, finished_directions, simplified
        if mode not in ('RECORDING', 'READY'):
            return
        finished_directions, simplified = decode_path(samples)
        if not (1 <= len(finished_directions) <= 3):
            print('[PATH] INVALID: need 1-3 clear strokes. Start again with FOUR/R.', flush=True)
            mode = 'IDLE'
            return
        duration = time.perf_counter() - start_record_time
        last_saved = str(save_recording(samples, finished_directions,
                                        simplified, duration, classes))
        mode = 'DONE'
        print('[FOUR 2/3] FINISHED: ' + ' -> '.join(finished_directions), flush=True)
        print('[PATH] Preview: ' + ' -> '.join(ROBOT_MAPPING[d] for d in finished_directions))
        print('[SAVED] ' + last_saved)
        print('[NEXT] Release hand, then show FOUR again to confirm.', flush=True)

    def confirm_path():
        nonlocal mode, sent_at
        if mode == 'DONE' and 1 <= len(finished_directions) <= 3:
            message = 'EXEC|' + ','.join(finished_directions)
            for _ in range(3):
                path_sock.sendto(message.encode('ascii'), destination)
            mode = 'SENT'
            sent_at = time.perf_counter()
            print('[FOUR 3/3] CONFIRMED: ' + message, flush=True)
        else:
            print('[PATH] Finish a valid path first; no command sent.', flush=True)

    def abort_path(source='KEYBOARD X'):
        """Stop the MuJoCo path immediately and re-enable static recognition.

        Handles RECORDING, READY, DONE, SENT, and IDLE. Sends both protocols
        so the existing simulator releases path priority and stands up.
        """
        nonlocal mode, last_saved, samples, classes, finished_directions, simplified
        nonlocal hand_lost_since, sent_at, static_last_sent, static_last_display
        send_abort_and_stand(path_sock, destination, static_sock, static_destination)
        mode = 'IDLE'
        sent_at = -1e10
        last_saved = ''  # JSON already written to disk remains available.
        samples = []
        classes = []
        finished_directions = []
        simplified = []
        hand_lost_since = None
        debouncer.candidate = None
        debouncer.count = 0
        debouncer.command = 'STAND'
        debouncer.last_confirmed = -1e10
        static_last_sent = -1e10
        static_last_display = None
        # Following the emergency SIX command, another FOUR must be deliberate.
        four_trigger.armed = False
        four_trigger.count = 0
        four_trigger.released_since = None
        print(f'[{source}] PATH ABORT + STATIC MODE restored (STAND).', flush=True)
        print('[STATIC] Six standard gestures enabled; FOUR starts a new path.', flush=True)

    try:
        with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, UI_WIDTH, UI_HEIGHT)
            while True:
                now_heartbeat = time.monotonic()
                if now_heartbeat - hb_sent >= 0.12:
                    path_sock.sendto(b'HB', destination)
                    hb_sent = now_heartbeat
                ok, frame = cap.read()
                if not ok:
                    print("[ERROR] Camera stopped delivering frames.")
                    break
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(last_timestamp_ms + 1,
                                   int((time.perf_counter() - started_at) * 1000))
                last_timestamp_ms = timestamp_ms
                result = recognizer.recognize_for_video(image, timestamp_ms)

                gesture = "None"
                confidence = 0.0
                tip = None
                is_four = False
                is_six = False
                if result.gestures and result.gestures[0]:
                    category = result.gestures[0][0]
                    gesture, confidence = category.category_name, float(category.score)
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    landmark = landmarks[8]  # INDEX_FINGER_TIP, MediaPipe #8
                    tip = (max(0., min(1., float(landmark.x))),
                           max(0., min(1., float(landmark.y))))
                    is_six = recognize_chinese_six(landmarks)
                    is_four = not is_six and recognize_four_fingers(landmarks)
                    if is_six:
                        gesture = 'Chinese_Six'
                        confidence = 1.0  # Geometric rule match, NOT ML probability.
                    elif is_four:
                        gesture = 'Four_Fingers'
                        confidence = 1.0  # Geometry match, NOT model probability.
                    elif mode == 'IDLE' and recognize_thumb_index_middle(landmarks):
                        gesture = 'Three_Fingers'
                        confidence = 1.0  # Geometry match, NOT model probability.

                now = time.perf_counter()
                # SIX is an interrupt in ANY dynamic phase (not just SENT).
                # A sustained pose triggers once. Open Palm is not an exit.
                six_pressed = six_trigger.update(is_six, now)
                if should_interrupt_path(mode, six_pressed):
                    abort_path(source='SIX / THUMB+PINKY')

                # Debounced FOUR: once held for seven frames, trigger ONE action.
                # To make another trigger, change pose / remove hand for .65sec.
                four_pressed = four_trigger.update(is_four, now)
                if four_pressed:
                    if mode == 'IDLE':
                        begin_recording()
                    elif mode in ('RECORDING', 'READY'):
                        finish_recording()
                    elif mode == 'DONE':
                        confirm_path()
                    elif mode == 'SENT':
                        print('[FOUR] Path already sent; X aborts and resets.', flush=True)

                static_enabled = (mode == 'IDLE' and not is_four and not is_six)
                static_command = debouncer.update(
                    gesture, confidence, time.monotonic(), enabled=static_enabled)
                if now - static_last_sent >= STATIC_SEND_INTERVAL or static_command != static_last_display:
                    static_sock.sendto(static_command.encode('ascii'), static_destination)
                    static_last_sent = now
                if static_command != static_last_display:
                    if static_enabled:
                        score_label = 'rule match' if gesture in ('Three_Fingers','Chinese_Six') else f'{confidence:.2f}'
                        print(f'[STATIC] {gesture} ({score_label}) => {static_command}', flush=True)
                    static_last_display = static_command

                if mode == "RECORDING":
                    if now - start_record_time > MAX_RECORD_SECONDS:
                        print("[INFO] Reached 15-second limit. Show FOUR to finish.")
                        # Finish automatically at time limit, saving finite data.
                        mode = "READY"
                    elif tip is None:
                        hand_lost_since = hand_lost_since or now
                        if hand_lost_since and now - hand_lost_since > 1.8:
                            # Avoid connecting trajectories across long tracking interruptions.
                            # Allow a brief hand release between FOUR and index pointing.
                            print("[WARN] Hand lost for 1.8 sec; recording cancelled. Show FOUR again.")
                            mode = "IDLE"
                            samples.clear()
                            classes.clear()
                    elif is_four or is_six:
                        # Do not append the tip of FOUR / SIX control hands to
                        # the path; otherwise the gesture could create a fake line.
                        hand_lost_since = None
                    else:
                        hand_lost_since = None
                        if (now - last_sample_time >= SAMPLE_INTERVAL and
                            len(samples) < MAX_POINTS and
                            (not samples or distance(tip, samples[-1]) >= MIN_POINT_DELTA)):
                            samples.append(tip)
                            classes.append(gesture)
                            last_sample_time = now
                        if len(samples) == MAX_POINTS:
                            print("[INFO] Max sample limit reached; show FOUR to finish.")
                            mode = "READY"

                preview, _ = decode_path(samples)
                current = finished_directions if mode in ("DONE", "SENT") else preview
                mapped = [ROBOT_MAPPING[d] for d in current]
                # Reuse v0.4 large dashboard. Show actual static command in IDLE.
                if mode == 'IDLE':
                    mapped = [f'STATIC: {static_command}']
                dashboard = draw_dashboard(cv2, frame, mode, gesture, confidence,
                                           tip, samples, current, mapped, last_saved)
                if mode == 'IDLE':
                    # Replace footer, keeping font legible and webcam viewport untouched.
                    cv2.rectangle(dashboard, (0, 699), (UI_WIDTH, UI_HEIGHT), (17, 22, 30), -1)
                    cv2.putText(dashboard, 'STATIC: Six gestures | FOUR fingers = RECORD path | X stop',
                                (20, 739), cv2.FONT_HERSHEY_SIMPLEX, .75,
                                (242, 246, 250), 2, cv2.LINE_AA)
                elif mode == 'SENT':
                    # A path can continue in MuJoCo. Static gestures resume on X.
                    cv2.rectangle(dashboard, (0, 699), (UI_WIDTH, UI_HEIGHT), (17, 22, 30), -1)
                    cv2.putText(dashboard, 'PATH SENT: Hold SIX (thumb+pinky) to STOP | X STOP',
                                (20, 739), cv2.FONT_HERSHEY_SIMPLEX, .75,
                                (242, 246, 250), 2, cv2.LINE_AA)
                cv2.imshow(name, dashboard)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('c'), ord('C')):
                    confirm_path()
                elif key in (ord('r'), ord('R')):
                    begin_recording()
                elif key in (ord('x'), ord('X')):
                    abort_path()
                elif key in (ord('f'), ord('F'), 13, 10) or mode == 'READY':
                    finish_recording()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by Ctrl+C")
    finally:
        try:
            for _ in range(3):
                path_sock.sendto(b'ABORT', destination)
                static_sock.sendto(b'STAND', static_destination)
        except OSError:
            pass
        path_sock.close()
        static_sock.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Camera startup interrupted (Ctrl+C).")
