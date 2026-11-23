#!/usr/bin/env python3
"""Go2 front camera -> MediaPipe -> v0.9 gesture/trajectory recognition.

SAFE LAB DIAGNOSTIC ONLY. No robot motion clients, publishers, UDP control,
or actuation APIs are initialized. All commands are PRINTED, never executed.

WSL2 with Windows mirrored networking: Robot NIC=eth0 (192.168.1.3),
robot peer=192.168.1.2. Adjust based on actual machine network.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

from go2_gesture_camera_combined_v09 import (
    FourFingerTrigger,
    SixInterruptTrigger,
    StaticGestureDebouncer,
    recognize_chinese_six,
    recognize_four_fingers,
    recognize_thumb_index_middle,
    decode_path,
    distance,
    ROBOT_MAPPING,
    MAX_RECORD_SECONDS,
    MAX_POINTS,
    SAMPLE_INTERVAL,
    MIN_POINT_DELTA,
)


def configure_unitree_dds(interface: str, peer: str) -> None:
    """Patch the exact SDK2 module configuration BEFORE channel initialization.

    SDK2 Python's ChannelFactoryInitialize creates its own CycloneDDS domain
    and overwrites CYCLONEDDS_URI; setting env alone did not fix this user's RPC.
    """
    from unitree_sdk2py.core import channel

    config = channel.ChannelConfigHasInterface
    if '<ParticipantIndex>' not in config:
        if '</General>' not in config:
            raise RuntimeError('Unexpected Unitree SDK2 channel configuration; stop and inspect version.')
        config = config.replace(
            '</General>',
            '</General><Discovery><ParticipantIndex>auto</ParticipantIndex>'
            f'<Peers><Peer Address="{peer}"/></Peers></Discovery>',
            1,
        )
    # Actual interface substitution is performed by the SDK on init.
    channel.ChannelConfigHasInterface = config
    channel.ChannelFactoryInitialize(0, interface)


def recognize_closed_fist(landmarks, image_aspect=1.0):
    """Conservative 2-D fist fallback, not a trained confidence estimate.

    Require all four fingertips folded toward the wrist and a bent, tucked
    thumb. Aspect correction makes distances refer to image pixels.
    """
    if landmarks is None or len(landmarks) != 21 or not math.isfinite(image_aspect) or image_aspect <= 0:
        return False
    points = [(float(p.x), float(p.y) / image_aspect) for p in landmarks]
    if not all(math.isfinite(v) for point in points for v in point):
        return False
    if any(not 0.01 <= float(p.x) <= 0.99 or not 0.01 <= float(p.y) <= 0.99 for p in landmarks):
        return False

    def dist(a, b):
        return math.dist(points[a], points[b])

    palm = dist(5, 17)
    if palm < 0.035 or dist(0, 9) < 0.035:
        return False
    for pip, tip in ((6, 8), (10, 12), (14, 16), (18, 20)):
        if dist(0, pip) < 0.02 or dist(0, tip) > 0.90 * dist(0, pip):
            return False
    if dist(4, 5) > 0.75 * palm or dist(4, 9) > 0.90 * palm:
        return False
    a = (points[2][0] - points[3][0], points[2][1] - points[3][1])
    b = (points[4][0] - points[3][0], points[4][1] - points[3][1])
    size = math.hypot(*a) * math.hypot(*b)
    if size < 1e-8:
        return False
    cosine = max(-1.0, min(1.0, (a[0]*b[0] + a[1]*b[1]) / size))
    return math.degrees(math.acos(cosine)) < 135.0


def classify(result, mode: str, *, fist_fallback=False, image_aspect=1.0, path_enabled=True):
    """Return interpreted v0.9 label, confid., index-tip and rule flags."""
    label, confidence, tip, is_four, is_six = 'None', 0.0, None, False, False
    rule = None
    if result.gestures and result.gestures[0]:
        category = result.gestures[0][0]
        label = category.category_name
        confidence = float(category.score)
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        tip = (max(0.0, min(1.0, float(lm[8].x))),
               max(0.0, min(1.0, float(lm[8].y))))
        is_six = recognize_chinese_six(lm)
        is_four = path_enabled and not is_six and recognize_four_fingers(lm)
        if is_six:
            label, confidence, rule = 'Chinese_Six', 1.0, 'landmark rule'
        elif is_four:
            label, confidence, rule = 'Four_Fingers', 1.0, 'landmark rule'
        elif mode == 'IDLE' and recognize_thumb_index_middle(lm):
            label, confidence, rule = 'Three_Fingers', 1.0, 'landmark rule'
        elif (fist_fallback and mode == 'IDLE' and label in ('None', 'Unrecognized', 'Closed_Fist')
              and recognize_closed_fist(lm, image_aspect)):
            label, confidence, rule = 'Closed_Fist', 1.0, 'fist landmark rule (not model confidence)'
    return label, confidence, tip, is_four, is_six, rule


class ReadOnlyGestureSession:
    def __init__(self, preview_only=True, path_enabled=True):
        self.preview_only = preview_only
        self.path_enabled = path_enabled
        self.db = StaticGestureDebouncer()
        self.four = FourFingerTrigger()
        self.six = SixInterruptTrigger()
        self.mode = 'IDLE'
        self.points = []
        self.planned = []
        self.record_started = 0.0
        self.last_sample = -1e10
        self.last_command = None
        self.last_raw = None

    def consume(self, label, score, tip, is_four, is_six, rule, now):
        is_four = is_four and self.path_enabled
        if self.six.update(is_six, now):
            self.mode = 'IDLE'
            self.points.clear()
            self.planned.clear()
            self.db = StaticGestureDebouncer()
            print('[SIX] CANCEL / RETURN TO STATIC.', flush=True)

        if self.path_enabled and self.four.update(is_four, now):
            if self.mode == 'IDLE':
                self.mode = 'RECORDING'
                self.points = []
                self.planned = []
                self.record_started = now
                self.last_sample = -1e10
                print('[FOUR 1/3] START: draw with INDEX fingertip.', flush=True)
            elif self.mode == 'RECORDING':
                self._finish()
            elif self.mode == 'DONE':
                self.mode = 'PREVIEW'
                print('[FOUR 3/3] PATH PREVIEW ONLY:',
                      [ROBOT_MAPPING.get(x, x) for x in self.planned], flush=True)
                print('[PATH] Preview only. SIX to reset.', flush=True)

        if self.mode == 'RECORDING':
            if now - self.record_started >= MAX_RECORD_SECONDS:
                self._finish()
            elif tip is not None and not (is_four or is_six) and len(self.points) < MAX_POINTS:
                if (now - self.last_sample >= SAMPLE_INTERVAL and
                        (not self.points or distance(tip, self.points[-1]) >= MIN_POINT_DELTA)):
                    self.points.append(tip)
                    self.last_sample = now

        enabled = self.mode == 'IDLE' and not (is_four or is_six)
        cmd = self.db.update(label, score, now, enabled=enabled)
        if cmd != self.last_command:
            source = rule or f'MediaPipe confidence {score:.2f}'
            output = 'STATIC PREVIEW' if self.preview_only else 'STATIC REQUEST'
            print(f'[{output}] {label} ({source}) -> {cmd}', flush=True)
            self.last_command = cmd
        if label != self.last_raw:
            if label not in ('None', 'Unrecognized'):
                print(f'[DETECTED] {label} ({rule or f"score {score:.2f}"})', flush=True)
            self.last_raw = label

    def _finish(self):
        self.planned, _ = decode_path(self.points)
        if 1 <= len(self.planned) <= 3:
            self.mode = 'DONE'
            print('[FOUR 2/3] FINISH: segments:', self.planned, flush=True)
            print('[NEXT] Release FOUR for 0.65s, then FOUR again to preview.', flush=True)
        else:
            self.mode = 'IDLE'
            print(f'[PATH INVALID] {len(self.points)} samples; decoded={self.planned}. Retry FOUR.', flush=True)


def self_test():
    from go2_gesture_camera_combined_v09 import (
        self_test as path_test, test_static, test_four_fingers,
        test_chinese_six, test_six_mode_gating,
    )
    path_test(); test_static(); test_four_fingers()
    test_chinese_six(); test_six_mode_gating()
    session = ReadOnlyGestureSession()
    assert session.mode == 'IDLE'
    for i in range(7):
        session.consume('Four_Fingers', 1.0, (.25, .4), True, False, 'landmark rule', i * .1)
    assert session.mode == 'RECORDING', session.mode
    print('PASS: read-only Go2 camera adapter, gesture state machine and v0.9 synthetic tests')


def main():
    parser = argparse.ArgumentParser(description='READ-ONLY Go2 front-camera gesture test (no motor control)')
    parser.add_argument('--interface', default='eth0', help='Robot-facing Linux interface; default eth0')
    parser.add_argument('--peer', default='192.168.1.2', help='Robot DDS peer IP')
    parser.add_argument('--model', default='gesture_recognizer.task')
    parser.add_argument('--width', type=int, default=960, help='Inference frame width; default 960')
    parser.add_argument('--fps', type=float, default=8, help='Target processing FPS; default 8')
    parser.add_argument('--max-seconds', type=float, default=60, help='Stop automatically after seconds; default 60')
    parser.add_argument('--save-first-frame', action='store_true', help='Save first Go2 image to go2_camera_first_frame.jpg')
    parser.add_argument('--self-test', action='store_true', help='No robot / MediaPipe needed')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.width < 320 or args.fps <= 0 or args.max_seconds <= 0:
        parser.error('--width >= 320, --fps > 0 and --max-seconds > 0 are required')
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f'[ERROR] Model not found: {model}')
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        from unitree_sdk2py.go2.video.video_client import VideoClient
    except ImportError as ex:
        raise SystemExit(f'[ERROR] Missing module: {ex}. Install MediaPipe in ~/go2_project/.venv.') from ex

    configure_unitree_dds(args.interface, args.peer)
    video = VideoClient()
    video.SetTimeout(4.0)
    video.Init()
    code, jpeg = video.GetImageSample()
    if code != 0 or not jpeg:
        raise SystemExit(f'[CAMERA ERROR] SDK2 code={code}; retry current DDS settings and connection.')
    frame = cv2.imdecode(np.frombuffer(bytes(jpeg), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit('[CAMERA ERROR] Invalid JPEG frame')
    print(f'[CONNECTED] Go2 camera {frame.shape[1]}x{frame.shape[0]} via {args.interface}', flush=True)
    print('[SAFE] READ ONLY: NO movement, no ROS command publishing, no UDP to robot.', flush=True)
    print('[GESTURES] 6 static commands, FOUR x3 path preview, SIX cancels path.', flush=True)
    if args.save_first_frame:
        image_path = Path('go2_camera_first_frame.jpg').resolve()
        image_path.write_bytes(bytes(jpeg))
        print('[IMAGE] First frame saved:', image_path, flush=True)

    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    start = time.monotonic()
    last_ms, count, failures = -1, 0, 0
    session = ReadOnlyGestureSession()
    with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
        try:
            while time.monotonic() - start < args.max_seconds:
                frame_start = time.monotonic()
                if count > 0:
                    code, jpeg = video.GetImageSample()
                    if code != 0 or not jpeg:
                        failures += 1
                        print(f'[CAMERA WARN] SDK code={code}, retry {failures}/3', flush=True)
                        if failures >= 3:
                            break
                        time.sleep(.2)
                        continue
                    failures = 0
                    frame = cv2.imdecode(np.frombuffer(bytes(jpeg), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        print('[CAMERA WARN] JPEG decode failed', flush=True)
                        continue
                frame = cv2.flip(frame, 1)  # same mirrored gesture convention as Windows v0.9
                h, w = frame.shape[:2]
                infer = cv2.resize(frame, (args.width, max(1, round(h * args.width / w))),
                                   interpolation=cv2.INTER_AREA) if w > args.width else frame
                rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now = time.monotonic()
                ms = max(last_ms + 1, int((now - start) * 1000))
                last_ms = ms
                result = recognizer.recognize_for_video(image, ms)
                label, score, tip, is_four, is_six, rule = classify(result, session.mode)
                session.consume(label, score, tip, is_four, is_six, rule, now)
                count += 1
                if count % 80 == 0:
                    print(f'[INFO] {count} frames processed, mode={session.mode}', flush=True)
                delay = max(0, 1.0 / args.fps - (time.monotonic() - frame_start))
                if delay:
                    time.sleep(delay)
        except KeyboardInterrupt:
            print('[EXIT] Stopped by Ctrl+C', flush=True)
    print(f'[DONE] {count} Go2 frames processed. No movement commands sent.', flush=True)


if __name__ == '__main__':
    main()
