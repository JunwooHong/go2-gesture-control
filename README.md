# Vision-Based Gesture Control for Unitree Go2

An undergraduate robotics project exploring a camera-based hand-gesture interface for supervised high-level control of a Unitree Go2 quadruped.

## Project overview

The prototype combines a camera, MediaPipe hand-gesture recognition, and Unitree SDK 2. It includes gesture stabilization, high-level command mapping, posture-state telemetry, fail-closed checks, and diagnostic logging.

## Repository contents

- Three Python program modules: camera gesture recognition, combined gesture/path preview, and Go2 high-level control.
- Offline regression tests in test_go2_real_gesture_move.py.
- MediaPipe model bundle in gesture_recognizer.task.
- An anonymized archive of diagnostic and offline test logs in logs_anonymized.zip.
- DEVELOPMENT_NOTES.md with current status and safety notes.
- THIRD_PARTY_NOTICES.md with model attribution and licensing information.

The example network addresses in the source are generic private-network placeholders. Set the values for your own network before connecting to a robot.

## Development status

This remains an active prototype. The latest posture-only changes are awaiting another physical-robot verification pass. Gesture-triggered recovery to standing has not yet been confirmed on the robot. The project does not claim a fully validated end-to-end gesture-control system.

## Safety and scope

Start in camera preview mode. Only enable physical motion in a clear area with an onsite operator and an accessible hardware stop. Posture commands remain restricted following an observed abrupt descent and RPC timeouts. This research prototype is not safety-rated control software.

## Privacy and media

Local paths and machine-specific network details were redacted from the public copy. The logs archive contains text diagnostics; image samples and hand-landmark captures were excluded. No video files were found in the project directory during this upload.

See DEVELOPMENT_NOTES.md for additional context.
