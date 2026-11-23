# Vision-Based Gesture Control for Unitree Go2

An undergraduate robotics project exploring how hand gestures can provide a natural, camera-based interface for high-level control of a Unitree Go2 quadruped.

## Project overview

The prototype uses a webcam, MediaPipe hand-gesture recognition, and Unitree SDK 2 to interpret static hand poses and map them to high-level robot commands. The work focuses on reliable command triggering, state-aware behavior, and supervised posture transitions.

## Current work

- Recognizes hand gestures from camera input and stabilizes detections across consecutive frames before acting.
- Maps selected gestures to high-level commands, with a dedicated posture-only mode for supervised testing.
- Monitors robot state and telemetry to check whether posture transitions have completed.
- Uses fail-closed checks, neutral-gesture re-arming, cancellation handling, and bounded command attempts to reduce unintended actions.
- Maintains diagnostic logs and offline tests to investigate state transitions and error handling.

## Development status

This is an active prototype. The latest posture-only changes are awaiting another physical-robot verification pass. In particular, gesture-triggered recovery to standing has not yet been confirmed on the robot. The project does not claim a fully validated end-to-end gesture-control system.

## Tools and technologies

- Python
- MediaPipe hand-gesture recognition and hand landmarks
- Unitree SDK 2 Python interface
- Unitree Go2 quadruped

## Safety and scope

Physical-robot tests are supervised and constrained to a small set of posture commands. Walking and path-control behavior are not described here as validated features. Machine-specific network settings, credentials, personal logs, and robot identifiers are intentionally excluded from this overview.

## Repository contents

This draft is intended as a project overview only. Source files, model assets, dependencies, and test instructions should be added only after a separate review for private configuration, safety implications, and redistribution terms.
