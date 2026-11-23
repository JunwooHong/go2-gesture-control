# Development notes

## Project status

This repository contains an in-progress thesis prototype for vision-based gesture interaction with a Unitree Go2 robot. The system uses MediaPipe hand landmarks and gesture labels with Unitree SDK2.

- Camera preview and offline gesture recognition are implemented.
- Motion control has explicit arming and telemetry checks.
- Posture commands remain restricted following an observed abrupt descent and RPC timeouts.
- The latest posture-only gesture routing change has passed offline regression tests in the source project but still needs physical verification.
- Gesture-triggered recovery to standing has not been confirmed on hardware.

## Safety

Use the preview mode first. Enable any physical motion only in a clear area with an onsite operator and an accessible hardware stop. Do not treat this research prototype as safety-rated control software.

## Public data notes

Machine-specific network addresses, local filesystem paths, image samples showing people, and original hand landmark captures are excluded or anonymized in this public copy. Log event timing and diagnostic outcomes are preserved where practical.