#!/usr/bin/env python3
"""Go2 camera gestures mapped to SDK2 high-level actions.

Run as a standalone WSL process. Default operation is camera preview.
Physical control requires --enable-motion, healthy telemetry and onsite arming.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import ipaddress
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

from go2_camera_gesture_wsl_v10 import configure_unitree_dds, classify, ReadOnlyGestureSession
from go2_gesture_camera_combined_v09 import STATIC_COMMANDS, STATIC_CONFIDENCE

MOTION = {
    'FORWARD': (0.20, 0.0, 0.0),
    'BACKWARD': (-0.16, 0.0, 0.0),
    'TURN_LEFT': (0.0, 0.0, 0.26),
    'TURN_RIGHT': (0.0, 0.0, -0.26),
}
PULSE_S = 1.20
CONTROL_PERIOD_S = 0.05
RPC_TIMEOUT_S = 0.25
FRAME_TIMEOUT_S = 0.60
STATE_TIMEOUT_S = 0.70
NEUTRAL_REARM_S = 0.40
MAX_ROLL_PITCH = 0.45
POSTURE_TIMEOUT_S = 8.0
POSTURE_RPC_TIMEOUT_S = 1.0  # Response wait only; does not change native action speed.
POSTURE_SETTLE_S = 0.5
# Incident 2026-11-18: operator reported abrupt/jumping motion after Sit;
# Sit and both subsequent StopMove RPCs timed out (3104). These native
# methods expose no speed argument. Keep actuation blocked until a supported
# controlled-descent API and its stopping behavior have been verified.
# After onsite confirmation of native prone/stand and hardware stop, the
# separate --native-posture-test permits one StandDown/StandUp pair only.
POSTURE_MOTION_BLOCKED_REASON = (
    'Posture actuation disabled after the Sit incident and RPC timeouts; '
    'no verified slow-posture API for this firmware')
# Official Go2 V2 interface: error_code carries an FSM id, not a fault code.
# https://support.unitree.com/home/zh/developer/Motion_Services_Interface_V2.0
V2_STANDING = frozenset((100, 1013))  # FreeWalk, balance stand
V2_UPRIGHT = V2_STANDING | {1002}  # locked stand is a posture, not permission to walk
V2_LOWERED = frozenset((1004, 2006))  # documented crouched/lowered states
V2_SUPPORTED = V2_UPRIGHT | V2_LOWERED
# Only the active fault bits defined in Fault_service. 0x01/0x20 are
# explicitly deprecated; unassigned bits are logged, not invented as faults.
LOWSTATE_FAULT_MASK = 0x40 | 0x80 | 0x10 | 0x02


def sport_code_fault(code, profile):
    if profile == 'mcf_v2':
        return '' if code in V2_SUPPORTED else f'unsupported V2 sport FSM={code}'
    return '' if code == 0 else f'unresolved robot error_code={code}'


class HardwareFaultMonitor:
    """Separate documented lowstate faults from the V2 sport FSM.

    rt/errorlist is an additional source when published, not assumed present.
    Lowstate freshness and its communication/motor fault fields are required.
    https://support.unitree.com/home/zh/developer/Fault_service
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.received = None
        self.error = ''
        self.errorlist_received = False
        self.bit_flag = None
        self.motor_faults = ()

    def lowstate_callback(self, msg):
        error = ''
        try:
            if len(msg.motor_state) < 12:
                raise ValueError('missing leg motor states')
            bit_flag = int(msg.bit_flag)
            if not 0 <= bit_flag <= 255:
                raise ValueError('bit_flag outside uint8 range')
            active_bits = bit_flag & LOWSTATE_FAULT_MASK
            if active_bits:
                error = f'lowstate communication fault bits=0x{active_bits:x} (raw=0x{bit_flag:x})'
            for index, motor in enumerate(msg.motor_state[:12]):
                if motor.reserve[0]:
                    error = error or f'motor {index} fault=0x{int(motor.reserve[0]):x}'
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            error = f'invalid lowstate fault data: {exc}'
        with self.lock:
            self.received = time.monotonic()
            self.error = self.error or error
            if not error:
                self.bit_flag = bit_flag
                self.motor_faults = tuple(int(m.reserve[0]) for m in msg.motor_state[:12])

    def status(self):
        with self.lock:
            return dict(bit_flag=self.bit_flag, motor_faults=self.motor_faults,
                        errorlist_received=self.errorlist_received)

    def errorlist_callback(self, msg):
        try:
            data = json.loads(msg.data)
            errors = data['errors']
            if not isinstance(errors, list):
                raise ValueError('errors must be a list')
            error = f'active rt/errorlist faults: {errors}' if errors else ''
        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            error = f'invalid rt/errorlist: {exc}'
        with self.lock:
            self.errorlist_received = True
            self.error = self.error or error

    def fault(self, now):
        with self.lock:
            # The callback may have updated received after the caller sampled
            # now. Read the clock under the same lock as the mutable state.
            now = max(now, time.monotonic())
            if self.error:
                return self.error
            if self.received is None or not 0 <= now - self.received <= STATE_TIMEOUT_S:
                return 'lowstate fault telemetry unavailable or stale'
        return ''


class EventLog:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('x', encoding='utf-8')
        self.lock = threading.Lock()
        print(f'[LOG] {path.resolve()}', flush=True)

    def __call__(self, event, **fields):
        row = {'time': datetime.now().astimezone().isoformat(),
               'monotonic': time.monotonic(), 'event': event, **fields}
        with self.lock:
            self.file.write(json.dumps(row, allow_nan=False) + '\n')
            self.file.flush()
        if event != 'state':
            print(f'[{event.upper()}] {json.dumps(fields)}', flush=True)

    def close(self):
        self.file.close()


@dataclass(frozen=True)
class RobotState:
    received: float
    error_code: int
    mode: int
    gait_type: int
    position: tuple
    velocity: tuple
    rpy: tuple
    body_height: float
    stamp: tuple

    @classmethod
    def from_message(cls, msg, now):
        state = cls(now, int(msg.error_code), int(msg.mode), int(msg.gait_type),
                    tuple(msg.position), tuple(msg.velocity), tuple(msg.imu_state.rpy),
                    float(msg.body_height), (int(msg.stamp.sec), int(msg.stamp.nanosec)))
        vectors = (state.position, state.velocity, state.rpy)
        if any(len(vector) != 3 for vector in vectors):
            raise ValueError('incomplete position, velocity or IMU vector')
        if not all(math.isfinite(x) for vector in vectors for x in vector):
            raise ValueError('non-finite telemetry')
        if not math.isfinite(state.body_height):
            raise ValueError('non-finite body height')
        return state

    def fault(self, now, profile='legacy', posture_codes=()):
        if not 0 <= now - self.received <= STATE_TIMEOUT_S:
            return 'sport state timed out'
        reason = sport_code_fault(self.error_code, profile)
        if reason and self.error_code not in posture_codes:
            return reason
        if any(abs(value) > MAX_ROLL_PITCH for value in self.rpy[:2]):
            return 'roll/pitch exceeds limit'
        return ''

    def standing(self, profile):
        return self.error_code in V2_UPRIGHT if profile == 'mcf_v2' else self.mode in (0, 1)

    def lowered(self, profile):
        return self.error_code in V2_LOWERED if profile == 'mcf_v2' else self.mode == 5

    def movable(self, profile):
        return self.error_code in V2_STANDING if profile == 'mcf_v2' else self.mode in (0, 1, 3)


class MotionPolicy:
    """Require neutral between bounded movement runs and posture requests."""
    def __init__(self, motion_seconds=PULSE_S, profile='legacy'):
        self.motion_seconds = motion_seconds
        self.profile = profile
        self.reset('waiting for neutral gesture')

    def reset(self, reason):
        self.blocked = True
        self.neutral_since = None
        self.active = None
        self.active_since = None
        self.reason = reason

    def decide(self, desired, now, frame_age, state_age, state_error=0, unsafe_tilt=False,
               posture_codes=()):
        if (not 0 <= frame_age <= FRAME_TIMEOUT_S
                or not 0 <= state_age <= STATE_TIMEOUT_S
                or (sport_code_fault(state_error, self.profile)
                    and (state_error not in posture_codes or desired in MOTION)) or unsafe_tilt):
            self.reset('camera/state watchdog, robot error or tilt')
            return None
        if desired in (None, 'STOP', 'CANCEL'):
            if self.active is not None:
                self.reset('gesture released')
            if self.neutral_since is None:
                self.neutral_since = now
            if now - self.neutral_since >= NEUTRAL_REARM_S:
                self.blocked = False
            self.reason = 'neutral / waiting for fresh gesture'
            return None
        self.neutral_since = None
        if self.blocked:
            self.reason = 'neutral required before next action'
            return None
        if desired in ('LIE_DOWN', 'STAND_UP'):
            self.reset('posture requested once')
            return desired
        if desired not in MOTION:
            self.reset('unsupported command')
            return None
        if self.active is None:
            self.active, self.active_since = desired, now
        if desired != self.active:
            self.reset('gesture changed; stop and rearm')
            return None
        if now - self.active_since >= self.motion_seconds:
            self.reset('motion duration reached; release and rearm')
            return None
        self.reason = 'bounded motion permitted'
        return self.active


class LeaseObserver:
    """Observe this process's SDK lease traffic without modifying the SDK.

    GetLeaseId alone can remain nonzero following a renewal timeout. Match our
    requests to responses and expire validity using the acknowledged term.
    """
    def __init__(self, context_name, log):
        self.context_name = context_name
        self.log = log
        self.lock = threading.Lock()
        self.pending = {}
        self.lease_id = 0
        self.term = 0.0
        self.valid_until = 0.0
        self.error = ''

    def request_callback(self, msg):
        now = time.monotonic()
        api = msg.header.identity.api_id
        try:
            parameter = json.loads(msg.parameter)
        except (TypeError, ValueError):
            return
        if not isinstance(parameter, dict):
            return
        with self.lock:
            ours = (api == 101 and parameter.get('name') == self.context_name)
            ours = ours or (api == 102 and self.lease_id != 0
                            and msg.header.lease.id == self.lease_id)
            if ours:
                self.pending = {key: value for key, value in self.pending.items()
                                if now - value[1] < 5.0}
                self.pending[msg.header.identity.id] = (api, now)

    def response_callback(self, msg):
        with self.lock:
            request = self.pending.pop(msg.header.identity.id, None)
            if request is None:
                return
            api, sent = request
            code = msg.header.status.code
            if api != msg.header.identity.api_id:
                self.error = 'lease response API mismatch'
            elif code != 0:
                self.error = f'lease API {api} returned {code}'
            else:
                try:
                    if api == 101:
                        data = json.loads(msg.data)
                        self.lease_id = int(data['id'])
                        self.term = float(data['term']) / 1_000_000
                    if self.lease_id <= 0 or not math.isfinite(self.term) or self.term <= 0:
                        raise ValueError('invalid lease id or term')
                    # Request-send time gives a conservative expiry bound.
                    self.valid_until = sent + self.term
                except (TypeError, ValueError, KeyError):
                    self.error = 'malformed lease response'
                    self.lease_id, self.term = 0, 0.0
            self.log('lease_response', api=api, code=code, lease_id=self.lease_id,
                     term_s=self.term, error=self.error)

    def fault(self, now, sdk_id):
        with self.lock:
            if self.error:
                return self.error
            if self.lease_id <= 0 or sdk_id != self.lease_id:
                return 'lease acquisition has not been confirmed'
            if now >= self.valid_until:
                return 'lease renewal acknowledgement expired'
            return ''


class RobotMotionBridge:
    """Single command writer, independent of camera RPC and inference.

    SDK calls have bounded timeouts. Stop and native posture commands are
    serialized with Move so a delayed action cannot race a cleanup StopMove.
    """
    def __init__(self, client=None, enable_motion=False, motion_seconds=PULSE_S,
                 log=None, lease=None, profile='legacy', hardware_faults=None,
                 posture_only=False, native_posture_test=False, recovery_stand_test=False):
        self.client = client
        self.enable_motion = enable_motion
        self.log = log or (lambda event, **fields: None)
        self.lease = lease
        self.profile = profile
        self.hardware_faults = hardware_faults
        self.posture_only = posture_only
        self.native_posture_test = native_posture_test
        self.recovery_stand_test = recovery_stand_test
        self.observed_descent = False
        self.recovery_left_damping = False
        self.posture_attempts = set()
        self.posture_settle_since = None
        self.lock = threading.Lock()
        self.state = None
        self.state_count = 0
        self.bad_state = ''
        self.desired = None
        self.last_frame = 0.0
        self.policy = MotionPolicy(motion_seconds, profile)
        self.closed = threading.Event()
        self.wake = threading.Event()
        self.thread = None
        self.failed = False
        self.was_moving = False
        self.send_count = 0
        self.last_move_send = -math.inf
        self.stop_count = 0
        self.last_desired = None
        self.last_control_status = None
        self.prone = False
        self.posture_pending = None
        self.last_state_log = 0.0
        self.session_deadline = math.inf

    def state_callback(self, msg):
        now = time.monotonic()
        try:
            state = RobotState.from_message(msg, now)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            with self.lock:
                self.bad_state = str(exc)
            self.wake.set()
            return
        with self.lock:
            self.state = state
            self.state_count += 1
            if self.recovery_stand_test and self.posture_pending:
                command, since = self.posture_pending
                if 0 <= now - since <= POSTURE_TIMEOUT_S:
                    if command == 'LIE_DOWN' and state.error_code == 2006:
                        self.observed_descent = True
                    if command == 'STAND_UP' and state.error_code != 1001:
                        self.recovery_left_damping = True
            # Latch even a single invalid/error frame during an armed run.
            reason = state.fault(now, self.profile, self._posture_codes(state, now))
            if self.enable_motion and reason:
                self.bad_state = self.bad_state or reason

    def _posture_codes(self, state, now):
        """Context-only allowances from this robot's recorded supervised sequence.

        2004 is observed during remote rise, not named in the official FSM table.
        Neither code enters V2_SUPPORTED or grants permission for Move.
        Caller holds self.lock when inspecting mutable callback state.
        """
        if (not self.recovery_stand_test or not self.native_posture_test
                or self.posture_block() or not self.observed_descent):
            return ()
        pending = self.posture_pending
        if pending and not 0 <= now - pending[1] <= POSTURE_TIMEOUT_S:
            return ()
        if state.error_code == 1001 and 0.04 <= state.body_height <= 0.13:
            if pending and pending[0] == 'LIE_DOWN':
                return (1001,)
            if self.prone and pending is None:
                return (1001,)
            if pending and pending[0] == 'STAND_UP' and not self.recovery_left_damping:
                return (1001,)
        if (state.error_code == 2004 and pending and pending[0] == 'STAND_UP'
                and 0.04 <= state.body_height <= 0.45):
            return (2004,)
        return ()

    def snapshot(self):
        with self.lock:
            return self.state

    def remote_callback(self, msg):
        if not self.enable_motion:
            return
        axes = (msg.lx, msg.ly, msg.rx, msg.ry)
        if msg.keys or any(not math.isfinite(value) or abs(value) > 0.15 for value in axes):
            with self.lock:
                self.bad_state = 'hardware remote input; automatic control cancelled'
            self.wake.set()

    def state_fault(self, now):
        with self.lock:
            state, bad_state = self.state, self.bad_state
            now = max(now, time.monotonic())
            reason = (state.fault(now, self.profile, self._posture_codes(state, now))
                      if state is not None else 'sport state unavailable')
        if bad_state:
            return bad_state
        if state is None:
            return 'sport state unavailable'
        return reason or (
            self.hardware_faults.fault(now) if self.hardware_faults is not None else '')

    def camera_command(self, desired, captured):
        if self.posture_only and desired in MOTION:
            desired = None
        with self.lock:
            self.desired = desired
            self.last_frame = captured
        self.wake.set()
        return desired

    def prepare_starting_posture(self):
        """Read posture at arming, including a robot already in native StandDown."""
        reason = self.state_fault(time.monotonic())
        if reason:
            raise RuntimeError(reason)
        state = self.snapshot()
        if self.recovery_stand_test and not state.standing(self.profile):
            raise RuntimeError('recovery gesture test must start standing; use verified remote first')
        if not (state.standing(self.profile) or state.lowered(self.profile)):
            raise RuntimeError('arm only while physically standing or in native StandDown')
        if state.lowered(self.profile) and self.posture_block():
            raise RuntimeError(POSTURE_MOTION_BLOCKED_REASON)
        self.prone = state.lowered(self.profile)
        self.policy.reset('waiting for neutral gesture')
        self.log('starting_posture', posture='PRONE' if self.prone else 'STANDING',
                 sport_code=state.error_code, mode=state.mode)

    def fail(self, reason):
        if not self.failed:
            self.log('abort', reason=reason)
        self.failed = True
        self.policy.reset(reason)

    def posture_block(self):
        if self.native_posture_test:
            if not self.posture_only or self.profile != 'mcf_v2' or self.lease is not None:
                return 'native posture test requires posture-only MCF control without Lease'
            return ''
        return POSTURE_MOTION_BLOCKED_REASON

    def _stop(self, reason='stop requested'):
        # This robot returned -1 for StopMove in damping. Only an already
        # confirmed, stationary prone state needs no locomotion-stop RPC.
        # Never turn a nonzero RPC result into success or skip an in-flight stop.
        with self.lock:
            state = self.state
            now = time.monotonic()
            already_damped = (state is not None and self.prone and self.posture_pending is None
                              and state.error_code == 1001
                              and 1001 in self._posture_codes(state, now)
                              and not state.fault(now, self.profile, (1001,))
                              and math.sqrt(sum(v*v for v in state.velocity)) <= 0.08)
        if already_damped:
            self.log('stop_not_sent', reason=reason, detail='confirmed stationary damping; hardware stop remains available')
            self.was_moving = False
            return
        if self.enable_motion and self.client is not None:
            try:
                code = self.client.StopMove()
                self.stop_count += 1
                self.log('stop', code=code, reason=reason)
                if code != 0:
                    self.fail(f'StopMove failed with {code}; use hardware stop')
            except Exception as exc:
                self.fail(f'StopMove exception {exc!r}; use hardware stop')
        self.was_moving = False

    def _posture(self, command, now):
        self._stop('before native posture action')
        block = self.posture_block()
        if block:
            self.fail(block)
            return
        if command not in ('LIE_DOWN', 'STAND_UP'):
            self.fail('unsupported native posture request')
            return
        if self.native_posture_test and command in self.posture_attempts:
            self.fail('native posture test permits each action once; no retry')
            return
        if time.monotonic() >= self.session_deadline:
            self.closed.set()
            return
        if self.failed or self.closed.is_set() or self.state_fault(time.monotonic()):
            return
        with self.lock:
            desired, captured = self.desired, self.last_frame
        expected = 'LIE_DOWN' if command == 'LIE_DOWN' else 'PALM'
        if desired != expected or not 0 <= time.monotonic() - captured <= FRAME_TIMEOUT_S:
            return
        if self.lease is not None and self.lease.fault(time.monotonic(), self.client.GetLeaseId()):
            self.fail('lease lost before posture request')
            return
        state = self.snapshot()
        valid = state.standing(self.profile) if command == 'LIE_DOWN' else self._prone_target(state)
        if not valid:
            self.fail('posture changed during StopMove; native action cancelled')
            return
        if self.native_posture_test and math.sqrt(sum(v*v for v in state.velocity)) > 0.08:
            self.fail('native posture test requires stationary telemetry')
            return
        api = ('StandDown' if command == 'LIE_DOWN' else
               'RecoveryStand' if self.recovery_stand_test else 'StandUp')
        method = getattr(self.client, api)
        self.posture_attempts.add(command)
        requested_at = time.monotonic()
        if self.recovery_stand_test:
            # DDS can publish the transition before the RPC response arrives.
            with self.lock:
                self.posture_pending = (command, requested_at)
                self.posture_settle_since = None
                if command == 'LIE_DOWN':
                    self.observed_descent = False
                else:
                    self.recovery_left_damping = False
        if self.native_posture_test:
            self.client.SetTimeout(POSTURE_RPC_TIMEOUT_S)
        try:
            code = method()
        finally:
            if self.native_posture_test:
                self.client.SetTimeout(RPC_TIMEOUT_S)
        self.log('native_action', command=command, api=api,
                 code=code, rpc_elapsed_s=time.monotonic() - requested_at)
        if code != 0:
            self.fail(f'{command} failed with {code}; posture is uncertain')
            return
        if not self.recovery_stand_test:
            self.posture_pending = (command, time.monotonic())
        self.posture_settle_since = None

    def _prone_target(self, state):
        if not self.recovery_stand_test:
            return state.lowered(self.profile)
        with self.lock:
            return (state.error_code == 1001
                    and 1001 in self._posture_codes(state, time.monotonic()))

    def step(self, now):
        if now >= self.session_deadline:
            self.closed.set()
            return
        state = self.snapshot()
        if state is not None and now - self.last_state_log >= 0.25:
            self.log('state', **asdict(state))
            self.last_state_log = now
        if not self.enable_motion or self.client is None:
            return
        with self.lock:
            desired, captured = self.desired, self.last_frame
            # Both DDS and camera callbacks can run after step(now) began.
            # Compare their timestamps against a clock sampled after the read.
            now = max(now, time.monotonic())
        if now >= self.session_deadline:
            self.closed.set()
            return
        fault = self.state_fault(now)
        if self.lease is not None:
            fault = fault or self.lease.fault(now, self.client.GetLeaseId())
        if fault:
            self.fail(fault)
        if self.failed or self.closed.is_set():
            return
        frame_age = now - captured
        if not 0 <= frame_age <= FRAME_TIMEOUT_S:
            desired = None
        changed = desired != self.last_desired
        self.last_desired = desired

        if self.recovery_stand_test and desired == 'CANCEL':
            self._stop('Chinese Six cancelled posture test')
            self.fail('posture test cancelled; no automatic stand-up')
            return

        if self.posture_pending is not None:
            command, since = self.posture_pending
            confirmed = self._prone_target(state) if command == 'LIE_DOWN' else state.standing(self.profile)
            if self.recovery_stand_test and command == 'STAND_UP':
                confirmed = state.error_code in V2_STANDING and 0.25 <= state.body_height <= 0.45
            if self.native_posture_test:
                settled = (confirmed and state.received > since
                           and math.sqrt(sum(v*v for v in state.velocity)) <= 0.08)
                if not settled:
                    self.posture_settle_since = None
                elif self.posture_settle_since is None:
                    self.posture_settle_since = now
                confirmed = (settled and self.posture_settle_since is not None
                             and now - self.posture_settle_since >= POSTURE_SETTLE_S)
            expected = 'LIE_DOWN' if command == 'LIE_DOWN' else 'PALM'
            if changed and desired != expected:
                self._stop('posture gesture cancelled or lost')
                self.fail('posture interrupted; confirm physical posture with remote')
            elif state.received > since and confirmed:
                with self.lock:
                    self.prone = command == 'LIE_DOWN'
                    self.posture_pending = None
                self.policy.reset('posture confirmed; neutral required')
                self.log('posture_state', command=command, mode=state.mode,
                         sport_code=state.error_code, profile=self.profile)
                if self.native_posture_test and command == 'STAND_UP':
                    self.log('posture_test_complete', reason='standing telemetry confirmed; verify physical result onsite')
                    self.closed.set()
            elif now - since >= POSTURE_TIMEOUT_S:
                self.fail('posture completion unconfirmed in telemetry')
            return

        if desired == 'PALM':
            desired = 'STAND_UP' if self.prone else 'STOP'
        if self.prone and desired in MOTION:
            desired = None
        with self.lock:
            posture_codes = self._posture_codes(state, now)
        selected = self.policy.decide(desired, now, frame_age,
                                      now - state.received, state.error_code,
                                      posture_codes=posture_codes)
        status = (selected, self.policy.blocked, self.policy.reason)
        if status != self.last_control_status:
            self.log('control', selected=selected, needs_release=self.policy.blocked,
                     reason=self.policy.reason)
            self.last_control_status = status
        if selected in MOTION:
            if not state.movable(self.profile):
                self.fail(f'movement requires standing/locomotion state; mode={state.mode}, sport_code={state.error_code}')
                return
            if now - self.last_move_send < CONTROL_PERIOD_S:
                return
            # A send can fail after reaching the robot. Cleanup must still stop.
            self.was_moving = True
            self.last_move_send = now
            code = self.client.Move(*MOTION[selected])
            self.send_count += 1
            self.log('move_send', command=selected, velocity=MOTION[selected], code=code)
            if code != 0:
                self.fail(f'Move transport error {code}')
        elif selected in ('LIE_DOWN', 'STAND_UP'):
            if selected == 'LIE_DOWN' and self.prone:
                return
            if selected == 'LIE_DOWN' and not state.standing(self.profile):
                self.fail('StandDown requires a standing state')
                return
            if selected == 'STAND_UP' and not self._prone_target(state):
                self.fail('StandUp requires a lowered state')
                return
            self._posture(selected, now)
        elif self.was_moving or (changed and desired in ('STOP', 'CANCEL', None)):
            self._stop(self.policy.reason)

    def _loop(self):
        try:
            while not self.closed.is_set() and not self.failed:
                self.wake.clear()
                self.step(time.monotonic())
                self.wake.wait(CONTROL_PERIOD_S)
        except Exception as exc:
            self.fail(f'control exception {exc!r}')
        finally:
            self.closed.set()
            reason = ('controller aborted: ' + self.policy.reason if self.failed else
                      'session duration reached' if time.monotonic() >= self.session_deadline else
                      'controller shutdown')
            self._stop(reason)
            self._stop(reason)

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True, name='Go2MotionWatchdog')
        self.thread.start()

    def shutdown(self):
        self.closed.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
            if self.thread.is_alive():
                self.fail('SDK command thread did not exit; use hardware stop')
        elif self.enable_motion:
            self._stop()


def stable_command(session, label, score, is_four, is_six):
    if is_six:
        return 'CANCEL'
    if session.mode != 'IDLE' or is_four:
        return None
    expected = STATIC_COMMANDS.get(label) if score >= STATIC_CONFIDENCE else None
    if (expected is None or session.db.command != expected
            or session.db.candidate != expected or session.db.count < session.db.frames):
        return None
    return {'CROUCH': 'LIE_DOWN', 'STAND': 'PALM'}.get(expected, expected)


def resolve_interface(interface, peer):
    if interface != 'auto':
        return interface
    result = subprocess.run(['ip', '-j', 'route', 'get', peer], check=True,
                            capture_output=True, text=True, timeout=3)
    routes = json.loads(result.stdout)
    if not routes or not routes[0].get('dev'):
        raise RuntimeError(f'no interface found for {peer}; pass --interface')
    return routes[0]['dev']


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--diagnose', action='store_true', help='Read services and both state topics')
    parser.add_argument('--interface', default='auto')
    parser.add_argument('--peer', default='192.168.1.2')
    parser.add_argument('--model', default=str(Path(__file__).with_name('gesture_recognizer.task')))
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--fps', type=float, default=8)
    parser.add_argument('--max-seconds', type=float, default=35)
    parser.add_argument('--motion-seconds', type=float, default=PULSE_S,
                        help='Maximum continuous hold, 0.2..3 seconds; default 1.2')
    parser.add_argument('--enable-motion', action='store_true')
    parser.add_argument('--posture-only', action='store_true',
                        help='Only fist=StandDown and palm=StandUp/Stop; ignore walking/turning gestures')
    parser.add_argument('--native-posture-test', action='store_true',
                        help='Supervised one-pair StandDown/StandUp test; native speed, no retries')
    parser.add_argument('--recovery-stand-test', action='store_true',
                        help='With native posture test: start standing, fist=StandDown, palm=RecoveryStand once')
    parser.add_argument('--capture-hand-samples', action='store_true',
                        help='Read-only: save camera JPEG and raw hand landmarks once per second')
    parser.add_argument('--fist-landmarks', action='store_true',
                        help='Opt in to tucked-thumb/folded-finger fist fallback; validate in preview first')
    parser.add_argument('--lease', action='store_true', help='Opt-in SDK lease during armed session')
    parser.add_argument('--allow-state-100', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--log-dir', type=Path, default=Path(__file__).with_name('logs'))
    return parser


def validate_args(parser, args):
    try:
        ipaddress.ip_address(args.peer)
    except ValueError as exc:
        parser.error(str(exc))
    if args.allow_state_100:
        parser.error('--allow-state-100 is not used; MCF V2 state semantics are selected by CheckMode')
    if args.lease and not args.enable_motion:
        parser.error('--lease requires --enable-motion and onsite arming')
    if args.diagnose and args.enable_motion:
        parser.error('--diagnose and --enable-motion are separate operating modes')
    if args.capture_hand_samples and (args.enable_motion or args.diagnose):
        parser.error('--capture-hand-samples requires camera preview without --enable-motion/--diagnose')
    if args.native_posture_test and (not args.enable_motion or not args.posture_only or args.lease):
        parser.error('--native-posture-test requires --enable-motion --posture-only and no --lease')
    if args.recovery_stand_test and not args.native_posture_test:
        parser.error('--recovery-stand-test requires --native-posture-test')
    if args.enable_motion and args.posture_only and not args.native_posture_test and POSTURE_MOTION_BLOCKED_REASON:
        parser.error(POSTURE_MOTION_BLOCKED_REASON)
    if not (320 <= args.width <= 1920 and 4 <= args.fps <= 12
            and 1 <= args.max_seconds <= 90 and 0.2 <= args.motion_seconds <= 3.0):
        parser.error('width 320..1920, fps 4..12, max-seconds 1..90, motion-seconds 0.2..3')


def run(args, log):
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, WirelessController_, LowState_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    interface = resolve_interface(args.interface, args.peer)
    log('connection', interface=interface, peer=args.peer, sdk_source=__import__(
        'unitree_sdk2py.go2.sport.sport_client', fromlist=['']).__file__)
    configure_unitree_dds(interface, args.peer)
    faults = HardwareFaultMonitor()
    bridge = RobotMotionBridge(motion_seconds=args.motion_seconds, log=log, hardware_faults=faults,
                               posture_only=args.posture_only, native_posture_test=args.native_posture_test,
                               recovery_stand_test=args.recovery_stand_test)
    subscribers = []
    previous_handlers = {}

    def subscribe(topic, message_type, callback):
        sub = ChannelSubscriber(topic, message_type)
        subscribers.append(sub)
        sub.Init(callback, 10)

    def interrupted(signum, frame):
        bridge.closed.set()
        bridge.wake.set()
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        subscribe('rt/sportmodestate', SportModeState_, bridge.state_callback)
        subscribe('rt/wirelesscontroller', WirelessController_, bridge.remote_callback)
        subscribe('rt/lf/lowstate', LowState_, faults.lowstate_callback)
        subscribe('rt/errorlist', String_, faults.errorlist_callback)
        switcher = MotionSwitcherClient()
        switcher.SetTimeout(2.0)
        switcher.Init()
        mode_code, mode = switcher.CheckMode()
        service = SportClient()
        service.SetTimeout(2.0)
        service.Init()
        version_code, version = service.GetServerApiVersion()
        log('services', mode_code=mode_code, mode=mode, version_code=version_code, version=version)
        # MCF identifies the new controller. Do not treat its FSM=0 as healthy,
        # and never extend the legacy fault-code whitelist globally.
        profile = 'mcf_v2' if mode_code == 0 and mode and mode.get('name') == 'mcf' else 'legacy'
        bridge.profile = bridge.policy.profile = profile
        log('state_profile', profile=profile, source='CheckMode',
            semantics='FSM id' if profile == 'mcf_v2' else 'legacy error code')
        telemetry_deadline = time.monotonic() + 2.0
        while (bridge.snapshot() is None or faults.received is None) and time.monotonic() < telemetry_deadline:
            time.sleep(0.05)
        log('hardware_fault_status', reason=faults.fault(time.monotonic()),
            **faults.status(),
            coverage='lowstate communication and 12 leg motors; errorlist when published')

        if args.diagnose:
            low_frequency = RobotMotionBridge(profile=profile, hardware_faults=faults)
            subscribe('rt/lf/sportmodestate', SportModeState_, low_frequency.state_callback)
            # Allow time for a supervised remote-controlled posture transition.
            # This branch only observes telemetry; it never arms the bridge.
            deadline = time.monotonic() + args.max_seconds
            last_fault_log = -math.inf
            log('diagnostic_ready', duration_s=args.max_seconds, enable_motion=False,
                instruction='Use the verified hardware remote; this program only records state')
            while time.monotonic() < deadline:
                for topic, source in (('rt/sportmodestate', bridge),
                                      ('rt/lf/sportmodestate', low_frequency)):
                    state = source.snapshot()
                    if state:
                        log('diagnostic_state', topic=topic, **asdict(state),
                            motion_block=source.state_fault(time.monotonic()))
                if time.monotonic() - last_fault_log >= 1.0:
                    log('hardware_fault_status', reason=faults.fault(time.monotonic()), **faults.status())
                    last_fault_log = time.monotonic()
                time.sleep(0.1)
            log('diagnostic_counts', regular=bridge.state_count, low_frequency=low_frequency.state_count)
            return 0

        if args.enable_motion:
            if args.native_posture_test and bridge.posture_block():
                raise RuntimeError(bridge.posture_block())
            reason = bridge.state_fault(time.monotonic())
            if reason:
                raise RuntimeError(reason)
            bridge.prepare_starting_posture()
            if mode_code != 0 or not mode or not mode.get('name') or version_code != 0:
                raise RuntimeError('motion service/mode query failed')
            print('Verify clear floor, slack cable, stable standing/prone robot and working hardware stop.', flush=True)
            if args.native_posture_test:
                print('Native StandDown/' + ('RecoveryStand' if args.recovery_stand_test else 'StandUp') +
                      ': firmware speed, NOT slow motion. Each action once. '
                      'Hardware stop must be ready; software stop can wait for the in-flight RPC.', flush=True)
            if args.lease:
                print('Lease changes ownership. Verify hardware-stop behavior for this firmware.', flush=True)
            if input('Enter GESTURE MOVE when the onsite operator is ready: ').strip() != 'GESTURE MOVE':
                log('cancelled', reason='onsite arming declined')
                return 0
            if bridge.state_fault(time.monotonic()):
                raise RuntimeError(bridge.state_fault(time.monotonic()))
            if args.lease:
                from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, Response_
                observer = LeaseObserver(f'{socket.gethostname()}/sport/{os.getpid()}', log)
                subscribe('rt/api/sport_lease/request', Request_, observer.request_callback)
                subscribe('rt/api/sport_lease/response', Response_, observer.response_callback)
                service = SportClient(enableLease=True)
                service.Init()
                bridge.lease = observer
                deadline = time.monotonic() + 3.0
                while observer.fault(time.monotonic(), service.GetLeaseId()) and time.monotonic() < deadline:
                    if observer.error or bridge.state_fault(time.monotonic()):
                        break
                    time.sleep(0.05)
                fault = observer.fault(time.monotonic(), service.GetLeaseId())
                if fault:
                    raise RuntimeError(fault)
                log('lease_acquired', lease_id=service.GetLeaseId())
            reason = bridge.state_fault(time.monotonic())
            if reason:
                raise RuntimeError(reason)
            bridge.prepare_starting_posture()
            service.SetTimeout(RPC_TIMEOUT_S)
            bridge.client = service
            bridge.enable_motion = True
            bridge._stop()
            if bridge.failed:
                raise RuntimeError('initial StopMove failed')

        import cv2
        import mediapipe as mp
        import numpy as np
        from unitree_sdk2py.go2.video.video_client import VideoClient

        video = VideoClient()
        video.SetTimeout(1.0)
        video.Init()
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(Path(args.model).resolve())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1)
        session = ReadOnlyGestureSession(preview_only=not args.enable_motion, path_enabled=not args.posture_only)
        count = 0
        failed_frames = 0
        last_ms = -1
        last_desired = object()
        last_hand_status = -math.inf
        sample_dir = None
        last_hand_sample = -math.inf
        if args.capture_hand_samples:
            sample_dir = args.log_dir / ('hand_samples_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            sample_dir.mkdir(parents=True, exist_ok=False)
            log('hand_capture', directory=str(sample_dir.resolve()), motion_enabled=False,
                note='Original JPEG; landmarks refer to horizontally mirrored inference image')
        started = time.monotonic()
        deadline = started + args.max_seconds
        bridge.session_deadline = deadline
        with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
            bridge.start()
            log('ready', enable_motion=args.enable_motion, max_seconds=args.max_seconds,
                motion_seconds=args.motion_seconds, lease=args.lease, posture_only=args.posture_only,
                fist_landmarks=args.fist_landmarks, native_posture_test=args.native_posture_test,
                recovery_stand_test=args.recovery_stand_test)
            log('posture_help', enabled=not bool(bridge.posture_block()), reason=bridge.posture_block(),
                fist='StandDown once; hold until POSTURE_STATE' if args.native_posture_test else 'Recognition only; StandDown disabled',
                palm=('RecoveryStand once after this session confirms prone; hold until POSTURE_STATE'
                      if args.recovery_stand_test else 'StandUp once when prone; StopMove when standing') if args.native_posture_test else
                     'StopMove while standing; native StandUp disabled')
            while time.monotonic() < deadline and not bridge.closed.is_set():
                # Timestamp BEFORE camera RPC. Late frames never acquire a fresh command age.
                captured = time.monotonic()
                code, jpeg = video.GetImageSample()
                image = None
                if code == 0 and jpeg:
                    image = cv2.imdecode(np.frombuffer(bytes(jpeg), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    failed_frames += 1
                    bridge.camera_command(None, 0.0)
                    session = ReadOnlyGestureSession(preview_only=not args.enable_motion, path_enabled=not args.posture_only)
                    log('camera_error', code=code, consecutive=failed_frames,
                        bytes_received=len(jpeg) if jpeg else 0)
                    if failed_frames >= 3:
                        raise RuntimeError('three consecutive camera/decode failures')
                    continue
                failed_frames = 0
                image = cv2.flip(image, 1)
                height, width = image.shape[:2]
                if width > args.width:
                    image = cv2.resize(image, (args.width, round(height * args.width / width)),
                                       interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                stamp = max(last_ms + 1, int((captured - started) * 1000))
                last_ms = stamp
                result = recognizer.recognize_for_video(mp_image, stamp)
                now = time.monotonic()
                label, score, tip, four, six, rule = classify(
                    result, session.mode, fist_fallback=args.fist_landmarks,
                    image_aspect=image.shape[1] / image.shape[0], path_enabled=not args.posture_only)
                if sample_dir is not None and now - last_hand_sample >= 1.0:
                    image_path = sample_dir / f'frame_{count:05d}.jpg'
                    image_path.write_bytes(bytes(jpeg))
                    sample = dict(image=str(image_path.resolve()),
                                  mirrored_for_inference=True, label=label, score=score, rule=rule,
                                  landmarks=[[[float(p.x), float(p.y), float(p.z)] for p in hand]
                                             for hand in result.hand_landmarks],
                                  categories=[[(c.category_name, float(c.score)) for c in hand]
                                              for hand in result.gestures])
                    image_path.with_suffix('.json').write_text(
                        json.dumps(sample, allow_nan=False, indent=2), encoding='utf-8')
                    log('hand_sample', image=sample['image'], label=label, score=score,
                        hand_count=len(sample['landmarks']))
                    last_hand_sample = now
                if now - captured > FRAME_TIMEOUT_S or now >= deadline:
                    desired = None
                    session = ReadOnlyGestureSession(preview_only=not args.enable_motion, path_enabled=not args.posture_only)
                else:
                    session.consume(label, score, tip, four, six, rule, now)
                    desired = stable_command(session, label, score, four, six)
                desired = bridge.camera_command(desired, captured)
                if now - last_hand_status >= 1.0:
                    if now - captured > FRAME_TIMEOUT_S:
                        hint = 'stale_frame'
                    elif six:
                        hint = 'cancel'
                    elif session.mode != 'IDLE' or four:
                        hint = 'path_preview_only'
                    elif label not in STATIC_COMMANDS:
                        hint = 'no_supported_gesture'
                    elif args.posture_only and STATIC_COMMANDS.get(label) in MOTION:
                        hint = 'walking_disabled_posture_only'
                    elif score < STATIC_CONFIDENCE:
                        hint = 'confidence_below_threshold'
                    elif desired is None:
                        hint = 'hold_steady_for_confirmation'
                    else:
                        hint = 'gesture_confirmed'
                    log('hand_status', label=label, score=round(score, 3),
                        source=rule or 'MediaPipe classifier',
                        threshold=STATIC_CONFIDENCE,
                        confirmed_frames=min(session.db.count, session.db.frames),
                        required_frames=session.db.frames, desired=desired, hint=hint)
                    last_hand_status = now
                if desired != last_desired:
                    log('gesture', label=label, score=score, desired=desired,
                        frame_age_s=now-captured, path_mode=session.mode)
                    last_desired = desired
                count += 1
                bridge.closed.wait(max(0, 1 / args.fps - (time.monotonic() - captured)))
        log('camera_done', frames=count)
        bridge.shutdown()
        return 1 if bridge.failed else 0
    except Exception as exc:
        bridge.fail(str(exc))
        raise
    finally:
        # Ignore a second interrupt while the bounded shutdown sends its stops.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        bridge.shutdown()
        for sub in subscribers:
            sub.Close()
        log('session_end', move_sends=bridge.send_count, stop_sends=bridge.stop_count,
            failed=bridge.failed)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def self_test():
    import unittest
    from go2_camera_gesture_wsl_v10 import self_test as inherited
    inherited()
    suite = unittest.defaultTestLoader.loadTestsFromName('test_go2_real_gesture_move')
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.self_test:
        self_test()
        return 0
    if not args.diagnose and not Path(args.model).is_file():
        parser.error(f'model file missing: {args.model}')
    log = EventLog(args.log_dir / f'go2_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl')
    try:
        return run(args, log)
    except KeyboardInterrupt:
        log('interrupted', reason='operator interrupt')
        return 130
    except Exception as exc:
        log('error', reason=str(exc))
        return 1
    finally:
        log.close()


if __name__ == '__main__':
    raise SystemExit(main())
