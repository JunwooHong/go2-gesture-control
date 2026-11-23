"""Offline regression tests. All robot clients and DDS messages are fakes."""
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import go2_real_gesture_move_v13 as control
from go2_camera_gesture_wsl_v10 import ReadOnlyGestureSession, recognize_closed_fist, classify


def state_message(error=0, mode=0, rpy=(0.0, 0.0, 0.0)):
    return SimpleNamespace(error_code=error, mode=mode, gait_type=0,
                           position=(0.0, 0.0, 0.32), velocity=(0.0, 0.0, 0.0),
                           imu_state=SimpleNamespace(rpy=rpy), body_height=0.32,
                           stamp=SimpleNamespace(sec=1, nanosec=1))


class FakeClient:
    def __init__(self):
        self.calls = []
        self.move_code = 0
        self.stop_code = 0
        self.sit_code = 0
        self.on_stop = None
        self.timeouts = []
        self.recovery_code = 0
        self.on_recovery = None

    def SetTimeout(self, seconds):
        self.timeouts.append(seconds)

    def Move(self, *velocity):
        self.calls.append(('Move', velocity))
        return self.move_code

    def StopMove(self):
        self.calls.append(('StopMove',))
        if self.on_stop:
            self.on_stop()
        return self.stop_code

    def StandDown(self):
        self.calls.append(('StandDown',))
        return self.sit_code

    def StandUp(self):
        self.calls.append(('StandUp',))
        return 0

    def RecoveryStand(self):
        self.calls.append(('RecoveryStand',))
        if self.on_recovery:
            self.on_recovery()
        return self.recovery_code

    def GetLeaseId(self):
        return 42


class GestureTests(unittest.TestCase):
    def test_all_static_gestures_through_existing_debouncer(self):
        expected = {'Thumb_Up': 'FORWARD', 'Thumb_Down': 'BACKWARD',
                    'Victory': 'TURN_RIGHT', 'Three_Fingers': 'TURN_LEFT',
                    'Closed_Fist': 'LIE_DOWN', 'Open_Palm': 'PALM'}
        for label, wanted in expected.items():
            with self.subTest(label=label):
                session = ReadOnlyGestureSession()
                for index in range(5):
                    session.consume(label, 0.9, None, False, False, None, index / 10)
                self.assertEqual(control.stable_command(session, label, 0.9, False, False), wanted)

    def test_lost_changed_and_low_confidence_hands_do_not_reuse_command(self):
        session = ReadOnlyGestureSession()
        session.db.command = 'FORWARD'
        for label, score in [('None', 0), ('Thumb_Down', 0.9), ('Thumb_Up', 0.2)]:
            self.assertIsNone(control.stable_command(session, label, score, False, False))

    def test_reappearing_same_gesture_requires_new_confirmation(self):
        session = ReadOnlyGestureSession()
        for index in range(5):
            session.consume('Thumb_Up', 0.9, None, False, False, None, index / 10)
        session.consume('None', 0, None, False, False, None, 0.5)
        session.consume('Thumb_Up', 0.9, None, False, False, None, 0.6)
        self.assertIsNone(control.stable_command(session, 'Thumb_Up', 0.9, False, False))

    def test_default_stand_value_does_not_confirm_first_palm_frame(self):
        session = ReadOnlyGestureSession()
        session.consume('Open_Palm', 0.9, None, False, False, None, 0)
        self.assertIsNone(control.stable_command(session, 'Open_Palm', 0.9, False, False))

    def test_six_cancels_every_path_mode_immediately(self):
        for mode in ('IDLE', 'RECORDING', 'DONE', 'PREVIEW'):
            session = ReadOnlyGestureSession()
            session.mode = mode
            self.assertEqual(control.stable_command(session, 'Chinese_Six', 1, False, True), 'CANCEL')
            self.assertIsNone(control.stable_command(session, 'Four_Fingers', 1, True, False))

    def test_six_returns_path_to_idle(self):
        session = ReadOnlyGestureSession()
        session.mode = 'PREVIEW'
        session.planned = ['UP', 'RIGHT']
        for index in range(10):
            session.consume('Chinese_Six', 1, None, False, True, 'rule', index / 10)
        self.assertEqual(session.mode, 'IDLE')
        self.assertEqual(session.planned, [])


class FistLandmarkTests(unittest.TestCase):
    def setUp(self):
        data = json.loads((control.Path(__file__).parent / 'testdata/fist_landmarks.json').read_text(encoding='utf-8-sig'))
        self.aspect = data['image_aspect']
        self.hand = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in data['landmarks']]

    def result(self, label='None'):
        return SimpleNamespace(hand_landmarks=[self.hand],
                               gestures=[[SimpleNamespace(category_name=label, score=0.8)]])

    def test_recorded_fist_requires_opt_in_then_five_frame_confirmation(self):
        self.assertTrue(recognize_closed_fist(self.hand, self.aspect))
        self.assertEqual(classify(self.result(), 'IDLE')[0], 'None')
        label, score, tip, four, six, rule = classify(self.result(), 'IDLE',
            fist_fallback=True, image_aspect=self.aspect)
        self.assertEqual(label, 'Closed_Fist')
        session = ReadOnlyGestureSession()
        for index in range(5):
            session.consume(label, score, tip, four, six, rule, index / 8)
            command = control.stable_command(session, label, score, four, six)
            self.assertEqual(command, 'LIE_DOWN' if index == 4 else None)

    def test_each_extended_finger_is_rejected(self):
        for pip, tip in ((6,8), (10,12), (14,16), (18,20)):
            original = self.hand[tip]
            self.hand[tip] = SimpleNamespace(x=2*self.hand[pip].x-self.hand[0].x,
                                             y=2*self.hand[pip].y-self.hand[0].y,z=0)
            self.assertFalse(recognize_closed_fist(self.hand, self.aspect))
            self.hand[tip] = original

    def test_straight_thumb_is_rejected(self):
        self.hand[4] = SimpleNamespace(x=2*self.hand[3].x-self.hand[2].x,
                                       y=2*self.hand[3].y-self.hand[2].y,z=0)
        self.assertFalse(recognize_closed_fist(self.hand, self.aspect))

    def test_other_model_commands_are_never_overridden(self):
        for label in ('Thumb_Up', 'Thumb_Down', 'Open_Palm', 'Victory'):
            self.assertEqual(classify(self.result(label), 'IDLE', fist_fallback=True,
                                     image_aspect=self.aspect)[0], label)
        self.assertEqual(classify(self.result(), 'RECORDING', fist_fallback=True,
                                 image_aspect=self.aspect)[0], 'None')

    def test_invalid_small_or_clipped_hand_is_rejected(self):
        self.assertFalse(recognize_closed_fist([], self.aspect))
        self.assertFalse(recognize_closed_fist(self.hand, math.nan))
        self.hand[4].x = math.nan
        self.assertFalse(recognize_closed_fist(self.hand, self.aspect))
        self.hand[4].x = 0
        self.assertFalse(recognize_closed_fist(self.hand, self.aspect))
        for p in self.hand:
            p.x, p.y = 0.5 + p.y * 0.001, 0.5 + p.y * 0.001
        self.assertFalse(recognize_closed_fist(self.hand, self.aspect))


class BridgeTests(unittest.TestCase):
    def setUp(self):
        # Exercise the retained posture controller only against FakeClient.
        # Production posture actuation remains blocked after the incident.
        patch.object(control, 'POSTURE_MOTION_BLOCKED_REASON', '').start()
        self.clock = patch.object(control.time, 'monotonic', return_value=10.0).start()
        self.addCleanup(patch.stopall)
        self.client = FakeClient()
        self.bridge = control.RobotMotionBridge(self.client, enable_motion=True)

    def tick(self, now, desired, mode=0, error=0, frame_time=None):
        self.clock.return_value = now
        self.bridge.state_callback(state_message(error, mode))
        self.bridge.camera_command(desired, now if frame_time is None else frame_time)
        self.bridge.step(now)

    def arm(self):
        self.tick(10, None)
        self.tick(10.5, None)

    def test_startup_requires_neutral(self):
        self.tick(10, 'FORWARD')
        self.assertEqual(self.client.calls, [])

    def test_repeat_move_until_duration_then_require_release(self):
        self.arm()
        self.tick(10.6, 'FORWARD')
        self.tick(10.7, 'FORWARD')
        self.tick(11.81, 'FORWARD')
        self.tick(12, 'FORWARD')
        self.assertEqual(self.client.calls, [('Move', control.MOTION['FORWARD']),
                                             ('Move', control.MOTION['FORWARD']), ('StopMove',)])
        self.tick(12.1, None)
        self.tick(12.6, None)
        self.tick(12.7, 'BACKWARD')
        self.assertEqual(self.client.calls[-1], ('Move', control.MOTION['BACKWARD']))

    def test_three_second_hold_keeps_sending_and_logs_its_limit(self):
        self.bridge.policy.motion_seconds = 3.0
        events = []
        self.bridge.log = lambda event, **fields: events.append((event, fields))
        self.arm()
        self.tick(10.6, 'FORWARD')
        self.tick(11.9, 'FORWARD')
        self.tick(13.5, 'FORWARD')
        self.assertEqual(self.bridge.send_count, 3)
        self.tick(13.61, 'FORWARD')
        self.assertEqual(self.client.calls[-1], ('StopMove',))
        self.assertIn('motion duration reached', events[-1][1]['reason'])
        self.tick(13.8, 'FORWARD')
        self.assertEqual(self.bridge.send_count, 3)

    def test_all_four_velocities(self):
        for command, velocity in control.MOTION.items():
            with self.subTest(command=command):
                self.bridge = control.RobotMotionBridge(self.client, enable_motion=True)
                self.arm()
                self.tick(10.6, command)
                self.assertEqual(self.client.calls[-1], ('Move', velocity))

    def test_camera_events_cannot_exceed_move_send_rate(self):
        self.arm()
        self.tick(10.6, 'FORWARD')
        self.tick(10.61, 'FORWARD')
        self.tick(10.62, 'FORWARD')
        self.assertEqual(self.bridge.send_count, 1)

    def test_six_palm_and_disappearing_hand_stop(self):
        for command in ('CANCEL', 'PALM', None):
            with self.subTest(command=command):
                self.bridge = control.RobotMotionBridge(self.client, enable_motion=True)
                self.arm()
                self.tick(10.6, 'FORWARD')
                self.tick(10.7, command)
                self.assertEqual(self.client.calls[-1], ('StopMove',))

    def test_late_camera_result_cannot_refresh_deadman(self):
        self.arm()
        self.tick(10.6, 'FORWARD')
        self.tick(11.3, 'FORWARD', frame_time=10.6)
        self.assertEqual(self.client.calls[-1], ('StopMove',))
        self.tick(11.4, 'FORWARD')
        self.assertEqual(sum(call[0] == 'Move' for call in self.client.calls), 1)

    def test_unknown_error_latches_even_if_followed_by_zero(self):
        self.arm()
        self.bridge.state_callback(state_message(error=100))
        self.tick(10.6, 'FORWARD')
        self.assertTrue(self.bridge.failed)
        self.assertFalse(any(call[0] == 'Move' for call in self.client.calls))

    def test_stale_state_aborts(self):
        self.arm()
        self.bridge.camera_command('FORWARD', 11.3)
        self.bridge.step(11.3)
        self.assertTrue(self.bridge.failed)

    def test_callback_after_tick_clock_does_not_look_like_timeout(self):
        self.arm()
        # step's argument was sampled before a newer DDS/camera callback.
        self.clock.return_value = 10.61
        code = 100 if self.bridge.profile == 'mcf_v2' else 0
        self.bridge.state_callback(state_message(error=code))
        self.bridge.camera_command('FORWARD', 10.61)
        self.bridge.step(10.60)
        self.assertFalse(self.bridge.failed)
        self.assertEqual(self.client.calls[-1], ('Move', control.MOTION['FORWARD']))

    def test_state_fault_uses_time_after_snapshot(self):
        self.clock.return_value = 10.01
        code = 100 if self.bridge.profile == 'mcf_v2' else 0
        self.bridge.state_callback(state_message(error=code))
        self.assertEqual(self.bridge.state_fault(10.0), '')
        self.clock.return_value = 10.8
        self.assertIn('timed out', self.bridge.state_fault(10.0))

    def test_invalid_and_tilted_state_blocks(self):
        for message in (SimpleNamespace(), state_message(rpy=(math.nan, 0, 0)),
                        state_message(rpy=(0, 0.5, 0))):
            with self.subTest(message=message):
                bridge = control.RobotMotionBridge(self.client, enable_motion=True)
                bridge.state_callback(message)
                self.assertTrue(bridge.state_fault(10))

    def test_remote_damping_mode_cannot_receive_move(self):
        self.arm()
        self.tick(10.6, 'FORWARD', mode=7)
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls, [])

    def test_remote_button_or_stick_latches_takeover(self):
        for keys, stick in ((1, 0), (0, 0.8)):
            with self.subTest(keys=keys, stick=stick):
                self.bridge = control.RobotMotionBridge(self.client, enable_motion=True)
                self.arm()
                self.bridge.remote_callback(SimpleNamespace(keys=keys, lx=stick, ly=0, rx=0, ry=0))
                self.tick(10.6, 'FORWARD')
                self.assertTrue(self.bridge.failed)
                self.assertFalse(any(call[0] == 'Move' for call in self.client.calls))

    def test_sit_and_rise_once_with_telemetry_confirmation(self):
        self.arm()
        self.tick(10.6, 'LIE_DOWN')
        self.tick(10.7, 'LIE_DOWN', mode=5)
        self.tick(10.8, 'LIE_DOWN', mode=5)
        self.assertTrue(self.bridge.prone)
        self.assertEqual(sum(call[0] == 'StandDown' for call in self.client.calls), 1)
        self.tick(10.9, None, mode=5)
        self.tick(11.4, None, mode=5)
        self.tick(11.5, 'PALM', mode=5)
        self.tick(11.6, 'PALM', mode=0)
        self.assertFalse(self.bridge.prone)
        self.assertEqual(sum(call[0] == 'StandUp' for call in self.client.calls), 1)

    def test_cannot_walk_while_seated(self):
        self.arm()
        self.bridge.prone = True
        self.tick(10.6, 'FORWARD', mode=5)
        self.assertFalse(any(call[0] == 'Move' for call in self.client.calls))

    def test_start_seated_then_palm_uses_native_rise(self):
        self.tick(10, None, mode=5)
        self.bridge.prepare_starting_posture()
        self.assertTrue(self.bridge.prone)
        self.tick(10.1, None, mode=5)
        self.tick(10.6, None, mode=5)
        self.tick(10.7, 'PALM', mode=5)
        self.assertEqual(self.client.calls[-2:], [('StopMove',), ('StandUp',)])
        self.tick(10.8, 'PALM', mode=0)
        self.assertFalse(self.bridge.prone)

    def test_posture_only_ignores_all_walk_turn_commands_but_sits(self):
        self.bridge.posture_only = True
        self.arm()
        for index, command in enumerate(control.MOTION):
            self.tick(10.6 + index * 0.1, command)
        self.assertFalse(any(call[0] == 'Move' for call in self.client.calls))
        self.tick(11.0, 'LIE_DOWN')
        self.assertEqual(self.client.calls[-1], ('StandDown',))

    def test_posture_change_during_stop_cancels_native_action(self):
        self.arm()
        code = 1004 if self.bridge.profile == 'mcf_v2' else 0
        self.client.on_stop = lambda: self.bridge.state_callback(state_message(error=code, mode=5))
        self.tick(10.6, 'LIE_DOWN')
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls, [('StopMove',)])

    def test_cancel_during_stop_prevents_sit(self):
        self.arm()
        self.client.on_stop = lambda: self.bridge.camera_command('CANCEL', 10.6)
        self.tick(10.6, 'LIE_DOWN')
        self.assertEqual(self.client.calls, [('StopMove',)])

    def test_deadline_during_stop_prevents_posture_action(self):
        self.arm()
        self.bridge.session_deadline = 10.7

        def finish_stop():
            self.clock.return_value = 10.8
            self.bridge.state_callback(state_message())

        self.client.on_stop = finish_stop
        self.tick(10.6, 'LIE_DOWN')
        self.assertTrue(self.bridge.closed.is_set())
        self.assertEqual(self.client.calls, [('StopMove',)])

    def test_posture_timeout_and_cancel_latch(self):
        for now, gesture in ((19.0, 'LIE_DOWN'), (10.7, 'CANCEL')):
            with self.subTest(gesture=gesture):
                self.bridge = control.RobotMotionBridge(self.client, enable_motion=True)
                self.arm()
                self.tick(10.6, 'LIE_DOWN')
                self.tick(now, gesture)
                self.assertTrue(self.bridge.failed)

    def test_cancel_and_palm_take_priority_over_posture_completion(self):
        for gesture in ('CANCEL', 'PALM'):
            with self.subTest(gesture=gesture):
                self.bridge = control.RobotMotionBridge(self.client, enable_motion=True)
                self.arm()
                self.tick(10.6, 'LIE_DOWN')
                self.tick(10.7, gesture, mode=5)
                self.assertTrue(self.bridge.failed)
                self.assertEqual(self.client.calls[-1], ('StopMove',))

    def test_native_action_error_never_falls_back_to_another_action(self):
        self.arm()
        self.client.sit_code = 3203
        self.tick(10.6, 'LIE_DOWN')
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls, [('StopMove',), ('StandDown',)])

    def test_failed_send_is_stopped_on_shutdown(self):
        self.arm()
        self.client.move_code = 3102
        self.tick(10.6, 'FORWARD')
        self.bridge.shutdown()
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls[-1], ('StopMove',))

    def test_stop_failure_prevents_posture_request(self):
        self.arm()
        self.client.stop_code = 3104
        self.tick(10.6, 'LIE_DOWN')
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls, [('StopMove',)])

    def test_preview_never_calls_motion_client(self):
        self.bridge.enable_motion = False
        self.arm()
        for i, command in enumerate(('FORWARD', 'LIE_DOWN', 'PALM', 'CANCEL')):
            self.tick(10.6 + i, command)
        self.bridge.shutdown()
        self.assertEqual(self.client.calls, [])

    def test_deadline_stops_worker_even_if_camera_is_blocked(self):
        self.arm()
        self.bridge.session_deadline = 10.8
        self.tick(10.6, 'FORWARD')
        self.tick(10.9, 'FORWARD')
        self.assertTrue(self.bridge.closed.is_set())
        self.bridge.shutdown()
        self.assertEqual(self.client.calls[-1], ('StopMove',))

    def test_worker_stops_on_unhandled_client_exception(self):
        self.arm()
        self.clock.return_value = 10.6
        self.bridge.camera_command('FORWARD', 10.6)
        with patch.object(self.client, 'Move', side_effect=RuntimeError('transport')):
            self.bridge._loop()
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls[-2:], [('StopMove',), ('StopMove',)])


def lease_request(api, identity, context='host/sport/123', lease_id=0):
    return SimpleNamespace(header=SimpleNamespace(
        identity=SimpleNamespace(api_id=api, id=identity), lease=SimpleNamespace(id=lease_id)),
        parameter=json.dumps({'name': context}))


def lease_response(api, identity, code=0, data=None):
    return SimpleNamespace(header=SimpleNamespace(
        identity=SimpleNamespace(api_id=api, id=identity), status=SimpleNamespace(code=code)),
        data=json.dumps(data if data is not None else {'id': 42, 'term': 1_000_000}))


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(control.time, 'monotonic', return_value=10).start()
        self.addCleanup(patch.stopall)
        self.observer = control.LeaseObserver('host/sport/123', lambda *args, **kwargs: None)

    def acquire(self):
        self.observer.request_callback(lease_request(101, 1))
        self.observer.response_callback(lease_response(101, 1))

    def test_acquire_renew_and_expire_despite_stale_sdk_id(self):
        self.acquire()
        self.assertEqual(self.observer.fault(10.5, 42), '')
        self.clock.return_value = 10.5
        self.observer.request_callback(lease_request(102, 2, lease_id=42))
        self.observer.response_callback(lease_response(102, 2))
        self.assertEqual(self.observer.fault(11.4, 42), '')
        self.assertTrue(self.observer.fault(11.6, 42))

    def test_other_clients_and_unmatched_responses_cannot_grant_lease(self):
        self.observer.request_callback(lease_request(101, 1, context='other'))
        self.observer.response_callback(lease_response(101, 1))
        self.assertTrue(self.observer.fault(10, 42))

    def test_malformed_unrelated_lease_payload_is_ignored(self):
        request = lease_request(101, 1)
        request.parameter = '[]'
        self.observer.request_callback(request)
        self.assertEqual(self.observer.pending, {})

    def test_denied_and_invalid_lease_never_ready(self):
        self.acquire()
        self.observer.request_callback(lease_request(102, 2, lease_id=42))
        self.observer.response_callback(lease_response(102, 2, code=3206))
        self.assertIn('3206', self.observer.fault(10.1, 42))
        self.assertTrue(self.observer.fault(10.1, 0))

    def test_bad_term_is_rejected(self):
        self.observer.request_callback(lease_request(101, 1))
        self.observer.response_callback(lease_response(101, 1, data={'id': 42, 'term': -1}))
        self.assertTrue(self.observer.fault(10, 42))

    def test_lease_loss_aborts_controller(self):
        bridge = control.RobotMotionBridge(FakeClient(), enable_motion=True, lease=self.observer)
        bridge.state_callback(state_message())
        bridge.camera_command(None, 10)
        bridge.step(10)
        self.assertTrue(bridge.failed)


class ArgumentTests(unittest.TestCase):
    def test_native_posture_test_is_explicit_and_isolated(self):
        parser = control.build_parser()
        control.validate_args(parser, parser.parse_args([
            '--enable-motion', '--posture-only', '--native-posture-test']))
        for flags in (['--native-posture-test'],
                      ['--native-posture-test', '--enable-motion'],
                      ['--native-posture-test', '--enable-motion', '--posture-only', '--lease']):
            with patch.object(parser, 'error', side_effect=ValueError), self.assertRaises(ValueError):
                control.validate_args(parser, parser.parse_args(flags))

    def test_posture_only_actuation_is_blocked_after_incident(self):
        parser = control.build_parser()
        self.assertTrue(control.POSTURE_MOTION_BLOCKED_REASON)
        with patch.object(parser, 'error', side_effect=ValueError), self.assertRaises(ValueError):
            control.validate_args(parser, parser.parse_args(['--enable-motion', '--posture-only']))

    def test_native_posture_methods_cannot_be_reached_in_normal_mode(self):
        self.assertTrue(control.POSTURE_MOTION_BLOCKED_REASON)
        for command in ('LIE_DOWN', 'STAND_UP'):
            client = FakeClient()
            bridge = control.RobotMotionBridge(client, enable_motion=True)
            bridge._posture(command, 10)
            self.assertTrue(bridge.failed)
            self.assertEqual(client.calls, [('StopMove',)])

    def test_removed_bypass_and_incompatible_modes_rejected(self):
        for arguments in (['--allow-state-100'], ['--lease'],
                          ['--capture-hand-samples', '--enable-motion'],
                          ['--capture-hand-samples', '--diagnose'],
                          ['--diagnose', '--enable-motion'], ['--motion-seconds', 'nan']):
            with self.subTest(arguments=arguments):
                parser = control.build_parser()
                with patch.object(parser, 'error', side_effect=ValueError), self.assertRaises(ValueError):
                    control.validate_args(parser, parser.parse_args(arguments))

    def test_default_model_is_resolved_beside_script(self):
        args = control.build_parser().parse_args([])
        self.assertTrue(control.Path(args.model).is_absolute())
        self.assertFalse(args.enable_motion)
        self.assertFalse(args.lease)

    def test_auto_interface_uses_robot_route(self):
        with patch.object(control.subprocess, 'run', return_value=SimpleNamespace(
                stdout='[{"dev":"eth1"}]')):
            self.assertEqual(control.resolve_interface('auto', '192.168.1.2'), 'eth1')


class InstalledSdkTests(unittest.TestCase):
    def test_native_methods_use_expected_official_apis_without_constructing_client(self):
        try:
            from unitree_sdk2py.go2.sport.sport_client import SportClient
        except ImportError:
            self.skipTest('SDK2 is unavailable in this Python environment')
        client = SportClient.__new__(SportClient)
        with patch.object(client, '_CallNoReply', return_value=0) as send:
            self.assertEqual(client.Move(0.2, 0, 0), 0)
            api, payload = send.call_args.args
            self.assertEqual(api, 1008)
            self.assertEqual(json.loads(payload), {'x': 0.2, 'y': 0, 'z': 0})
        for method, api in ((client.StopMove, 1003), (client.StandDown, 1005),
                            (client.StandUp, 1004), (client.RecoveryStand, 1006)):
            with patch.object(client, '_Call', return_value=(0, '{}')) as call:
                self.assertEqual(method(), 0)
                self.assertEqual(call.call_args.args[0], api)

    def test_main_error_gate_precedes_arming_and_lease_construction(self):
        try:
            from unitree_sdk2py.core import channel
            from unitree_sdk2py.go2.sport import sport_client
            from unitree_sdk2py.comm.motion_switcher import motion_switcher_client
        except ImportError:
            self.skipTest('SDK2 is unavailable in this Python environment')

        class Subscriber:
            def __init__(self, topic, message_type):
                self.topic = topic

            def Init(self, callback, depth):
                if self.topic == 'rt/sportmodestate':
                    callback(state_message(error=9999))
                elif self.topic == 'rt/lf/lowstate':
                    callback(lowstate_message())

            def Close(self):
                pass

        args = control.build_parser().parse_args(['--enable-motion', '--lease', '--interface', 'eth1'])
        with patch.object(channel, 'ChannelSubscriber', Subscriber), \
                patch.object(control, 'configure_unitree_dds'), \
                patch.object(sport_client, 'SportClient') as factory, \
                patch.object(motion_switcher_client, 'MotionSwitcherClient') as switcher, \
                patch('builtins.input') as prompt:
            factory.return_value.GetServerApiVersion.return_value = (0, '1.0.0.1')
            switcher.return_value.CheckMode.return_value = (0, {'name': 'mcf'})
            with self.assertRaisesRegex(RuntimeError, 'unsupported V2 sport FSM=9999'):
                control.run(args, lambda *args, **kwargs: None)
            factory.assert_called_once_with()
            factory.return_value.Move.assert_not_called()
            factory.return_value.StopMove.assert_not_called()
            prompt.assert_not_called()


def lowstate_message(bit_flag=0, motor_fault=0):
    return SimpleNamespace(bit_flag=bit_flag, motor_state=[
        SimpleNamespace(reserve=[motor_fault if index == 3 else 0, 0]) for index in range(20)])


class V2Tests(BridgeTests):
    """Run the same watchdog/gesture tests with the documented MCF FSM fields."""
    def setUp(self):
        super().setUp()
        self.bridge.profile = self.bridge.policy.profile = 'mcf_v2'

    def tick(self, now, desired, mode=0, error=0, frame_time=None):
        if self.bridge.profile == 'mcf_v2':
            error = error or {0: 100, 1: 1013, 3: 100, 7: 1001, 5: 1004}.get(mode, 9999)
        super().tick(now, desired, mode, error, frame_time)

    def test_unknown_error_latches_even_if_followed_by_zero(self):
        self.arm()
        self.bridge.state_callback(state_message(error=9999))
        self.tick(10.6, 'FORWARD')
        self.assertTrue(self.bridge.failed)
        self.assertFalse(any(call[0] == 'Move' for call in self.client.calls))

    def test_zero_is_not_a_valid_v2_fsm(self):
        self.bridge.state_callback(state_message(error=0))
        self.assertIn('FSM=0', self.bridge.state_fault(10))

    def test_v2_ignores_obsolete_mode_for_sit_confirmation(self):
        self.arm()
        self.tick(10.6, 'LIE_DOWN')
        self.tick(10.7, 'LIE_DOWN', mode=0, error=1004)
        self.assertTrue(self.bridge.prone)

    def test_v2_seated_state_cannot_move_without_local_sitting_flag(self):
        self.arm()
        self.tick(10.6, 'FORWARD', mode=0, error=1004)
        self.assertTrue(self.bridge.failed)
        self.assertFalse(any(call[0] == 'Move' for call in self.client.calls))

    def test_v2_damping_lock_and_acrobatics_are_blocked(self):
        for code in (1001, 1007, 2012, 9999):
            state = control.RobotState.from_message(state_message(error=code), 10)
            self.assertTrue(state.fault(10, 'mcf_v2'))

    def test_native_stand_up_confirms_locked_state_without_enabling_walk(self):
        self.tick(10, None, error=1004)
        self.bridge.prepare_starting_posture()
        self.tick(10.1, None, error=1004)
        self.tick(10.6, None, error=1004)
        self.tick(10.7, 'PALM', error=1004)
        self.assertEqual(self.client.calls[-1], ('StandUp',))
        self.tick(10.8, 'PALM', error=1002)
        self.assertFalse(self.bridge.prone)
        self.assertIsNone(self.bridge.posture_pending)
        self.tick(10.9, None, error=1002)
        self.tick(11.4, None, error=1002)
        self.tick(11.5, 'FORWARD', error=1002)
        self.assertTrue(self.bridge.failed)
        self.assertFalse(any(call[0] == 'Move' for call in self.client.calls))

    def test_only_documented_lowered_states_confirm_stand_down(self):
        for code in (1004, 2006):
            state = control.RobotState.from_message(state_message(error=code), 10)
            self.assertTrue(state.lowered('mcf_v2'))
            self.assertFalse(state.movable('mcf_v2'))
        state = control.RobotState.from_message(state_message(error=1007), 10)
        self.assertFalse(state.lowered('mcf_v2'))
        self.assertTrue(state.fault(10, 'mcf_v2'))


class NativeProneTestTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(control.time, 'monotonic', return_value=10.0).start()
        self.addCleanup(patch.stopall)
        self.client = FakeClient()
        self.bridge = control.RobotMotionBridge(self.client, enable_motion=True,
            profile='mcf_v2', posture_only=True, native_posture_test=True)
        self.assertTrue(control.POSTURE_MOTION_BLOCKED_REASON)  # ordinary mode stays blocked

    def tick(self, now, gesture, fsm=100):
        self.clock.return_value = now
        self.bridge.state_callback(state_message(error=fsm))
        self.bridge.camera_command(gesture, now)
        self.bridge.step(now)

    def lower(self):
        self.tick(10, None)
        self.tick(10.5, None)
        self.tick(10.6, 'LIE_DOWN')

    def test_one_pair_waits_for_settled_state_and_exits_after_up(self):
        self.lower()
        self.assertEqual(self.client.calls[-1], ('StandDown',))
        self.tick(10.7, 'LIE_DOWN', 1004)
        self.assertFalse(self.bridge.prone)
        self.tick(11.3, 'LIE_DOWN', 1004)
        self.assertTrue(self.bridge.prone)
        self.tick(11.4, None, 1004)
        self.tick(12, None, 1004)
        self.tick(12.1, 'PALM', 1004)
        self.assertEqual(self.client.calls[-1], ('StandUp',))
        self.tick(12.2, 'PALM', 1002)
        self.assertFalse(self.bridge.closed.is_set())
        self.tick(12.8, 'PALM', 1002)
        self.assertTrue(self.bridge.closed.is_set())
        self.assertFalse(self.bridge.failed)
        self.assertEqual(self.client.timeouts, [1.0, 0.25, 1.0, 0.25])
        self.assertFalse(any(c[0] == 'Move' for c in self.client.calls))

    def test_rpc_failure_latches_and_restores_stop_timeout(self):
        self.client.sit_code = 3104
        self.lower()
        self.assertTrue(self.bridge.failed)
        self.tick(11, 'LIE_DOWN')
        self.assertEqual(self.client.calls.count(('StandDown',)), 1)
        self.assertEqual(self.client.timeouts, [1.0, 0.25])

    def test_each_native_request_cannot_be_retried(self):
        self.lower()
        self.bridge._posture('LIE_DOWN', 10.6)
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls.count(('StandDown',)), 1)

    def test_remote_takeover_cancels_pending_action(self):
        self.lower()
        self.bridge.remote_callback(SimpleNamespace(keys=1, lx=0, ly=0, rx=0, ry=0))
        self.tick(10.7, 'PALM', 1004)
        self.assertTrue(self.bridge.failed)
        self.assertNotIn(('StandUp',), self.client.calls)

    def test_unsupported_state_still_blocks_test(self):
        self.tick(10, 'LIE_DOWN', 1007)
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls, [])

    def test_unsettled_state_does_not_complete(self):
        self.lower()
        self.tick(10.7, 'LIE_DOWN', 1004)
        self.tick(11.2, 'LIE_DOWN', 100)  # target state did not stay stable
        self.tick(11.3, 'LIE_DOWN', 1004)
        self.assertFalse(self.bridge.prone)
        self.tick(11.9, 'LIE_DOWN', 1004)
        self.assertTrue(self.bridge.prone)


class PostureStaticRecognitionTests(unittest.TestCase):
    def setUp(self):
        # Reproduce the conflict: a valid model palm and a matching FOUR rule.
        self.result = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=.5, y=.5, z=0) for _ in range(21)]],
            gestures=[[SimpleNamespace(category_name='Open_Palm', score=.59)]])
        patch('go2_camera_gesture_wsl_v10.recognize_chinese_six', return_value=False).start()
        patch('go2_camera_gesture_wsl_v10.recognize_four_fingers', return_value=True).start()
        patch('go2_camera_gesture_wsl_v10.recognize_thumb_index_middle', return_value=False).start()
        self.addCleanup(patch.stopall)

    def test_posture_mode_retains_model_palm_and_confirms_five_frames(self):
        session = ReadOnlyGestureSession(path_enabled=False)
        for index in range(5):
            label, score, tip, four, six, rule = classify(self.result, session.mode, path_enabled=False)
            session.consume(label, score, tip, four, six, rule, index * .15)
            desired = control.stable_command(session, label, score, four, six)
            self.assertEqual(desired, 'PALM' if index == 4 else None)
        self.assertEqual(session.mode, 'IDLE')
        self.assertEqual(score, .59)

    def test_low_confidence_palm_is_not_promoted_to_rule_confidence(self):
        self.result.gestures[0][0].score = .54
        session = ReadOnlyGestureSession(path_enabled=False)
        for index in range(10):
            values = classify(self.result, session.mode, path_enabled=False)
            session.consume(*values, index * .15)
        label, score, _, four, six, _ = values
        self.assertIsNone(control.stable_command(session, label, score, four, six))
        self.assertEqual(score, .54)

    def test_four_cannot_enter_path_mode_in_posture_session(self):
        session = ReadOnlyGestureSession(path_enabled=False)
        for index in range(10):
            session.consume('Four_Fingers', 1, (.5, .5), True, False, 'rule', index * .15)
        self.assertEqual(session.mode, 'IDLE')
        self.assertIsNone(control.stable_command(session, 'Four_Fingers', 1, True, False))

    def test_default_path_preview_and_six_cancel_remain_available(self):
        session = ReadOnlyGestureSession()
        for index in range(10):
            values = classify(self.result, session.mode)
            session.consume(*values, index * .15)
        self.assertEqual(session.mode, 'RECORDING')
        with patch('go2_camera_gesture_wsl_v10.recognize_chinese_six', return_value=True):
            label, score, tip, four, six, rule = classify(self.result, 'IDLE', path_enabled=False)
        self.assertEqual(label, 'Chinese_Six')
        self.assertEqual(control.stable_command(ReadOnlyGestureSession(path_enabled=False),
                         label, score, four, six), 'CANCEL')


class RecoveryGestureTests(unittest.TestCase):
    """Exercise the observed descent/damping/rise sequence and its boundaries."""
    def setUp(self):
        self.clock = patch.object(control.time, 'monotonic', return_value=10.0).start()
        self.addCleanup(patch.stopall)
        self.client = FakeClient()
        self.events = []
        self.bridge = control.RobotMotionBridge(self.client, True, profile='mcf_v2',
            posture_only=True, native_posture_test=True, recovery_stand_test=True,
            log=lambda event, **fields: self.events.append((event, fields)))

    def tick(self, now, gesture, fsm=100, height=.32, speed=0, rpy=(0, 0, 0)):
        self.clock.return_value = now
        msg = state_message(error=fsm, rpy=rpy)
        msg.body_height = height
        msg.velocity = (0, 0, speed)
        self.bridge.state_callback(msg)
        self.bridge.camera_command(gesture, now)
        self.bridge.step(now)

    def down(self):
        self.tick(10, None)
        self.tick(10.5, None)
        self.tick(10.6, 'LIE_DOWN')
        self.tick(10.7, 'LIE_DOWN', 2006, .24, -.15)

    def prone(self):
        self.down()
        self.tick(12, 'LIE_DOWN', 1001, .082)
        self.tick(12.6, 'LIE_DOWN', 1001, .082)
        self.assertTrue(self.bridge.prone)
        self.assertFalse(self.bridge.failed)

    def rise(self):
        self.prone()
        self.tick(12.7, None, 1001, .082)
        self.tick(13.2, None, 1001, .082)
        self.tick(13.3, 'PALM', 1001, .082)

    def test_observed_pair_requires_settling_and_exits_after_balance_stand(self):
        self.rise()
        self.tick(13.4, 'PALM', 1001, .082)
        self.tick(13.5, 'PALM', 2004, .10, .079)
        self.tick(14.2, 'PALM', 2004, .24, .15)
        self.tick(15, 'PALM', 100, .323)
        self.assertFalse(self.bridge.closed.is_set())
        self.tick(15.6, 'PALM', 100, .315)
        self.assertTrue(self.bridge.closed.is_set())
        self.assertFalse(self.bridge.failed)
        self.assertEqual(self.client.calls.count(('StandDown',)), 1)
        self.assertEqual(self.client.calls.count(('RecoveryStand',)), 1)
        self.assertNotIn(('StandUp',), self.client.calls)
        self.assertFalse(any(call[0] == 'Move' for call in self.client.calls))
        self.assertEqual(self.client.timeouts, [1, .25, 1, .25])

    def test_transition_published_before_rpc_response_is_handled(self):
        def transition():
            self.clock.return_value = 13.31
            msg = state_message(error=2004)
            msg.body_height = .10
            self.bridge.state_callback(msg)
        self.client.on_recovery = transition
        self.rise()
        self.assertEqual(self.bridge.bad_state, '')
        self.tick(14, 'PALM', 100)
        self.tick(14.6, 'PALM', 100)
        self.assertTrue(self.bridge.closed.is_set())

    def test_start_in_damping_is_rejected_without_any_rpc(self):
        self.tick(10, None, 1001, .082)
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls, [])

    def test_damping_without_observed_descent_is_rejected(self):
        self.tick(10, None)
        self.tick(10.5, None)
        self.tick(10.6, 'LIE_DOWN')
        self.tick(10.7, 'LIE_DOWN', 1001, .082)
        self.assertTrue(self.bridge.failed)

    def test_high_body_damping_is_rejected(self):
        self.down()
        self.tick(11, 'LIE_DOWN', 1001, .32)
        self.assertTrue(self.bridge.failed)

    def test_descent_state_alone_never_confirms_prone(self):
        self.down()
        self.tick(11, 'LIE_DOWN', 2006, .10)
        self.tick(11.6, 'LIE_DOWN', 2006, .10)
        self.assertFalse(self.bridge.prone)

    def test_confirmed_damping_does_not_send_known_unsupported_stopmove(self):
        self.prone()
        calls = list(self.client.calls)
        self.client.stop_code = -1
        self.tick(12.7, None, 1001, .082)
        self.assertEqual(self.client.calls, calls)
        self.assertFalse(self.bridge.failed)
        self.assertTrue(any(e == 'stop_not_sent' for e, _ in self.events))

    def test_stop_failure_while_standing_still_prevents_descent(self):
        self.client.stop_code = -1
        self.down()
        self.assertTrue(self.bridge.failed)
        self.assertNotIn(('StandDown',), self.client.calls)

    def test_recovery_timeout_is_not_retried(self):
        self.client.recovery_code = 3104
        self.rise()
        self.tick(14, 'PALM', 2004, .10)
        self.assertTrue(self.bridge.failed)
        self.assertEqual(self.client.calls.count(('RecoveryStand',)), 1)
        self.assertEqual(self.client.timeouts[-1], .25)

    def test_unknown_state_is_not_allowed_during_recovery(self):
        self.rise()
        self.tick(14, 'PALM', 9999)
        self.assertTrue(self.bridge.failed)

    def test_unrequested_rise_is_rejected(self):
        self.prone()
        self.tick(12.7, None, 2004, .10)
        self.assertTrue(self.bridge.failed)
        self.assertNotIn(('RecoveryStand',), self.client.calls)

    def test_return_to_damping_after_rise_begins_aborts(self):
        self.rise()
        self.tick(13.5, 'PALM', 2004, .10)
        self.tick(13.6, 'PALM', 1001, .082)
        self.assertTrue(self.bridge.failed)

    def test_rise_window_expires(self):
        self.rise()
        self.tick(21.4, 'PALM', 2004, .10)
        self.assertTrue(self.bridge.failed)

    def test_low_height_standing_code_does_not_confirm_success(self):
        self.rise()
        self.tick(14, 'PALM', 100, .10)
        self.tick(14.6, 'PALM', 100, .10)
        self.assertFalse(self.bridge.closed.is_set())

    def test_remote_cancellation_survives_valid_damping_callback(self):
        self.prone()
        self.bridge.remote_callback(SimpleNamespace(keys=1, lx=0, ly=0, rx=0, ry=0))
        self.tick(12.7, 'PALM', 1001, .082)
        self.assertTrue(self.bridge.failed)
        self.assertIn('hardware remote', self.bridge.policy.reason)
        self.assertNotIn(('RecoveryStand',), self.client.calls)

    def test_cancel_never_automatically_stands(self):
        self.prone()
        self.tick(12.7, 'CANCEL', 1001, .082)
        self.assertTrue(self.bridge.failed)
        self.assertNotIn(('RecoveryStand',), self.client.calls)

    def test_lost_hand_aborts_pending_rise(self):
        self.rise()
        self.tick(13.5, None, 2004, .10)
        self.assertTrue(self.bridge.failed)

    def test_stale_state_and_tilt_are_not_masked_by_damping_context(self):
        self.prone()
        self.clock.return_value = 13.4
        self.assertIn('timed out', self.bridge.state_fault(13.4))
        self.tick(13.5, 'PALM', 1001, .082, rpy=(.5, 0, 0))
        self.assertTrue(self.bridge.failed)
        self.assertNotIn(('RecoveryStand',), self.client.calls)

    def test_hardware_fault_still_prevents_recovery(self):
        self.prone()
        monitor = control.HardwareFaultMonitor()
        monitor.lowstate_callback(lowstate_message(motor_fault=1))
        self.bridge.hardware_faults = monitor
        self.tick(12.7, 'PALM', 1001, .082)
        self.assertTrue(self.bridge.failed)
        self.assertNotIn(('RecoveryStand',), self.client.calls)

    def test_test_flags_are_mandatory_and_walking_stays_blocked(self):
        parser = control.build_parser()
        with patch.object(parser, 'error', side_effect=ValueError), self.assertRaises(ValueError):
            control.validate_args(parser, parser.parse_args(['--recovery-stand-test']))
        control.validate_args(parser, parser.parse_args([
            '--enable-motion', '--posture-only', '--native-posture-test', '--recovery-stand-test']))
        self.prone()
        self.tick(12.7, 'FORWARD', 1001, .082)
        self.tick(13.3, 'FORWARD', 1001, .082)
        self.assertFalse(any(call[0] == 'Move' for call in self.client.calls))
        self.assertTrue(control.sport_code_fault(1001, 'mcf_v2'))
        self.assertTrue(control.sport_code_fault(2004, 'mcf_v2'))


class HardwareFaultTests(unittest.TestCase):
    def test_callback_after_caller_clock_does_not_look_stale(self):
        monitor = control.HardwareFaultMonitor()
        with patch.object(control.time, 'monotonic', return_value=10.01):
            monitor.lowstate_callback(lowstate_message())
            self.assertEqual(monitor.fault(10.0), '')
        with patch.object(control.time, 'monotonic', return_value=10.8):
            self.assertTrue(monitor.fault(10.0))

    def test_only_documented_active_bitflag_faults_are_decoded(self):
        with patch.object(control.time, 'monotonic', return_value=10):
            monitor = control.HardwareFaultMonitor()
            monitor.lowstate_callback(lowstate_message(bit_flag=0x24))
            self.assertEqual(monitor.fault(10), '')
            self.assertEqual(monitor.status()['bit_flag'], 0x24)
            for bit in (0x02, 0x10, 0x40, 0x80):
                monitor = control.HardwareFaultMonitor()
                monitor.lowstate_callback(lowstate_message(bit_flag=bit))
                self.assertTrue(monitor.fault(10))

    def test_missing_stale_and_active_faults_block(self):
        monitor = control.HardwareFaultMonitor()
        self.assertTrue(monitor.fault(10))
        with patch.object(control.time, 'monotonic', return_value=10):
            monitor.lowstate_callback(lowstate_message())
            self.assertEqual(monitor.fault(10), '')
            self.assertTrue(monitor.fault(11))
            monitor.lowstate_callback(lowstate_message(motor_fault=0x100))
            monitor.lowstate_callback(lowstate_message())
            self.assertIn('motor 3 fault=0x100', monitor.fault(10))

    def test_communication_fault_blocks_even_in_freewalk(self):
        monitor = control.HardwareFaultMonitor()
        with patch.object(control.time, 'monotonic', return_value=10):
            monitor.lowstate_callback(lowstate_message(bit_flag=0x40))
            bridge = control.RobotMotionBridge(FakeClient(), True, profile='mcf_v2', hardware_faults=monitor)
            bridge.state_callback(state_message(error=100))
            bridge.camera_command('FORWARD', 10)
            bridge.step(10)
            self.assertTrue(bridge.failed)
            self.assertEqual(bridge.client.calls, [])

    def test_errorlist_active_and_malformed_payloads_latch(self):
        for payload in ('{"errors":[[123,300,1]]}', '{}', '{', '{"errors":null}'):
            monitor = control.HardwareFaultMonitor()
            with patch.object(control.time, 'monotonic', return_value=10):
                monitor.lowstate_callback(lowstate_message())
                monitor.errorlist_callback(SimpleNamespace(data=payload))
                monitor.errorlist_callback(SimpleNamespace(data='{"errors":[]}'))
                self.assertTrue(monitor.fault(10))


if __name__ == '__main__':
    unittest.main(verbosity=2)
