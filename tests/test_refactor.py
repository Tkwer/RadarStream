import importlib.util
import os
import unittest
from ctypes import c_int
from multiprocessing import RawArray
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app_config import AppConfig, RadarFrameConfig
from signal_processor import RadarSignalProcessor
from data_pipeline import DataProcessor, UdpListener
from hardware_interfaces import (
    Dca1000Controller,
    HardwareConnectionError,
    select_preferred_cli_port,
)
from radar_dsp import utils
from radar_dsp.utils import Window
from radar_profile import RadarProfileShape, parse_radar_profile_shape
from runtime_state import RuntimeState


UI_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("PyQt5", "pyqtgraph", "serial")
)


class StubCaptureBuffer:
    active_half = 0


class ProcessStubCaptureBuffer:
    def __init__(self):
        self.value = RawArray(c_int, 1)

    def capture(self):
        self.value[0] = 7


class WinErrorSocket:
    """Socket double that reproduces Windows bind error 10049."""

    def __init__(self):
        self.closed = False
        self.timeout = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def bind(self, address):
        error = OSError("requested address is not valid in its context")
        error.winerror = 10049
        raise error

    def close(self):
        self.closed = True


class RuntimeStateTests(unittest.TestCase):
    def test_gesture_events_are_consumed_once(self):
        state = RuntimeState()

        self.assertFalse(state.processing_enabled)
        state.set_processing_enabled(True)
        self.assertTrue(state.processing_enabled)

        state.open_gesture_interval()
        self.assertTrue(state.claim_gesture_interval())
        self.assertFalse(state.claim_gesture_interval())

        state.mark_gesture_ready()
        self.assertTrue(state.consume_gesture())
        self.assertFalse(state.consume_gesture())


class ConfigurationTests(unittest.TestCase):
    def test_default_frame_size_is_derived_from_dimensions(self):
        radar = RadarFrameConfig()

        self.assertEqual(192, radar.chirps_per_frame)
        self.assertEqual(12, radar.virtual_antennas)
        self.assertEqual(98304, radar.raw_values_per_frame)

    def test_invalid_frame_dimensions_fail_early(self):
        with self.assertRaises(ValueError):
            RadarFrameConfig(adc_samples=0)

    def test_dsp_channel_mapping_is_validated_against_radar(self):
        with self.assertRaises(ValueError):
            AppConfig(radar=RadarFrameConfig(tx_antennas=2))

    def test_current_iwr6843_cfg_can_build_its_own_runtime_config(self):
        shape = parse_radar_profile_shape("radar_configs/iwr6843.cfg")

        runtime_config = AppConfig().with_radar_shape(*shape.as_tuple())

        self.assertEqual(3, shape.tx_antennas)
        self.assertEqual(4, shape.rx_antennas)
        self.assertEqual(shape.as_tuple(), (
            runtime_config.radar.adc_samples,
            runtime_config.radar.chirps_per_tx,
            runtime_config.radar.tx_antennas,
            runtime_config.radar.rx_antennas,
        ))

    def test_app_config_adapts_to_selected_radar_cfg_shape(self):
        shape = RadarProfileShape(32, 64, 3, 4)

        runtime_config = AppConfig().with_radar_shape(*shape.as_tuple())

        self.assertEqual((32, 64, 3, 4), shape.as_tuple())
        self.assertEqual(49152, runtime_config.radar.raw_values_per_frame)

    def test_profile_with_insufficient_virtual_antennas_is_rejected(self):
        shape = parse_radar_profile_shape("radar_configs/iwr1843.cfg")

        with self.assertRaisesRegex(ValueError, "天线映射"):
            AppConfig().with_radar_shape(*shape.as_tuple())

    def test_enhanced_cp2105_port_is_selected_for_radar_cli(self):
        ports = [
            SimpleNamespace(
                device="COM12",
                description="Standard Serial over Bluetooth link (COM12)",
                manufacturer="Microsoft",
                product=None,
                interface=None,
                hwid="BTHENUM\\device",
            ),
            SimpleNamespace(
                device="COM11",
                description=(
                    "Silicon Labs Dual CP2105 USB to UART Bridge: "
                    "Enhanced COM Port (COM11)"
                ),
                manufacturer="Silicon Labs",
                product=None,
                interface=None,
                hwid="USB VID:PID=10C4:EA70",
            ),
            SimpleNamespace(
                device="COM14",
                description=(
                    "Silicon Labs Dual CP2105 USB to UART Bridge: "
                    "Standard COM Port (COM14)"
                ),
                manufacturer="Silicon Labs",
                product=None,
                interface=None,
                hwid="USB VID:PID=10C4:EA70",
            ),
        ]

        preferred = select_preferred_cli_port(ports)

        self.assertEqual("COM11", preferred.device)

    def test_dca_command_builder_rejects_unknown_command(self):
        with self.assertRaises(ValueError):
            Dca1000Controller.build_command("unknown")

    def test_dca_start_command_has_expected_wire_format(self):
        self.assertEqual(
            "5aa505000000aaee", Dca1000Controller.build_command("5").hex()
        )

    def test_dca_bind_error_is_explained_and_socket_is_closed(self):
        fake_socket = WinErrorSocket()

        with patch("hardware_interfaces.socket.socket", return_value=fake_socket):
            with self.assertRaisesRegex(
                HardwareConnectionError, "NetworkConfig.host_address"
            ):
                Dca1000Controller("test")

        self.assertTrue(fake_socket.closed)


class DataProcessorTests(unittest.TestCase):
    def test_udp_listener_can_share_and_exit_cleanly(self):
        capture_buffer = ProcessStubCaptureBuffer()
        listener = UdpListener("test-listener", capture_buffer)

        listener.start()
        listener.join(timeout=5)

        self.assertFalse(listener.is_alive())
        self.assertEqual(7, capture_buffer.value[0])

    def test_decode_frame_uses_injected_radar_shape(self):
        radar = RadarFrameConfig(
            adc_samples=2,
            chirps_per_tx=2,
            tx_antennas=2,
            rx_antennas=2,
        )
        processor = DataProcessor(
            "test",
            StubCaptureBuffer(),
            signal_processor=object(),
            output_queue=Queue(),
            radar_config=radar,
        )
        raw = np.arange(radar.raw_values_per_frame, dtype=np.int16)

        decoded = processor._decode_frame(raw)

        self.assertEqual((2, 2, 4), decoded.shape)
        expected_first = raw.reshape(-1, 4)[0, 0] + 1j * raw.reshape(-1, 4)[0, 2]
        self.assertEqual(expected_first, decoded[0, 0, 0])

    def test_decode_frame_rejects_wrong_length(self):
        radar = RadarFrameConfig()
        processor = DataProcessor(
            "test",
            StubCaptureBuffer(),
            signal_processor=object(),
            output_queue=Queue(),
            radar_config=radar,
        )

        with self.assertRaises(ValueError):
            processor._decode_frame(np.zeros(4, dtype=np.int16))

    def test_default_decoded_frame_runs_windowed_range_processing(self):
        radar = RadarFrameConfig()
        processor = DataProcessor(
            "test",
            StubCaptureBuffer(),
            signal_processor=object(),
            output_queue=Queue(),
            radar_config=radar,
        )
        decoded = processor._decode_frame(
            np.zeros(radar.raw_values_per_frame, dtype=np.int16)
        )
        signal_processor = RadarSignalProcessor(AppConfig(), RuntimeState())

        rti, rdi, dti = signal_processor.process_time_features(
            decoded, window_type_1d=Window.HANNING
        )

        self.assertEqual((64, 64, 12), decoded.shape)
        self.assertEqual((64, 64, 1), rti.shape)
        self.assertEqual((1, 64, 64, 12), rdi.shape)
        self.assertEqual((1, 64), dti.shape)

    def test_32_sample_profile_keeps_standard_feature_shapes(self):
        config = AppConfig().with_radar_shape(32, 64, 3, 4)
        radar = config.radar
        processor = DataProcessor(
            "test",
            StubCaptureBuffer(),
            signal_processor=object(),
            output_queue=Queue(),
            radar_config=radar,
        )
        decoded = processor._decode_frame(
            np.zeros(radar.raw_values_per_frame, dtype=np.int16)
        )
        signal_processor = RadarSignalProcessor(config, RuntimeState())

        rti, rdi, dti = signal_processor.process_time_features(
            decoded, window_type_1d=Window.HANNING
        )
        rai, rei = signal_processor.process_angle_features(decoded)

        self.assertEqual((64, 32, 12), decoded.shape)
        self.assertEqual((64, 64, 1), rti.shape)
        self.assertEqual((1, 64, 64, 12), rdi.shape)
        self.assertEqual((1, 64), dti.shape)
        self.assertEqual((1, 91, 64), rai.shape)
        self.assertEqual((1, 91, 64), rei.shape)

    def test_non_64_chirp_profile_is_processed_without_shape_error(self):
        config = AppConfig().with_radar_shape(32, 32, 3, 4)
        radar = config.radar
        processor = DataProcessor(
            "test",
            StubCaptureBuffer(),
            signal_processor=object(),
            output_queue=Queue(),
            radar_config=radar,
        )
        decoded = processor._decode_frame(
            np.zeros(radar.raw_values_per_frame, dtype=np.int16)
        )
        signal_processor = RadarSignalProcessor(config, RuntimeState())

        rti, rdi, dti = signal_processor.process_time_features(
            decoded, window_type_1d=Window.HANNING
        )

        self.assertEqual((32, 64, 1), rti.shape)
        self.assertEqual((1, 64, 32, 12), rdi.shape)
        self.assertEqual((1, 32), dti.shape)


class DspUtilityTests(unittest.TestCase):
    def test_windowing_broadcasts_on_non_last_axis(self):
        data = np.ones((2, 3, 4))

        result = utils.windowing(data, Window.HANNING, axis=1)

        expected = np.hanning(3).reshape(1, 3, 1) * data
        np.testing.assert_array_equal(expected, result)


class SignalProcessorTests(unittest.TestCase):
    def test_default_processor_emits_expected_feature_shapes(self):
        random = np.random.RandomState(7)
        data = random.randn(64, 64, 12) + 1j * random.randn(64, 64, 12)
        processor = RadarSignalProcessor(AppConfig(), RuntimeState())

        rti, rdi, dti = processor.process_time_features(data)
        rai, rei = processor.process_angle_features(data)

        self.assertEqual((64, 64, 1), rti.shape)
        self.assertEqual((1, 64, 64, 12), rdi.shape)
        self.assertEqual((1, 64), dti.shape)
        self.assertEqual((1, 91, 64), rai.shape)
        self.assertEqual((1, 91, 64), rei.shape)

    def test_zero_frame_does_not_crash_angle_processing(self):
        processor = RadarSignalProcessor(AppConfig(), RuntimeState())
        data = np.zeros((64, 64, 12), dtype=np.complex128)

        rai, rei = processor.process_angle_features(data)

        self.assertTrue(np.isfinite(rai).all())
        self.assertTrue(np.isfinite(rei).all())

    def test_noncanonical_adc_layout_is_rejected(self):
        processor = RadarSignalProcessor(AppConfig(), RuntimeState())

        with self.assertRaisesRegex(ValueError, "ADC frame must use"):
            processor.process_time_features(
                np.zeros((64, 12, 64)), window_type_1d=Window.HANNING
            )


@unittest.skipUnless(UI_DEPENDENCIES_AVAILABLE, "optional UI dependencies unavailable")
class ApplicationLifecycleTests(unittest.TestCase):
    def test_application_run_does_not_initialize_capture_hardware(self):
        import main

        class ImmediateQtApplication:
            @staticmethod
            def exec_():
                return 0

        application = main.RadarStreamApplication()

        def build_ui_without_event_loop():
            application.qt_app = ImmediateQtApplication()

        application._build_ui = build_ui_without_event_loop
        with patch.object(
            main,
            "CaptureBuffer",
            side_effect=AssertionError("capture hardware initialized at startup"),
        ), patch.object(
            main,
            "Dca1000Controller",
            side_effect=AssertionError("DCA1000 initialized at startup"),
        ):
            self.assertEqual(0, application.run())

        self.assertIsNone(application.capture_buffer)
        self.assertIsNone(application.dca1000)

    def test_default_cli_port_is_enhanced_port(self):
        import main

        ports = [
            SimpleNamespace(
                device="COM14",
                description="CP2105 Standard COM Port",
                manufacturer="Silicon Labs",
                product=None,
                interface=None,
                hwid="USB VID:PID=10C4:EA70",
            ),
            SimpleNamespace(
                device="COM11",
                description="CP2105 Enhanced COM Port",
                manufacturer="Silicon Labs",
                product=None,
                interface=None,
                hwid="USB VID:PID=10C4:EA70",
            ),
        ]
        with patch.object(main.list_ports, "comports", return_value=ports):
            application = main.RadarStreamApplication()
            application._build_ui()
        try:
            self.assertEqual("COM11", application.cli_port_name)
            self.assertEqual("COM11", application.ui.comboBox_8.currentText())
        finally:
            application.refresh_timer.stop()
            application.gesture_interval_timer.stop()
            application.main_window.close()
            application.shutdown()

    def test_profile_change_rebuilds_runtime_processing_config(self):
        import main

        application = main.RadarStreamApplication()
        application.ui = SimpleNamespace(config=application.config)
        with patch.object(application, "_stop_hardware") as stop_hardware:
            application._apply_radar_profile(RadarProfileShape(32, 64, 3, 4))

        stop_hardware.assert_called_once_with()
        self.assertEqual(32, application.config.radar.adc_samples)
        self.assertEqual(49152, application.config.radar.raw_values_per_frame)
        self.assertEqual(application.config, application.signal_processor.config)
        self.assertEqual(application.config, application.ui.config)

    def test_feature_views_fill_responsive_grid_cells(self):
        import main

        application = main.RadarStreamApplication()
        application._build_ui()
        application.qt_app.processEvents()
        try:
            view_names = {
                "rdi": "graphicsView_6",
                "rai": "graphicsView_4",
                "rti": "graphicsView",
                "dti": "graphicsView_2",
                "rei": "graphicsView_3",
            }
            for feature_name, widget_name in view_names.items():
                widget = getattr(application.ui, widget_name)
                view = application.feature_views[feature_name]
                self.assertIs(view, widget.centralWidget)
                self.assertGreater(widget.maximumWidth(), 255)
                self.assertAlmostEqual(widget.width(), view.width(), delta=2)
                self.assertAlmostEqual(widget.height(), view.height(), delta=2)

                image = application.images[feature_name]
                image.setImage(np.ones((10, 20)))

            application.qt_app.processEvents()
            for feature_name in view_names:
                image = application.images[feature_name]
                view_range = application.feature_views[feature_name].viewRange()
                image_rect = image.boundingRect()
                np.testing.assert_allclose(
                    view_range[0], [image_rect.left(), image_rect.right()]
                )
                np.testing.assert_allclose(
                    view_range[1], [image_rect.top(), image_rect.bottom()]
                )
        finally:
            application.refresh_timer.stop()
            application.gesture_interval_timer.stop()
            application.main_window.close()
            application.shutdown()


if __name__ == "__main__":
    unittest.main()
