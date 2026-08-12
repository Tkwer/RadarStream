"""RadarStream application entry point and UI controller."""

import html
import os
import sys
import time
from queue import Empty, Queue

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore, QtGui
from serial.tools import list_ports

from app_config import DEFAULT_CONFIG
from colormap_utils import pg_get_cmap
from data_pipeline import CaptureBuffer, DataProcessor, UdpListener
from hardware_interfaces import (
    Dca1000Controller,
    RadarCliClient,
    select_preferred_cli_port,
)
from radar_profile import parse_radar_profile_shape
from radar_tlv import Iwr6843TlvParser
from runtime_state import RuntimeState
from signal_processor import RadarSignalProcessor
from generated_ui import GestureOverlayWindow, Ui_MainWindow


GESTURE_NAMES = {
    0: "backward",
    1: "dbclick",
    2: "down",
    3: "front",
    4: "Left",
    5: "Right",
    6: "up",
    7: "NO",
}


class RadarStreamApplication:
    """Compose hardware, processing, runtime state and PyQt widgets."""

    def __init__(
        self,
        config=DEFAULT_CONFIG,
        model_factory=None,
        gesture_predictor=None,
    ):
        self.base_config = config
        self.config = config
        self.runtime_state = RuntimeState()
        self.model_factory = model_factory
        self.gesture_predictor = gesture_predictor
        self.model = None

        self.feature_queue = Queue(maxsize=config.feature_queue_size)
        self.signal_processor = RadarSignalProcessor(config, self.runtime_state)
        self.capture_buffer = None
        self.dca1000 = None
        self.collector = None
        self.processor = None
        self.radar_ctrl = None
        self.latest_features = None

        self.cli_port_name = ""
        self.dataset_scene_dir = None
        self.capture_index = 0

        self.qt_app = None
        self.main_window = None
        self.ui = None
        self.sub_window = None
        self.images = {}
        self.feature_views = {}

    def run(self):
        exit_code = 1
        try:
            self._build_ui()
            exit_code = self.qt_app.exec_()
        finally:
            self.shutdown()
        return exit_code

    def _ensure_capture_backend(self):
        """Connect DCA1000 and create capture threads on first hardware use."""

        if self.dca1000 is not None:
            return

        capture_buffer = None
        dca1000 = None
        collector = None
        try:
            capture_buffer = CaptureBuffer(
                self.config.radar.raw_values_per_frame,
                self.config.paths.capture_library,
            )
            dca1000 = Dca1000Controller(
                "Dca1000Controller", settings=self.config.network
            )
            collector = UdpListener("Listener", capture_buffer)
            processor = DataProcessor(
                "Processor",
                capture_buffer,
                self.signal_processor,
                self.feature_queue,
                self.config.radar,
            )
            collector.start()
        except Exception:
            if dca1000 is not None:
                try:
                    dca1000.close()
                except Exception:
                    pass
            if collector is not None and collector.ident is not None:
                collector.stop()
                collector.join(timeout=1)
            raise

        # Commit only after every initialization step succeeds. A failed
        # connection therefore leaves the application in a retryable state.
        self.capture_buffer = capture_buffer
        self.dca1000 = dca1000
        self.collector = collector
        self.processor = processor

    def _build_ui(self):
        self.qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.main_window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow(self.config, self.runtime_state)
        self.ui.setupUi(self.main_window)
        self.sub_window = GestureOverlayWindow(
            self.main_window, self.config.paths.gesture_icon_dir
        )

        self._configure_feature_views()
        self._connect_signals()
        self.update_com_ports()

        self.refresh_timer = QtCore.QTimer(self.main_window)
        self.refresh_timer.timeout.connect(self.update_figure)
        self.refresh_timer.start(self.config.ui_refresh_milliseconds)

        self.gesture_interval_timer = QtCore.QTimer(self.main_window)
        self.gesture_interval_timer.timeout.connect(
            self.runtime_state.open_gesture_interval
        )
        self.gesture_interval_timer.start(
            self.config.gesture_interval_milliseconds
        )
        self.runtime_state.open_gesture_interval()
        self.main_window.show()

    def _configure_feature_views(self):
        view_names = {
            "rdi": "graphicsView_6",
            "rai": "graphicsView_4",
            "rti": "graphicsView",
            "dti": "graphicsView_2",
            "rei": "graphicsView_3",
        }
        color_map = pg_get_cmap("customize")
        lookup_table = color_map.getLookupTable(0.0, 1.0, 256)

        # The generated UI used fixed 255 x 255 panels. Make all six feature
        # cells participate in the grid layout so they tile the available
        # window space instead of leaving unused margins.
        feature_widgets = [
            getattr(self.ui, widget_name) for widget_name in view_names.values()
        ] + [self.ui.graphicsView_5]
        for widget in feature_widgets:
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            widget.setMinimumSize(QtCore.QSize(160, 160))
            widget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        for column in range(3):
            self.ui.gridLayout.setColumnStretch(column, 1)

        for feature_name, widget_name in view_names.items():
            widget = getattr(self.ui, widget_name)
            # GraphicsLayoutWidget already owns a GraphicsLayout. Adding a
            # ViewBox to that layout and then setting the same ViewBox as the
            # central widget makes two layout systems fight over its geometry,
            # shrinking it to a strip in the top-left corner. Replace the
            # central item once with a standalone ViewBox instead.
            view = pg.ViewBox(enableMenu=False)
            view.setDefaultPadding(0.0)
            view.setAspectLocked(False)
            view.setMouseEnabled(x=False, y=False)
            widget.setCentralItem(view)
            image = pg.ImageItem(border=None)
            image.setLookupTable(lookup_table)
            view.addItem(image)
            view.enableAutoRange(x=True, y=True)
            self.feature_views[feature_name] = view
            self.images[feature_name] = image

        neutral_icon = str(self.config.paths.gesture_icon(7))
        self.ui.graphicsView_5.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.graphicsView_5.setPixmap(QtGui.QPixmap(neutral_icon))

    def _connect_signals(self):
        self.ui.comboBox_8.arrowClicked.connect(self.update_com_ports)
        self.ui.comboBox_8.currentIndexChanged.connect(self.set_serial_port)
        self.ui.comboBox.currentIndexChanged.connect(self.set_color)
        self.ui.comboBox_2.currentIndexChanged.connect(self.load_model)
        self.ui.comboBox_7.currentIndexChanged.connect(self.show_radar_parameters)
        self.ui.comboBox_3.currentIndexChanged.connect(self.select_dataset_scene)
        self.ui.lineEdit_6.editingFinished.connect(self.select_dataset_scene)
        self.ui.pushButton_11.clicked.connect(self.send_radar_config)
        self.ui.actionload.triggered.connect(self.show_sub_window)
        self.ui.pushButton_12.clicked.connect(self.qt_app.exit)

    def update_figure(self):
        try:
            while True:
                self.latest_features = self.feature_queue.get_nowait()
        except Empty:
            pass

        if self.latest_features is None:
            return

        features = self.latest_features
        dsp_config = self.config.dsp
        history_slice = slice(
            dsp_config.angle_history_start, dsp_config.angle_history_stop
        )
        rti_feature = features.rti.sum(2)[::dsp_config.rti_display_stride, :]
        self.images["rti"].setImage(rti_feature, levels=self.config.rti_levels)
        self.images["rdi"].setImage(
            features.rdi.sum(0)[:, :, 0].T, levels=self.config.rdi_levels
        )
        self.images["rei"].setImage(
            features.rei[history_slice].sum(0).T,
            levels=self.config.angle_levels,
        )
        self.images["dti"].setImage(features.dti, levels=self.config.dti_levels)
        self.images["rai"].setImage(
            features.rai[history_slice].sum(0),
            levels=self.config.angle_levels,
        )

        if self.runtime_state.consume_gesture():
            self._handle_gesture(features, rti_feature)

    def _handle_gesture(self, features, rti_feature):
        feature_views = (
            rti_feature,
            features.dti,
            features.rdi[:, :, :, 0],
            features.rai,
            features.rei,
        )
        if self.ui.pushButton_15.isChecked():
            start_time = time.time()
            result = self.judge_gesture(*feature_views)
            if result is not None:
                elapsed = time.time() - start_time
                self.print_log(
                    "识别时间:{:.4f}s, 识别结果:{}".format(elapsed, result),
                    "blue",
                )
        elif self.ui.pushButton.isChecked() and self.dataset_scene_dir is not None:
            self._save_feature_views(feature_views)

    def load_model(self):
        model_path = self.ui.comboBox_2.currentText()
        if model_path in ("", "--select--"):
            self.model = None
            return
        if self.model_factory is None:
            self.print_log("尚未配置 model_factory，无法实例化模型", "red")
            return

        import torch

        checkpoint = torch.load(model_path, map_location="cpu")
        self.model = self.model_factory()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.print_log("加载{}模型成功!".format(model_path), "blue")

    def judge_gesture(self, rti, dti, rdi, rai, rei):
        if self.model is None or self.gesture_predictor is None:
            self.print_log("模型或 gesture_predictor 尚未配置", "red")
            return None

        gesture_id = int(
            self.gesture_predictor(self.model, rai, dti, rei, rdi, rti)
        )
        gesture_name = GESTURE_NAMES.get(gesture_id, "unknown")
        icon_path = str(self.config.paths.gesture_icon(gesture_id))
        self.ui.graphicsView_5.setPixmap(QtGui.QPixmap(icon_path))
        self.sub_window.img_update(icon_path)
        QtCore.QTimer.singleShot(
            self.config.gesture_interval_milliseconds, self.clear_gesture_icon
        )
        self.print_log("输出:{}".format(gesture_name), "blue")
        return gesture_name

    def clear_gesture_icon(self):
        icon_path = str(self.config.paths.gesture_icon(7))
        self.ui.graphicsView_5.setPixmap(QtGui.QPixmap(icon_path))
        self.sub_window.img_update(icon_path)

    def _save_feature_views(self, feature_views):
        self.capture_index += 1
        names = ("RT", "DT", "RDT", "ART", "ERT")
        for name, feature in zip(names, feature_views):
            file_name = "{}_feature_{:05d}.npy".format(name, self.capture_index)
            np.save(str(self.dataset_scene_dir / file_name), feature)
        self.print_log(
            "采集到特征:{}-{:05d}".format(
                self.ui.comboBox_3.currentText(), self.capture_index
            ),
            "blue",
        )

    def select_dataset_scene(self):
        subject = self.ui.lineEdit_6.text().strip()
        scene = self.ui.comboBox_3.currentText().strip()
        if not subject or not scene:
            self.dataset_scene_dir = None
            return

        self.dataset_scene_dir = self.config.paths.dataset_dir / subject / scene
        self.dataset_scene_dir.mkdir(parents=True, exist_ok=True)
        self.capture_index = len(list(self.dataset_scene_dir.glob("DT_feature_*.npy")))

    def show_radar_parameters(self):
        config_path = self.ui.comboBox_7.currentText()
        if self.ui.comboBox_7.currentIndex() < 0 or config_path == "--select--":
            return
        self.ui.comboBox_7.setToolTip(config_path)
        parameters = Iwr6843TlvParser().parse_config(config_path)
        self.ui.label_14.setText(
            "{}m".format(parameters["rangeResolutionMeters"])
        )
        self.ui.label_35.setText(
            "{}m/s".format(parameters["dopplerResolutionMps"])
        )
        self.ui.label_16.setText("{}m".format(parameters["maxRange"]))
        self.ui.label_37.setText("{}m/s".format(parameters["maxVelocity"]))

    def update_com_ports(self):
        ports = list(list_ports.comports())
        preferred = select_preferred_cli_port(ports)
        previous = self.cli_port_name

        self.ui.comboBox_8.blockSignals(True)
        try:
            self.ui.comboBox_8.clear()
            for port in ports:
                self.ui.comboBox_8.addItem(port.device)
                index = self.ui.comboBox_8.count() - 1
                self.ui.comboBox_8.setItemData(
                    index, port.description, QtCore.Qt.ToolTipRole
                )

            available_devices = [port.device for port in ports]
            selected_device = previous
            if selected_device not in available_devices:
                selected_device = preferred.device if preferred is not None else ""
            selected_index = self.ui.comboBox_8.findText(selected_device)
            self.ui.comboBox_8.setCurrentIndex(selected_index)
            self.cli_port_name = selected_device if selected_index >= 0 else ""
        finally:
            self.ui.comboBox_8.blockSignals(False)

    def set_serial_port(self):
        if self.ui.comboBox_8.currentIndex() >= 0:
            self.cli_port_name = self.ui.comboBox_8.currentText()

    def send_radar_config(self):
        config_path = self.ui.comboBox_7.currentText()
        if not self.cli_port_name or config_path == "--select--":
            self.print_log("发送失败", "red")
            return
        try:
            profile_shape = parse_radar_profile_shape(config_path)
            self._apply_radar_profile(profile_shape)
            self.open_radar(config_path, self.cli_port_name)
        except Exception as error:
            self.print_log("发送失败: {}".format(error), "red")
            return
        self.print_log(
            "发送成功；已按配置使用 ADC/chirp/TX/RX={}，帧长度={} int16".format(
                profile_shape.as_tuple(), self.config.radar.raw_values_per_frame
            ),
            "green",
        )

    def _apply_radar_profile(self, profile_shape):
        runtime_config = self.base_config.with_radar_shape(
            *profile_shape.as_tuple()
        )
        if runtime_config.radar == self.config.radar:
            return

        feature_queue = Queue(maxsize=runtime_config.feature_queue_size)
        signal_processor = RadarSignalProcessor(
            runtime_config, self.runtime_state
        )

        # Native capture owns a frame-sized double buffer. Stop the previous
        # capture process before replacing it with the selected cfg's exact
        # frame length.
        self._stop_hardware()
        self.config = runtime_config
        if self.ui is not None:
            self.ui.config = runtime_config
        self.feature_queue = feature_queue
        self.signal_processor = signal_processor
        self.latest_features = None

    def open_radar(self, config_path, com_port):
        self._ensure_capture_backend()

        if self.radar_ctrl is not None:
            try:
                self.radar_ctrl.disconnect()
            finally:
                self.radar_ctrl = None

        radar_ctrl = RadarCliClient(
            name="ConnectRadar", cli_port=com_port, settings=self.config.serial
        )
        try:
            radar_ctrl.stop_radar()
            radar_ctrl.send_config(config_path)
        except Exception:
            try:
                radar_ctrl.disconnect()
            except Exception:
                # disconnect() always closes the serial port in its finally
                # block. Keep the original configuration error for the UI.
                pass
            raise
        self.radar_ctrl = radar_ctrl

        if self.processor.ident is None:
            self.processor.start()
        elif not self.processor.is_alive():
            self.processor = DataProcessor(
                "Processor",
                self.capture_buffer,
                self.signal_processor,
                self.feature_queue,
                self.config.radar,
            )
            self.processor.start()

    def set_color(self):
        color_name = self.ui.comboBox.currentText()
        if color_name in ("", "--select--"):
            return
        if color_name == "customize":
            color_map = pg_get_cmap(color_name)
        else:
            import matplotlib.pyplot as plt

            color_map = pg_get_cmap(plt.cm.get_cmap(color_name))
        lookup_table = color_map.getLookupTable(0.0, 1.0, 256)
        for image in self.images.values():
            image.setLookupTable(lookup_table)

    def show_sub_window(self):
        self.sub_window.show()
        self.main_window.hide()

    def print_log(self, message, color="green"):
        self.ui.textEdit.moveCursor(QtGui.QTextCursor.End)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.ui.textEdit.append(
            '<font color="{}">{}-->{}</font>'.format(
                html.escape(color), timestamp, html.escape(str(message))
            )
        )

    def shutdown(self):
        self._stop_hardware()

    def _stop_hardware(self):
        if self.radar_ctrl is not None:
            try:
                if self.radar_ctrl.is_open:
                    self.radar_ctrl.disconnect()
            except Exception:
                pass
            finally:
                self.radar_ctrl = None
        if self.processor is not None:
            self.processor.stop()
        if self.dca1000 is not None:
            try:
                self.dca1000.close()
            except Exception:
                pass
            finally:
                self.dca1000 = None
        if self.processor is not None and self.processor.ident is not None:
            self.processor.join(timeout=1)
        self.processor = None
        if self.collector is not None:
            if self.collector.ident is not None:
                self.collector.stop()
                self.collector.join(timeout=1)
        self.collector = None
        self.capture_buffer = None


def main():
    return RadarStreamApplication().run()


if __name__ == "__main__":
    sys.exit(main())
