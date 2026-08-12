"""Threads and buffer adapters for real-time radar capture."""

import time
from ctypes import POINTER, c_int, c_short, cast, cdll
from dataclasses import dataclass
from multiprocessing import Process, RawArray
from pathlib import Path
from queue import Empty, Full
from threading import Event, Thread

import numpy as np

from app_config import DEFAULT_CONFIG, RadarFrameConfig
from radar_dsp.utils import Window


@dataclass(frozen=True)
class FeatureFrame:
    """All feature views derived from the same ADC frame."""

    sequence: int
    rti: np.ndarray
    dti: np.ndarray
    rdi: np.ndarray
    rai: np.ndarray
    rei: np.ndarray


class CaptureBuffer:
    """Own the double buffer shared with the native UDP capture library."""

    def __init__(self, frame_length, library_path):
        self.frame_length = frame_length
        self.library_path = str(Path(library_path).resolve())
        if not Path(self.library_path).is_file():
            raise FileNotFoundError(
                "Native capture library not found: {}".format(self.library_path)
            )
        # RawArray keeps the two native buffers visible to the parent while
        # captureudp runs in a child process. A process can be terminated and
        # restarted when a selected radar cfg changes the frame length.
        self._active_half = RawArray(c_int, 1)
        self._samples = RawArray(c_short, frame_length * 2)

    @property
    def active_half(self):
        return self._active_half[0]

    def capture(self):
        library = cdll.LoadLibrary(self.library_path)
        library.captureudp(
            cast(self._active_half, POINTER(c_int)),
            cast(self._samples, POINTER(c_short)),
            self.frame_length,
        )

    def copy_latest_frame(self, active_half=None):
        if active_half is None:
            active_half = self.active_half
        start = self.frame_length * (1 - active_half)
        end = start + self.frame_length
        samples = np.frombuffer(self._samples, dtype=np.int16)
        return samples[start:end].copy()


class UdpListener(Process):
    """Run blocking native UDP capture in a restartable child process."""

    def __init__(self, name, capture_buffer):
        super().__init__(name=name, daemon=True)
        self.capture_buffer = capture_buffer

    def run(self):
        self.capture_buffer.capture()

    def stop(self):
        if self.is_alive():
            self.terminate()


class DataProcessor(Thread):
    """Decode raw frames and publish synchronized DSP feature frames."""

    def __init__(
        self,
        name,
        capture_buffer,
        signal_processor,
        output_queue,
        radar_config=DEFAULT_CONFIG.radar,
    ):
        super().__init__(name=name, daemon=True)
        if not isinstance(radar_config, RadarFrameConfig):
            raise TypeError("radar_config must be a RadarFrameConfig")
        self.capture_buffer = capture_buffer
        self.signal_processor = signal_processor
        self.output_queue = output_queue
        self.radar_config = radar_config
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        sequence = 0
        last_active_half = self.capture_buffer.active_half
        while not self._stop_event.is_set():
            active_half = self.capture_buffer.active_half
            if active_half == last_active_half:
                time.sleep(0.001)
                continue

            last_active_half = active_half
            adc_frame = self._decode_frame(
                self.capture_buffer.copy_latest_frame(active_half)
            )
            sequence += 1
            rti, rdi, dti = self.signal_processor.process_time_features(
                adc_frame, window_type_1d=Window.HANNING
            )
            rai, rei = self.signal_processor.process_angle_features(adc_frame)
            self._publish_latest(FeatureFrame(sequence, rti, dti, rdi, rai, rei))

    def _decode_frame(self, raw_frame):
        expected = self.radar_config.raw_values_per_frame
        if raw_frame.size != expected:
            raise ValueError(
                "Expected {} raw values, received {}".format(expected, raw_frame.size)
            )

        # DCA1000 complex mode packs [I0, I1, Q0, Q1] repeatedly.
        packed = raw_frame.reshape(-1, 4)
        complex_samples = packed[:, 0:2] + 1j * packed[:, 2:4]
        adc_data = complex_samples.reshape(
            self.radar_config.chirps_per_frame,
            self.radar_config.rx_antennas,
            self.radar_config.adc_samples,
        )
        adc_data = adc_data.transpose((0, 2, 1))

        tx_channels = [
            adc_data[tx_index::self.radar_config.tx_antennas]
            for tx_index in range(self.radar_config.tx_antennas)
        ]
        return np.concatenate(tx_channels, axis=2)

    def _publish_latest(self, feature_frame):
        try:
            self.output_queue.put_nowait(feature_frame)
            return
        except Full:
            pass

        try:
            self.output_queue.get_nowait()
        except Empty:
            pass
        self.output_queue.put_nowait(feature_frame)
