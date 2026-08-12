"""Central application configuration for RadarStream.

Only values that describe the installation or processing policy belong here.
Mutable, cross-thread flags live in :mod:`runtime_state` instead.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RadarFrameConfig:
    """Shape of one raw ADC frame produced by the radar."""

    adc_samples: int = 64
    chirps_per_tx: int = 64
    tx_antennas: int = 3
    rx_antennas: int = 4
    iq_components: int = 2

    def __post_init__(self):
        values = (
            self.adc_samples,
            self.chirps_per_tx,
            self.tx_antennas,
            self.rx_antennas,
            self.iq_components,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Radar frame dimensions must all be positive")
        if self.iq_components != 2:
            raise ValueError("RadarStream currently expects complex I/Q samples")

    @property
    def chirps_per_frame(self):
        return self.chirps_per_tx * self.tx_antennas

    @property
    def virtual_antennas(self):
        return self.tx_antennas * self.rx_antennas

    @property
    def raw_values_per_frame(self):
        """Number of int16 values written by the capture library per frame."""

        return (
            self.adc_samples
            * self.chirps_per_tx
            * self.tx_antennas
            * self.rx_antennas
            * self.iq_components
        )


@dataclass(frozen=True)
class DspConfig:
    """Feature extraction and gesture-trigger parameters."""

    history_frames: int = 12
    range_fft_bins: int = 64
    input_sample_limit: int = 64
    rti_sample_step: int = 2
    angle_sample_step: int = 3
    angle_range_degrees: int = 90
    angle_resolution_degrees: int = 2
    angle_fft_bins: int = 64
    bins_processed: int = 64
    beamforming_antennas: int = 4
    azimuth_channels: Tuple[int, ...] = (7, 4, 3, 0)
    elevation_channels: Tuple[int, ...] = (7, 6, 11, 10)
    elevation_phase_signs: Tuple[int, ...] = (1, -1, 1, -1)
    detection_range_start: int = 36
    detection_range_stop: int = 62
    detection_threshold: float = 3e3
    detection_min_points: int = 14
    detection_required_frames: int = 2
    gesture_delay_frames: int = 8
    rti_noise_floor: float = 3e3
    micro_doppler_noise_floor: float = 20.0
    angle_snr_offset: float = 14.7
    angle_snr_dead_zone: float = 1.8
    angle_range_cutoff: int = 40
    rti_display_stride: int = 16
    angle_history_start: int = 4
    angle_history_stop: int = 12

    def __post_init__(self):
        if self.history_frames <= 0 or self.range_fft_bins <= 0:
            raise ValueError("DSP history and FFT sizes must be positive")
        if self.history_frames < 12:
            raise ValueError("history_frames must be at least 12 for UI feature windows")
        if self.range_fft_bins != 64 or self.bins_processed != 64:
            raise ValueError(
                "The current feature model expects 64 range FFT bins; "
                "update DSP and model preprocessing together before changing them"
            )
        if self.bins_processed > self.range_fft_bins:
            raise ValueError("bins_processed cannot exceed range_fft_bins")
        if len(self.azimuth_channels) != self.beamforming_antennas:
            raise ValueError("azimuth_channels must match beamforming_antennas")
        if len(self.elevation_channels) != self.beamforming_antennas:
            raise ValueError("elevation_channels must match beamforming_antennas")
        if len(self.elevation_phase_signs) != self.beamforming_antennas:
            raise ValueError("elevation_phase_signs must match beamforming_antennas")
        if not 0 <= self.angle_history_start < self.angle_history_stop <= self.history_frames:
            raise ValueError("angle history display window must fit within history_frames")

    @property
    def angle_bins(self):
        return (self.angle_range_degrees * 2) // self.angle_resolution_degrees + 1


@dataclass(frozen=True)
class NetworkConfig:
    host_address: Tuple[str, int] = ("192.168.33.30", 4096)
    fpga_address: Tuple[str, int] = ("192.168.33.180", 4096)
    response_timeout_seconds: float = 2.0


@dataclass(frozen=True)
class SerialPortConfig:
    cli_baud_rate: int = 115200
    response_timeout_seconds: float = 0.1
    line_delay_seconds: float = 0.01


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    radar_config_dir: Path = PROJECT_ROOT / "radar_configs"
    model_dir: Path = PROJECT_ROOT / "model_checkpoints"
    dataset_dir: Path = PROJECT_ROOT / "dataset"
    assets_dir: Path = PROJECT_ROOT / "assets"
    media_dir: Path = PROJECT_ROOT / "assets" / "media"
    gesture_icon_dir: Path = PROJECT_ROOT / "assets" / "gesture_icons"
    cad_dir: Path = PROJECT_ROOT / "assets" / "cad"
    capture_library: Path = PROJECT_ROOT / "native" / "UDPCAPTUREADCRAWDATA.dll"

    def gesture_icon(self, gesture_id):
        return self.gesture_icon_dir / (str(gesture_id) + ".jpg")


@dataclass(frozen=True)
class AppConfig:
    radar: RadarFrameConfig = field(default_factory=RadarFrameConfig)
    dsp: DspConfig = field(default_factory=DspConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    serial: SerialPortConfig = field(default_factory=SerialPortConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    feature_queue_size: int = 2
    ui_refresh_milliseconds: int = 10
    gesture_interval_milliseconds: int = 2000
    rti_levels: Tuple[float, float] = (0, 1e4)
    rdi_levels: Tuple[float, float] = (2e4, 4e5)
    angle_levels: Tuple[float, float] = (0, 8)
    dti_levels: Tuple[float, float] = (0, 1000)

    def __post_init__(self):
        channel_indices = self.dsp.azimuth_channels + self.dsp.elevation_channels
        if max(channel_indices) >= self.radar.virtual_antennas:
            raise ValueError(
                "当前 DSP 天线映射至少需要 {} 路虚拟天线，但所选配置只有 {} 路；"
                "不同雷达的方位角/俯仰角阵列布局无法仅从 cfg 自动推断，请配置对应的 "
                "DspConfig 天线通道映射。".format(
                    max(channel_indices) + 1, self.radar.virtual_antennas
                )
            )
        if self.radar.chirps_per_tx < self.dsp.beamforming_antennas:
            raise ValueError(
                "chirps_per_tx must be at least the beamforming antenna count"
            )
        if self.feature_queue_size <= 0:
            raise ValueError("feature_queue_size must be positive")

    def with_radar_shape(
        self, adc_samples, chirps_per_tx, tx_antennas, rx_antennas
    ):
        """Return a runtime config whose frame shape comes from a radar cfg."""

        radar = RadarFrameConfig(
            adc_samples=adc_samples,
            chirps_per_tx=chirps_per_tx,
            tx_antennas=tx_antennas,
            rx_antennas=rx_antennas,
        )
        return replace(self, radar=radar)


# This is the fallback shape before a radar CLI file is selected. At runtime,
# the selected .cfg file is the source of truth and replaces ``radar``.
DEFAULT_CONFIG = AppConfig()
