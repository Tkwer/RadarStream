"""Parsing and validation helpers for TI mmWave CLI configuration files."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RadarProfileShape:
    adc_samples: int
    chirps_per_tx: int
    tx_antennas: int
    rx_antennas: int

    def as_tuple(self):
        return (
            self.adc_samples,
            self.chirps_per_tx,
            self.tx_antennas,
            self.rx_antennas,
        )


def parse_radar_profile_shape(config_path):
    lines = Path(config_path).read_text(encoding="utf-8").splitlines()
    profile = _find_command(lines, "profileCfg")
    channel = _find_command(lines, "channelCfg")
    frame = _find_command(lines, "frameCfg")
    chirp_lines = _find_commands(lines, "chirpCfg")
    if profile is None or channel is None or frame is None or not chirp_lines:
        raise ValueError(
            "Radar configuration requires channelCfg, profileCfg, chirpCfg and frameCfg"
        )

    frame_chirp_start = int(frame[1])
    frame_chirp_end = int(frame[2])
    tx_masks = {
        int(chirp[8])
        for chirp in chirp_lines
        if int(chirp[1]) <= frame_chirp_end
        and int(chirp[2]) >= frame_chirp_start
    }
    tx_antennas = len(
        {
            bit
            for mask in tx_masks
            for bit in range(mask.bit_length())
            if mask & (1 << bit)
        }
    )
    return RadarProfileShape(
        adc_samples=int(profile[10]),
        chirps_per_tx=int(frame[3]),
        tx_antennas=tx_antennas,
        rx_antennas=bin(int(channel[1])).count("1"),
    )


def validate_radar_profile(config_path, expected):
    actual = parse_radar_profile_shape(config_path).as_tuple()
    expected_shape = (
        expected.adc_samples,
        expected.chirps_per_tx,
        expected.tx_antennas,
        expected.rx_antennas,
    )
    if actual != expected_shape:
        raise ValueError(
            "Radar file ADC/chirp/TX/RX={} does not match app_config.py {}. "
            "Keep them consistent to prevent capture-buffer misalignment."
            .format(actual, expected_shape)
        )


def _find_command(lines, command):
    return next(iter(_find_commands(lines, command)), None)


def _find_commands(lines, command):
    return [
        line.split()
        for line in lines
        if line.strip().startswith(command + " ")
    ]
