"""Hardware communication adapters for the radar EVM and DCA1000."""

import socket
import time

from app_config import NetworkConfig, SerialPortConfig


class HardwareConnectionError(ConnectionError):
    """Raised when a radar capture interface cannot be initialized."""


def select_preferred_cli_port(ports):
    """Choose the radar configuration UART from discovered serial ports.

    TI XDS110 boards expose an Application/User UART, while CP2105-based
    boards expose Enhanced and Standard ports. The Application/User or
    Enhanced port is the CLI/configuration port.
    """

    def score(port):
        fields = (
            getattr(port, "description", ""),
            getattr(port, "manufacturer", ""),
            getattr(port, "product", ""),
            getattr(port, "interface", ""),
            getattr(port, "hwid", ""),
        )
        text = " ".join(str(field or "") for field in fields).lower()
        value = 0
        if "application/user" in text or "application uart" in text:
            value += 120
        if "enhanced" in text or "增强" in text:
            value += 110
        if "xds110" in text:
            value += 40
        if "cp2105" in text or "10c4:ea70" in text:
            value += 30
        if "standard" in text or "data port" in text or "auxiliary" in text:
            value -= 100
        if "bluetooth" in text or "bthenum" in text or "蓝牙" in text:
            value -= 200
        return value

    candidates = list(ports)
    if not candidates:
        return None
    _, preferred = max(
        enumerate(candidates), key=lambda item: (score(item[1]), -item[0])
    )
    return preferred if score(preferred) > 0 else None


class RadarCliClient:
    """Send CLI configuration commands to a TI radar evaluation module."""

    def __init__(self, name, cli_port, baud_rate=None, settings=None):
        import serial

        self.name = name
        self.settings = settings or SerialPortConfig()
        self.cli_port = serial.Serial(
            cli_port,
            baudrate=baud_rate or self.settings.cli_baud_rate,
        )

    @property
    def is_open(self):
        return self.cli_port.is_open

    def send_config(self, config_file_name):
        with open(config_file_name, encoding="utf-8") as config_file:
            for line in config_file:
                self._send_line(line)

    def _send_line(self, line):
        self.cli_port.write((line.rstrip("\r\n") + "\n").encode())
        print("Sent: {}".format(line.strip()))

        start_time = time.time()
        response = b""
        while time.time() - start_time < self.settings.response_timeout_seconds:
            if self.cli_port.in_waiting > 0:
                response += self.cli_port.read(self.cli_port.in_waiting)
            time.sleep(self.settings.line_delay_seconds)

        print("Received: {}".format(response.decode(errors="ignore").strip()))
        time.sleep(self.settings.line_delay_seconds)

    def start_radar(self):
        self.cli_port.write(b"sensorStart\n")

    def stop_radar(self):
        self.cli_port.write(b"sensorStop\n")

    def close(self):
        self.cli_port.close()

    def disconnect(self):
        try:
            self.stop_radar()
        finally:
            self.close()


class Dca1000Controller:
    """Configure and control the DCA1000 FPGA over UDP."""

    INITIAL_COMMANDS = ("9", "E", "3", "B", "5")
    COMMAND_CODES = {
        "3": 0x03,
        "5": 0x05,
        "6": 0x06,
        "9": 0x09,
        "B": 0x0B,
        "E": 0x0E,
    }
    COMMAND_PAYLOADS = {
        "3": (0x01020102031E).to_bytes(6, byteorder="big", signed=False),
        "B": (0xC005350C0000).to_bytes(6, byteorder="big", signed=False),
    }

    def __init__(
        self,
        name,
        config_address=None,
        fpga_address=None,
        settings=None,
    ):
        settings = settings or NetworkConfig()
        self.name = name
        self.config_address = config_address or settings.host_address
        self.fpga_address = fpga_address or settings.fpga_address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(settings.response_timeout_seconds)
        try:
            self.socket.bind(self.config_address)
            for command in self.INITIAL_COMMANDS:
                self._send_and_receive(command)
        except OSError as error:
            self.socket.close()
            if getattr(error, "winerror", None) == 10049:
                raise HardwareConnectionError(
                    "无法绑定DCA1000本机地址 {}:{}；请先将采集网卡IPv4配置为该地址，"
                    "或修改 app_config.py 中的 NetworkConfig.host_address。".format(
                        *self.config_address
                    )
                ) from error
            raise HardwareConnectionError(
                "连接DCA1000失败（本机 {}:{}，设备 {}:{}）：{}".format(
                    *self.config_address, *self.fpga_address, error
                )
            ) from error
        except Exception:
            self.socket.close()
            raise

    def _send_and_receive(self, command):
        self.socket.sendto(self.build_command(command), self.fpga_address)
        time.sleep(0.1)
        self.socket.recvfrom(2048)

    def close(self):
        try:
            self.socket.sendto(self.build_command("6"), self.fpga_address)
        finally:
            self.socket.close()

    @classmethod
    def build_command(cls, command):
        try:
            command_code = cls.COMMAND_CODES[command]
        except KeyError:
            raise ValueError("Unsupported DCA1000 command: {}".format(command))

        header = (0xA55A).to_bytes(2, byteorder="little", signed=False)
        footer = (0xEEAA).to_bytes(2, byteorder="little", signed=False)
        code = command_code.to_bytes(2, byteorder="little", signed=False)
        payload = cls.COMMAND_PAYLOADS.get(command, b"")
        data_size = len(payload).to_bytes(2, byteorder="little", signed=False)
        return header + code + data_size + payload + footer
