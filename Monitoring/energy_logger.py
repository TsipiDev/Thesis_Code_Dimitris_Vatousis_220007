"""
--- Εξήγηση Κώδικα ---

Περιγραφή:
Συνδέεται με το Atorch UD18 μέσω Bluetooth SPP (/dev/rfcomm0)
και διαβάζει τα δεδομένα κατανάλωσης ισχύος σε πραγματικό χρόνο.

Πρωτόκολλο UD18:
  - 36-byte binary packets, big-endian
  - Starts: 0xFF 0x55
  - Verified offsets from real device packet:
    [4-6]   voltage  uint24 / 100   -> V
    [7-9]   current  uint24 / 100   -> A
    [10-13] capacity uint32         -> mAh (accumulated)
    [14-17] energy   uint32 / 100   -> Wh  (accumulated)
    [22-25] power    uint32 / 1000  -> W
    [26-27] USB D-   uint16 / 100   -> V
    [28-29] USB D+   uint16 / 100   -> V
    [35]    checksum byte
"""

import struct
import time
import serial

DEVICE_PORT  = "/dev/rfcomm0"
BAUD_RATE    = 9600
PACKET_LEN   = 36
START_BYTE_1 = 0xFF
START_BYTE_2 = 0x55
MSG_TYPE_DATA = 0x01


def _uint24(data: bytes, offset: int) -> int:
    return (data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2]


def _validate_checksum(packet: bytes) -> bool:
    return True


class UD18Data:
    def __init__(self, raw: bytes):
        self.voltage_V    = _uint24(raw, 4)  / 100.0   # offset 4-6
        self.current_A    = _uint24(raw, 7)  / 100.0   # offset 7-9  (divisor /100 not /1000)
        self.capacity_mAh = struct.unpack_from(">I", raw, 10)[0]          # offset 10-13
        self.energy_Wh    = struct.unpack_from(">I", raw, 14)[0] / 100.0  # offset 14-17
        self.power_W      = struct.unpack_from(">I", raw, 22)[0] / 1000.0 # offset 22-25 (not 10!)
        self.usb_dm_V     = struct.unpack_from(">H", raw, 26)[0] / 100.0  # offset 26-27
        self.usb_dp_V     = struct.unpack_from(">H", raw, 28)[0] / 100.0  # offset 28-29

    def __repr__(self):
        return (
            f"UD18Data(voltage={self.voltage_V}V, current={self.current_A}A, "
            f"power={self.power_W}W, capacity={self.capacity_mAh}mAh, "
            f"energy={self.energy_Wh}Wh)"
        )


class EnergyLogger:

    def __init__(self, port: str = DEVICE_PORT, max_retries: int = 10):
        self.port        = port
        self.max_retries = max_retries
        self.ser         = serial.Serial(port, BAUD_RATE, timeout=2)
        time.sleep(0.5)

    def _sync_and_read_packet(self) -> bytes:
        if not self.ser.is_open:
            self.ser.open()
            time.sleep(0.5)

        for attempt in range(self.max_retries):
            b = self.ser.read(1)
            if not b:
                print(f"[UD18] Attempt {attempt + 1}/{self.max_retries}: no data received")
                continue
            if b[0] != START_BYTE_1:
                print(f"[UD18] Attempt {attempt + 1}/{self.max_retries}: bad start byte 0x{b[0]:02X}, expected 0xFF")
                continue

            b2 = self.ser.read(1)
            if not b2 or b2[0] != START_BYTE_2:
                print(f"[UD18] Attempt {attempt + 1}/{self.max_retries}: bad second byte, expected 0x55")
                continue

            rest = self.ser.read(PACKET_LEN - 2)
            if len(rest) < PACKET_LEN - 2:
                print(f"[UD18] Attempt {attempt + 1}/{self.max_retries}: short packet ({len(rest)} bytes)")
                continue

            packet = bytes([START_BYTE_1, START_BYTE_2]) + rest

            if packet[2] != MSG_TYPE_DATA:
                print(f"[UD18] Attempt {attempt + 1}/{self.max_retries}: not a data packet (type=0x{packet[2]:02X})")
                continue

            return packet

        raise RuntimeError(
            f"[UD18] Failed to receive valid packet after {self.max_retries} attempts"
        )

    def read(self) -> UD18Data:
        packet = self._sync_and_read_packet()
        return UD18Data(packet)

    def get_power(self) -> float:
        return self.read().power_W

    def get_voltage(self) -> float:
        return self.read().voltage_V

    def get_current(self) -> float:
        return self.read().current_A

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()