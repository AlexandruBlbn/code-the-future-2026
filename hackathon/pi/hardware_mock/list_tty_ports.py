#!/usr/bin/env python3
"""List the tty/serial ports currently visible on this Raspberry Pi.

Usage:
  python hardware_mock/list_tty_ports.py
  python hardware_mock/list_tty_ports.py --filter usb
  python hardware_mock/list_tty_ports.py --include-links
"""

from __future__ import annotations

import argparse
import re
import sys

from serial.tools import list_ports


def _matches(port, needle: str | None) -> bool:
    if not needle:
        return True

    text = " ".join(
        str(value)
        for value in (
            port.device,
            port.name,
            port.description,
            port.hwid,
            port.manufacturer,
            port.product,
            port.serial_number,
            port.location,
        )
        if value
    ).lower()
    return needle.lower() in text


def _format_row(port) -> str:
    description = port.description or "-"
    hwid = port.hwid or "-"
    manufacturer = port.manufacturer or "-"
    product = port.product or "-"
    serial_number = port.serial_number or "-"
    location = port.location or "-"
    return (
        f"{port.device:18} | {description:30.30} | {hwid:24.24} | "
        f"{manufacturer:14.14} | {product:16.16} | {serial_number:12.12} | {location}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="List active tty/serial ports")
    parser.add_argument(
        "--filter",
        default=None,
        help="Only show ports whose metadata contains this text",
    )
    parser.add_argument(
        "--include-links",
        action="store_true",
        help="Include symlinked ports reported by pyserial",
    )
    args = parser.parse_args()

    ports = list(list_ports.comports(include_links=args.include_links))
    ports = [port for port in ports if _matches(port, args.filter)]

    print("Detected tty/serial ports")
    print("-" * 120)
    print(
        f"{'DEVICE':18} | {'DESCRIPTION':30} | {'HWID':24} | {'MANUFACTURER':14} | "
        f"{'PRODUCT':16} | {'SERIAL':12} | LOCATION"
    )
    print("-" * 120)

    if not ports:
        print("No tty/serial ports found.")
        return 0

    for port in ports:
        print(_format_row(port))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())