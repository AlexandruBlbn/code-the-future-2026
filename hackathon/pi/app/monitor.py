#!/usr/bin/env python3
"""
MediTwin AI — Raspberry Pi Monitor
====================================
Primeste JSON de la ESP32 pe UART, decide starea FSM si trimite comenzi inapoi.

Protocol:
  ESP32 → Pi  (JSON line, la fiecare 2 s):
    {"ts":12345,"temp":25.30,"hum":60.5,"air":2450,"light":1800,
     "dist":45.2,"accel":1.02,"gx":0.01,"gy":0.00,"gz":0.00,
     "cur":0.000,"dht":1,"mpu":1,"state":"SAFE","risk":5.0}

  Pi → ESP32  (CMD:<TOKEN>\n):
    CMD:FAN_ON  CMD:FAN_OFF          — control manual ventilator
    CMD:HC_ON   CMD:HC_OFF           — activare/dezactivare alarma HC-SR04
    CMD:ALARM_ON  CMD:ALARM_OFF      — alarma automata FSM
    CMD:FSM_RESET                    — recalibrare

Rulare:
  python3 raspberry/monitor.py --port /dev/ttyAMA0 --baud 115200
  python3 raspberry/monitor.py --port /dev/ttyUSB0 --baud 115200   # via USB-UART

Instalare dependinte:
  pip3 install pyserial
"""

import argparse
import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import serial  # pip3 install pyserial
from serial.tools import list_ports

# ─── Configuratie praguri ───────────────────────────────────────────────────
TEMP_WARNING_C   = 25.5    # avertizare timpurie sub pragul critic
TEMP_CRITICAL_C  = 26.0
AIR_WARNING      = 2700    # ADC raw (fallback fara calibrare ESP32)
AIR_CRITICAL     = 3100
RISK_WARNING     = 25.0    # risk score (0-100)
RISK_CRITICAL    = 55.0
FAN_ON_RISK      = 40.0    # porneste fan cand risk depaseste pragul
FAN_OFF_RISK     = 20.0    # opreste fan cand risk scade sub prag
TEMP_TREND_RISE  = 1.0     # °C/min — crestere rapida = WARNING
TEMP_HISTORY_LEN = 10      # numarul de masuratori pastrate pentru trend
RECONNECT_DELAY_S = 2.0    # reconectare UART automata
SUMMARY_INTERVAL_S = 2.0   # interval linie compacta de date
BACKEND_INGEST_URL = "http://127.0.0.1:8000/ingest"
BACKEND_TIMEOUT_S = 2.0
UNIX_TS_MIN = 1_600_000_000

# ─── State machine ──────────────────────────────────────────────────────────
class FsmState:
    SAFE     = "SAFE"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("meditwin")


@dataclass
class MonitorState:
    fsm_state:  str   = FsmState.SAFE
    fan_on:     bool  = False
    fan_manual_override: bool = False
    fan_command_style: str = "both"
    uart_debug: bool = False
    alarm_on:   bool  = False
    hc_enabled: bool  = False
    # istoric temperatura (ts_s, temp_c)
    temp_history: deque = field(default_factory=lambda: deque(maxlen=TEMP_HISTORY_LEN))


def _resolve_port(requested_port: str) -> str:
    """Alege portul serial efectiv.

    Prioritizeaza portul cerut explicit; daca este "auto", incearca porturile
    disponibile si prefera UART-ul intern Raspberry Pi.
    """
    if requested_port and requested_port.lower() != "auto":
        return requested_port

    available_ports = [port.device for port in list_ports.comports()]
    preferred_ports = (
        "/dev/ttyAMA0",
        "/dev/serial0",
        "/dev/ttyS0",
    )

    for candidate in preferred_ports:
        if candidate in available_ports:
            return candidate

    if available_ports:
        return available_ports[0]

    raise FileNotFoundError(
        "Nu exista niciun port serial disponibil. "
        "Conecteaza ESP32/USB-UART sau porneste cu --port explicit."
    )


def _temp_trend_c_per_min(history: deque) -> float:
    """Calculeaza panta temperaturii in °C/min pe ultimele masuratori."""
    if len(history) < 2:
        return 0.0
    oldest_ts, oldest_t = history[0]
    newest_ts, newest_t = history[-1]
    dt_min = (newest_ts - oldest_ts) / 60.0
    if dt_min < 0.05:
        return 0.0
    return (newest_t - oldest_t) / dt_min


def decide(data: dict, state: MonitorState, ser: serial.Serial, lock: threading.Lock) -> None:
    """Aplica logica de decizie si trimite comenzi ESP32 daca e necesar."""
    now_s   = time.time()
    risk    = float(data.get("risk", 0))
    temp    = float(data.get("temp", 0))
    air     = int(data.get("air", 0))
    dht_ok  = bool(data.get("dht", 0))
    esp_fsm = data.get("state", "SAFE")

    if dht_ok:
        state.temp_history.append((now_s, temp))

    trend = _temp_trend_c_per_min(state.temp_history)

    # ── Determinare stare noua ───────────────────────────────────────────────
    new_state = FsmState.SAFE

    if (risk >= RISK_CRITICAL
            or (dht_ok and temp >= TEMP_CRITICAL_C)
            or air >= AIR_CRITICAL
            or esp_fsm == FsmState.CRITICAL):
        new_state = FsmState.CRITICAL

    elif (risk >= RISK_WARNING
            or (dht_ok and temp >= TEMP_WARNING_C)
            or air >= AIR_WARNING
            or trend >= TEMP_TREND_RISE
            or esp_fsm == FsmState.WARNING):
        new_state = FsmState.WARNING

    # ── Tranzitii de stare ───────────────────────────────────────────────────
    if new_state != state.fsm_state:
        log.info("FSM: %s → %s  (risk=%.0f temp=%.1fC air=%d trend=%.2f°C/min)",
                 state.fsm_state, new_state, risk, temp, air, trend)
        state.fsm_state = new_state

        if new_state == FsmState.CRITICAL:
            _send(ser, "CMD:ALARM_ON", lock, state)
            if not state.alarm_on:
                state.alarm_on = True
        elif new_state == FsmState.WARNING:
            # la WARNING nu declanseaza alarma — doar logam
            pass
        else:  # SAFE
            if state.alarm_on:
                _send(ser, "CMD:ALARM_OFF", lock, state)
                state.alarm_on = False

    # ── Control fan independent de stare ────────────────────────────────────
    # Daca utilizatorul a pornit manual ventilatorul, nu mai intervenim automat
    # pana cand primim explicit FAN_OFF.
    if not state.fan_manual_override:
        if risk >= FAN_ON_RISK and not state.fan_on:
            _send_fan_command(ser, lock, state, True)
            state.fan_on = True
            log.info("Fan ON  (risk=%.0f)", risk)
        elif risk < FAN_OFF_RISK and state.fan_on:
            _send_fan_command(ser, lock, state, False)
            state.fan_on = False
            log.info("Fan OFF (risk=%.0f)", risk)

    # Log periodic status
    log.debug("  risk=%.0f  temp=%.1fC(trend%.2f)  air=%d  esp=%s  pi=%s  fan=%s",
              risk, temp, trend, air, esp_fsm, state.fsm_state,
              "ON" if state.fan_on else "OFF")


def _send(ser: serial.Serial, cmd: str, lock: threading.Lock, state: MonitorState) -> None:
    line = cmd + "\n"
    with lock:
        payload = line.encode()
        written = ser.write(payload)
        ser.flush()
    log.info("\u2192 Pi cmd: %s", cmd)
    if state.uart_debug:
        log.info(
            "UART TX debug: wrote=%d bytes hex=%s ascii=%r out_waiting=%d",
            written,
            payload.hex(" "),
            payload,
            getattr(ser, "out_waiting", -1),
        )


def _send_fan_command(ser: serial.Serial, lock: threading.Lock, state: MonitorState, enabled: bool) -> None:
    if state.fan_command_style == "raw":
        commands = ["FAN_ON" if enabled else "FAN_OFF"]
    elif state.fan_command_style == "both":
        commands = [
            "CMD:FAN_ON" if enabled else "CMD:FAN_OFF",
            "FAN_ON" if enabled else "FAN_OFF",
        ]
    else:
        commands = ["CMD:FAN_ON" if enabled else "CMD:FAN_OFF"]

    for command in commands:
        _send(ser, command, lock, state)


def _send_hc_command(ser: serial.Serial, lock: threading.Lock, state: MonitorState, enabled: bool) -> None:
    if state.fan_command_style == "raw":
        commands = ["HC_ON" if enabled else "HC_OFF"]
    elif state.fan_command_style == "both":
        commands = [
            "CMD:HC_ON" if enabled else "CMD:HC_OFF",
            "HC_ON" if enabled else "HC_OFF",
        ]
    else:
        commands = ["CMD:HC_ON" if enabled else "CMD:HC_OFF"]

    for command in commands:
        _send(ser, command, lock, state)


def _reset_safety(ser: serial.Serial, lock: threading.Lock, state: MonitorState) -> None:
    """Opreste alarmele si cere recalibrare FSM/baseline pe ESP32."""
    _send(ser, "CMD:ALARM_OFF", lock, state)
    _send_fan_command(ser, lock, state, False)
    _send(ser, "CMD:FSM_RESET", lock, state)

    state.alarm_on = False
    state.fan_on = False
    state.fan_manual_override = False
    state.fsm_state = FsmState.SAFE
    state.temp_history.clear()


def _to_backend_reading(data: dict) -> dict:
    """Mapeaza payload-ul ESP32 pe schema FastAPI SensorReading."""
    raw_ts = data.get("ts", time.time())
    try:
        ts = int(raw_ts)
    except (TypeError, ValueError):
        ts = int(time.time())

    # ESP32 trimite uneori uptime in milisecunde; backend-ul asteapta epoch sec.
    if ts < UNIX_TS_MIN:
        ts = int(time.time())

    payload: dict = {
        "ts": ts,
    }

    if "temp" in data:
        payload["airTempC"] = float(data["temp"])
    if "hum" in data:
        payload["humidityPct"] = float(data["hum"])
    if "air" in data:
        payload["airQualityRaw"] = int(data["air"])
    if "light" in data:
        payload["lightRaw"] = int(data["light"])
    if "dist" in data:
        payload["lidDistanceCm"] = float(data["dist"])
        payload["lidOpen"] = float(data["dist"]) > 5.0
    if "accel" in data:
        # ESP trimite magnitudinea acceleratiei; o mapam pe axa Z ca fallback.
        payload["accelZ"] = float(data["accel"])
    if "gx" in data:
        payload["gyroX"] = float(data["gx"])
    if "gy" in data:
        payload["gyroY"] = float(data["gy"])
    if "gz" in data:
        payload["gyroZ"] = float(data["gz"])
    if "cur" in data:
        payload["heaterCurrentA"] = float(data["cur"])
        payload["heaterActive"] = float(data["cur"]) > 0.1

    return payload


def _post_backend_ingest(payload: dict) -> int:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BACKEND_INGEST_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=BACKEND_TIMEOUT_S) as resp:
        return resp.status


def _cli_thread(serial_ref: dict, lock: threading.Lock, state: MonitorState) -> None:
    """Thread interactiv — citeste comenzi manuale de la tastatura (stdin)."""
    print("\n+-- Comenzi manuale disponiblile: -------------------+")
    print("|  +  sau FAN_ON   => porneste ventilatorul (30 s)   |")
    print("|  -  sau FAN_OFF  => opreste ventilatorul           |")
    print("|  h  sau HC       => toggle alarma HC-SR04 ON/OFF   |")
    print("|  b  sau BTN      => opreste + recalibrare FSM      |")
    print("|  q  sau QUIT     => opreste monitorul              |")
    print("+----------------------------------------------------+\n")

    for raw_line in sys.stdin:
        cmd_in = raw_line.strip().lower()
        ser = serial_ref.get("ser")
        if cmd_in in ("+", "fan_on", "fan+", "-", "fan_off", "fan-", "h", "hc", "hc_on", "hc_off", "b", "btn", "reset") and ser is None:
            print("[CLI] UART deconectat. Astept reconectare...")
            continue
        if cmd_in in ("+", "fan_on", "fan+"):
            _send_fan_command(ser, lock, state, True)
            state.fan_on = True
            state.fan_manual_override = True
            print("[CLI] FAN_ON trimis")
        elif cmd_in in ("-", "fan_off", "fan-"):
            _send_fan_command(ser, lock, state, False)
            state.fan_on = False
            state.fan_manual_override = False
            print("[CLI] FAN_OFF trimis")
        elif cmd_in in ("h", "hc", "hc_on", "hc_off"):
            if not state.hc_enabled:
                _send_hc_command(ser, lock, state, True)
                state.hc_enabled = True
                print("[CLI] HC_ON trimis — alarma HC-SR04 activa")
            else:
                _send_hc_command(ser, lock, state, False)
                state.hc_enabled = False
                print("[CLI] HC_OFF trimis — alarma HC-SR04 dezactivata")
        elif cmd_in in ("b", "btn", "reset"):
            _reset_safety(ser, lock, state)
            print("[CLI] OPRIRE + RECALIBRARE trimisa")
        elif cmd_in in ("q", "quit", "exit"):
            print("[CLI] Oprire monitor...")
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)
            break
        elif cmd_in == "?" or cmd_in == "help":
            print("+  FAN_ON | -  FAN_OFF | h  HC toggle | b reset | q quit")
        elif cmd_in:
            print(f"[CLI] Comanda necunoscuta: '{cmd_in}' (? = help)")


def run(
    port: str,
    baud: int,
    fan_command_style: str = "both",
    uart_debug: bool = False,
) -> None:
    port = _resolve_port(port)
    log.info("MediTwin Monitor pornit pe %s @ %d baud", port, baud)

    state = MonitorState(fan_command_style=fan_command_style, uart_debug=uart_debug)
    serial_ref = {"ser": None}
    lock  = threading.Lock()
    buf   = ""
    last_rx_at = time.monotonic()
    last_idle_log_at = 0.0
    last_summary_at = 0.0
    last_backend_error_log_at = 0.0

    # Porneste thread interactiv CLI
    t = threading.Thread(target=_cli_thread, args=(serial_ref, lock, state), daemon=True)
    t.start()

    stopped_by_user = False
    try:
        while True:
            if serial_ref["ser"] is None:
                try:
                    serial_ref["ser"] = serial.Serial(port, baud, timeout=3.0)
                    log.info("UART conectat pe %s", port)
                    buf = ""
                    last_rx_at = time.monotonic()
                    last_idle_log_at = 0.0
                except serial.SerialException as e:
                    log.warning("Nu pot deschide %s: %s. Reincerc in %.1fs...", port, e, RECONNECT_DELAY_S)
                    time.sleep(RECONNECT_DELAY_S)
                    continue

            try:
                ser = serial_ref["ser"]
                raw = ser.read(ser.in_waiting or 1).decode("utf-8", errors="replace")
            except serial.SerialException as e:
                log.error("Eroare UART pe %s: %s", port, e)
                log.warning("Portul a fost pierdut. Reincerc conectarea...")
                try:
                    if serial_ref["ser"] and serial_ref["ser"].is_open:
                        serial_ref["ser"].close()
                except Exception:
                    pass
                serial_ref["ser"] = None
                time.sleep(RECONNECT_DELAY_S)
                continue

            if raw:
                if state.uart_debug:
                    payload = raw.encode("utf-8", errors="replace")
                    log.info(
                        "UART RX debug: got=%d bytes hex=%s ascii=%r",
                        len(payload),
                        payload.hex(" "),
                        payload,
                    )
                last_rx_at = time.monotonic()
            elif time.monotonic() - last_rx_at >= 5.0 and time.monotonic() - last_idle_log_at >= 5.0:
                log.info("Astept date de la ESP32 pe %s...", port)
                last_idle_log_at = time.monotonic()
            buf += raw
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)

                    # Forward UART telemetry to backend for frontend display.
                    try:
                        status = _post_backend_ingest(_to_backend_reading(data))
                        if status >= 400:
                            now = time.monotonic()
                            if now - last_backend_error_log_at >= 5.0:
                                log.warning("Backend /ingest returned HTTP %s", status)
                                last_backend_error_log_at = now
                    except (urllib.error.URLError, TimeoutError, ValueError) as e:
                        now = time.monotonic()
                        if now - last_backend_error_log_at >= 5.0:
                            log.warning("Nu pot trimite la backend (%s): %s", BACKEND_INGEST_URL, e)
                            last_backend_error_log_at = now

                    decide(data, state, ser, lock)
                    now = time.monotonic()
                    if now - last_summary_at >= SUMMARY_INTERVAL_S:
                        log.info(
                            "DATA T=%.1fC H=%.1f%% AQ=%s L=%s D=%.1fcm A=%.3f I=%.3fA risk=%.1f esp=%s pi=%s fan=%s",
                            float(data.get("temp", 0.0)),
                            float(data.get("hum", 0.0)),
                            data.get("air", "-"),
                            data.get("light", "-"),
                            float(data.get("dist", 0.0)),
                            float(data.get("accel", 0.0)),
                            float(data.get("cur", 0.0)),
                            float(data.get("risk", 0.0)),
                            data.get("state", "?"),
                            state.fsm_state,
                            "ON" if state.fan_on else "OFF",
                        )
                        last_summary_at = now
                except json.JSONDecodeError:
                    # Poate fi linie de debug de la ESP32.
                    if state.uart_debug:
                        log.info("UART RX text (non-JSON): %s", line[:120])
                    else:
                        log.debug("Non-JSON: %s", line[:80])
    except KeyboardInterrupt:
        stopped_by_user = True
        log.info("Oprit de utilizator.")
    finally:
        # Opreste alarma si fan doar la oprire controlata.
        if stopped_by_user and serial_ref["ser"] is not None:
            try:
                _send(serial_ref["ser"], "CMD:ALARM_OFF", lock, state)
                _send(serial_ref["ser"], "CMD:FAN_OFF", lock, state)
            except Exception:
                pass

        try:
            if serial_ref["ser"] and serial_ref["ser"].is_open:
                serial_ref["ser"].close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediTwin AI — Raspberry Pi Monitor")
    parser.add_argument("--port",  default="auto",
                        help="Port serial (default: auto)")
    parser.add_argument("--baud",  type=int, default=115200,
                        help="Viteza UART (default: 115200)")
    parser.add_argument(
        "--fan-protocol",
        choices=("prefixed", "raw", "both"),
        default="both",
        help="Stilul comenzii de ventilator trimise catre ESP32",
    )
    parser.add_argument(
        "--uart-debug",
        action="store_true",
        help="Afiseaza bytes-ii scrisi pe UART pentru fiecare comanda trimisa",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Afiseaza toate liniile JSON primite")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(
        args.port,
        args.baud,
        args.fan_protocol,
        args.uart_debug,
    )