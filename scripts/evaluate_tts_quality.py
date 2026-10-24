#!/usr/bin/env python3
"""Run a bounded, cloud-side smoke/technical-quality evaluation of Shan Voice TTS."""
import array
import csv
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = os.getenv("TTS_API_URL", "https://shan-voice.onrender.com").rstrip("/")
VOICE = os.getenv("TTS_VOICE", "shan-mms")
SPEED = float(os.getenv("TTS_SPEED", "1.0"))
LIMIT = max(1, min(184, int(os.getenv("SAMPLE_COUNT", "10"))))
DELAY = max(0, int(os.getenv("DELAY_SECONDS", "8")))
API_KEY = os.getenv("TTS_API_KEY", "").strip()
CSV_PATH = Path("shan_dataset_fixed.csv")
REPORT_PATH = Path("tts_quality_report.csv")
SUMMARY_PATH = Path("tts_quality_summary.json")


def request(url, data=None, headers=None, timeout=180):
    req_headers = {"User-Agent": "Shan-Voice-Dataset-Evaluator/1.0"}
    if headers:
        req_headers.update(headers)
    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=req_headers)
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url):
    with request(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wav_metrics(content):
    with wave.open(io.BytesIO(content), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        duration = frames / rate if rate else 0.0
        raw = wav.readframes(frames)
    if not frames or not rate or not raw:
        raise ValueError("WAV has no audio frames")
    # Calculate RMS for standard PCM sample widths without extra dependencies.
    samples = []
    if width == 1:
        samples = [b - 128 for b in raw]
        scale = 128.0
    elif width == 2:
        usable = len(raw) - (len(raw) % 2)
        vals = array.array("h")
        vals.frombytes(raw[:usable])
        if os.sys.byteorder != "little":
            vals.byteswap()
        samples = vals
        scale = 32768.0
    elif width == 3:
        usable = len(raw) - (len(raw) % 3)
        for i in range(0, usable, 3):
            value = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            if value & 0x800000:
                value -= 1 << 24
            samples.append(value)
        scale = 8388608.0
    elif width == 4:
        usable = len(raw) - (len(raw) % 4)
        vals = array.array("i")
        vals.frombytes(raw[:usable])
        if os.sys.byteorder != "little":
            vals.byteswap()
        samples = vals
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {width}")
    if not samples:
        raise ValueError("Could not read PCM samples")
    rms = math.sqrt(sum(float(v) * float(v) for v in samples) / len(samples)) / scale
    peak = max(abs(float(v)) for v in samples) / scale
    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "rms_normalized": round(rms, 6),
        "peak_normalized": round(peak, 6),
        "silent": peak < 0.0001,
    }


def source_duration(path):
    try:
        with wave.open(str(path), "rb") as wav:
            return round(wav.getnframes() / wav.getframerate(), 3)
    except Exception:
        return ""


def main():
    started = datetime.now(timezone.utc).isoformat()
    health_ok = False
    voices_ok = False
    health_detail = ""
    try:
        health = get_json(BASE_URL + "/health")
        health_ok = health.get("status") == "ok"
        health_detail = json.dumps(health, ensure_ascii=False)
    except Exception as exc:
        health_detail = f"Health check failed: {type(exc).__name__}: {exc}"

    try:
        voices = get_json(BASE_URL + "/api/voices")
        voice_ids = [v.get("id") for v in voices.get("voices", [])]
        voices_ok = VOICE in voice_ids
        if not voices_ok:
            health_detail += f"; configured voice {VOICE!r} not found in {voice_ids!r}"
    except Exception as exc:
        voice_ids = []
        health_detail += f"; voice check failed: {type(exc).__name__}: {exc}"

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {CSV_PATH}")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("transcription", "").strip()][:LIMIT]
    if not rows:
        raise RuntimeError("No rows with transcription found in dataset manifest")

    fields = [
        "row_idx", "audio_file", "text_characters", "source_duration_seconds",
        "http_status", "result", "generated_bytes", "generated_duration_seconds",
        "sample_rate", "channels", "rms_normalized", "peak_normalized",
        "silent", "latency_seconds", "error"
    ]
    results = []
    successes = 0

    with REPORT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for index, row in enumerate(rows):
            text = row["transcription"].strip()
            audio_path = Path(row.get("audio_file", ""))
            ref_duration = source_duration(audio_path) if audio_path.is_file() else ""
            report = {
                "row_idx": row.get("row_idx", index),
                "audio_file": row.get("audio_file", ""),
                "text_characters": len(text),
                "source_duration_seconds": ref_duration,
                "http_status": "",
                "result": "FAIL",
                "generated_bytes": 0,
                "generated_duration_seconds": "",
                "sample_rate": "",
                "channels": "",
                "rms_normalized": "",
                "peak_normalized": "",
                "silent": "",
                "latency_seconds": "",
                "error": "",
            }
            started_one = time.monotonic()
            try:
                headers = {"Origin": BASE_URL}
                if API_KEY:
                    headers["Authorization"] = "Bearer " + API_KEY
                with request(
                    BASE_URL + "/api/tts",
                    data={"text": text, "voice": VOICE, "speed": SPEED},
                    headers=headers,
                    timeout=180,
                ) as resp:
                    audio = resp.read()
                    report["http_status"] = resp.status
                report["generated_bytes"] = len(audio)
                metrics = wav_metrics(audio)
                report["generated_duration_seconds"] = metrics["duration_seconds"]
                report["sample_rate"] = metrics["sample_rate"]
                report["channels"] = metrics["channels"]
                report["rms_normalized"] = metrics["rms_normalized"]
                report["peak_normalized"] = metrics["peak_normalized"]
                report["silent"] = metrics["silent"]
                if metrics["silent"]:
                    report["error"] = "Audio is effectively silent"
                else:
                    report["result"] = "PASS"
                    successes += 1
            except urllib.error.HTTPError as exc:
                report["http_status"] = exc.code
                try:
                    report["error"] = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    report["error"] = str(exc)
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"[:500]
            report["latency_seconds"] = round(time.monotonic() - started_one, 2)
            writer.writerow(report)
            f.flush()
            print(
                f"[{index + 1}/{len(rows)}] row={report['row_idx']} "
                f"result={report['result']} http={report['http_status']} "
                f"latency={report['latency_seconds']}s"
            )
            if index < len(rows) - 1 and DELAY:
                time.sleep(DELAY)

    summary = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_base_url": BASE_URL,
        "health_ok": health_ok,
        "health_detail": health_detail,
        "voices_ok": voices_ok,
        "available_voice_ids": voice_ids,
        "requested_voice": VOICE,
        "speed": SPEED,
        "dataset_rows_available": len(list(csv.DictReader(CSV_PATH.open("r", encoding="utf-8-sig", newline="")))),
        "samples_tested": len(rows),
        "passed": successes,
        "failed": len(rows) - successes,
        "pass_rate_percent": round(successes * 100 / len(rows), 2),
        "report": str(REPORT_PATH),
        "note": "Technical WAV validation only; this does not assess pronunciation correctness or intelligibility.",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not health_ok or not voices_ok or successes == 0:
        raise SystemExit("API health/voice check failed or no TTS samples passed. See report.")
    if successes < len(rows):
        print("WARNING: Some samples failed; inspect tts_quality_report.csv.")
    print("Evaluation finished. No generated WAV files were saved; only CSV/JSON reports were retained.")


if __name__ == "__main__":
    main()
