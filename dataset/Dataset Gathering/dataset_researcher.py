"""
Dataset Researcher — Autonomous Driving Dataset Metadata Extractor
Fixed version: JSON parse + rate limit handling
"""

import anthropic
import json
import csv
import time
import os
import sys
from pathlib import Path
from datetime import datetime

DATASETS_FILE  = "datasets.txt"
OUTPUT_CSV     = "dataset_research.csv"
PROGRESS_FILE  = "progress.txt"
COST_LOG_FILE  = "cost_log.txt"

MODEL          = "claude-sonnet-4-6"
MAX_TOKENS     = 1500
DELAY_SECONDS  = 22
MAX_RETRIES    = 3

CSV_FIELDS = [
    "dataset_name","real_synthetic_mixed","public_availability",
    "monocular_camera","stereo_camera","fisheye_camera","event_camera",
    "thermal_camera","lidar","radar","sensor_imu","gps_gnss",
    "ego_motion_imu_ins","can_bus_vehicle_odometry","ego_pose_trajectory",
    "source_url","confidence","notes","error",
]

SYSTEM_PROMPT = """You are a precise autonomous driving dataset researcher.
RULES:
1. Always search the web before answering. Find the official paper, project page, and GitHub.
2. Only mark a sensor as true if you found EXPLICIT documentation.
3. If no evidence found after searching: use "unknown" NOT false.
4. "false" = you found proof it is NOT included.
5. public_availability: "Public-Free"/"Public-Registration"/"Restricted"/"Not Public"/"unknown"
6. real_synthetic_mixed: "Real"/"Synthetic"/"Mixed"/"unknown"
7. confidence: "high"=official docs, "medium"=paper/secondary, "low"=inferred
8. Return ONLY valid JSON — no markdown, no preamble, nothing else."""

def build_prompt(name):
    return f"""Research this autonomous driving dataset: "{name}"
Search the web for its paper, project page, and GitHub. Return ONLY this JSON:
{{
  "dataset_name": "{name}",
  "real_synthetic_mixed": "Real or Synthetic or Mixed or unknown",
  "public_availability": "Public-Free or Public-Registration or Restricted or Not Public or unknown",
  "monocular_camera": true or false or "unknown",
  "stereo_camera": true or false or "unknown",
  "fisheye_camera": true or false or "unknown",
  "event_camera": true or false or "unknown",
  "thermal_camera": true or false or "unknown",
  "lidar": true or false or "unknown",
  "radar": true or false or "unknown",
  "sensor_imu": true or false or "unknown",
  "gps_gnss": true or false or "unknown",
  "ego_motion_imu_ins": true or false or "unknown",
  "can_bus_vehicle_odometry": true or false or "unknown",
  "ego_pose_trajectory": true or false or "unknown",
  "source_url": "URL of your main source",
  "confidence": "high or medium or low",
  "notes": "one sentence caveat or empty string"
}}"""

def research_dataset(client, name):
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 4}],
        messages=[{"role": "user", "content": build_prompt(name)}]
    )

    block_types = [b.type for b in response.content]
    print(f"    blocks: {block_types}  stop: {response.stop_reason}")

    text_output = ""
    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            text_output += block.text

    if not text_output.strip():
        raise ValueError(f"Empty response. stop={response.stop_reason} blocks={block_types}")

    text_output = text_output.strip()

    if "```" in text_output:
        parts = text_output.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text_output = part
                break

    start = text_output.find("{")
    end   = text_output.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found: {text_output[:200]}")
    text_output = text_output[start:end]

    result = json.loads(text_output)
    result["dataset_name"] = name

    usage = response.usage
    web_searches = 0
    if hasattr(usage, "server_tool_use") and usage.server_tool_use:
        web_searches = getattr(usage.server_tool_use, "web_search_requests", 0)
    result["_in"]  = usage.input_tokens
    result["_out"] = usage.output_tokens
    result["_ws"]  = web_searches
    return result

class CostTracker:
    def __init__(self):
        self.inp = self.out = self.ws = 0
    def add(self, r):
        self.inp += r.pop("_in", 0)
        self.out += r.pop("_out", 0)
        self.ws  += r.pop("_ws", 0)
    @property
    def cost(self):
        return (self.inp/1e6)*3 + (self.out/1e6)*15 + (self.ws/1000)*10
    def summary(self):
        return (f"  Input: {self.inp:,} tokens\n"
                f"  Output: {self.out:,} tokens\n"
                f"  Searches: {self.ws:,}\n"
                f"  Cost: ${self.cost:.4f}")

def load_done(p):
    if not Path(p).exists(): return set()
    return set(l.strip() for l in open(p) if l.strip())

def mark_done(p, n):
    open(p, "a").write(n + "\n")

def write_row(result, csv_file, header):
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if header: w.writeheader()
        w.writerow({k: result.get(k, "") for k in CSV_FIELDS})

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY first")
        sys.exit(1)
    if not Path(DATASETS_FILE).exists():
        print(f"ERROR: {DATASETS_FILE} not found")
        sys.exit(1)

    all_ds    = [l.strip() for l in open(DATASETS_FILE) if l.strip()]
    done      = load_done(PROGRESS_FILE)
    remaining = [d for d in all_ds if d not in done]
    need_hdr  = not Path(OUTPUT_CSV).exists()
    eta       = len(remaining) * DELAY_SECONDS / 60

    print("=" * 50)
    print(f"  Total: {len(all_ds)}  Done: {len(done)}  Remaining: {len(remaining)}")
    print(f"  ETA: ~{eta:.0f} min  |  Delay: {DELAY_SECONDS}s per call")
    print("=" * 50)

    if not remaining:
        print("All done! Open", OUTPUT_CSV)
        return

    client  = anthropic.Anthropic()
    tracker = CostTracker()
    failed  = []

    for i, name in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}]  {name}")
        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result  = research_dataset(client, name)
                tracker.add(result)
                write_row(result, OUTPUT_CSV, need_hdr)
                need_hdr = False
                mark_done(PROGRESS_FILE, name)
                print(f"    ✓  conf={result.get('confidence','?')}  src={str(result.get('source_url',''))[:55]}")
                success = True
                break

            except anthropic.RateLimitError as e:
                wait = float(getattr(e, "retry_after", 0) or 30 * attempt) + 2
                print(f"    ⚠  Rate limit — waiting {wait:.0f}s (attempt {attempt})")
                time.sleep(wait)

            except anthropic.APIStatusError as e:
                print(f"    ⚠  API {e.status_code} — waiting {10*attempt}s (attempt {attempt})")
                time.sleep(10 * attempt)

            except (json.JSONDecodeError, ValueError) as e:
                print(f"    ✗  Parse error: {e}")
                write_row({"dataset_name": name, "error": str(e)[:200]}, OUTPUT_CSV, need_hdr)
                need_hdr = False
                mark_done(PROGRESS_FILE, name)
                failed.append(name)
                success = True
                break

            except Exception as e:
                print(f"    ✗  Error: {e}")
                time.sleep(5)

        if not success:
            write_row({"dataset_name": name, "error": "max retries"}, OUTPUT_CSV, need_hdr)
            need_hdr = False
            mark_done(PROGRESS_FILE, name)
            failed.append(name)

        if (i + 1) % 10 == 0:
            print(f"\n── Cost after {i+1} datasets ──")
            print(tracker.summary())

        time.sleep(DELAY_SECONDS)

    print("\n" + "=" * 50)
    print(f"  {len(remaining)-len(failed)} succeeded  |  {len(failed)} failed")
    print(tracker.summary())
    if failed:
        print("\n  Failed datasets:")
        for f in failed: print(f"    - {f}")
    open(COST_LOG_FILE, "a").write(f"\n{datetime.now().isoformat()}\n{tracker.summary()}\n")

if __name__ == "__main__":
    main()