# run_all.py
from pathlib import Path
import subprocess
import sys
import time

# Folder where this script is located
ROOT = Path(__file__).resolve().parent

# Scripts you probably want to run in this order
# Edit this order if some scripts depend on outputs from others
ORDER = [
    "distributional_analysis.py",
    "ridgeline.py",
    "density_band_analysis.py",
    "trajectory_fan.py",
    "phase_potrait.py",
    "boxplot_comparison.py",
    "ecdf_analysis.py",
    "regime_classification.py",
    "time_series_analysis.py",
    "heatmap_summary.py"
]

# Scripts to skip
SKIP = {
    "run_all.py",
    "standardize_inertial_data.py",
    "compute_std.py",
    "Test.py",   # usually test/debug file
    "tum_colors.py"
}

def run_script(script_path: Path):
    print("\n" + "=" * 100)
    print(f"🚀 Running: {script_path.name}")
    print("=" * 100)

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    duration = time.time() - start

    if result.stdout:
        print("\n--- STDOUT ---")
        print(result.stdout)

    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)

    if result.returncode == 0:
        print(f"✅ Finished: {script_path.name} in {duration:.2f} s")
        return True
    else:
        print(f"❌ Failed: {script_path.name} in {duration:.2f} s")
        return False


def main():
    print(f"📁 Working folder: {ROOT}")

    success = []
    failed = []

    for script_name in ORDER:
        script_path = ROOT / script_name

        if script_name in SKIP:
            print(f"⏭️ Skipping: {script_name}")
            continue

        if not script_path.exists():
            print(f"⚠️ Missing script, skipped: {script_name}")
            continue

        ok = run_script(script_path)

        if ok:
            success.append(script_name)
        else:
            failed.append(script_name)

            # Harsh truth: usually better to stop after failure,
            # because later plots may depend on missing outputs.
            print("\n🛑 Stopping because one script failed.")
            break

    print("\n" + "=" * 100)
    print("📊 SUMMARY")
    print("=" * 100)

    print("\n✅ Successful:")
    for s in success:
        print(f"  - {s}")

    print("\n❌ Failed:")
    for s in failed:
        print(f"  - {s}")

    if not failed:
        print("\n🎉 All scripts completed successfully.")
    else:
        print("\n⚠️ Fix the failed script first, then rerun run_all.py.")


if __name__ == "__main__":
    main()