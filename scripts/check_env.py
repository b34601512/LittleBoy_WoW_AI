import shutil, subprocess, sys, platform, json, torch
from pathlib import Path
def cmd(x): return subprocess.check_output(x, shell=True, text=True).strip()
report = {
    "python": sys.version,
    "torch": torch.__version__,
    "cuda_ok": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "ffmpeg": shutil.which("ffmpeg") is not None,
    "docker": cmd("docker --version") if shutil.which("docker") else None,
}
Path("env_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
