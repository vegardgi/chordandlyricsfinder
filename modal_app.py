"""Akkordverktøy på nett – Modal (https://modal.com).

    modal run modal_app.py::download_models     # én gang: legg modellene i et Modal-volum
    modal serve modal_app.py                    # test med midlertidig URL (endringer lastes på nytt)
    modal deploy modal_app.py                   # publiser

To deler:
  * web      – FastAPI-appen fra app.py i nettmodus (AKKORD_ONLINE=1), på CPU. Én container,
               så køen og øktene kan ligge i minnet. Ingen lagring: alt slettes når siden lukkes.
  * process  – kjører songsheet.py på en GPU for én låt. Skalerer til null når ingen bruker
               den; betaler per sekund. Fremdriftslinjene sendes tilbake via en modal.Queue.

Kostnadstak: GPU_MAX samtidige GPU-containere. Modellene (~5 GB) ligger i volumet
«akkord-models», så en kald GPU-container slipper å laste dem ned.
"""
import os

import modal

GPU = os.environ.get("AKKORD_GPU", "L4")          # L4 (24 GB) er et godt pris/fart-valg; T4 er billigere
GPU_MAX = int(os.environ.get("AKKORD_GPU_MAX", 3))
APP_DIR, MODELS = "/app", "/models"
HERE = os.path.dirname(os.path.abspath(__file__))

app = modal.App("akkordverktoy")
models = modal.Volume.from_name("akkord-models", create_if_missing=True)
progress = modal.Queue.from_name("akkord-progress", create_if_missing=True)

model_env = {"HF_HOME": f"{MODELS}/hf", "TORCH_HOME": f"{MODELS}/torch",
             "HF_HUB_DISABLE_PROGRESS_BARS": "0", "TOKENIZERS_PARALLELISM": "false"}

# CUDA 12 + cuDNN 9: kreves av CTranslate2 (faster-whisper) på GPU
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")
    .pip_install("torch", "lv-chordia>=1.1", "faster-whisper>=1.1", "transformers>=4.56",
                 "librosa", "demucs>=4.1", "av", "soundfile")
    .env(model_env)
    .add_local_file(os.path.join(HERE, "songsheet.py"), f"{APP_DIR}/songsheet.py")
    .add_local_file(os.path.join(HERE, "chords.py"), f"{APP_DIR}/chords.py")
    .add_local_file(os.path.join(HERE, "chart.py"), f"{APP_DIR}/chart.py")
)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "uvicorn", "python-multipart")
    .env({"AKKORD_ONLINE": "1", "AKKORD_JOBS_DIR": "/tmp/jobber", "AKKORD_WORKERS": str(GPU_MAX)})
    .add_local_file(os.path.join(HERE, "app.py"), f"{APP_DIR}/app.py")
    .add_local_file(os.path.join(HERE, "index.html"), f"{APP_DIR}/index.html")
    .add_local_file(os.path.join(HERE, "vilkar.html"), f"{APP_DIR}/vilkar.html")
    .add_local_dir(os.path.join(HERE, "static"), f"{APP_DIR}/static")
)

# ------------------------------------------------------------------ GPU


@app.function(image=gpu_image, gpu=GPU, volumes={MODELS: models}, timeout=15 * 60,
              scaledown_window=5 * 60, max_containers=GPU_MAX, memory=16384)
def process(audio: bytes, filename: str, lang: str, separate: bool, jid: str, mode: str = "both") -> dict:
    """Én låt: songsheet.py i en midlertidig mappe. Linjene sendes til køen underveis,
    lydfila og alle mellomresultater forsvinner med mappa."""
    import re, subprocess, sys, tempfile, time
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, os.path.basename(filename))
        with open(path, "wb") as f:
            f.write(audio)
        del audio
        cmd = [sys.executable, "-u", f"{APP_DIR}/songsheet.py", path, "--lang", lang, "--quiet"]
        if not separate:
            cmd.append("--no-separate")
        if mode in ("lyrics", "chords"):
            cmd += ["--only", mode]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONWARNINGS="ignore",
                   TRANSFORMERS_VERBOSITY="error")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, cwd=d, env=env, bufsize=0)
        log, batch, buf, last = [], [], b"", time.time()
        while True:
            chunk = p.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"(\r\n|\n|\r)", buf)
            buf = parts.pop()
            for text, sep in zip(parts[0::2], parts[1::2]):
                line = text.decode("utf-8", "replace").strip()
                if line:
                    final = sep != b"\r"
                    batch.append((line, final))
                    if final and not line.startswith("% "):
                        log.append(line)
            if batch and time.time() - last > 0.5:
                progress.put_many(batch, partition=jid, partition_ttl=3600)
                batch, last = [], time.time()
        if buf.strip():
            line = buf.decode("utf-8", "replace").strip()
            batch.append((line, True))
            log.append(line)
        if batch:
            progress.put_many(batch, partition=jid, partition_ttl=3600)
        p.wait()
        cho_path = os.path.splitext(path)[0] + ".cho"
        cho = open(cho_path, encoding="utf-8").read() if os.path.exists(cho_path) else None
        chart_path = os.path.splitext(path)[0] + ".chart.cho"
        chart = open(chart_path, encoding="utf-8").read() if os.path.exists(chart_path) else None
    models.commit()                       # nye modellnedlastinger blir liggende til neste gang
    return {"returncode": p.returncode, "cho": cho, "chart": chart, "log": "\n".join(log[-80:])}


@app.function(image=gpu_image, gpu=GPU, volumes={MODELS: models}, timeout=60 * 60)
def download_models():
    """Last ned alle modellene til volumet én gang (ellers skjer det ved første låt)."""
    import sys
    sys.path.insert(0, APP_DIR)
    import songsheet
    from demucs.api import Separator
    for name in ("NbAiLab/nb-whisper-medium", songsheet.DETECT_MODEL):
        print("•", name)
        songsheet.fw_load(name)
    print("• htdemucs")
    Separator("htdemucs", device="cpu")
    models.commit()
    print("Ferdig.")

# ------------------------------------------------------------------ web


def run_job_modal(jid):
    """RUNNER for app.py: send låta til GPU-funksjonen og følg fremdriften via køen."""
    import time
    import app as akkord                  # app.py (lastet i web-containeren)
    from modal.exception import TimeoutError as ModalTimeout
    with akkord.lock:
        j = akkord.jobs[jid]
        j.update(status="running", stage="Starter GPU", progress=0.01, detail="", error=None,
                 log=None, started=time.time(), finished=None, cho=None, timing=None)
        path, base = akkord.audio_path(j), akkord.out_base(j)
        lang, separate, filename = j["lang"], j["separate"], j["audio"]
        mode = j.get("mode") or "both"
    with open(path, "rb") as f:
        audio = f.read()
    progress.clear(partition=jid)
    call = process.spawn(audio, filename, lang, separate, jid, mode)
    del audio
    akkord.set_job(jid, remote_cancel=lambda: call.cancel())
    tracker = akkord.Tracker(jid, separate, False, mode)
    result = None
    while result is None:
        try:
            for line, final in progress.get_many(200, block=True, timeout=1.0, partition=jid):
                tracker.line(line, final)
        except Exception:  # noqa: BLE001 – køen er bare fremdrift; resultatet kommer uansett
            time.sleep(1)
        with akkord.lock:
            j = akkord.jobs.get(jid)
            if j is None or j.get("status") == "cancelled":
                try:
                    call.cancel()
                except Exception:  # noqa: BLE001
                    pass
                break
        try:
            result = call.get(timeout=0)
        except (ModalTimeout, TimeoutError):
            continue
        except Exception as e:  # noqa: BLE001
            result = {"returncode": -1, "cho": None, "log": f"{type(e).__name__}: {e}"}
    try:
        for line, final in progress.get_many(1000, block=False, partition=jid):
            tracker.line(line, final)
        progress.clear(partition=jid)
    except Exception:  # noqa: BLE001
        pass
    with akkord.lock:
        j = akkord.jobs.get(jid)
        if j is None:                     # slettet (siden lukket) mens den kjørte
            return
        j.pop("remote_cancel", None)
    if j.get("status") == "cancelled" or result is None:
        akkord.set_job(jid, stage="Avbrutt", finished=time.time())
        return
    if result["returncode"] == 0 and result["cho"]:
        with open(base + ".cho", "w", encoding="utf-8") as f:
            f.write(result["cho"])
        if result.get("chart"):
            with open(base + ".chart.cho", "w", encoding="utf-8") as f:
                f.write(result["chart"])
        akkord.set_job(jid, status="done", stage="Ferdig", progress=1.0, cho=result["cho"],
                       chart=result.get("chart"), detail="", finished=time.time())
    else:
        lines = [ln for ln in result["log"].splitlines() if not akkord.NOISE.match(ln)]
        tail = "\n".join(lines[-14:])
        akkord.set_job(jid, status="error", stage="Feilet", log=tail, finished=time.time(),
                       error=akkord.friendly_error(result["returncode"], tail))


@app.function(image=web_image, max_containers=1, scaledown_window=10 * 60, timeout=60 * 60)
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def web():
    import sys
    sys.path.insert(0, APP_DIR)
    import app as akkord
    akkord.RUNNER = run_job_modal
    return akkord.app
