#!/usr/bin/env python3
"""Akkordverktøy – lokal dropzone-app.

Start:  python app.py   (eller dobbeltklikk Start.command / start.bat)
Åpner http://localhost:8765 i nettleseren. Alt kjører lokalt.

Én jobb kjøres om gangen (modellene er store). Hver jobb kjører songsheet.py som en
egen prosess, så minnet frigjøres helt mellom låtene og en jobb kan avbrytes.
"""
import atexit, json, os, queue, re, shutil, socket, subprocess, sys, threading, time, uuid
import urllib.request, webbrowser

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(HERE, "jobber")
OUT_DIR = os.path.join(HERE, "ferdige")
PORT = int(os.environ.get("AKKORD_PORT", 8765))
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".aiff", ".aif", ".opus"}
LANGS = {"auto", "no", "nn", "en"}
META_KEYS = ("id", "name", "audio", "lang", "separate", "status", "stage", "progress",
             "error", "created", "started", "finished", "lang_used", "timing")

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI(title="Akkordverktøy")
jobs = {}            # id -> dict
order = []           # eldste først
work = queue.Queue()
lock = threading.RLock()

# Steg: nøkkel -> (visningsnavn, vekt). Vektene er målt andel av tida på CPU (3-min-låt).
STEPS = {"sep": ("Skiller ut vokal", 0.48), "lang": ("Finner språk", 0.07),
         "text": ("Gjenkjenner tekst", 0.36), "chords": ("Finner akkorder", 0.09)}


def classify(line):
    """'• ...'-linje fra songsheet.py -> (stegnøkkel, ferdig-uten-arbeid?)."""
    body = line.lstrip("• ").strip()
    if body.startswith("Vokalseparasjon"):
        return "sep", "gjenbruker" in body
    if body.startswith(("Språk", "Bruker språk", "Tekst: leser lyd")):
        return "lang", False
    if body.startswith("Tekst"):
        return "text", "gjenbruker" in body
    if body.startswith("Akkorder"):
        return "chords", "gjenbruker" in body
    if body.startswith("Skrev"):
        return "done", False
    return None, False


def safe_name(name):
    base, ext = os.path.splitext(os.path.basename(name or ""))
    base = re.sub(r"[^\w\-. ]", "_", base).strip(" .") or "sang"
    return base[:80], ext.lower()


def job_dir(jid):
    return os.path.join(JOBS_DIR, jid)


def audio_path(j):
    return os.path.join(job_dir(j["id"]), j["audio"])


def out_base(j):
    return os.path.splitext(audio_path(j))[0]


def save_meta(j):
    try:
        with open(os.path.join(job_dir(j["id"]), "jobb.json"), "w", encoding="utf-8") as f:
            json.dump({k: j.get(k) for k in META_KEYS}, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


NOISE = re.compile(r"(Loading weights|Warning: You are sending unauthenticated|Inference: |"
                   r"\[transformers\] (Using `chunk_length_s`|Passing `generation_config`))")


def log_tail(j, n=14):
    try:
        with open(os.path.join(job_dir(j["id"]), "logg.txt"), encoding="utf-8") as f:
            lines = [ln.rstrip() for ln in f if ln.strip() and not NOISE.match(ln)]
    except OSError:
        return ""
    tb = next((i for i, ln in enumerate(lines) if ln.startswith("Traceback")), None)
    if tb is not None:                       # vis starten av feilen + siste linje
        return "\n".join(lines[-n:]) if len(lines) - tb <= n else \
            "\n".join(lines[tb:tb + 3] + ["…"] + lines[-(n - 4):])
    return "\n".join(lines[-n:])


def public(j):
    d = {k: j.get(k) for k in META_KEYS if k != "audio"}
    d["detail"] = j.get("detail", "")
    d["cho"] = j.get("cho")
    if j.get("status") in ("error", "cancelled"):
        d["log"] = j.get("log") or ""
    return d


def set_job(jid, persist=False, **kw):
    with lock:
        j = jobs.get(jid)
        if j is None:
            return None
        j.update(kw)
        if persist:
            save_meta(j)
        return j


def friendly_error(code, log):
    low = log.lower()
    if code in (-9, 137, 9) or "out of memory" in low or "memoryerror" in low \
            or "cannot allocate memory" in low:
        return "For lite minne – lukk andre programmer, eller prøv igjen uten vokalseparasjon"
    if "no space left" in low:
        return "Disken er full – slett gamle jobber eller frigjør plass"
    if any(s in low for s in ("connectionerror", "max retries", "name resolution",
                              "offline", "couldn't connect", "failed to establish")):
        return "Kunne ikke laste ned modellene – sjekk internett (trengs bare første gang)"
    if "invaliddataerror" in low or "fant ingen lyd" in low or "no suitable format" in low:
        return "Klarte ikke å lese lydfila – er den skadet eller i et uvanlig format?"
    last = next((ln for ln in reversed(log.splitlines()) if ln.strip()), "")
    return f"Prosessen stoppet ({last[:160]})" if last else "Prosessen stoppet uventet"

# ------------------------------------------------------------------ jobbkjøring


DOWNLOAD = re.compile(r"(?:Downloading|Fetching|\.safetensors|\.bin|\.th\b|\.pt\b|model).*?(\d{1,3})%\|")


class Tracker:
    """Gjør songsheet.py-utskrift om til steg, fremdrift og detalj for én jobb."""

    def __init__(self, jid, separate, cached_words):
        self.jid = jid
        plan = (["sep"] if separate and not cached_words else []) + \
               (["lang", "text"] if not cached_words else ["text"]) + ["chords"]
        total = sum(STEPS[s][1] for s in plan) or 1
        self.span, acc = {}, 0.0
        for s in plan:
            w = STEPS[s][1] / total
            self.span[s] = (acc, acc + w)
            acc += w
        self.step = None
        self.inference = 0

    def at(self, frac):
        lo, hi = self.span.get(self.step, (None, None))
        if lo is None:
            return
        p = round(lo + (hi - lo) * max(0.0, min(1.0, frac)), 4)
        with lock:
            j = jobs.get(self.jid)
            if j and p > (j.get("progress") or 0):
                j["progress"] = p

    def line(self, line, final):
        if line.startswith("•"):
            key, skipped = classify(line)
            body = line.lstrip("• ").strip()
            if body.startswith("Tid:"):
                set_job(self.jid, timing=body[4:].strip())
                return
            if key == "done":
                return
            if key in self.span:
                if key != self.step:
                    self.step, self.inference = key, 0
                    self.at(0)
                label = STEPS[key][0]
                if skipped:
                    label = {"sep": "Gjenbruker vokal", "text": "Gjenbruker tekst",
                             "chords": "Gjenbruker akkorder"}.get(key, label)
                    self.at(1)
                detail = body.split(":", 1)[1].strip() if ":" in body else ""
                m = re.match(r"Bruker språk:\s*(\S+)", body)
                if m:
                    set_job(self.jid, lang_used=m.group(1))
                set_job(self.jid, stage=label, detail=detail[:140])
        elif line.startswith("!"):
            set_job(self.jid, detail=line[1:].strip()[:160])
        elif line.startswith("% "):
            try:
                self.at(int(line[2:].strip()) / 100)
            except ValueError:
                pass
            with lock:
                j = jobs.get(self.jid)
                if j and str(j.get("detail", "")).startswith("Laster ned"):
                    j["detail"] = ""
        elif line.startswith("Inference: ") and self.step == "chords":
            self.inference += 1                                   # lv-chordia: 5 modeller
            self.at(0.15 + 0.85 * (self.inference - 1) / 5)
        else:
            m = DOWNLOAD.search(line)
            if m and "Loading weights" not in line and "seconds/s" not in line:
                set_job(self.jid, detail=f"Laster ned modell (bare første gang) – {m.group(1)} %")


def run_job(jid):
    with lock:
        j = jobs[jid]
        j.update(status="running", stage="Starter", progress=0.01, detail="", error=None,
                 log=None, started=time.time(), finished=None, cho=None, timing=None)
        save_meta(j)
        audio = audio_path(j)
        base = out_base(j)
        lang, separate = j["lang"], j["separate"]
    cached_words = os.path.exists(base + ".words.json")
    cmd = [sys.executable, "-u", os.path.join(HERE, "songsheet.py"), audio,
           "--lang", lang, "--quiet"]
    if not separate:
        cmd.append("--no-separate")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONWARNINGS="ignore", TOKENIZERS_PARALLELISM="false",
               TRANSFORMERS_VERBOSITY="error")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=env, cwd=job_dir(jid), bufsize=0, creationflags=flags)
    set_job(jid, proc=p)
    tracker = Tracker(jid, separate, cached_words)
    buf = b""
    with open(os.path.join(job_dir(jid), "logg.txt"), "w", encoding="utf-8") as log:
        while True:
            chunk = p.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"(\r\n|\n|\r)", buf)
            buf = parts.pop()                       # siste (ufullstendige) bit
            for text, sep in zip(parts[0::2], parts[1::2]):
                line = text.decode("utf-8", "replace").strip()
                if not line:
                    continue
                final = sep != b"\r"                # \r = tqdm som oppdaterer samme linje
                if final and not line.startswith("% "):
                    log.write(line + "\n")
                    log.flush()
                tracker.line(line, final)
        if buf.strip():
            line = buf.decode("utf-8", "replace").strip()
            log.write(line + "\n")
            tracker.line(line, True)
    p.wait()

    with lock:
        j = jobs.get(jid)
        if j is None:                               # slettet mens den kjørte
            return
        j.pop("proc", None)
    if j.get("status") == "cancelled":
        set_job(jid, stage="Avbrutt", finished=time.time(), log=log_tail(j), persist=True)
        return
    cho_path = base + ".cho"
    if p.returncode == 0 and os.path.exists(cho_path):
        with open(cho_path, encoding="utf-8") as f:
            cho = f.read()
        try:
            shutil.copyfile(cho_path, os.path.join(OUT_DIR, j["name"] + ".cho"))
        except OSError:
            pass
        set_job(jid, status="done", stage="Ferdig", progress=1.0, cho=cho, detail="",
                finished=time.time(), persist=True)
    else:
        tail = log_tail(j)
        set_job(jid, status="error", stage="Feilet", error=friendly_error(p.returncode, tail),
                log=tail, finished=time.time(), persist=True)


def worker():
    while True:
        jid = work.get()
        try:
            with lock:
                ready = jobs.get(jid, {}).get("status") == "queued"
            if ready:
                run_job(jid)
        except Exception as e:  # noqa: BLE001
            set_job(jid, status="error", stage="Feilet", error=f"Intern feil: {e}",
                    finished=time.time(), persist=True)
        finally:
            work.task_done()


def kill(j):
    p = j.get("proc")
    if p and p.poll() is None:
        try:
            p.kill()
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass


@atexit.register
def _stop_running():
    with lock:
        running = [j for j in jobs.values() if j.get("proc")]
    for j in running:
        kill(j)


def load_existing():
    """Vis tidligere jobber etter omstart."""
    for jid in sorted(os.listdir(JOBS_DIR)):
        meta = os.path.join(JOBS_DIR, jid, "jobb.json")
        if not os.path.exists(meta):
            continue
        try:
            with open(meta, encoding="utf-8") as f:
                j = json.load(f)
        except (OSError, ValueError):
            continue
        j["id"] = jid
        j["audio"] = os.path.basename(j.get("audio") or "")      # eldre format: full sti
        j.setdefault("lang", "auto")
        j.setdefault("separate", True)
        cho = out_base(j) + ".cho"
        if j.get("status") == "done" and os.path.exists(cho):
            with open(cho, encoding="utf-8") as f:
                j.update(status="done", stage="Ferdig", progress=1.0, cho=f.read())
        elif j.get("status") in ("error", "cancelled"):
            j["log"] = log_tail(j)
        else:
            j.update(status="cancelled", stage="Avbrutt", progress=0.0,
                     error="Programmet ble lukket før jobben ble ferdig", log=log_tail(j))
        jobs[jid] = j
        order.append(jid)


load_existing()
threading.Thread(target=worker, daemon=True).start()

# ------------------------------------------------------------------ API


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read(), headers={"Cache-Control": "no-store"})


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), lang: str = Form("auto"),
                     separate: bool = Form(True)):
    name, ext = safe_name(file.filename)
    if ext not in AUDIO_EXT:
        raise HTTPException(400, f"Ukjent filtype «{ext or 'ingen'}». Bruk mp3, wav, m4a, flac, ogg eller aiff.")
    if lang not in LANGS:
        lang = "auto"
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    os.makedirs(job_dir(jid))
    j = {"id": jid, "name": name, "audio": name + ext, "lang": lang, "separate": separate,
         "status": "queued", "stage": "I kø", "progress": 0.0, "error": None,
         "created": time.time(), "started": None, "finished": None, "lang_used": None}
    size = 0
    with open(audio_path(j), "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
            size += len(chunk)
    if size == 0:
        shutil.rmtree(job_dir(jid), ignore_errors=True)
        raise HTTPException(400, "Fila er tom.")
    save_meta(j)
    with lock:
        jobs[jid] = j
        order.append(jid)
    work.put(jid)
    return public(j)


@app.get("/api/jobs")
def list_jobs():
    with lock:
        return [public(jobs[i]) for i in reversed(order) if i in jobs]


def get_job(jid):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "Fant ikke jobben")
    return j


@app.put("/api/jobs/{jid}/cho")
def save_cho(jid: str, body: dict = Body(...)):
    j = get_job(jid)
    if j["status"] != "done":
        raise HTTPException(409, "Jobben er ikke ferdig")
    text = str(body.get("cho", ""))
    if text and not text.endswith("\n"):
        text += "\n"
    path = out_base(j) + ".cho"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    shutil.copyfile(path, os.path.join(OUT_DIR, j["name"] + ".cho"))
    set_job(jid, cho=text)
    return {"ok": True}


@app.get("/api/jobs/{jid}/cho")
def download_cho(jid: str):
    j = get_job(jid)
    path = out_base(j) + ".cho"
    if not os.path.exists(path):
        raise HTTPException(404, "Ingen ChordPro-fil ennå")
    return FileResponse(path, filename=j["name"] + ".cho", media_type="text/plain; charset=utf-8")


@app.post("/api/jobs/{jid}/retry")
def retry(jid: str, body: dict | None = Body(None)):
    """Kjør på nytt med (ev. nye) valg. Teksten gjenkjennes på nytt; akkorder og
    utskilt vokal gjenbrukes fra forrige runde når det er mulig."""
    with lock:
        j = get_job(jid)
        if j["status"] in ("queued", "running"):
            raise HTTPException(409, "Jobben kjører allerede")
        body = body or {}
        if "separate" in body:
            j["separate"] = bool(body["separate"])
        if body.get("lang") in LANGS:
            j["lang"] = body["lang"]
        base = out_base(j)
        for ext in (".cho", ".words.json"):
            try:
                os.remove(base + ext)
            except OSError:
                pass
        j.update(status="queued", stage="I kø", progress=0.0, error=None, cho=None, log=None,
                 detail="", started=None, finished=None, lang_used=None)
        save_meta(j)
        order.remove(jid)
        order.append(jid)                          # øverst i lista igjen
    work.put(jid)
    return public(j)


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    with lock:
        j = get_job(jid)
        if j["status"] not in ("queued", "running"):
            return public(j)
        j.update(status="cancelled", stage="Avbrutt", error=None, finished=time.time())
        save_meta(j)
    kill(j)
    return public(j)


@app.delete("/api/jobs/{jid}")
def delete(jid: str):
    with lock:
        j = get_job(jid)
        j["status"] = "cancelled"
        jobs.pop(jid, None)
        if jid in order:
            order.remove(jid)
    kill(j)                                        # må være stoppet før mappa kan slettes
    shutil.rmtree(job_dir(jid), ignore_errors=True)
    return {"ok": True}


@app.post("/api/open-folder")
def open_folder():
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", OUT_DIR])
        elif sys.platform.startswith("win"):
            os.startfile(OUT_DIR)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", OUT_DIR])
    except (OSError, FileNotFoundError) as e:
        raise HTTPException(500, f"{e} – mappa ligger her: {OUT_DIR}")
    return {"ok": True, "path": OUT_DIR}


@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "akkordverktoy"

# ------------------------------------------------------------------ oppstart


def already_running(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as r:
            return r.read() == b"akkordverktoy"
    except Exception:  # noqa: BLE001
        return False


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


if __name__ == "__main__":
    import uvicorn
    no_browser = "--no-browser" in sys.argv
    port = PORT
    if already_running(port):
        print(f"\n  Akkordverktøy kjører allerede – åpner http://localhost:{port}\n")
        if not no_browser:
            webbrowser.open(f"http://localhost:{port}")
        sys.exit(0)
    while not port_free(port) and port < PORT + 20:
        port += 1
    url = f"http://localhost:{port}"
    if not no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  Akkordverktøy kjører på {url}\n"
          f"  Ferdige filer: {OUT_DIR}\n"
          f"  Lukk dette vinduet (eller trykk Ctrl+C) for å stoppe.\n", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
