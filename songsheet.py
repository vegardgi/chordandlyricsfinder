#!/usr/bin/env python3
"""Lydfil inn -> ChordPro med tekst og akkorder over riktig ord.

  1. (valgfritt) Demucs skiller ut vokal  -> bedre tekstgjenkjenning
  2. Språk gjenkjennes automatisk; tekst + tid per ord med
     NB-Whisper (norsk) eller Whisper large-v3-turbo (engelsk/andre)
  3. lv-chordia finner akkorder           -> C, C/B, Em7, D/F# ...
  4. Akkordene settes inn foran ordet de faller på -> .cho

Bruk:
  python songsheet.py sang.mp3                 # auto-språk, med vokalseparasjon
  python songsheet.py sang.mp3 --lang en       # tving engelsk
  python songsheet.py sang.mp3 --no-separate   # raskere, uten Demucs
  python songsheet.py sang.mp3 --model NbAiLab/nb-whisper-large --lang nn
Tekst lagres i sang.words.json; neste kjøring gjenbruker den (bruk --redo-lyrics for ny).
Akkorder lagres i sang.chords.raw.json og gjenbrukes på samme måte (--redo-chords).

Fremdrift på stderr:  "• Steg: ..."  (nytt steg),  "! ..."  (advarsel),  "% 42"  (prosent i steget)
"""
import argparse, gc, json, math, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chords import recognize, clean, estimate_key, names_for_key, key_name, MIXED, FLAT, SHARP  # noqa: E402


def say(msg):
    print(msg, file=sys.stderr, flush=True)


class Timer:
    """Måler tid per steg: t.lap("tekst") etter hvert steg."""
    def __init__(self):
        self.t0 = self.last = time.time()
        self.laps = []

    def lap(self, name):
        now = time.time()
        self.laps.append((name, now - self.last))
        self.last = now

    def report(self):
        parts = [f"{n} {d:.0f} s" for n, d in self.laps if d >= 0.5]
        return ", ".join(parts + [f"totalt {time.time() - self.t0:.0f} s"])


class Progress:
    """Skriver "% n" på stderr når prosenten i et steg endrer seg."""
    def __init__(self):
        self.last = -1

    def __call__(self, frac):
        p = max(0, min(100, int(frac * 100)))
        if p != self.last:
            self.last = p
            say(f"% {p}")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data, indent=1):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

# ---------------------------------------------------------------- lyd

def load_audio(path, sr, mono):
    """Dekod hvilken som helst lydfil (mp3, m4a, wav, flac, ogg ...) til float32 numpy.

    Bruker PyAV (ffmpeg-bibliotekene følger med pakken), så ffmpeg trenger ikke være
    installert. Returnerer (kanaler, samples), eller (samples,) når mono=True.
    """
    import numpy as np
    try:
        import av
    except ImportError:                                   # reserve: librosa/soundfile
        import librosa
        y, _ = librosa.load(path, sr=sr, mono=mono)
        return y.astype("float32")
    chunks = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        res = av.AudioResampler(format="fltp", layout="mono" if mono else "stereo", rate=sr)
        for frame in container.decode(stream):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray())
        for f in res.resample(None):
            chunks.append(f.to_ndarray())
    if not chunks:
        raise ValueError(f"Fant ingen lyd i {os.path.basename(path)}")
    y = np.concatenate(chunks, axis=1)
    return y[0] if mono else y


def audio_tags(path):
    """(tittel, artist) fra filas metadata, eller (None, None)."""
    try:
        import av
        with av.open(path) as c:
            meta = {k.lower(): v for k, v in c.metadata.items()}
            if not meta.get("title") and c.streams.audio:
                meta.update({k.lower(): v for k, v in c.streams.audio[0].metadata.items()})
        return (meta.get("title") or "").strip() or None, (meta.get("artist") or "").strip() or None
    except Exception:
        return None, None


def soundfile_can_read(path):
    try:
        import soundfile
        soundfile.info(path)
        return True
    except Exception:
        return False


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

# ---------------------------------------------------------------- vokal

def separate_vocals(audio, outdir):
    """Demucs (htdemucs) -> sti til vocals.wav (16 kHz mono), eller None hvis det feiler."""
    name = os.path.splitext(os.path.basename(audio))[0]
    target = os.path.join(outdir, "htdemucs", name)
    voc = os.path.join(target, "vocals.wav")
    if os.path.exists(voc):
        say("• Vokalseparasjon: gjenbruker vocals.wav")
        return voc
    say("• Vokalseparasjon: Demucs")
    try:
        import soundfile, torch
        from demucs.api import Separator
        dev = device()
        mix = load_audio(audio, 44100, mono=False)
        prog = Progress()

        def cb(d):
            if d.get("state") == "end" and d.get("audio_length"):
                prog((d["segment_offset"] + 1) / d["audio_length"])

        def run(dev):
            # overlap 0.25 -> 0.1: ~10 % raskere; shifts=0: ingen tilfeldig forskyvning -> samme resultat hver gang
            sep = Separator("htdemucs", device=dev, callback=cb, overlap=0.1, shifts=0)
            with torch.inference_mode():
                _, stems = sep.separate_tensor(torch.from_numpy(mix), 44100)
            return stems["vocals"].mean(0).cpu().numpy()

        try:
            vocals = run(dev)
        except Exception as e:
            if dev == "cpu":
                raise
            say(f"! Demucs på {dev} feilet ({type(e).__name__}) – prøver CPU")
            vocals = run("cpu")
        import librosa
        vocals = librosa.resample(vocals, orig_sr=44100, target_sr=16000)
        os.makedirs(target, exist_ok=True)
        soundfile.write(voc, vocals, 16000, subtype="PCM_16")
    except Exception as e:  # noqa: BLE001
        say(f"! Demucs feilet ({type(e).__name__}: {e}) – bruker originallyd til tekst")
        return None
    finally:
        free_memory()
    return voc if os.path.exists(voc) else None

# ---------------------------------------------------------------- språk

MODELS = {
    "no": "NbAiLab/nb-whisper-medium",     # bokmål
    "nn": "NbAiLab/nb-whisper-medium",     # nynorsk
    "en": "openai/whisper-large-v3-turbo", # engelsk
}
OTHER_MODEL = "openai/whisper-large-v3-turbo"
NORDIC = {"no", "nn", "nb", "da", "sv"}   # sang på norsk feiltolkes ofte som dansk/svensk
DETECT_MODEL = "openai/whisper-large-v3-turbo"   # whisper-base bommer ofte på sang


def pick_clips(y, sr=16000, n=3, length=30):
    """Velg n klipp à 30 sek med mest lydenergi (= mest sang i vokalsporet)."""
    import numpy as np
    half = length * sr // 2
    if len(y) <= 2 * half:
        return [y]
    cands = []
    for frac in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        mid = int(len(y) * frac)
        clip = y[max(0, mid - half): mid + half]
        cands.append((float(np.sqrt(np.mean(clip ** 2))), frac, clip))
    best = sorted(cands, key=lambda c: -c[0])[:n]
    return [c[2] for c in sorted(best, key=lambda c: c[1])]


def detect_language(y, sr=16000):
    """Språk fra klippene med mest sang. Sannsynlighetene summeres, og no/nn/da/sv
    regnes sammen (norsk sang feiltolkes ofte som svensk/dansk)."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from transformers.models.whisper.tokenization_whisper import LANGUAGES
    dev = device()
    dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
    proc = WhisperProcessor.from_pretrained(DETECT_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(DETECT_MODEL, dtype=dtype).to(dev).eval()
    codes = list(LANGUAGES)
    tok = proc.tokenizer
    ids = tok.convert_tokens_to_ids([f"<|{c}|>" for c in codes])
    sot = torch.tensor([[tok.convert_tokens_to_ids("<|startoftranscript|>")]], device=dev)
    total = torch.zeros(len(codes))
    for clip in pick_clips(y, sr):
        feats = proc(clip, sampling_rate=sr, return_tensors="pt").input_features.to(dev, dtype)
        with torch.no_grad():
            logits = model(input_features=feats, decoder_input_ids=sot).logits[0, -1, ids]
        total += torch.softmax(logits.float(), -1).cpu()
    total /= total.sum()
    prob = dict(zip(codes, total.tolist()))
    nordic = sum(prob.get(c, 0) for c in NORDIC)
    best = max(prob, key=prob.get)
    lang = "no" if nordic >= prob[best] else best
    top = sorted(prob.items(), key=lambda kv: -kv[1])[:3]
    say("• Språk: " + ", ".join(f"{c} {p:.0%}" for c, p in top) + f" (nordisk {nordic:.0%}) -> {lang}")
    del model
    free_memory()
    return lang


def resolve_language(y, lang, model):
    """Returnerer (whisper-språkkode, modell)."""
    if lang == "auto":
        say("• Språk: lytter på klippene med mest sang")
        try:
            lang = detect_language(y)
        except Exception as e:
            say(f"! Språkgjenkjenning feilet ({e}) – antar norsk")
            lang = "no"
        if lang in NORDIC and lang not in ("no", "nn"):
            lang = "no"
        say(f"• Bruker språk: {lang}")
    return lang, model or MODELS.get(lang, OTHER_MODEL)


def ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    try:                                                  # Windows
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullTotalPhys / 1e9
    except Exception:
        return 8

# ---------------------------------------------------------------- tekst

def repair_words(words, per_word=0.45, max_word=8.0):
    """Rett vanlige Whisper-feil i ord-tidsstempler."""
    # 1) overlapp mellom biter: tiden hopper bakover -> behold den siste versjonen
    norm = lambda s: re.sub(r"[^\wæøå]", "", s.lower())
    kept = []
    for idx, w in enumerate(words):
        if kept and w["start"] < kept[-1]["start"] - 1.0:
            cut = [k for k in kept if k["start"] >= w["start"] - 0.2]
            ahead = {norm(x["text"]) for x in words[idx: idx + len(cut) + 3]}
            same = sum(norm(k["text"]) in ahead for k in cut)
            if cut and same / len(cut) >= 0.5:          # samme tekst gjentatt -> dropp gammel
                del kept[len(kept) - len(cut):]
        kept.append(dict(w))
    words = kept
    # 2) ord uten varighet (f.eks. alle på 0:00): fordel mellom nærmeste gode naboer
    bad = lambda w: w["end"] - w["start"] < 0.05
    i = 0
    while i < len(words):
        if not bad(words[i]):
            i += 1
            continue
        j = i
        while j < len(words) and bad(words[j]):
            j += 1
        left = words[i - 1]["end"] if i > 0 else None
        right = words[j]["start"] if j < len(words) else None
        n = j - i
        if left is not None and right is not None and right - left >= 0.1 * n:
            step = (right - left) / n
            t0 = left
        elif right is not None:                      # starten av låta
            step = per_word
            t0 = max(0.0, right - n * per_word)
        else:                                        # slutten av låta
            step = per_word
            t0 = left or 0.0
        for k in range(n):
            words[i + k]["start"] = round(t0 + k * step, 2)
            words[i + k]["end"] = round(t0 + (k + 1) * step, 2)
        i = j
    # 3) urimelig lange ord (hallusinasjon over outro): kutt
    for w in words:
        if w["end"] - w["start"] > max_word:
            w["end"] = round(w["start"] + 2.0, 2)
    # 4) foreldreløse fragmenter: 1–3 ord etter setningsslutt, rett før en lang pause,
    #    er egentlig starten på neste linje -> flytt dem inntil neste ord
    ends = lambda w: re.search(r"[.!?]$", w["text"])
    i = 0
    while i < len(words):
        for n in (1, 2, 3):
            j = i + n                                    # første ord etter fragmentet
            if j >= len(words):
                break
            frag = words[i:j]
            if any(ends(w) for w in frag):
                break
            after_sentence = i == 0 or ends(words[i - 1]) or \
                words[i]["start"] - words[i - 1]["end"] > 2.5
            if after_sentence and words[j]["start"] - frag[-1]["end"] > 2.5:
                t = words[j]["start"]
                for m, w in enumerate(frag):
                    w["start"] = round(t - (n - m) * 0.4, 2)
                    w["end"] = round(w["start"] + 0.4, 2)
                break
        i += 1
    # 5) avkuttede ord ("f" foran "frelse"): fjern
    out = []
    for k, w in enumerate(words):
        nxt = words[k + 1]["text"].lower() if k + 1 < len(words) else ""
        bare = w["text"].lower().strip(",.")
        if len(bare) <= 2 and bare.isalpha() and not re.search(r"[aeiouyæøåé]", bare) \
                and nxt.startswith(bare) and len(nxt) > len(bare) + 2:
            continue
        out.append(w)
    return out


def transcribe(y, model, lang, beams, chunked=True, sr=16000):
    """y: 16 kHz mono float32. Returnerer ord med start/slutt (sek)."""
    import torch
    from transformers import pipeline
    free_memory()
    dev = device()

    def dtype_for(d):
        if d in ("cuda", "mps"):
            return torch.float16
        return torch.bfloat16 if ram_gb() < 15 else torch.float32   # halv minnebruk
    say(f"• Tekst: {model} på {dev} ({str(dtype_for(dev)).replace('torch.', '')}), "
        f"beams={beams}, {'biter' if chunked else 'sekvensiell'}")

    # fremdrift: tell hvor mange 28-sek-biter pipelinen sender til modellen
    chunk_s = 28
    step = chunk_s - 2 * chunk_s / 6
    total = max(1, math.ceil((len(y) / sr - chunk_s) / step) + 1) if chunked else 1
    prog = Progress()

    def run(d):
        asr = pipeline("automatic-speech-recognition", model, device=d, dtype=dtype_for(d))
        gen = asr.model.generate
        done = [0]

        def counting_generate(*a, **kw):
            out = gen(*a, **kw)
            feats = kw.get("input_features", a[0] if a else None)
            done[0] += feats.shape[0] if hasattr(feats, "shape") else 1
            prog(min(done[0] / total, 0.99))
            return out
        asr.model.generate = counting_generate
        kw = {"chunk_length_s": chunk_s} if chunked else {}
        try:
            return asr({"raw": y, "sampling_rate": sr}, return_timestamps="word", **kw,
                       generate_kwargs={"task": "transcribe", "language": lang,
                                        "num_beams": beams})
        finally:
            del asr
            free_memory()
    prog(0)
    try:
        res = run(dev)
    except Exception as e:  # MPS mangler av og til ops for ord-tidsstempler
        if dev == "cpu":
            raise
        say(f"! {dev} feilet ({type(e).__name__}) – prøver CPU")
        prog.last = -1
        res = run("cpu")
    prog(1)

    words = []
    for c in res.get("chunks", []):
        s, e = c["timestamp"]
        t = c["text"].strip()
        if not t or s is None:
            continue
        words.append({"start": float(s), "end": float(e if e is not None else s + 0.3),
                      "text": t})
    return repair_words(words)

# ---------------------------------------------------------------- rask tekst (CTranslate2)
#
# faster-whisper (CTranslate2, int8 på CPU) er 2–4x raskere enn transformers med samme modell.
# Låta deles i overlappende 28-sek-vinduer som dekodes i batch (som transformers' chunking –
# Whispers egen lange modus hopper over vers i sang), og ordene flettes ved skjøtene.

FW_ALIASES = {"openai/whisper-large-v3-turbo": "large-v3-turbo", "openai/whisper-large-v3": "large-v3",
              "openai/whisper-medium": "medium", "openai/whisper-small": "small"}
WIN, STRIDE = 28.0, 2.0              # vindu og overlapp på hver side (4,67 s gav ingen bedre tekst, 30 % tregere)


def have_fast():
    try:
        import ctranslate2, faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def fw_load(name):
    """HF-modellnavn -> faster_whisper.WhisperModel (int8 på CPU/Mac, float16 på CUDA)."""
    from faster_whisper import WhisperModel
    path = FW_ALIASES.get(name, name)
    if name.startswith("NbAiLab/"):                       # NB publiserer CT2-vekter i ct2/
        from huggingface_hub import snapshot_download
        path = os.path.join(snapshot_download(name, allow_patterns=["ct2/*"]), "ct2")
        if not os.path.exists(os.path.join(path, "model.bin")):
            raise FileNotFoundError(f"{name} har ingen CT2-versjon")
    try:
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda = False
    return WhisperModel(path, device="cuda" if cuda else "cpu",
                        compute_type="float16" if cuda else "int8",
                        cpu_threads=os.cpu_count() or 4)


def fw_detect(model, y, sr=16000, with_conf=False):
    """Språk med faster-whisper: klippene med mest sang, stopper tidlig når den er sikker."""
    total, n = {}, 0
    for clip in pick_clips(y, sr):
        _, _, probs = model.detect_language(audio=clip)
        n += 1
        for c, p in probs:
            total[c] = total.get(c, 0.0) + p
        best = max(total.values()) / n
        nordic = sum(total.get(c, 0) for c in NORDIC) / n
        if max(best, nordic) >= 0.75:                     # sikker nok – spar resten
            break
    prob = {c: p / n for c, p in total.items()}
    nordic = sum(prob.get(c, 0) for c in NORDIC)
    best = max(prob, key=prob.get)
    lang = "no" if nordic >= prob[best] else best
    top = sorted(prob.items(), key=lambda kv: -kv[1])[:3]
    say("• Språk: " + ", ".join(f"{c} {p:.0%}" for c, p in top) +
        f" (nordisk {nordic:.0%}, {n} klipp) -> {lang}")
    if with_conf:
        return lang, (nordic if lang == "no" else prob[best])
    return lang


def norwegian_variant(model, y, sr=16000):
    """Bokmål eller nynorsk? NB-Whisper er trent på begge (NRK-teksting), så språkgjenkjenningen
    i selve NB-modellen skiller dem. Ved tvil: bokmål."""
    no = nn = 0.0
    for clip in pick_clips(y, sr, n=2):
        _, _, probs = model.detect_language(audio=clip)
        p = dict(probs)
        no += p.get("no", 0.0) + p.get("nb", 0.0)
        nn += p.get("nn", 0.0)
    say(f"• Språk: bokmål {no / (no + nn or 1):.0%} / nynorsk {nn / (no + nn or 1):.0%}")
    return "nn" if nn > no else "no"


LANG_NAMES = {"no": "norsk", "nn": "norsk", "en": "engelsk", "sv": "svensk", "da": "dansk", "de": "tysk",
              "fr": "fransk", "es": "spansk", "it": "italiensk", "pt": "portugisisk", "nl": "nederlandsk",
              "pl": "polsk", "fi": "finsk", "is": "islandsk"}


def check_language(det, y, chosen):
    """Brukeren valgte språk selv: lytt kjapt og si fra hvis låta høres ut som et annet språk.
    Skriver «• Språksjekk: valgt X, hørt Y NN%» (app.py viser en advarsel med bytt-knapp)."""
    heard, conf = fw_detect(det, y, with_conf=True)
    want = "no" if chosen in NORDIC else chosen
    if heard != want and conf >= 0.6:
        say(f"• Språksjekk: valgt {chosen}, hørt {heard} {conf:.0%}")
        say(f"! Låta høres ut som {LANG_NAMES.get(heard, heard)} ({conf:.0%}), men du valgte "
            f"{LANG_NAMES.get(chosen, chosen)} – teksten kan bli feil")
    return heard


def windows(dur):
    step = WIN - 2 * STRIDE
    out, t = [], 0.0
    while True:
        out.append({"start": round(t, 3), "end": round(min(t + WIN, dur), 3)})
        if t + WIN >= dur:
            return out
        t += step


def sung_windows(y, clips, sr=16000):
    """Dropp vinduer nesten uten sang (intro/mellomspill) – raskere og ingen «hallusinert» tekst.
    Et vindu beholdes hvis minst 1,5 sek har vokalenergi over 10 % av låtas sterke partier."""
    import numpy as np
    hop = sr // 20                                         # 50 ms
    n = len(y) // hop
    if n == 0:
        return clips
    rms = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1))
    thr = 0.1 * float(np.percentile(rms, 95))
    keep = [c for c in clips
            if (rms[int(c["start"] * 20): int(c["end"] * 20)] > thr).sum() >= 30]
    return keep or clips


def merge_windows(clips, per_clip):
    """Flett ord fra overlappende vinduer. Hvert ord tilhører vinduet der midtpunktet ligger
    i vinduets «kjerne» (overlappen deles på midten). Ord rett før kuttet som forrige vindu
    ikke fikk med, tas også med; står samme ordrekke på begge sider av skjøten, fjernes den doble."""
    norm = lambda t: re.sub(r"[^\wæøå]", "", t.lower())
    out = []
    for k, (c, words) in enumerate(zip(clips, per_clip)):
        prev = clips[k - 1] if k > 0 else None
        nxt = clips[k + 1] if k + 1 < len(clips) else None
        lo = (c["start"] + prev["end"]) / 2 if prev and prev["end"] > c["start"] else -1.0
        hi = (c["end"] + nxt["start"]) / 2 if nxt and c["end"] > nxt["start"] else float("inf")
        last = out[-1]["end"] if out else -1.0
        kept = [w for w in words
                if (lo <= (w["start"] + w["end"]) / 2 < hi)
                or (lo - 2.0 <= w["start"] < lo and w["start"] >= last - 0.05)]
        if out and kept:                                  # dobbel ordrekke over skjøten?
            for m in range(min(4, len(out), len(kept)), 0, -1):
                tail, head = out[-m:], kept[:m]
                if [norm(w["text"]) for w in tail] == [norm(w["text"]) for w in head] and \
                        abs(head[0]["start"] - tail[0]["start"]) < 1.5:
                    kept = kept[m:]
                    break
        out.extend(kept)
    return out


# Whisper «finner på» tekst i instrumentale partier og stille slutter. NB-Whisper er trent på
# NRK-teksting og dikter ofte opp TV-fraser; Whisper generelt «takk for at du så på» o.l.
HALLUCINATIONS = re.compile(
    r"likhet med virkelige|virkelige hendelser|hendelser og personer|er tilfeldig\b|"
    r"teksting|tekstet av|undertekst|norsk tekst|oversatt av|oversettelse|nrk\b|"
    r"takk for at du (så|lyttet|har sett)|takk for oppmerksomheten|abonner|"
    r"thanks? for watching|subscribe|subtitles?\b|transcribed by|amara\.org|"
    r"etterforskning|programmet|episode|sesong \d|vi sees (igjen|neste)|god natt alle|"
    r"lik og del|følg oss|lyttere|seere|reklame|sponset|"
    r"www\.|https?://|\.com\b|\.no\b", re.IGNORECASE)


def is_hallucination(seg, loud=None):
    """Segment fra faster-whisper som trolig ikke er sunget. Nulltoleranse: heller miste en
    usikker linje enn å vise påfunnet tekst. Grensene er satt ut fra målt ekte sang
    (ekte linjer: no_speech 0,0, ordsikkerhet ≥ 0,50, logprob ≥ −0,95).
      - kjent TV-/teksting-frase
      - modellen tror det ikke er tale, eller er usikker (logprob / ordsannsynlighet)
      - løkke som gjentar seg (høy kompresjon)
      - nesten stille lyd under segmentet (utfading, pause)"""
    text = (seg.text or "").strip()
    if not re.search(r"\w", text):
        return True
    if HALLUCINATIONS.search(text):
        return True
    if (seg.no_speech_prob or 0) > 0.3:
        return True
    if (seg.avg_logprob or 0) < -1.0:
        return True
    words = seg.words or []
    if words and sum(w.probability for w in words) / len(words) < 0.45:
        return True
    if (seg.compression_ratio or 0) > 2.4:
        return True
    if loud is not None and loud(seg.start, seg.end) < 0.3:
        return True
    return False


def loudness_gate(y, sr=16000):
    """-> funksjon(start, slutt) = andel av 50 ms-rammer med lyd over 5 % av låtas sterke partier."""
    import numpy as np
    hop = sr // 20
    n = len(y) // hop
    if n == 0:
        return None
    rms = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1))
    thr = 0.05 * float(np.percentile(rms, 95))

    def frac(a, b):
        seg = rms[int(a * 20): max(int(a * 20) + 1, int(b * 20))]
        return float((seg > thr).mean()) if len(seg) else 0.0
    return frac


def transcribe_fast(y, model, lang, beams, sr=16000, label=None, prompt=None):
    """y: 16 kHz mono float32 -> ord med start/slutt. model: navn eller lastet WhisperModel."""
    import faster_whisper.transcribe as fwt
    from faster_whisper import BatchedInferencePipeline
    if isinstance(model, str):
        name = model
        model = fw_load(model)
    else:
        name = label or "turbo"
    dev = model.model.device
    say(f"• Tekst: {name} på {dev} (CTranslate2), beams={beams}, overlappende biter")
    # en logglinje i faster-whisper tåler ikke overlappende klipp (negativ «VAD-tid»)
    if not getattr(fwt, "_akkord_patch", False):
        orig = fwt.format_timestamp
        fwt.format_timestamp = lambda t, *a, **k: orig(max(0.0, t), *a, **k)
        fwt._akkord_patch = True
    dur = len(y) / sr
    clips = sung_windows(y, windows(dur), sr)
    prog = Progress()
    prog(0)
    segs = BatchedInferencePipeline(model).transcribe(
        y, language=lang, task="transcribe", beam_size=beams, batch_size=8, initial_prompt=prompt,
        word_timestamps=True, vad_filter=False, clip_timestamps=clips,
        without_timestamps=False,          # med tidsstempel-tokens avslutter ikke Whisper vinduet for tidlig
        condition_on_previous_text=False, no_speech_threshold=None,
        log_prob_threshold=None, compression_ratio_threshold=None)[0]
    per_clip = [[] for _ in clips]
    dropped = []
    loud = loudness_gate(y, sr)
    for seg in segs:                                       # generator: én batch om gangen
        offset = seg.seek / model.frames_per_second           # vinduets start (sek)
        k = min(range(len(clips)), key=lambda i: abs(clips[i]["start"] - offset))
        if is_hallucination(seg, loud):
            dropped.append((round(seg.start), seg.text.strip()))
            prog(min(seg.end / dur, 0.99))
            continue
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                per_clip[k].append({"start": round(float(w.start), 2),
                                    "end": round(float(w.end), 2), "text": t})
        prog(min(seg.end / dur, 0.99))
    prog(1)
    for t, text in dropped:
        say(f"! Fjernet trolig påfunnet tekst ved {t // 60}:{t % 60:02d}: «{text[:80]}»")
    for ws in per_clip:
        ws.sort(key=lambda w: w["start"])
    return repair_words(merge_windows(clips, per_clip))

# ---------------------------------------------------------------- linjer

# Ord som alltid skrives med stor forbokstav – de starter ikke nødvendigvis en ny frase
ALWAYS_CAPS = {"i", "i'm", "i'll", "i've", "i'd", "god", "jesus", "lord", "christ", "father", "spirit",
               "gud", "herre", "herren", "kristus", "jesu", "ånd", "ånden", "faderen", "hallelujah",
               "halleluja", "hosanna", "israel", "zion", "sion"}


def split_lines(words, gap=0.9, max_words=12):
    """Del ordstrømmen i linjer på pauser og punktum. Whisper skriver ofte sang uten punktum,
    men med stor forbokstav der en ny frase starter («…in the sand Where they say…») – del der også."""
    lines, cur = [], []
    for i, w in enumerate(words):
        if cur:
            pause = w["start"] - cur[-1]["end"]
            punct = re.search(r"[.!?;:]$", cur[-1]["text"]) and len(cur) >= 3
            comma = cur[-1]["text"].endswith(",") and len(cur) >= 6
            word = re.sub(r"[^\wæøåÆØÅ']", "", w["text"])
            capital = len(cur) >= 3 and word[:1].isupper() and word.lower() not in ALWAYS_CAPS
            if pause > gap or punct or comma or capital or len(cur) >= max_words \
               or (pause > 0.45 and len(cur) >= 5):
                lines.append(cur); cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    out = []
    for line in lines:
        out.extend(_split_long(line))
    return out


LONG_LINE = 8          # flere ord enn dette -> del på beste sted


def _split_long(line):
    """Del lange linjer der en ny frase mest sannsynlig starter: lengst pause, ny setning,
    stor forbokstav («I'll», «And»), etter komma – og helst nær midten."""
    if len(line) <= LONG_LINE:
        return [line]
    best, at = None, None
    for k in range(3, len(line) - 2):
        prev, w = line[k - 1], line[k]
        word = re.sub(r"[^\wæøåÆØÅ']", "", w["text"])
        score = min(w["start"] - prev["end"], 1.0) * 2
        if re.search(r"[.!?;:]$", prev["text"]):
            score += 2
        elif prev["text"].endswith(","):
            score += 1.2
        if word[:1].isupper():
            score += 1.0 if word.lower() not in ALWAYS_CAPS else 0.6
        if word.lower() in LINE_STARTS:
            score += 0.4
        score -= abs(k - len(line) / 2) / len(line)      # balanserte halvdeler
        if best is None or score > best:
            best, at = score, k
    return _split_long(line[:at]) + _split_long(line[at:])


LINE_STARTS = {"and", "but", "so", "cause", "'cause", "yeah", "oh", "when", "where", "i'll", "i",
               "og", "men", "så", "for", "når", "der", "jeg", "du", "vi"}

# ---------------------------------------------------------------- fletting

LEAD = 0.35   # akkord som kommer litt før ordet regnes til ordet
GAP = 1.5     # mer enn dette før neste ord -> instrumentalt parti


def merge(lines, chords):
    """Returnerer liste av ('chords', [akk]) og ('line', [(akk-liste, ord)], hale-akk)."""
    words = [(li, wi, w) for li, line in enumerate(lines) for wi, w in enumerate(line)]
    attach = {}                     # (li, wi) -> [akkord]
    tail = {}                       # li -> [akkord] etter siste ord
    before = {}                     # li -> [akkord] i instrumentalt parti før linja
    outro = []
    for c in chords:
        t = c["start"]
        # ord som klinger når akkorden kommer (midt i et langt ord)
        inside = next(((li, wi, w) for li, wi, w in words
                       if w["start"] + LEAD < t < w["end"]), None)
        nxt = next(((li, wi, w) for li, wi, w in words if w["start"] >= t - LEAD), None)
        prev = next(((li, wi, w) for li, wi, w in reversed(words) if w["end"] <= t), None)
        in_gap = nxt and prev and prev[0] != nxt[0] and inside is None
        if in_gap and nxt[2]["start"] - t > 0.8 and t - prev[2]["end"] <= 1.0:
            tail.setdefault(prev[0], []).append(c["chord"])     # rett etter linja
        elif nxt and nxt[2]["start"] - t <= GAP and \
           not (inside and inside[0] != nxt[0]):
            attach.setdefault(nxt[:2], []).append(c["chord"])
        elif inside and inside[1] == len(lines[inside[0]]) - 1:
            tail.setdefault(inside[0], []).append(c["chord"])
        elif nxt:
            before.setdefault(nxt[0], []).append(c["chord"])
        else:
            outro.append(c["chord"])

    out = []
    for li, line in enumerate(lines):
        if li in before:
            out.append(("chords", before[li]))
        out.append(("line", [(attach.get((li, wi), []), w["text"])
                             for wi, w in enumerate(line)], tail.get(li, [])))
    if outro:
        out.append(("chords", outro))
    return out


def to_chordpro(blocks, title, artist=None, key=None):
    o = [f"{{title: {title}}}"]
    if artist:
        o.append(f"{{artist: {artist}}}")
    if key:
        o.append(f"{{key: {key}}}")
    o.append("{comment: Auto-generert (lv-chordia + NB-Whisper) – sjekk på øret}")
    o.append("")
    for b in blocks:
        if b[0] == "break":                         # ny del i låta (vers/refreng …): tom linje
            if o and o[-1] != "":
                o.append("")
            continue
        if b[0] == "open":                          # start på vers/refreng/bro med tittel
            if o and o[-1] != "":
                o.append("")
            o.append(f"{{start_of_{b[1]}: {b[2]}}}")
            continue
        if b[0] == "close":
            o.append(f"{{end_of_{b[1]}}}")
            continue
        if b[0] == "label":                         # overskrift for instrumentalt parti
            if o and o[-1] != "":
                o.append("")
            o.append(f"{{comment: {b[1]}}}")
            continue
        if b[0] == "chords":
            if o and o[-1] != "" and not o[-1].startswith("{comment:"):   # instrumentalt parti: egen del
                o.append("")
            seq = b[1]
            for i in range(0, len(seq), 8):
                o.append(" ".join(f"[{c}]" for c in seq[i:i + 8]))
            o.append("")
        else:
            parts = ["".join(f"[{c}]" for c in ch) + w for ch, w in b[1]]
            line = " ".join(parts)
            if b[2]:
                line += " " + " ".join(f"[{c}]" for c in b[2])
            o.append(line)
    return "\n".join(o).rstrip() + "\n"

# ---------------------------------------------------------------- akkorder

CHORD_WIN, CHORD_OV = 40.0, 10.0     # akkorder i biter: 40 s vinduer, 10 s overlapp


def find_chords(audio, cache, redo=False, quiet=False):
    """Rå lv-chordia-segmenter (Harte-labels). Bufres i <navn>.chords.raw.json.

    Låta analyseres i overlappende biter på 40 s. På hele låta under ett «drukner» stille
    partier (et sparsomt første vers) mot de høye refrengene, og modellen hører én akkord der
    det er flere (målt: «Build a Boat» ga Db i 22 s på hele låta, men Db–Gb–Bbm–Ab–Gb–Db i
    biter – slik låta faktisk går). Bitene settes sammen på midten av overlappen."""
    if os.path.exists(cache) and not redo:
        say("• Akkorder: gjenbruker " + os.path.basename(cache))
        return read_json(cache)
    if not quiet:
        say("• Akkorder: lv-chordia")
    import soundfile, tempfile
    sr = 22050
    y = load_audio(audio, sr, mono=True)
    dur = len(y) / sr
    starts, t = [], 0.0
    while True:
        starts.append(t)
        if t + CHORD_WIN >= dur:
            break
        t += CHORD_WIN - CHORD_OV
    prog = None if quiet else Progress()
    segs = []
    with tempfile.TemporaryDirectory() as d:
        for k, s0 in enumerate(starts):
            e0 = min(dur, s0 + CHORD_WIN)
            lo = s0 + CHORD_OV / 2 if k > 0 else 0.0
            hi = e0 - CHORD_OV / 2 if k + 1 < len(starts) else float("inf")
            path = os.path.join(d, f"bit{k}.wav")
            soundfile.write(path, y[int(s0 * sr): int(e0 * sr)], sr, subtype="PCM_16")
            for seg in recognize(path, "full"):
                a, b = float(seg["start_time"]) + s0, float(seg["end_time"]) + s0
                a, b = max(a, lo), min(b, hi)
                if b - a > 0.01:
                    segs.append({"start_time": round(a, 2), "end_time": round(b, 2), "chord": seg["chord"]})
            if prog:
                prog((k + 1) / len(starts))
    write_json(cache, segs)
    free_memory()
    return segs

def fast_lyrics(y, a, timer):
    """Språk + tekst med faster-whisper. Gjenbruker turbo-modellen fra språkgjenkjenningen
    når låta ikke er norsk (da trengs bare én modell)."""
    lang, det = a.lang, None
    known = bool(getattr(a, "prompt", None))           # kjent tekst: språket kommer fra teksten
    if lang == "auto" or not known:
        say("• Språk: lytter på klippene med mest sang")
        det = fw_load(DETECT_MODEL)
    if lang == "auto":
        lang = fw_detect(det, y)
    elif not known:
        try:
            check_language(det, y, lang)
        except Exception as e:  # noqa: BLE001 – bare en sjekk
            say(f"! Språksjekk feilet ({type(e).__name__}: {e})")
    name = a.model or MODELS.get(lang, OTHER_MODEL)
    if name == DETECT_MODEL and det is not None:
        model = det                                       # samme modell – ikke last på nytt
    else:
        del det
        free_memory()
        model = fw_load(name)
        if lang == "no" and name == MODELS["no"]:         # «Norsk»: finn ut om det er bokmål eller nynorsk
            try:
                lang = norwegian_variant(model, y)
            except Exception as e:  # noqa: BLE001
                say(f"! Fant ikke bokmål/nynorsk ({type(e).__name__}) – bruker bokmål")
    say(f"• Bruker språk: {lang}")
    timer.lap("språk")
    words = transcribe_fast(y, model, lang, a.beams or 5, label=name.split("/")[-1], prompt=getattr(a, "prompt", None))
    del model
    free_memory()
    return words

SOLO_SECONDS = 20     # instrumentalt parti midt i låta så langt eller lenger -> «Solo / instrumental»
INTERLUDE_SECONDS = 6 # kortere akkordpartier mellom linjer er bare overganger, ikke egne deler


def with_sections(blocks, lines, words, parts=None):
    """Del låta i seksjoner med tittel (samme inndeling som akkordarket):
    sungne deler -> ('open', kind, tittel) … ('close', kind) rundt linjene (verse/chorus/bridge),
    rene akkordpartier -> ('label', tittel): Intro, Mellomspill / Interlude, Solo, Outro.
    Titlene følger låtas språk. Hver del får en tittel."""
    import chart
    lang = chart.song_lang(words)
    L = chart.LABELS[lang]
    if parts is None:
        try:
            parts = chart.label_blocks(chart.vocal_blocks(lines, 2.0), lang)
        except Exception:  # noqa: BLE001
            parts = []
    first_word = {id(ln[0]): i for i, ln in enumerate(lines)}
    starts = {}                                       # linjenr -> (kind, tittel)
    for p in parts:
        i = first_word.get(id(p["words"][0]))
        if i is not None:
            starts[i] = (p.get("kind", "verse"), p["label"])
    if 0 not in starts:
        starts[0] = ("verse", f"{L['verse']} 1")
    n_lines = sum(1 for b in blocks if b[0] == "line")
    out, li, current = [], 0, None                    # current = (kind, tittel) for delen som er åpen
    is_open = False
    for b in blocks:
        if b[0] == "line":
            if li in starts or not is_open:
                if is_open:
                    out.append(("close", current[0]))
                current = starts.get(li, current or ("verse", f"{L['verse']} 1"))
                out.append(("open", current[0], current[1]))
                is_open = True
            out.append(b)
            li += 1
            continue
        if b[0] == "chords":
            gap = lines[li][0]["start"] - lines[li - 1][-1]["end"] if 0 < li < n_lines else 0
            if 0 < li < n_lines and gap < INTERLUDE_SECONDS:
                out.append(b)                           # kort overgang: blir stående i delen
                continue
            if is_open:                                 # instrumentalt parti avslutter delen
                out.append(("close", current[0]))
                is_open = False
            if li == 0:
                name = L["intro"]
            elif li >= n_lines:
                name = L["outro"]
            else:
                name = L["solo"] if gap >= SOLO_SECONDS else L["interlude"]
            out.append(("label", name))
        out.append(b)
    if is_open:
        out.append(("close", current[0]))
    return out


def start_background_chords(audio, base, redo):
    import threading, torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    class Job(threading.Thread):
        error = None

        def run(self):
            try:
                find_chords(audio, base + ".chords.raw.json", redo, quiet=True)
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"
    job = Job(daemon=True)
    job.start()
    return job

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--model", default=None,
                    help="overstyr modell, f.eks. NbAiLab/nb-whisper-large eller openai/whisper-large-v3")
    ap.add_argument("--lang", default="auto",
                    help="auto (standard), no (bokmål), nn (nynorsk), en, eller annen Whisper-kode")
    ap.add_argument("--beams", type=int, default=None,
                    help="5 = best, 1 = raskest. Standard: 5 på GPU/MPS med ≥15 GB RAM, ellers 1")
    ap.add_argument("--slow", action="store_true",
                    help="bruk transformers i stedet for faster-whisper (tregere, samme modeller)")
    ap.add_argument("--sequential", action="store_true",
                    help="Whispers egen lange modus i stedet for 28-sek-biter (krever mye RAM)")
    ap.add_argument("--no-separate", action="store_true", help="hopp over Demucs")
    ap.add_argument("--redo-lyrics", action="store_true", help="transkriber teksten på nytt")
    ap.add_argument("--redo-chords", action="store_true", help="finn akkordene på nytt")
    ap.add_argument("--min-len", type=float, default=0.4)
    ap.add_argument("--flats", action="store_true", help="alltid b (Db, Eb ...)")
    ap.add_argument("--sharps", action="store_true", help="alltid # (C#, D# ...)")
    ap.add_argument("--key", help="overstyr {key: ...} i ChordPro (standard: estimert)")
    ap.add_argument("--quiet", action="store_true", help="ikke skriv ChordPro til stdout")
    ap.add_argument("--lyrics", help="fil med riktig tekst: brukes i stedet for det Whisper hører; "
                    "Whisper finner bare når ordene synges. Overskrifter (Vers, Refreng …) blir deltitler")
    ap.add_argument("--only", choices=["lyrics", "chords"], default=None,
                    help="lyrics = bare tekst (ingen akkorder), chords = bare akkorder (ingen tekst, mye raskere)")
    a = ap.parse_args()

    audio = os.path.abspath(a.audio)
    if not os.path.exists(audio):
        sys.exit(f"Fant ikke {audio}")
    base = os.path.splitext(audio)[0]

    timer = Timer()
    chord_job = None
    want_lyrics, want_chords = a.only != "chords", a.only != "lyrics"
    wfile = base + ".words.json"
    stanzas = None
    if a.lyrics and want_lyrics:
        import lyrics as known
        with open(a.lyrics, encoding="utf-8") as f:
            stanzas = known.parse_lyrics(f.read())
        if stanzas:
            a.prompt = known.prompt_text(stanzas)
            if a.lang == "auto":
                a.lang = known.lyrics_lang(stanzas)
            say(f"• Tekst: bruker innlimt tekst ({sum(len(s['lines']) for s in stanzas)} linjer, "
                f"{len(stanzas)} deler, språk {a.lang})")
        else:
            stanzas = None
    if not want_lyrics:
        words = []
        say("• Tekst: hoppes over (bare akkorder)")
    elif os.path.exists(wfile) and not a.redo_lyrics:
        words = repair_words(read_json(wfile))
        say(f"• Tekst: gjenbruker {os.path.basename(wfile)}")
    else:
        src = audio
        if not a.no_separate:
            src = separate_vocals(audio, os.path.join(os.path.dirname(audio), "separated")) or audio
            timer.lap("vokal")
        say("• Tekst: leser lyd")
        y = load_audio(src, 16000, mono=True)             # dekodes én gang, brukes til begge
        fast = have_fast() and not a.slow and not a.sequential
        words = None
        if fast and want_chords and not (os.path.exists(base + ".chords.raw.json") and not a.redo_chords):
            # akkordene bruker ~2 kjerner og tekstgjenkjenningen utnytter ikke alle hele
            # tida -> kjør akkordene i bakgrunnen samtidig (lite minne, CTranslate2 er eget)
            chord_job = start_background_chords(audio, base, a.redo_chords)
        if fast:
            try:
                words = fast_lyrics(y, a, timer)
            except Exception as e:  # noqa: BLE001
                say(f"! Rask tekstgjenkjenning feilet ({type(e).__name__}: {e}) – bruker transformers")
                free_memory()
        if words is None:
            lang, model = resolve_language(y, a.lang, a.model)
            timer.lap("språk")
            # beams=5 gir litt bedre tekst, men er 3–5x tregere på CPU -> bare på GPU/MPS
            beams = a.beams or (5 if device() != "cpu" and ram_gb() >= 15 else 1)
            words = transcribe(y, model, lang, beams, not a.sequential)
        del y
        write_json(wfile, words)
        free_memory()
        timer.lap("tekst")

    if chord_job is not None:
        say("• Akkorder: lv-chordia (startet i bakgrunnen)")
        chord_job.join()
        if chord_job.error:
            say(f"! Akkorder i bakgrunnen feilet ({chord_job.error}) – prøver på nytt")
    if want_chords:
        raw = find_chords(audio, base + ".chords.raw.json", a.redo_chords and chord_job is None)
        key = estimate_key(raw)                           # staver Db/Gb eller C#/F# etter toneart
        names = FLAT if a.flats else SHARP if a.sharps else names_for_key(key)
        chords = clean(raw, a.min_len, names)
        write_json(base + ".chords.json", chords, indent=2)
    else:
        key, names, chords = None, MIXED, []
        say("• Akkorder: hoppes over (bare tekst)")

    title, artist = audio_tags(audio)
    title = title or os.path.basename(base)
    key_str = a.key or (key_name(key, names) if key else None)
    lines, parts = split_lines(words), None
    if stanzas and words:                             # kjent tekst: legg den på tid, bruk den
        import lyrics as known
        import chart
        k_lines, k_parts = known.align(stanzas, words, say=say)
        if k_lines:
            lines = k_lines
            words = [w for ln in lines for w in ln]
            if k_parts:
                parts = known.label_parts(k_parts, chart.song_lang(words))
    blocks = merge(lines, chords)
    if lines:                                         # bare akkorder: ingen deler å sette titler på
        blocks = with_sections(blocks, lines, words, parts)
    cho = to_chordpro(blocks, title, artist, key_str)
    with open(base + ".cho", "w", encoding="utf-8") as f:
        f.write(cho)
    timer.lap("akkorder")

    # band-blekke: takter (beat tracking), deler og repetisjoner -> <navn>.chart.cho
    if want_chords:
        say("• Akkordark: takter og deler")
    try:
        if not want_chords:
            raise StopIteration
        import chart
        sheet = chart.make_chart(load_audio(audio, 22050, mono=True), 22050, words, lines,
                                 chords, title, artist, key_str,
                                 lang=chart.song_lang(words) if words else None, parts=parts)
        if sheet:
            with open(base + ".chart.cho", "w", encoding="utf-8") as f:
                f.write(sheet)
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001 – blekka er et tillegg; teksten er viktigst
        say(f"! Akkordark feilet ({type(e).__name__}: {e})")
    timer.lap("akkordark")
    if not a.quiet:
        try:
            print(cho)
        except UnicodeEncodeError:                        # gammel Windows-konsoll
            print(cho.encode("ascii", "replace").decode())
    say(f"• Tid: {timer.report()}")
    say(f"• Skrev {base}.cho")


if __name__ == "__main__":
    main()
