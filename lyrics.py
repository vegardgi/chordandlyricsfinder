"""Kjent tekst: brukeren limer inn riktig tekst, og vi finner bare *når* hvert ord synges.

Whisper kjører som vanlig (med teksten som hint), og ordene den hører brukes som tidsankre:
den innlimte teksten justeres mot det Whisper hørte (som DNA-sekvenser, tåler feilhørte,
manglende og ekstra ord). Ord uten anker får tid mellom nabo-ankrene.

Overskrifter i teksten (Vers, Refreng, Bro, Pre-refreng, Chorus, Verse, Bridge …) blir
titlene på delene. En overskrift uten tekst under («Refreng») gjentar den forrige delen med
samme navn. Uten overskrifter deles det på tomme linjer, og like strofer = refreng.
"""
import difflib
import re

HEAD_RE = re.compile(
    r"^\s*\[?\s*(?P<name>(?:pre[- ]?)?(?:refreng|chorus|kor)|vers|verse|bro|bridge|"
    r"intro|outro|tag|mellomspill|interlude|slutt|ending)\s*(?P<num>\d+)?\s*\]?\s*:?\s*(?:x\s*\d+)?\s*$",
    re.I)


def _kind(name):
    n = name.lower()
    if n.startswith("pre"):
        return "verse"
    if n in ("refreng", "chorus", "kor"):
        return "chorus"
    if n in ("bro", "bridge"):
        return "bridge"
    return "verse"


def parse_lyrics(text):
    """Tekst -> strofer [{"label": str|None, "kind": str|None, "lines": [str]}]."""
    stanzas, cur = [], None
    seen = {}                                          # overskrift -> linjer (for gjentakelse)

    def close():
        nonlocal cur
        if cur is not None:
            if cur["lines"]:
                stanzas.append(cur)
                if cur["label"]:
                    seen[cur["label"].lower()] = cur["lines"]
            elif cur["label"] and cur["label"].lower() in seen:   # «Refreng» uten tekst = gjenta
                stanzas.append(dict(cur, lines=list(seen[cur["label"].lower()])))
        cur = None

    for raw in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        m = HEAD_RE.match(line)
        if m:
            close()
            label = line.strip("[]: ").strip()
            label = re.sub(r"\s*x\s*\d+$", "", label, flags=re.I).strip()
            label = label[:1].upper() + label[1:]
            cur = {"label": label, "kind": _kind(m.group("name")), "lines": []}
            continue
        if not line:
            if cur is not None and cur["lines"]:
                close()
            continue
        if cur is None:
            cur = {"label": None, "kind": None, "lines": []}
        cur["lines"].append(line)
    close()
    return stanzas


def lyrics_lang(stanzas):
    """'en' eller 'no' ut fra vanlige småord."""
    en_w = {"the", "and", "you", "i", "is", "to", "of", "my", "your", "a", "in", "it", "me", "we", "be",
            "will", "with", "that", "this", "for", "on", "are", "love", "all", "so"}
    no_w = {"og", "jeg", "er", "du", "det", "på", "i", "en", "et", "som", "til", "har", "meg", "deg",
            "ikke", "han", "hun", "vi", "min", "din", "med", "for", "av", "den", "kan", "skal", "vil", "eg", "ikkje"}
    toks = [_norm(t) for s in stanzas for ln in s["lines"] for t in ln.split()]
    return "en" if sum(t in en_w for t in toks) > sum(t in no_w for t in toks) else "no"


def prompt_text(stanzas, limit=600):
    """Unike linjer som hint til Whisper (Whisper tar ~220 tokens forhåndstekst)."""
    out, seen = [], set()
    for s in stanzas:
        for ln in s["lines"]:
            k = _norm(ln)
            if k and k not in seen:
                seen.add(k)
                out.append(ln)
    txt = " ".join(out)
    return txt[:limit].rsplit(" ", 1)[0] if len(txt) > limit else txt


def _norm(t):
    return re.sub(r"[^\wæøåäöüé]", "", t.lower())


def align(stanzas, heard, say=print):
    """Legg den kjente teksten på tid. heard: ord fra Whisper [{start, end, text}].
    Returnerer (lines, parts): lines = [[{start, end, text}]], parts = deler med tittel
    (samme format som chart.label_blocks) eller None når teksten ikke har strofer."""
    K = []                                             # (tekst, strofe, linje)
    lines_idx = []
    for si, s in enumerate(stanzas):
        for ln in s["lines"]:
            li = len(lines_idx)
            lines_idx.append(si)
            for t in ln.split():
                if _norm(t):
                    K.append((t, si, li))
    R = [w for w in heard if _norm(w["text"])]
    if not K:
        return None, None
    kn, rn = [_norm(t) for t, _, _ in K], [_norm(w["text"]) for w in R]
    cache = {}

    def sim(a, b):
        v = cache.get((a, b))
        if v is None:
            v = cache[(a, b)] = 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()
        return v

    n, m = len(K), len(R)
    GAP_K, GAP_R = -0.7, -0.3
    NEG = float("-inf")
    # semiglobal: gratis å hoppe over hørte ord før første og etter siste kjente ord
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    B = [[0] * (m + 1) for _ in range(n + 1)]         # 1 = diagonal, 2 = opp (kjent ord uten anker), 3 = venstre
    for i in range(1, n + 1):
        S[i][0], B[i][0] = S[i - 1][0] + GAP_K, 2
    for i in range(1, n + 1):
        Si, Sp, Bi, a = S[i], S[i - 1], B[i], kn[i - 1]
        last = i == n
        for j in range(1, m + 1):
            s = sim(a, rn[j - 1])
            d = Sp[j - 1] + (2.0 if s >= 0.75 else 1.0 if s >= 0.5 else -1.0)
            u = Sp[j] + GAP_K
            l = Si[j - 1] + (0.0 if last else GAP_R)
            if d >= u and d >= l:
                Si[j], Bi[j] = d, 1
            elif u >= l:
                Si[j], Bi[j] = u, 2
            else:
                Si[j], Bi[j] = l, 3
    i, j = n, max(range(m + 1), key=lambda x: S[n][x]) if m else 0
    anchor = [None] * n
    while i > 0:
        if j > 0 and B[i][j] == 1:
            if sim(kn[i - 1], rn[j - 1]) >= 0.5:
                anchor[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif j == 0 or B[i][j] == 2:
            i -= 1
        else:
            j -= 1
    found = sum(a is not None for a in anchor)
    say(f"• Tekst: din tekst lagt på tid – {found} av {n} ord gjenkjent i lyden")
    if found == 0:
        return None, None

    # tider: ankre direkte, resten fordeles mellom ankrene
    times = [None] * n
    for k, r in enumerate(anchor):
        if r is not None:
            times[k] = (R[r]["start"], R[r]["end"])
    rpos = list(anchor)                               # siste hørte ord brukt (for grensene videre)
    k = 0
    while k < n:
        if times[k] is not None:
            k += 1
            continue
        e = k                                         # løpet stopper ved slutten av strofa
        while e < n and times[e] is None and K[e][1] == K[k][1]:
            e += 1
        run = list(range(k, e))
        prev_k = k - 1 if k > 0 else None
        nk = next((x for x in range(e, n) if anchor[x] is not None), None)
        lo_r = next((rpos[x] + 1 for x in range(k - 1, -1, -1) if rpos[x] is not None), 0)
        hi_r = anchor[nk] if nk is not None else m
        between = R[lo_r:hi_r]
        t_prev = times[prev_k][1] if prev_k is not None else None
        t_next = times[nk][0] if nk is not None else None
        stanza_end = e < n and K[e][1] != K[k][1]                   # neste ord er i en ny strofe
        stanza_start = prev_k is None or K[prev_k][1] != K[k][1]
        used = None
        if between:
            if stanza_end and not stanza_start and t_prev is not None:
                chain = []                            # bare hørte ord som henger på forrige ord
                last = t_prev
                for w in between:
                    if w["start"] - last > 3.0 or len(chain) > len(run) + 2:
                        break
                    chain.append(w)
                    last = w["end"]
                between = chain
            elif stanza_start and not stanza_end and t_next is not None and prev_k is not None:
                chain = []                            # hørte ord rett før neste anker
                last = t_next
                for w in reversed(between):
                    if last - w["end"] > 3.0 or len(chain) > len(run) + 2:
                        break
                    chain.insert(0, w)
                    last = w["start"]
                between = chain
        if between:
            t0, t1 = between[0]["start"], between[-1]["end"]
            used = lo_r + R[lo_r:].index(between[-1])
        else:
            t0, t1 = t_prev, t_next
            need = 0.45 * len(run) + 0.2
            if t0 is None:
                t0 = max(0.0, t1 - need)
            if t1 is None:
                t1 = t0 + need
            if t1 - t0 > need + 1.0:                  # lang pause: legg ordene inntil riktig nabo
                prev_line = prev_k is not None and K[prev_k][2] == K[k][2]
                after_prev = prev_line or nk is None or (stanza_end and not stanza_start)
                if after_prev:
                    t1 = t0 + need
                else:
                    t0 = t1 - need
        weights = [max(1, len(_norm(K[x][0]))) + 1 for x in run]
        tot, acc = sum(weights), 0.0
        for x, wgt in zip(run, weights):
            a = t0 + (t1 - t0) * acc / tot
            acc += wgt
            times[x] = (round(a, 2), round(t0 + (t1 - t0) * acc / tot, 2))
        rpos[e - 1] = used
        k = e
    # monoton tid (ankre kan krysse ved feilhørte ord)
    for x in range(1, n):
        if times[x][0] < times[x - 1][0]:
            times[x] = (times[x - 1][0], max(times[x - 1][0], times[x][1]))

    lines = [[] for _ in lines_idx]
    for (t, si, li), (a, b) in zip(K, times):
        lines[li].append({"start": a, "end": max(b, a + 0.05), "text": t})
    keep = [i for i, ln in enumerate(lines) if ln]
    lines_out = [lines[i] for i in keep]
    stanza_of = [lines_idx[i] for i in keep]

    # deler: én per strofe
    if len(stanzas) < 2 and not any(s["label"] for s in stanzas):
        return lines_out, None
    parts = []
    for si, s in enumerate(stanzas):
        ws = [w for ln, st in zip(lines_out, stanza_of) if st == si for w in ln]
        if ws:
            parts.append({"start": ws[0]["start"], "end": ws[-1]["end"], "words": ws, "rep": 1,
                          "cue": " ".join(w["text"] for w in ws[:6]),
                          "label": s["label"], "kind": s["kind"]})
    return lines_out, parts


def label_parts(parts, lang):
    """Strofer uten overskrift: like strofer = refreng osv. (chart.label_blocks). Delene som
    har overskrift fra brukeren beholder den. To like titler på rad slås ikke sammen."""
    import chart
    if all(p["label"] for p in parts):
        return parts
    if not any(p["label"] for p in parts):
        texts = [" ".join(chart.norm_text(p["words"])) for p in parts]
        for p, t in zip(parts, texts):
            p["rep"] = sum(1 for u in texts if chart._sim(t, u) >= 0.8)
        labelled = chart.label_blocks([dict(p, label=None, words=list(p["words"])) for p in parts], lang)
        if len(labelled) == len(parts):
            return labelled
        # label_blocks slo sammen naboer: legg titlene tilbake per strofe
        out, li = [], 0
        for p in parts:
            while li + 1 < len(labelled) and p["start"] >= labelled[li + 1]["start"]:
                li += 1
            out.append(dict(p, label=labelled[li]["label"], kind=labelled[li].get("kind", "verse")))
        return out
    last = None                                       # noen har overskrift: resten arver forrige
    for p in parts:
        if p["label"]:
            last = (p["label"], p["kind"])
        else:
            p["label"], p["kind"] = last or (chart.LABELS[lang]["verse"], "verse")
    return parts
