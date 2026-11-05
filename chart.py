#!/usr/bin/env python3
"""Band-blekke: takter, deler og repetisjoner ut fra akkorder (med tid) og tekst (med tid).

Resultatet er ChordPro med grid-seksjoner (ChordPro 6):

    {start_of_grid: label="Refreng"}
    |: Db . . . | Gb . . . | Ab . . . | Db . . . :| x2
    {end_of_grid}

Én celle per slag, «.» = samme akkord videre, «|: … :| xN» = repetisjon.

  1. Slag finnes med librosa (beat tracking). Taktart (4/4 eller 3/4) og hvor taktstreken
     står, velges slik at flest mulig akkordskifter havner på et taktslag.
  2. Teksten deles i blokker på pauser i sangen. Blokker med (nesten) samme tekst er
     refrenget; resten er vers. Spill uten sang blir Intro / Mellomspill / Outro.
  3. Like takter etter hverandre i en del, eller like deler etter hverandre, blir |: … :| xN.
"""
import bisect, difflib, re

BAR_LINE = 4                       # takter per linje


def detect_beats(y, sr):
    import librosa
    import numpy as np
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return float(np.atleast_1d(tempo)[0]), [float(b) for b in beats]


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.5


def _grids(beats):
    """Slagnettet slik beat-trackeren fant det, pluss dobbelt og halvt tempo."""
    doubled = [x for a, b in zip(beats, beats[1:]) for x in (a, (a + b) / 2)] + beats[-1:]
    return {"normal": beats, "dobbel": doubled, "halv": beats[::2]}


def choose_meter(beats, chords):
    """(slagnett, slag per takt, fase, slaglengde): flest akkordskifter på taktstreken,
    justert for hvor ofte det ville truffet tilfeldig. 3/4 må vinne tydelig over 4/4."""
    best = None
    for name, grid in _grids(beats).items():
        if len(grid) < 8:
            continue
        period = _median([b - a for a, b in zip(grid, grid[1:])])
        bpm = 60 / period
        if not 50 <= bpm <= 200:
            continue
        changes = [c["start"] for c in chords[1:] if c["end"] - c["start"] >= period * 0.9]
        if not changes:
            continue
        for n in (4, 3):
            for ph in range(n):
                starts = grid[ph::n]
                hits = 0
                for t in changes:
                    i = bisect.bisect_left(starts, t)
                    near = min(abs(t - starts[j]) for j in (i - 1, i) if 0 <= j < len(starts))
                    hits += near < 0.3 * period
                score = hits / len(changes) - 0.6 / n          # minus treff ved ren flaks
                score += 0.04 if n == 4 else 0                 # 4/4 er vanligst
                score += 0.02 if name == "normal" else 0
                score -= 0.1 if not 60 <= bpm <= 160 else 0     # urimelig tempo for en blekke
                if best is None or score > best[0]:
                    best = (score, grid, n, ph, period)
    if best is None:
        return beats, 4, 0, _median([b - a for a, b in zip(beats, beats[1:])])
    return best[1], best[2], best[3], best[4]


def bar_grid(beats, n, ph, period, t_end):
    """Taktstart-tider + slagtider per takt. Forlenger før første og etter siste slag."""
    bar_len = n * period
    starts = beats[ph::n]
    while starts and starts[0] > 0.5 * bar_len:
        starts.insert(0, starts[0] - bar_len)
    while starts and starts[-1] + bar_len < t_end:
        starts.append(starts[-1] + bar_len)
    bars = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else s + bar_len
        bars.append({"start": s, "end": e, "beats": [s + k * (e - s) / n for k in range(n)]})
    return bars


def chord_at(chords, starts, t):
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return None
    c = chords[i]
    if t < c["end"] + 1.0:                # kort hull etter en akkord: den klinger fortsatt
        return c["chord"]
    return None


def bar_cells(chords, bars):
    starts = [c["start"] for c in chords]
    out = []
    for b in bars:
        beat = (b["end"] - b["start"]) / len(b["beats"])
        out.append(tuple(chord_at(chords, starts, t + 0.15 * beat) for t in b["beats"]))
    return out


def fmt_bar(cells, prev_cells=None):
    """('C','C','G','G') -> 'C . G .'; samme takt som forrige -> '%'."""
    if prev_cells is not None and cells == prev_cells and len(set(cells)) == 1:
        return "%"
    out, prev = [], object()
    for c in cells:
        out.append("." if c == prev else (c or "N.C."))
        prev = c
    return " ".join(out)


def find_repeat(bars):
    """Hele delen er ett mønster som gjentas: (mønster, antall). Ellers (bars, 1)."""
    n = len(bars)
    for p in range(1, n // 2 + 1):
        if n % p == 0 and bars == bars[:p] * (n // p):
            return bars[:p], n // p
    return bars, 1


def compress(bars):
    """Del opp i biter: (takter, antall). Gjentatte mønstre på 2–8 takter blir én bit xN."""
    items, i, plain = [], 0, []
    while i < len(bars):
        found = None
        for p in range(min(8, (len(bars) - i) // 2), 1, -1):
            if len(set(bars[i:i + p])) == 1:                 # bare lik takt: skrives med %
                continue
            reps = 1
            while bars[i + reps * p: i + (reps + 1) * p] == bars[i:i + p]:
                reps += 1
            if reps >= 2:
                found = (p, reps)
                break
        if found:
            if plain:
                items.append((plain, 1)); plain = []
            p, reps = found
            items.append((bars[i:i + p], reps))
            i += p * reps
        else:
            plain.append(bars[i]); i += 1
    if plain:
        items.append((plain, 1))
    return items


def grid_lines(items):
    lines = []
    for bars, times in items:
        prev = None
        for i in range(0, len(bars), BAR_LINE):
            chunk = bars[i:i + BAR_LINE]
            first, last = i == 0, i + BAR_LINE >= len(bars)
            left = "|:" if times > 1 and first else "|"
            right = f":| x{times}" if times > 1 and last else "|"
            cells = []
            for b in chunk:
                cells.append(fmt_bar(b, prev))
                prev = b
            lines.append(f"{left} " + " | ".join(cells) + f" {right}")
    return lines


def norm_text(words):
    return [re.sub(r"[^\wæøå]", "", w["text"].lower()) for w in words if w["text"].strip()]


def _sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def repeat_counts(flat, owner, n_lines, N=3, min_run=6):
    """Hvor mange ganger hver linjes tekst forekommer i låta (1 = unik). Finner lange gjentatte
    ordsekvenser (tåler hørefeil og ulik linjedeling), slår sammen treff som ligger rett ved
    siden av hverandre til én forekomst, og glatter så tallene over nabolinjene."""
    n = len(flat)
    grams = {}
    for i in range(n - N + 1):
        grams.setdefault(tuple(flat[i:i + N]), []).append(i)
    pairs = sorted((occ[a], occ[b]) for occ in grams.values() if len(occ) <= 16
                   for a in range(len(occ)) for b in range(a + 1, len(occ)) if occ[b] - occ[a] >= 6)
    runs = []                                         # [p0, p1, q0, q1] – samme tekst to steder
    for p, q in pairs:
        for r in runs:
            if 0 < p - r[1] <= 6 and abs((q - p) - (r[3] - r[1])) <= 3 and q > r[3]:
                r[1], r[3] = p, q
                break
        else:
            runs.append([p, p, q, q])
    match = [set() for _ in range(n_lines)]
    for p0, p1, q0, q1 in runs:
        if p1 + N - p0 < min_run:
            continue
        for k in range(p1 + N - p0):
            a = p0 + k
            b = q0 + round(k * (q1 - q0) / max(1, p1 - p0))
            if a < n and b < n and owner[a] != owner[b]:
                match[owner[a]].add(owner[b])
                match[owner[b]].add(owner[a])
    raw = []
    for i in range(n_lines):
        occ, last = 1, None
        for j in sorted(match[i]):
            if abs(j - i) > 2 and (last is None or j - last > 2):
                occ += 1
            last = j
        raw.append(min(occ, 3))
    return [sorted(raw[max(0, i - 1):i + 2])[len(raw[max(0, i - 1):i + 2]) // 2] if 0 < i < n_lines - 1 else raw[i]
            for i in range(n_lines)]


def vocal_blocks(lines, bar_len):
    """Linjer -> blokker. En linje regnes som «gjentatt» hvis den har en sekvens på 4 ord som
    også finnes et annet sted i låta (tåler at Whisper deler linjene ulikt og hører litt ulikt
    fra gang til gang). Blokker skilles der sangen går mellom gjentatte og unike linjer, og ved
    lange pauser. Enkeltlinjer slås sammen med naboen."""
    flat, owner = [], []
    for li, ln in enumerate(lines):
        for w in ln:
            t = re.sub(r"[^\wæøå']", "", w["text"].lower())
            if t:
                flat.append(t)
                owner.append(li)
    count = repeat_counts(flat, owner, len(lines))
    rep = [c >= 2 for c in count]
    for i, ln in enumerate(lines):                   # korte linjer («Hei!») arver fra linja før
        if len(norm_text(ln)) < 3 and i:
            rep[i] = rep[i - 1]
    blocks, cur = [], []
    for i, ln in enumerate(lines):
        if cur:
            gap = ln[0]["start"] - cur[-1][-1]["end"]
            level = lambda k: min(count[k], 3) if rep[k] else 0      # 0 = unik, 2 = 2 ganger, 3 = 3+
            steady = all(level(k) == level(i) for k in range(i, min(i + 3, len(lines))))
            if gap > max(4.0, 2 * bar_len) or rep[i] != rep[i - 1] or \
               (rep[i] and level(i) != level(i - 1) and len(cur) >= 2 and steady):
                blocks.append(cur)
                cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)
    tidy = []
    for b in blocks:
        if tidy and len(b) < 2 and b[0][0]["start"] - tidy[-1][-1][-1]["end"] < max(4.0, 2 * bar_len):
            tidy[-1] = tidy[-1] + b
        else:
            tidy.append(b)
    if len(tidy) > 1 and len(tidy[0]) < 2:
        tidy[1] = tidy[0] + tidy[1]
        tidy.pop(0)
    idx = {id(ln): i for i, ln in enumerate(lines)}
    return [{"start": b[0][0]["start"], "end": b[-1][-1]["end"],
             "words": [w for ln in b for w in ln],
             "rep": max(count[idx[id(ln)]] for ln in b),
             "cue": " ".join(w["text"] for w in b[0][:6])} for b in tidy]


# Navn på delene – følger låtas språk (engelske låter får Verse/Chorus/Bridge)
LABELS = {
    "no": {"verse": "Vers", "chorus": "Refreng", "bridge": "Bro", "pre": "Pre-refreng", "intro": "Intro", "outro": "Outro",
           "interlude": "Mellomspill", "solo": "Solo / instrumental", "chords": "Akkorder"},
    "en": {"verse": "Verse", "chorus": "Chorus", "bridge": "Bridge", "pre": "Pre-Chorus", "intro": "Intro", "outro": "Outro",
           "interlude": "Interlude", "solo": "Solo / Instrumental", "chords": "Chords"},
}
EN_WORDS = {"the", "and", "you", "i", "is", "to", "of", "my", "your", "a", "in", "it", "me", "we", "be",
            "will", "with", "that", "this", "for", "on", "are", "love", "all", "so", "i'm", "i'll"}
NO_WORDS = {"og", "jeg", "er", "du", "det", "på", "i", "en", "et", "som", "til", "har", "meg", "deg",
            "ikke", "han", "hun", "vi", "min", "din", "med", "for", "av", "den", "kan", "skal", "vil"}


def song_lang(words):
    """'en' hvis teksten er engelsk, ellers 'no' (enkel ordtelling – godt nok for titlene)."""
    toks = [re.sub(r"[^\wæøå']", "", w["text"].lower()) for w in words]
    en = sum(t in EN_WORDS for t in toks)
    no = sum(t in NO_WORDS for t in toks)
    return "en" if en > no else "no"


def label_blocks(blocks, lang="no"):
    """Grupper blokker med lik tekst. Gruppa som gjentas mest = refreng; en del som bare kommer
    én gang, etter minst to refrenger, = bro/bridge; resten vers (samme tekst = samme nummer).
    Like naboer slås sammen. Hver blokk får «kind» (verse/chorus/bridge) og «label»."""
    L = LABELS.get(lang, LABELS["no"])
    texts = [norm_text(b["words"]) for b in blocks]
    grams = [{tuple(t[k:k + 4]) for k in range(len(t) - 3)} for t in texts]

    def alike(i, j):                       # delte 4-ordsekvenser tåler ulik linjedeling/hørefeil
        if grams[i] and grams[j]:
            return len(grams[i] & grams[j]) / min(len(grams[i]), len(grams[j])) >= 0.5
        return _sim(texts[i], texts[j]) >= 0.55
    group = list(range(len(blocks)))
    for i in range(len(blocks)):
        for j in range(i):
            if group[j] == j and alike(i, j):
                group[i] = j
                break
    sizes = {}                             # forekomster (to like blokker på rad = én forekomst)
    for k, g in enumerate(group):
        if k == 0 or group[k - 1] != g:
            sizes[g] = sizes.get(g, 0) + 1
    # refrenget: den gjentatte gruppa med flest forekomster (og mest tekst)
    reps = {g for g in sizes if sizes[g] >= 2}
    weight = {g: sum(b.get("rep", 1) for b, x in zip(blocks, group) if x == g) for g in reps}
    chorus = max(reps, key=lambda g: (weight[g], sizes[g], len(texts[g])), default=None)
    # pre-refreng: annen gjentatt gruppe som (nesten) alltid kommer rett før refrenget
    pre = None
    for g in reps - {chorus}:
        idxs = [i for i, x in enumerate(group) if x == g]
        before = sum(1 for i in idxs if i + 1 < len(group) and group[i + 1] == chorus)
        if before >= max(2, len(idxs) - 1):
            pre = g
            break
    names, verse, choruses = {}, 0, 0
    for b, g in zip(blocks, group):
        if g == chorus:
            b["kind"], b["label"] = "chorus", L["chorus"]
            choruses += 1
        elif g == pre:
            b["kind"], b["label"] = "verse", L["pre"]
        elif sizes[g] == 1 and choruses >= 2 and verse >= 2 and g not in names:
            b["kind"], b["label"] = "bridge", L["bridge"]
            names[g] = b["label"]
        else:
            if g not in names:
                verse += 1
                names[g] = f"{L['verse']} {verse}"
            b["kind"], b["label"] = "verse", names[g]
    def restarts(prev, b):                 # delen begynner på nytt (f.eks. siste refreng etter et break)
        a, c = norm_text(prev["words"][:8]), norm_text(b["words"][:8])
        return b["start"] - prev["end"] >= 3 and len(a) >= 4 and _sim(a, c) >= 0.6

    merged = []
    for b in blocks:
        if merged and merged[-1]["label"] == b["label"] and not restarts(merged[-1], b):
            merged[-1]["end"] = b["end"]
            merged[-1]["words"] += b["words"]
        else:
            merged.append(b)
    return merged


def make_sections(blocks, bars, lang="no"):
    L = LABELS.get(lang, LABELS["no"])
    starts = [b["start"] for b in bars]
    bar_len = _median([b["end"] - b["start"] for b in bars])

    def first_bar(t):                      # opptakt inntil 35 % av en takt før streken
        return max(0, bisect.bisect_right(starts, t + 0.35 * bar_len) - 1)

    def after_bar(t):
        return min(len(bars), bisect.bisect_right(starts, t - 0.05))

    sections, pos = [], 0
    for k, blk in enumerate(blocks):
        sb = max(first_bar(blk["start"]), pos)
        if sb - pos >= 4 or (k == 0 and sb > pos):
            secs = bars[sb - 1]["end"] - bars[pos]["start"]
            label = L["intro"] if k == 0 else (L["solo"] if secs >= 20 else L["interlude"])
            sections.append({"label": label, "a": pos, "b": sb, "cue": ""})
        elif sections:                      # én takt mellom deler hører til forrige del
            sections[-1]["b"] = sb
        else:
            sb = pos
        eb = max(after_bar(blk["end"]), sb + 1)
        sections.append({"label": blk["label"], "a": sb, "b": eb, "cue": blk["cue"]})
        pos = eb
    if pos < len(bars):
        if len(bars) - pos >= 2 or not sections:
            sections.append({"label": L["outro"], "a": pos, "b": len(bars), "cue": ""})
        else:
            sections[-1]["b"] = len(bars)
    return sections


def make_chart(y, sr, words, lines, chords, title=None, artist=None, key=None, lang=None, parts=None):
    """-> ChordPro-tekst med grid-seksjoner (band-blekke)."""
    if not chords:
        return None
    t_end = max(c["end"] for c in chords)
    tempo, beats = detect_beats(y, sr)
    if len(beats) < 8:
        return None
    beats, n, ph, period = choose_meter(beats, chords)
    bars = bar_grid(beats, n, ph, period, t_end)
    cells = bar_cells(chords, bars)
    while cells and all(c is None for c in cells[0]):      # stille start/slutt
        cells.pop(0); bars.pop(0)
    while cells and all(c is None for c in cells[-1]):
        cells.pop(); bars.pop()
    if not bars:
        return None
    lang = lang or (song_lang(words) if words else "no")
    blocks = parts if parts else label_blocks(vocal_blocks(lines, n * period), lang) if lines else []
    sections = make_sections(blocks, bars, lang) if blocks else \
        [{"label": LABELS[lang]["chords"], "a": 0, "b": len(bars), "cue": ""}]

    # like deler rett etter hverandre -> én del xN
    merged = []
    for s in sections:
        body = cells[s["a"]:s["b"]]
        if not body:
            continue
        if merged and merged[-1]["label"] == s["label"] and merged[-1]["body"] == body:
            merged[-1]["times"] += 1
            continue
        merged.append({"label": s["label"], "cue": s["cue"], "body": body, "times": 1})

    bpm = round(60 / period) if period else None
    o = []
    if title:
        o.append(f"{{title: {title}}}")
    if artist:
        o.append(f"{{artist: {artist}}}")
    if key:
        o.append(f"{{key: {key}}}")
    o.append(f"{{time: {n}/4}}")
    if bpm:
        o.append(f"{{tempo: {bpm}}}")
    o.append("{comment: Band-blekke – auto-generert, sjekk på øret}")
    for m in merged:
        pattern, times = find_repeat(m["body"])
        if len(pattern) == 1 and m["times"] == 1:          # bare lik takt: skriv den ut med %
            pattern, times = m["body"], 1
        items = [(pattern, times * m["times"])] if times > 1 or m["times"] > 1 else compress(m["body"])
        o.append("")
        if m["cue"]:
            o.append(f"{{comment: «{m['cue']} …»}}")
        o.append(f'{{start_of_grid: label="{m["label"]}"}}')
        o.extend(grid_lines(items))
        o.append("{end_of_grid}")
    return "\n".join(o) + "\n"
