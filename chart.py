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


def vocal_blocks(lines, bar_len):
    """Linjer -> blokker. Skiller der sangen går fra linjer som gjentas i låta (refreng-
    aktige) til unike linjer (vers-aktige) og omvendt, og ved lange pauser."""
    texts = [norm_text(ln) for ln in lines]
    rep = []
    for i, t in enumerate(texts):
        if len(t) < 3:                                   # «Hei!» o.l. arver fra linja før
            rep.append(rep[-1] if rep else False)
            continue
        rep.append(any(j != i and len(texts[j]) >= 3 and _sim(t, texts[j]) >= 0.7
                       for j in range(len(texts))))
    blocks, cur = [], []
    for i, ln in enumerate(lines):
        if cur:
            gap = ln[0]["start"] - cur[-1][-1]["end"]
            if gap > max(4.0, 2 * bar_len) or rep[i] != rep[i - 1]:
                blocks.append(cur)
                cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)
    # enkeltlinjer blir ikke egne deler: slå sammen med delen før (eller etter)
    tidy = []
    for b in blocks:
        if tidy and len(b) < 2 and b[0][0]["start"] - tidy[-1][-1][-1]["end"] < max(4.0, 2 * bar_len):
            tidy[-1] = tidy[-1] + b
        else:
            tidy.append(b)
    if len(tidy) > 1 and len(tidy[0]) < 2:
        tidy[1] = tidy[0] + tidy[1]
        tidy.pop(0)
    blocks = tidy
    return [{"start": b[0][0]["start"], "end": b[-1][-1]["end"],
             "words": [w for ln in b for w in ln],
             "cue": " ".join(w["text"] for w in b[0][:6])} for b in blocks]


def label_blocks(blocks):
    """Grupper blokker med lik tekst. Gruppa som gjentas mest = refreng; andre grupper er
    vers (samme tekst = samme nummer). Like naboer slås sammen."""
    texts = [norm_text(b["words"]) for b in blocks]
    group = list(range(len(blocks)))
    for i in range(len(blocks)):
        for j in range(i):
            if group[j] == j and _sim(texts[i], texts[j]) >= 0.55:
                group[i] = j
                break
    sizes = {}
    for g in group:
        sizes[g] = sizes.get(g, 0) + 1
    chorus = max((g for g in sizes if sizes[g] >= 2),
                 key=lambda g: (sizes[g], len(texts[g])), default=None)
    names, verse = {}, 0
    for b, g in zip(blocks, group):
        if g == chorus:
            b["label"] = "Refreng"
        else:
            if g not in names:
                verse += 1
                names[g] = f"Vers {verse}"
            b["label"] = names[g]
    merged = []
    for b in blocks:
        if merged and merged[-1]["label"] == b["label"]:
            merged[-1]["end"] = b["end"]
            merged[-1]["words"] += b["words"]
        else:
            merged.append(b)
    return merged


def make_sections(blocks, bars):
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
            sections.append({"label": "Intro" if k == 0 else "Mellomspill", "a": pos, "b": sb, "cue": ""})
        elif sections:                      # én takt mellom deler hører til forrige del
            sections[-1]["b"] = sb
        else:
            sb = pos
        eb = max(after_bar(blk["end"]), sb + 1)
        sections.append({"label": blk["label"], "a": sb, "b": eb, "cue": blk["cue"]})
        pos = eb
    if pos < len(bars):
        if len(bars) - pos >= 2 or not sections:
            sections.append({"label": "Outro", "a": pos, "b": len(bars), "cue": ""})
        else:
            sections[-1]["b"] = len(bars)
    return sections


def make_chart(y, sr, words, lines, chords, title=None, artist=None, key=None):
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
    blocks = label_blocks(vocal_blocks(lines, n * period)) if lines else []
    sections = make_sections(blocks, bars) if blocks else \
        [{"label": "Akkorder", "a": 0, "b": len(bars), "cue": ""}]

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
