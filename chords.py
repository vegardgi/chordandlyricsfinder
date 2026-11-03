#!/usr/bin/env python3
"""Lydfil inn -> akkorder som C, C/B, Em7, D/F# ut (tekst + ChordPro + JSON).

Bruk:  python chords.py sang.mp3 [--min-len 0.4] [--flats]
Krever lv-chordia (pip install lv-chordia).
"""
import argparse, json, os, sys

SHARP = "C C# D D# E F F# G G# A A# B".split()
FLAT  = "C Db D Eb E F Gb G Ab A Bb B".split()
MIXED = "C C# D Eb E F F# G Ab A Bb B".split()   # vanligste skrivemåte
PC = {n: i for names in (SHARP, FLAT) for i, n in enumerate(names)}
PC.update({"Cb": 11, "Fb": 4, "E#": 5, "B#": 0})

# Harte-kvalitet -> visningsnavn
QUAL = {
    "maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7",
    "dim": "dim", "aug": "aug", "dim7": "dim7", "hdim7": "m7b5",
    "sus2": "sus2", "sus4": "sus4", "7sus4": "7sus4", "6": "6", "min6": "m6",
    "9": "9", "maj9": "maj9", "min9": "m9", "11": "11", "min11": "m11",
    "13": "13", "maj13": "maj13", "min13": "m13", "minmaj7": "m(maj7)",
    "add9": "add9", "madd9": "madd9", "maj6": "6", "1": "5", "5": "5",
    "sus4(b7)": "7sus4", "sus2(b7)": "7sus2", "sus4(b7,9)": "9sus4",
    "maj(9)": "add9", "min(9)": "madd9", "maj(2)": "add2", "maj(4)": "add4",
    "maj6(9)": "6/9", "min6(9)": "m6/9", "7(b9)": "7b9", "7(#9)": "7#9",
    "7(#11)": "7#11", "7(b13)": "7b13", "dim(b7)": "m7b5",
}
# Intervall (Harte) -> halvtoner
IV = {"1": 0, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4, "4": 5, "#4": 6,
      "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9, "bb7": 9, "b7": 10, "7": 11,
      "b9": 1, "9": 2, "#9": 3, "11": 5, "#11": 6, "b13": 8, "13": 9}


def name(pc, names):
    return names[pc % 12]


def harte_to_display(label, names=MIXED):
    """'C:maj/7' -> 'C/B', 'E:min7' -> 'Em7', 'N' -> None."""
    if label in ("N", "X", ""):
        return None
    root_part, _, bass = label.partition("/")
    root, _, qual = root_part.partition(":")
    qual = qual or "maj"
    rpc = PC[root]
    if qual.startswith("(") or qual not in QUAL:  # ukjent/grad-liste: behold råtekst
        q = qual.replace("(", "").replace(")", "")
        disp = name(rpc, names) + ("" if q in ("1,3,5", "") else f"({q})")
    else:
        disp = name(rpc, names) + QUAL[qual]
    if bass and bass in IV and IV[bass] != 0:
        disp += "/" + name(rpc + IV[bass], names)
    return disp


def recognize(path, chord_dict="full"):
    from lv_chordia.chord_recognition import chord_recognition
    return chord_recognition(audio_path=os.path.abspath(path), chord_dict_name=chord_dict)


def clean(segs, min_len, names):
    out = []
    for s in segs:
        c = harte_to_display(s["chord"], names)
        start, end = float(s["start_time"]), float(s["end_time"])
        if c is None or end - start <= 0:
            continue
        if out and out[-1]["chord"] == c:           # slå sammen like naboer
            out[-1]["end"] = end
            continue
        out.append({"start": round(start, 2), "end": round(end, 2), "chord": c,
                    "raw": s["chord"]})
    # fjern korte blipp (sannsynlig støy), slå sammen igjen
    kept = [s for s in out if s["end"] - s["start"] >= min_len] or out
    merged = []
    for s in kept:
        if merged and merged[-1]["chord"] == s["chord"]:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    return merged


def mmss(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--min-len", type=float, default=0.4,
                    help="dropp akkorder kortere enn dette (sek)")
    ap.add_argument("--dict", default="full", choices=["full", "submission", "ismir2017"])
    ap.add_argument("--flats", action="store_true", help="skriv b i stedet for #")
    ap.add_argument("--sharps", action="store_true", help="skriv # i stedet for b")
    a = ap.parse_args()
    names = FLAT if a.flats else SHARP if a.sharps else MIXED

    segs = clean(recognize(a.audio, a.dict), a.min_len, names)
    base = os.path.splitext(a.audio)[0]
    title = os.path.basename(base)

    with open(base + ".chords.json", "w") as f:
        json.dump(segs, f, indent=2, ensure_ascii=False)

    with open(base + ".chords.txt", "w") as f:
        for s in segs:
            f.write(f"{mmss(s['start'])}  {s['chord']}\n")

    with open(base + ".cho", "w") as f:
        f.write(f"{{title: {title}}}\n{{comment: Auto-akkorder (lv-chordia) – sjekk på øret}}\n\n")
        line = []
        for s in segs:
            line.append(f"[{s['chord']}]")
            if len(line) == 8:
                f.write(" ".join(line) + "\n"); line = []
        if line:
            f.write(" ".join(line) + "\n")

    print(" | ".join(s["chord"] for s in segs))
    print(f"\nSkrev: {base}.chords.txt, {base}.chords.json, {base}.cho", file=sys.stderr)


if __name__ == "__main__":
    main()
