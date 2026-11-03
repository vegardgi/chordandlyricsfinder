# Akkordverktøy

Lydfil inn → ChordPro med tekst og akkorder over riktig ord. Alt kjører lokalt.

- **Akkorder:** lv-chordia (C, C/B, Em7, D/F#, G7sus4 …)
- **Tekst:** språket gjenkjennes automatisk
  - norsk → NB-Whisper fra Nasjonalbiblioteket (bokmål/nynorsk)
  - engelsk og andre språk → OpenAI Whisper large-v3-turbo
  - tidsstempel per ord
- **Vokalseparasjon:** Demucs (valgfritt, gir mye bedre tekst på låter med band)

## Appen (dra og slipp)

**Mac:** dobbeltklikk `Start.command`  ·  **Windows:** dobbeltklikk `start.bat`

Første gang lages et Python-miljø i `.venv` og avhengighetene installeres – det tar noen
minutter (ca. 1–2 GB). Skriptet bruker [uv](https://docs.astral.sh/uv/) hvis det finnes,
ellers Python 3.10–3.13, og installerer uv selv hvis ingen av delene finnes.
Deretter åpnes appen i nettleseren på <http://localhost:8765>.

- Slipp én eller flere lydfiler hvor som helst i vinduet (eller klikk for å velge).
  mp3, wav, m4a, flac, ogg, aiff – ffmpeg trengs ikke.
- Velg språk (Auto / Bokmål / Nynorsk / Engelsk) og om vokalen skal skilles ut.
- Låtene kjøres én om gangen i en kø, med steg, fremdrift og tid brukt.
- Ferdig låt vises med akkordene over teksten. Du kan **redigere** ChordPro direkte
  (Ctrl/⌘+S lagrer), **laste ned** `.cho`, **kopiere**, **kjøre på nytt** med andre valg,
  eller **slette**.
- Alle ferdige `.cho`-filer kopieres til mappa `ferdige/` («Åpne mappe med ferdige filer»).
- Feiler en jobb, vises årsaken og et logg-utdrag, med knapp for å prøve igjen uten
  vokalseparasjon.

Lukk terminalvinduet for å stoppe appen. Starter du den når den allerede kjører,
åpnes bare nettleseren. Jobbene ligger i `jobber/<id>/` og vises igjen etter omstart.

Første låt laster i tillegg ned modellene: Whisper base ~150 MB (språk), NB-Whisper medium
~1,5 GB, Whisper turbo ~1,6 GB første gang du kjører en engelsk låt, Demucs ~80 MB.

## Kommandolinje

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
source .venv/bin/activate

python songsheet.py sang.mp3                         # auto-språk: tekst + akkorder → sang.cho
python songsheet.py sang.mp3 --lang en               # tving engelsk
python songsheet.py sang.mp3 --lang nn               # nynorsk
python songsheet.py sang.mp3 --model NbAiLab/nb-whisper-large   # best, tregere
python songsheet.py sang.mp3 --no-separate           # raskere, uten Demucs
python songsheet.py sang.mp3 --redo-lyrics           # transkriber teksten på nytt
python songsheet.py sang.mp3 --redo-chords           # finn akkordene på nytt

python chords.py sang.mp3                            # bare akkorder (txt/json/cho)
python app.py [--no-browser]                         # appen uten startskript
```

Filer som lages ved siden av lydfila:

| Fil | Innhold |
|---|---|
| `sang.cho` | ChordPro med tekst og akkorder |
| `sang.chords.json` | akkorder med start/slutt (sek) |
| `sang.chords.raw.json` | rå lv-chordia-resultat – gjenbrukes ved neste kjøring |
| `sang.words.json` | ord med tid – gjenbrukes ved neste kjøring, kan redigeres for hånd |
| `separated/htdemucs/sang/vocals.wav` | utskilt vokal (16 kHz mono) – gjenbrukes |

**Tips:** Rett feil i `sang.words.json` (feilhørte ord) og kjør på nytt. Akkordene flettes inn
på nytt uten ny transkripsjon eller ny akkordgjenkjenning – det tar bare sekunder.

## Minne og fart

- Under 15 GB RAM: tekstmodellen kjører i bfloat16 med `beams=1` (~4,5 GB, ~7 min for en 4-min-låt på 2 CPU-kjerner).
- 15 GB eller mer: `beams=5`, som gir best tekst. Apple Silicon (MPS) og NVIDIA (CUDA) er mye raskere
  (float16; både Demucs og Whisper bruker GPU-en, med automatisk reserve til CPU).
- Overstyr med `--beams 1..5`.
- Modellene lastes aldri samtidig: vokal → språk → tekst → akkorder, med minnet frigjort mellom hvert steg.
  Appen kjører hver låt i en egen prosess, så alt minne gis tilbake når låta er ferdig.

## Fremdriftsprotokoll (for appen)

`songsheet.py` skriver på stderr: `• Steg: …` når et steg starter, `! …` ved advarsler og
`% 42` for prosent innen steget (Demucs og Whisper). `app.py` bruker dette til fremdriftslinja.

## Kjente begrensninger

- Korte gjennomgangsakkorder (f.eks. C#dim7) kan få litt feil navn.
- Whisper kan av og til hoppe over eller «finne på» tekst, særlig i instrumentale partier.
- Låter med både norsk og engelsk tekst får ett språk (flertallet). Velg språk manuelt ved behov.
- Linjeskift gjettes ut fra pauser i sangen – vers/refreng-merking kommer ikke automatisk.

## Senere

- Kobling til songbook-appen (ChordPro-basert, OnSong-lignende).
- Når teksten finnes fra før (CCLI/worship-repoer): align kjent tekst i stedet for Whisper.
- Automatisk vers/refreng ut fra tekst som gjentar seg.
