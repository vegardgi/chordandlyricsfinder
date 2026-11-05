# Akkordverktøy

Lydfil inn → ChordPro med tekst og akkorder over riktig ord. Alt kjører lokalt.

- **Akkorder:** lv-chordia (C, C/B, Em7, D/F#, G7sus4 …)
- **Tekst:** språket gjenkjennes automatisk (Whisper turbo på klippene med mest sang;
  no/nn/da/sv regnes sammen fordi norsk sang ofte feiltolkes som svensk)
  - norsk → NB-Whisper fra Nasjonalbiblioteket (bokmål/nynorsk)
  - engelsk og andre språk → OpenAI Whisper large-v3-turbo
  - kjøres med faster-whisper (CTranslate2, int8) – 2x raskere enn transformers, med beam 5
  - tidsstempel per ord
- **Toneart:** estimeres fra akkordene; akkordene staves etter den (Db/Gb i Db-dur, C#/F# i A-dur)
  og `{key: …}` skrives i ChordPro
- **Vokalseparasjon:** Demucs (valgfritt, gir mye bedre tekst på låter med band)

## Appen (dra og slipp)

**Mac:** dobbeltklikk `Start.command`  ·  **Windows:** dobbeltklikk `start.bat`

Første gang lages et Python-miljø i `.venv` og avhengighetene installeres – det tar noen
minutter (ca. 1–2 GB). Skriptet bruker [uv](https://docs.astral.sh/uv/) hvis det finnes,
ellers Python 3.10–3.13, og installerer uv selv hvis ingen av delene finnes.
Deretter åpnes appen i nettleseren på <http://localhost:8765>.

- Slipp én eller flere lydfiler hvor som helst i vinduet (eller klikk for å velge). Filene vises
  med navn, lengde og størrelse under «Klar til å starte» – ingenting starter før du trykker **Start**.
  mp3, wav, m4a, flac, ogg, aiff – ffmpeg trengs ikke.
- Velg hva du vil ha: **Tekst og akkorder**, **Kun tekst** eller **Kun akkorder** (raskest – ca. 15 s
  for en 3-min-låt; språk og Rask/Grundig gråes da ut). Knapper og visninger som ikke gjelder for låta
  (f.eks. transponer/akkordark for «Kun tekst»), gråes ut på resultatet.
- Velg språk (Auto / Bokmål / Nynorsk / Engelsk) og fart: **Rask** (standard, ofte godt resultat)
  eller **Grundig** (tregere: skiller ut sangstemmen med Demucs først – bedre på låter med fullt band).
  Ferdige Rask-låter har en snarvei: «Ikke bra nok? Prøv Grundig».
- Låtene kjøres én om gangen i en kø, med steg, fremdrift og tid brukt.
- Ferdig låt vises i tre visninger: **Tekst med akkorder** (akkordene over ordene), **Akkordark**
  (band-blekke: Intro/Vers/Refreng/Mellomspill/Outro, taktstreker, én celle per slag, `%` for lik
  takt, `|: … :| x2` for repetisjoner, taktart og tempo) og **ChordPro**.
- **Transponer** (− / +, staving etter ny toneart) og **Capo** (med forslag til capo som gir lette grep).
  Gjelder visning, kopiering, nedlasting og PDF.
- Knapper: **Kopier tekst med akkorder over** (for melding/e-post/Word), **Kopier ChordPro**,
  **Kopier akkordark** (ChordPro grid), **Last ned .cho**, **PDF: tekst med akkorder**,
  **PDF: akkordark** (forhåndsvisning av A4-sidene med 1 eller 2 kolonner, størrelse −/+ og «få plass
  på N sider»; deretter «Lagre som PDF»), **Rediger** (redigerer visningen du står i;
  Ctrl/⌘+S lagrer), **Kjør på nytt**, **Slett**.
- Nederst: lenke til vilkår, opphavsrett og personvern (`/vilkar`).
- Alle ferdige `.cho`-filer kopieres til mappa `ferdige/` («Åpne mappe med ferdige filer»).
- Feiler en jobb, vises årsaken og et logg-utdrag, med knapp for å prøve igjen med Rask.

Lukk terminalvinduet for å stoppe appen. Starter du den når den allerede kjører,
åpnes bare nettleseren. Jobbene ligger i `jobber/<id>/` og vises igjen etter omstart.

Første låt laster i tillegg ned modellene: Whisper turbo ~1,6 GB (språkgjenkjenning og engelsk
tekst), NB-Whisper medium ~3 GB (norsk tekst), Demucs ~80 MB, akkordmodellen følger med pakken.

## På nett (Modal)

Samme app kan ligge på nett via [Modal](https://modal.com): siden og køen kjører på en liten
CPU-container, og analysen på en GPU som bare koster noe mens den jobber (per sekund, skalerer til null).

**Ingen brukere, ingen lagring.** Hver nettleserfane er en anonym økt som bare ser sine egne låter.
Lydfila og alle mellomresultater slettes så snart akkordarket er ferdig. Akkordarket slettes 20 sek
etter at siden lukkes, og senest etter 15 minutter uten aktivitet. Resultatet sendes fra GPU-en via
en kø som tømmes med en gang (returverdier fra `spawn()` lagrer Modal i 7 dager, så de inneholder
bare status). Man laster opp, kopierer og er ferdig. Modellene er det eneste som lagres
(i Modal-volumet `akkord-models`). Misbruksgrense: 20 opplastinger per IP per time (bare i minnet).

Før lansering: sett et **kostnadstak** i Modal (Settings → Usage/Billing) og les Modals vilkår og
databehandleravtale (modal.com/legal/terms, modal.com/legal/dpa). Vilkårssiden (`/vilkar`) beskriver
personvern, lagringstider og opphavsrett for nettversjonen.

```bash
brew install uv                               # Mac; se docs.astral.sh/uv for andre systemer
uv tool install --python 3.12 modal           # Modal-CLI i eget miljø (ingen pip nødvendig)
uv tool update-shell                          # så «modal» finnes i PATH – åpne ny terminal etterpå
modal setup                                   # logg inn (én gang)
modal run modal_app.py::download_models       # legg modellene i volumet (én gang, ~5 GB)
modal serve modal_app.py                      # test på en midlertidig URL
modal deploy modal_app.py                     # publiser – Modal skriver ut adressen
```

Innstillinger (miljøvariabler når du kjører `modal deploy`):

| Variabel | Standard | Betydning |
|---|---|---|
| `AKKORD_GPU` | `L4` | GPU-type (`T4` er billigere og tregere) |
| `AKKORD_GPU_MAX` | `3` | maks samtidige GPU-er = kostnadstak |
| `AKKORD_MAX_MB` | `60` | maks filstørrelse |
| `AKKORD_MAX_ACTIVE` | `3` | maks låter i kø per fane |
| `AKKORD_MAX_PER_HOUR` | `20` | maks opplastinger per IP per time |
| `AKKORD_SESSION_TTL` | `900` | sekunder uten aktivitet før alt slettes |

Nettmodus kan også testes lokalt uten Modal: `AKKORD_ONLINE=1 python app.py` (analysen kjører
da på maskinen din).

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
python songsheet.py sang.mp3 --only chords           # bare akkorder (+ akkordark), ingen tekst
python songsheet.py sang.mp3 --only lyrics           # bare tekst, ingen akkorder
python songsheet.py sang.mp3 --beams 1               # enda raskere tekst, litt dårligere
python songsheet.py sang.mp3 --flats                 # alltid b (standard: etter toneart)

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
| `sang.chart.cho` | band-blekke (ChordPro grid): deler, taktstreker, repetisjoner |
| `separated/htdemucs/sang/vocals.wav` | utskilt vokal (16 kHz mono) – gjenbrukes |

**Tips:** Rett feil i `sang.words.json` (feilhørte ord) og kjør på nytt. Akkordene flettes inn
på nytt uten ny transkripsjon eller ny akkordgjenkjenning – det tar bare sekunder.

## Fart og minne

Målt på en skymaskin med 4 CPU-kjerner uten GPU (den tregeste plattformen appen er laget for):

| Låt | Vokal (Demucs) | Språk | Tekst | Akkorder | Totalt |
|---|---|---|---|---|---|
| Aldri Langt Borte (3:07) | 82 s | 10 s | 61 s | 13 s | **2:46** |
| På fjell (3:37) | 93 s | 17 s | ~100 s (akkorder parallelt) | – | **~3:35** |
| Original-koden, samme maskin (2:48-låt) | 110 s | 30 s | 270 s | 15 s | 7:00 |

Minnetopp: ~7 GB (før: 11,7 GB). På Apple Silicon kjører Demucs på GPU (MPS) og faster-whisper
bruker Apples Accelerate, så det bør gå klart raskere enn tallene over (ikke målt ennå).
Med **Rask** spares hele Demucs-steget. Hver låt rapporterer tid per steg
(«Tid: vokal …, språk …, tekst …, akkorder …») nederst i forhåndsvisningen og i loggen.

Slik er det gjort raskt:

- Tekst med **faster-whisper** (CTranslate2 int8 på CPU/Mac, float16 på NVIDIA). NB-Whisper
  publiserer CT2-vekter selv (`ct2/`). Låta deles i 28-sek-vinduer med 2 s overlapp som dekodes i
  batch (Whispers egen lange modus hopper over vers i sang), og ordene flettes i skjøtene.
- **Vinduer uten sang** (intro, mellomspill, outro) hoppes over – raskere og ingen «hallusinert» tekst.
- **Språk** stopper etter første klipp når den er ≥80 % sikker.
- **Akkordene** kjøres i bakgrunnen mens teksten gjenkjennes (de bruker ~2 kjerner).
- Demucs kjører i samme prosess (ingen ekstra oppstart), overlapp 0,1, på GPU når det finnes.
- Lyden dekodes én gang med PyAV (ingen ffmpeg nødvendig).
- Modellene lastes aldri samtidig (bortsett fra den lille akkordmodellen). Appen kjører hver
  låt i en egen prosess, så alt minne gis tilbake når låta er ferdig.
- «Kjør på nytt» gjenbruker utskilt vokal og akkorder, så bare teksten gjøres på nytt.
- Reserve: `--slow` bruker transformers-løypa (samme modeller) hvis faster-whisper ikke virker.

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
