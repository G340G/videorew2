# VHS/PPT Horror Generator (GitHub-only)

This repo generates an **always-different** analogue-horror MP4 that looks like a **found VHS job‑training PowerPoint**:
- Web‑scraped text & images (Wikipedia + Wikimedia Commons)
- Mixed **normal** wellness slides + **scary** protocol / Jane Doe / fatal error beats
- Strong VHS look: tracking line, scanlines, chroma bleed, vignette, noise, timecode
- **Redactions/censorship** in text and images (unease + "classified" vibe)
- Up to **3** brief pop‑up distorted images (~0.5s) + one‑frame flashes (jump cuts)
- Audio bed + abrupt bursts + **TTS** narration (espeak), muxed into a single MP4
- Optional random **transmission failure** (freeze frame and/or early cut)

## GitHub Actions (recommended)
1. Put any images you want to include into `assets/images/` (jpg/png/webp).
2. Copy `config.example.yaml` to `config.yaml` and tweak.
3. Push to GitHub — the workflow will build `out.mp4` as an artifact.

## Run locally (optional)
```bash
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml
python generate.py --config config.yaml --out out.mp4
```

## Configuration highlights
- `theme_key`: leave empty to let the generator pick a new theme key every run.
- `web.enable`: turn scraping on/off.
- `render.vhs_strength`: stronger tape look.
- `render.redaction_strength`: more censorship bars + mosaic blur.
- `transmission.*`: probability of freeze / early‑end transmission errors.
- `render.max_popups` + `render.popup_seconds`: pop‑up "wrong" images.

## Notes
- The generator only uses public sources and local images you provide.
- You can safely commit placeholder images or keep the folder empty; the web pool will still work.
