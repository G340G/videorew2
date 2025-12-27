#!/usr/bin/env python3
"""
generate.py — "Found VHS Job Training" analogue horror generator (GitHub-friendly)

What it does
- Builds an always-different horror video that blends "normal corporate wellness" slides with unsettling
  safety bulletins, "Jane Doe" intermissions, redactions, and VHS tracking errors.
- Scrapes theme text + images from the web (Wikipedia + Wikimedia Commons) when enabled.
- Mixes in local images from your repo (assets/images) into the slide collage.
- Adds pop-up distorted images (max N) for ~0.5s each.
- Adds stronger VHS feel: timecode, scanlines, chroma bleed, tracking line, tape noise, and hard cutouts.
- Adds a random "transmission error" that may freeze and/or end the tape early.
- Generates noisy music bed + abrupt noises + TTS narration with randomized voices.

Designed for running in GitHub Actions only (no local machine required).

Usage
  python generate.py --config config.yaml --out out.mp4

Notes
- Requires ffmpeg and espeak to be available in the environment (GitHub ubuntu-latest has them).
- Scraping is kept lightweight + polite (small number of requests).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav
import imageio.v2 as imageio

# ----------------------------
# Config + parsing utilities
# ----------------------------

DEFAULTS: Dict[str, Any] = {
    "seed": 1337,
    "theme_key": "",
    "workdir": ".work",
    "local_images_dir": "assets/images",
    "web": {"enable": True, "text_paragraphs": 10, "image_limit": 12, "timeout_s": 15},
    "render": {
        "width": 640,
        "height": 480,  # 4:3 VHS by default
        "fps": 15,
        "slide_seconds": 3.2,
        "max_popups": 3,
        "popup_seconds": 0.5,
        "vhs_strength": 1.0,
        "redaction_strength": 1.0,
        "flashes": 12,
    },
    "audio": {
        "sr": 44100,
        "music": True,
        "tts": True,
        "tts_speed": 155,
        "tts_pitch": 32,
        "tts_amp": 170,
        "voices": ["en-us", "en", "en-uk-rp", "en-uk-north", "en-sc"],
    },
    "transmission": {
        "enable": True,
        "error_probability": 0.35,   # chance the tape has a failure moment
        "freeze_probability": 0.6,   # given an error, chance it's a freeze
        "early_end_probability": 0.45,  # given an error, chance it ends early
        "freeze_seconds_min": 0.8,
        "freeze_seconds_max": 2.2,
    },
    "story": {
        "slide_count": 10,
        "normal_ratio": 0.45,  # how many "normal wellness" slides to keep
        "include_intro_outro": True,
        "include_jane_doe": True,
        "include_fatal": True,
        "easter_egg_probability": 0.25,
    },
}

UA = "pptex-vhs-generator/1.0 (+github-actions; educational/art project)"


def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if _is_mapping(v) and _is_mapping(out.get(k)):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_env_pat = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def resolve_env_value(v: Any) -> Any:
    """Resolve ${ENV} placeholders. If missing env var, return empty string."""
    if isinstance(v, str):
        m = _env_pat.match(v.strip())
        if m:
            return os.environ.get(m.group(1), "")
    return v


def coerce_int(v: Any, default: int) -> int:
    v = resolve_env_value(v)
    if v is None:
        return default
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return default
        try:
            return int(s, 10)
        except ValueError:
            # allow "1337.0"
            try:
                return int(float(s))
            except Exception:
                return default
    return default


def coerce_float(v: Any, default: float) -> float:
    v = resolve_env_value(v)
    if v is None or isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return default
        try:
            return float(s)
        except ValueError:
            return default
    return default


def coerce_bool(v: Any, default: bool) -> bool:
    v = resolve_env_value(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    return default


def as_dict(v: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    # Fixes the user's error: cfg["web"] might accidentally be set to true/false in YAML.
    if isinstance(v, dict):
        return v
    return dict(default)


def load_config(path: Path) -> Dict[str, Any]:
    raw = {}
    if path and path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = deep_merge(DEFAULTS, raw)

    # normalize nested dicts (handles cases where user set "web: true")
    cfg["web"] = as_dict(cfg.get("web"), DEFAULTS["web"])
    cfg["render"] = as_dict(cfg.get("render"), DEFAULTS["render"])
    cfg["audio"] = as_dict(cfg.get("audio"), DEFAULTS["audio"])
    cfg["transmission"] = as_dict(cfg.get("transmission"), DEFAULTS["transmission"])
    cfg["story"] = as_dict(cfg.get("story"), DEFAULTS["story"])

    # coerce key fields
    cfg["seed"] = coerce_int(cfg.get("seed"), DEFAULTS["seed"])
    cfg["web"]["enable"] = coerce_bool(cfg["web"].get("enable"), True)
    cfg["web"]["text_paragraphs"] = coerce_int(cfg["web"].get("text_paragraphs"), 10)
    cfg["web"]["image_limit"] = coerce_int(cfg["web"].get("image_limit"), 12)
    cfg["web"]["timeout_s"] = coerce_int(cfg["web"].get("timeout_s"), 15)

    cfg["render"]["width"] = coerce_int(cfg["render"].get("width"), 640)
    cfg["render"]["height"] = coerce_int(cfg["render"].get("height"), 480)
    cfg["render"]["fps"] = coerce_int(cfg["render"].get("fps"), 15)
    cfg["render"]["slide_seconds"] = coerce_float(cfg["render"].get("slide_seconds"), 3.2)
    cfg["render"]["max_popups"] = coerce_int(cfg["render"].get("max_popups"), 3)
    cfg["render"]["popup_seconds"] = coerce_float(cfg["render"].get("popup_seconds"), 0.5)
    cfg["render"]["vhs_strength"] = coerce_float(cfg["render"].get("vhs_strength"), 1.0)
    cfg["render"]["redaction_strength"] = coerce_float(cfg["render"].get("redaction_strength"), 1.0)
    cfg["render"]["flashes"] = coerce_int(cfg["render"].get("flashes"), 12)

    cfg["audio"]["sr"] = coerce_int(cfg["audio"].get("sr"), 44100)
    cfg["audio"]["music"] = coerce_bool(cfg["audio"].get("music"), True)
    cfg["audio"]["tts"] = coerce_bool(cfg["audio"].get("tts"), True)
    cfg["audio"]["tts_speed"] = coerce_int(cfg["audio"].get("tts_speed"), 155)
    cfg["audio"]["tts_pitch"] = coerce_int(cfg["audio"].get("tts_pitch"), 32)
    cfg["audio"]["tts_amp"] = coerce_int(cfg["audio"].get("tts_amp"), 170)
    if not isinstance(cfg["audio"].get("voices"), list):
        cfg["audio"]["voices"] = DEFAULTS["audio"]["voices"]

    cfg["transmission"]["enable"] = coerce_bool(cfg["transmission"].get("enable"), True)
    cfg["transmission"]["error_probability"] = coerce_float(cfg["transmission"].get("error_probability"), 0.35)
    cfg["transmission"]["freeze_probability"] = coerce_float(cfg["transmission"].get("freeze_probability"), 0.6)
    cfg["transmission"]["early_end_probability"] = coerce_float(cfg["transmission"].get("early_end_probability"), 0.45)
    cfg["transmission"]["freeze_seconds_min"] = coerce_float(cfg["transmission"].get("freeze_seconds_min"), 0.8)
    cfg["transmission"]["freeze_seconds_max"] = coerce_float(cfg["transmission"].get("freeze_seconds_max"), 2.2)

    cfg["story"]["slide_count"] = coerce_int(cfg["story"].get("slide_count"), 10)
    cfg["story"]["normal_ratio"] = coerce_float(cfg["story"].get("normal_ratio"), 0.45)
    cfg["story"]["include_intro_outro"] = coerce_bool(cfg["story"].get("include_intro_outro"), True)
    cfg["story"]["include_jane_doe"] = coerce_bool(cfg["story"].get("include_jane_doe"), True)
    cfg["story"]["include_fatal"] = coerce_bool(cfg["story"].get("include_fatal"), True)
    cfg["story"]["easter_egg_probability"] = coerce_float(cfg["story"].get("easter_egg_probability"), 0.25)

    # safe theme_key
    cfg["theme_key"] = str(resolve_env_value(cfg.get("theme_key", "")) or "").strip()

    cfg["workdir"] = str(resolve_env_value(cfg.get("workdir", ".work")) or ".work")
    cfg["local_images_dir"] = str(resolve_env_value(cfg.get("local_images_dir", "assets/images")) or "assets/images")
    return cfg


# ----------------------------
# Web scraping (lightweight)
# ----------------------------

def _http_get(url: str, timeout: int) -> requests.Response:
    return requests.get(url, headers={"User-Agent": UA}, timeout=timeout)


def choose_theme_key(rng: random.Random) -> str:
    # A curated pool that tends to yield good public-domain imagery on Commons.
    pool = [
        "workplace safety", "industrial training", "office equipment", "employee handbook",
        "memory", "amnesia", "cognitive psychology", "identity", "surveillance camera",
        "public health poster", "telephone", "keys", "stairs", "hospital corridor", "portrait photograph",
        "street signage", "warning label", "VHS", "analog television", "computer error", "missing person",
    ]
    return rng.choice(pool)


def wiki_extract(query: str, max_paragraphs: int, timeout_s: int) -> str:
    # Use Wikipedia REST summary then optionally a couple more sentences from extracts API.
    # Keep it short and creepy by selectively clipping.
    q = query.strip()
    if not q:
        return ""
    # 1) Search for a page title
    s_url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": q, "format": "json"}
    r = _http_get(s_url + "?" + requests.compat.urlencode(params), timeout_s)
    data = r.json()
    hits = (data.get("query", {}).get("search") or [])
    if not hits:
        return ""
    title = hits[0].get("title", q)

    # 2) Get extract
    params2 = {
        "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain",
        "titles": title, "format": "json"
    }
    r2 = _http_get(s_url + "?" + requests.compat.urlencode(params2), timeout_s)
    data2 = r2.json()
    pages = (data2.get("query", {}).get("pages") or {})
    page = next(iter(pages.values()), {})
    txt = page.get("extract", "") or ""
    # split to paragraphs
    paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
    paras = paras[:max(1, max_paragraphs)]
    out = "\n\n".join(paras)
    return out


def commons_images(query: str, limit: int, timeout_s: int) -> List[str]:
    """
    Return a list of direct image URLs (jpg/png) from Wikimedia Commons related to the query.
    """
    q = query.strip()
    if not q:
        return []
    api = "https://commons.wikimedia.org/w/api.php"
    # search files
    params = {"action": "query", "list": "search", "srsearch": q, "srnamespace": 6, "format": "json"}
    r = _http_get(api + "?" + requests.compat.urlencode(params), timeout_s)
    data = r.json()
    hits = (data.get("query", {}).get("search") or [])[: max(5, limit)]
    titles = [h.get("title") for h in hits if h.get("title")]
    urls: List[str] = []
    for t in titles[:limit]:
        # imageinfo for url
        params2 = {"action": "query", "titles": t, "prop": "imageinfo", "iiprop": "url", "format": "json"}
        r2 = _http_get(api + "?" + requests.compat.urlencode(params2), timeout_s)
        d2 = r2.json()
        pages = d2.get("query", {}).get("pages", {})
        p = next(iter(pages.values()), {})
        ii = (p.get("imageinfo") or [])
        if not ii:
            continue
        u = ii[0].get("url", "")
        if re.search(r"\.(jpg|jpeg|png)$", u, re.I):
            urls.append(u)
    # dedupe
    seen=set(); out=[]
    for u in urls:
        if u in seen: 
            continue
        seen.add(u); out.append(u)
    return out


def download_image(url: str, timeout_s: int) -> Optional[Image.Image]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout_s)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content))
        im = im.convert("RGB")
        return im
    except Exception:
        return None


# ----------------------------
# "Brain" — story + slide spec
# ----------------------------

@dataclass
class Slide:
    kind: str                 # "intro", "normal", "protocol", "face", "janedoe", "fatal", "outro"
    title: str
    body: str
    bg_imgs: List[Image.Image]  # 0..n
    face_imgs: List[Image.Image]  # 0..n
    seconds: float


def _clamp(x, a, b):
    return max(a, min(b, x))


def redact_text(rng: random.Random, s: str, strength: float) -> str:
    """Redact parts of text with [REDACTED] and ███ bars."""
    if not s:
        return s
    strength = _clamp(strength, 0.0, 3.0)
    prob = 0.12 * strength
    words = s.split()
    for i in range(len(words)):
        if rng.random() < prob and len(words[i]) > 3:
            if rng.random() < 0.55:
                words[i] = "[REDACTED]"
            else:
                words[i] = "█" * rng.randint(4, 10)
    return " ".join(words)


def build_story_slides(
    rng: random.Random,
    theme_key: str,
    scraped_text: str,
    web_images: List[Image.Image],
    local_images: List[Image.Image],
    cfg: Dict[str, Any],
) -> List[Slide]:
    story_cfg = cfg["story"]
    render_cfg = cfg["render"]

    slide_n = max(6, int(story_cfg["slide_count"]))
    normal_ratio = _clamp(float(story_cfg["normal_ratio"]), 0.1, 0.9)
    normal_n = max(2, int(round(slide_n * normal_ratio)))
    scary_n = max(2, slide_n - normal_n)

    # theme-driven "anchors" to sprinkle
    anchors = [
        "keys", "telephone", "name badge", "calendar", "stairwell", "camera", "door handle", "ID card"
    ]
    rng.shuffle(anchors)

    # slice scraped text into short lines
    lines = []
    if scraped_text:
        txt = re.sub(r"\s+", " ", scraped_text).strip()
        # break into sentence-ish fragments
        frags = re.split(r"(?<=[\.\!\?])\s+", txt)
        for f in frags:
            f = f.strip()
            if 20 <= len(f) <= 140:
                lines.append(f)
    if not lines:
        lines = [
            "This module is designed to keep your workday stable.",
            "If you notice something that feels incorrect, you are already involved.",
            "Recognition is the failure mode. Do not complete the pattern.",
            "Touch an anchor object. Leave. Do not describe what you saw.",
        ]

    # helper to pick images
    def pick(pool: List[Image.Image], kmin: int, kmax: int) -> List[Image.Image]:
        if not pool:
            return []
        k = rng.randint(kmin, min(kmax, len(pool)))
        return [pool[rng.randrange(len(pool))] for _ in range(k)]

    # Intro/outro template
    slides: List[Slide] = []
    if story_cfg.get("include_intro_outro", True):
        slides.append(Slide(
            kind="intro",
            title="PLAYBACK / TRAINING ARCHIVE",
            body=f"JOB TRAINING TAPE // {rng.randint(1985, 1996)}\nCHANNEL {rng.randint(1, 12):02d}  SP  TRACKING: OK\nTHEME KEY: {theme_key.upper()}",
            bg_imgs=pick(web_images + local_images, 1, 2),
            face_imgs=[],
            seconds=2.2
        ))

    # Normal slides (fitness/happiness) with mild weird bleed
    normal_titles = [
        "WORKPLACE WELLNESS", "HAPPINESS HABITS", "STRESS MANAGEMENT",
        "PRODUCTIVITY TIP", "POSTURE CHECK", "SLEEP ROUTINE"
    ]
    normal_bullets = [
        "Hydrate every hour.",
        "Breathe in for 4, out for 6.",
        "Short walk after lunch.",
        "Write down 3 good things.",
        "Reduce distractions.",
        "Keep your desk tidy.",
        "Smile (optional).",
    ]

    # Scary protocol slides
    protocol_titles = [
        "MEMORY SAFETY BULLETIN", "RECOGNITION HAZARD", "ANCHOR OBJECTS",
        "INCIDENT RESPONSE", "ENTITY AVOIDANCE PROTOCOL"
    ]
    protocol_lines = [
        "If a face looks unfamiliar: look away.",
        "Do not describe it. Description teaches it.",
        "Touch an anchor object you can name.",
        "Count backward from forty.",
        "Leave immediately. Do not run.",
        "If it speaks in your voice: do not answer.",
    ]

    def mk_normal() -> Slide:
        t = rng.choice(normal_titles)
        bs = rng.sample(normal_bullets, k=rng.randint(3, 5))
        # small wrong note
        if rng.random() < 0.35:
            bs.append(rng.choice([
                "If the slide blinks, blink back once.",
                "If you feel watched, do not react.",
                "Do not read the notes out loud.",
            ]))
        body = "• " + "\n• ".join(bs)
        body = redact_text(rng, body, render_cfg["redaction_strength"]*0.4)
        return Slide(
            kind="normal",
            title=t,
            body=body,
            bg_imgs=pick(web_images + local_images, 1, 3),
            face_imgs=pick(web_images + local_images, 0, 1),
            seconds=float(render_cfg["slide_seconds"])
        )

    def mk_protocol() -> Slide:
        t = rng.choice(protocol_titles)
        ls = rng.sample(protocol_lines, k=rng.randint(3, 5))
        # theme anchors
        if anchors:
            ls.append(f"Anchor object: {anchors.pop(0)}.")
        # cryptic code / easter egg
        if rng.random() < story_cfg.get("easter_egg_probability", 0.25):
            code = hashlib.sha1(f"{theme_key}-{rng.random()}".encode()).hexdigest()[:10].upper()
            ls.append(f"CODE {code} (do not repeat).")
        body = "\n".join([f"{i+1}) {l}" for i, l in enumerate(ls)])
        body = redact_text(rng, body, render_cfg["redaction_strength"]*1.1)
        return Slide(
            kind="protocol",
            title=t,
            body=body,
            bg_imgs=pick(web_images + local_images, 1, 2),
            face_imgs=pick(web_images + local_images, 1, 2),
            seconds=float(render_cfg["slide_seconds"])
        )

    # Jane Doe + Fatal screens
    def mk_janedoe() -> Slide:
        body = "\n".join([
            "SUBJECT: JANE DOE",
            "IDENTITY: [REDACTED]",
            f"LAST STABLE MEMORY: {rng.randint(0,59):02d}:{rng.randint(0,59):02d}",
            f"WITNESS COUNT: {rng.randint(1,4)}",
            f"COMPLIANCE: {'PARTIAL' if rng.random()<0.7 else 'FAILED'}",
            "NEXT INSTRUCTION: DO NOT DESCRIBE.",
        ])
        return Slide(
            kind="janedoe",
            title="JANE DOE INTERMISSION",
            body=body,
            bg_imgs=pick(web_images + local_images, 0, 1),
            face_imgs=pick(web_images + local_images, 1, 3),
            seconds=float(render_cfg["slide_seconds"])
        )

    def mk_fatal() -> Slide:
        stop = f"0x{rng.randint(0, 0xFFFFFF):06X}"
        body = "\n".join([
            "FATAL ERROR: MNEMONIC_LEAK.EXE",
            f"STOP: {stop}",
            "ESC DISABLED",
            "DO NOT RESTART",
            "DO NOT REWIND",
        ])
        return Slide(
            kind="fatal",
            title="TRACKING LOST",
            body=body,
            bg_imgs=[],
            face_imgs=pick(web_images + local_images, 0, 1),
            seconds=float(render_cfg["slide_seconds"])
        )

    # Compose: alternate to keep tension
    normals = [mk_normal() for _ in range(normal_n)]
    protocols = [mk_protocol() for _ in range(scary_n)]
    rng.shuffle(normals)
    rng.shuffle(protocols)

    # Interleave
    while normals or protocols:
        if normals and (not protocols or rng.random() < 0.55):
            slides.append(normals.pop())
        else:
            slides.append(protocols.pop())

        # occasional intermissions
        if story_cfg.get("include_jane_doe", True) and rng.random() < 0.18:
            slides.append(mk_janedoe())
        if story_cfg.get("include_fatal", True) and rng.random() < 0.12:
            slides.append(mk_fatal())

    # Outro
    if story_cfg.get("include_intro_outro", True):
        outro_body = "\n".join([
            "END OF MODULE",
            "Thank you.",
            "Do not replay this tape.",
            "The tape will replay you."
        ])
        slides.append(Slide(
            kind="outro",
            title="END OF TRANSMISSION",
            body=outro_body,
            bg_imgs=pick(web_images + local_images, 1, 2),
            face_imgs=[],
            seconds=2.0
        ))

    # keep at most ~18
    return slides[:18]


# ----------------------------
# Visuals (90s PPT + VHS)
# ----------------------------

@dataclass
class RenderContext:
    W: int
    H: int
    FPS: int
    vhs_strength: float
    redaction_strength: float


def _font_try(names: List[str], size: int) -> ImageFont.FreeTypeFont:
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()


def cover_resize(im: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = im.size
    s = max(w/iw, h/ih)
    nw, nh = int(iw*s), int(ih*s)
    im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
    x0 = (nw - w)//2
    y0 = (nh - h)//2
    return im2.crop((x0, y0, x0+w, y0+h))


def vibrant_bg(rng: random.Random, W: int, H: int) -> np.ndarray:
    # early 90s PPT: intense gradient
    c1 = np.array([rng.randint(40,255), rng.randint(40,255), rng.randint(40,255)], dtype=np.float32)
    c2 = np.array([rng.randint(40,255), rng.randint(40,255), rng.randint(40,255)], dtype=np.float32)
    gx = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    gy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    mix = np.clip(0.65*gx + 0.35*gy, 0, 1)
    arr = c1*(1-mix) + c2*mix
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_ui_layer(ctx: RenderContext, rng: random.Random, slide: Slide) -> np.ndarray:
    W, H = ctx.W, ctx.H
    layer = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(layer)

    fontT = _font_try(["DejaVuSans.ttf", "Arial.ttf"], 28)
    fontB = _font_try(["DejaVuSans.ttf", "Arial.ttf"], 18)
    fontM = _font_try(["DejaVuSansMono.ttf", "Courier New.ttf"], 16)

    # theme color
    if slide.kind in ("fatal",):
        hdr = (255, 50, 60, 235)
    elif slide.kind in ("protocol", "face", "janedoe"):
        hdr = (255, 220, 60, 235)
    else:
        hdr = (60, 210, 255, 235)

    d.rectangle((0,0,W,60), fill=hdr)
    d.text((16, 16), slide.title, fill=(10,10,10,255), font=fontT)

    # content panel
    px0, py0, px1, py1 = 20, 92, int(W*0.76), 92+290
    d.rounded_rectangle((px0, py0, px1, py1), radius=16, fill=(255,255,255,220), outline=(0,0,0,70), width=2)

    # body text (wrap)
    y = py0 + 16
    for raw_line in slide.body.split("\n"):
        line = raw_line.strip()
        if not line:
            y += 10
            continue
        wrapped = textwrap_wrap(line, 44)
        for wline in wrapped:
            d.text((px0+18, y), wline, fill=(10,10,10,255), font=fontB)
            y += 24
        y += 2

    # footer
    d.rectangle((0, H-52, W, H), fill=(0,0,0,120))
    d.text((16, H-40), "VHS TRAINING ARCHIVE  //  DO NOT DUPLICATE", fill=(255,255,255,255), font=fontM)

    # Jane Doe label
    if slide.kind == "janedoe":
        d.rectangle((px0+18, py1-54, px1-18, py1-22), fill=(230,230,230,235))
        d.text((px0+24, py1-50), "FILE: JANE_DOE.TAPE  //  ACCESS: DENIED", fill=(0,0,0,255), font=fontM)

    return np.array(layer, dtype=np.uint8)


def textwrap_wrap(s: str, width: int) -> List[str]:
    # cheap wrap without importing textwrap globally
    words = s.split()
    out = []
    line = []
    n = 0
    for w in words:
        if n + len(w) + (1 if line else 0) > width:
            out.append(" ".join(line))
            line = [w]
            n = len(w)
        else:
            line.append(w)
            n += len(w) + (1 if line else 0)
    if line:
        out.append(" ".join(line))
    return out


def alpha_over(bg: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    a = fg_rgba[:,:,3:4].astype(np.float32)/255.0
    return np.clip(bg.astype(np.float32)*(1-a) + fg_rgba[:,:,:3].astype(np.float32)*a, 0, 255).astype(np.uint8)


def make_popup(ctx: RenderContext, rng: random.Random, im: Image.Image) -> np.ndarray:
    W, H = ctx.W, ctx.H
    arr = np.array(cover_resize(im, 220, 220).convert("RGB"), dtype=np.uint8)
    # "wrong" treatment
    if rng.random() < 0.55:
        pil = Image.fromarray(arr).convert("L")
        pil = ImageOps.autocontrast(ImageOps.posterize(pil, 3))
        pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=220, threshold=2))
        arr = np.array(pil.convert("RGB").resize((180,180), Image.Resampling.NEAREST), dtype=np.uint8)
    else:
        pil = Image.fromarray(arr)
        pil = ImageOps.autocontrast(pil)
        pil = pil.filter(ImageFilter.GaussianBlur(radius=0.9))
        arr = np.array(pil.resize((180,180), Image.Resampling.NEAREST), dtype=np.uint8)
    return arr


def stamp_popup(ctx: RenderContext, rng: random.Random, frame: np.ndarray, popup: np.ndarray) -> np.ndarray:
    out = frame.copy()
    ph, pw = popup.shape[:2]
    x = rng.randint(0, ctx.W - pw)
    y = rng.randint(70, ctx.H - ph - 70)
    out[y:y+ph, x:x+pw] = popup
    # thick border like a window
    out[y:y+3, x:x+pw] = 0
    out[y+ph-3:y+ph, x:x+pw] = 0
    out[y:y+ph, x:x+3] = 0
    out[y:y+ph, x+pw-3:x+pw] = 0
    return out


def redact_image(ctx: RenderContext, rng: random.Random, frame: np.ndarray) -> np.ndarray:
    """Add censorship bars + mosaic blur squares."""
    out = frame.copy()
    strength = _clamp(ctx.redaction_strength, 0.0, 3.0)
    bars = rng.randint(1, 2 + int(2*strength))
    for _ in range(bars):
        w = rng.randint(int(ctx.W*0.22), int(ctx.W*0.65))
        h = rng.randint(14, 30)
        x = rng.randint(18, ctx.W - w - 18)
        y = rng.randint(72, ctx.H - 90)
        out[y:y+h, x:x+w] = 0
    # mosaic blocks
    if rng.random() < 0.45*strength:
        x = rng.randint(0, ctx.W-160); y = rng.randint(60, ctx.H-180)
        block = out[y:y+160, x:x+160]
        pil = Image.fromarray(block).resize((40,40), Image.Resampling.BILINEAR).resize((160,160), Image.Resampling.NEAREST)
        out[y:y+160, x:x+160] = np.array(pil, dtype=np.uint8)
    return out


def vhs_stack(ctx: RenderContext, rng: random.Random, frame: np.ndarray) -> np.ndarray:
    """Apply stronger VHS look."""
    W, H = ctx.W, ctx.H
    s = _clamp(ctx.vhs_strength, 0.0, 3.0)

    out = frame.copy()

    # chroma bleed
    amt = 2 + int(rng.random()*4*s)
    out[:,:,0] = np.roll(out[:,:,0], -amt, axis=1)
    out[:,:,2] = np.roll(out[:,:,2],  amt, axis=1)

    # scanlines
    if s > 0:
        out = (out.astype(np.float32) * (0.86 + 0.14*np.sin(np.arange(H)[:,None]*math.pi))).astype(np.uint8)

    # tracking line
    if rng.random() < 0.35*s:
        y = rng.randint(int(H*0.55), H-12)
        hh = rng.randint(4, 10)
        out[y:y+hh] = np.clip(out[y:y+hh].astype(np.int16) + rng.randint(40, 90), 0, 255).astype(np.uint8)
        out[y:y+hh] = np.roll(out[y:y+hh], rng.randint(-80, 80), axis=1)

    # horizontal slice glitch
    if rng.random() < 0.22*s:
        bands = 2 + int(2*s)
        for _ in range(bands):
            y = rng.randint(0, H-18)
            hh = rng.randint(6, 24)
            shift = rng.randint(-70, 70)
            out[y:y+hh] = np.roll(out[y:y+hh], shift, axis=1)

    # noise
    level = int(10 + 10*s)
    n = rng.randint(-level, level+1, size=out.shape, dtype=np.int16)
    out = np.clip(out.astype(np.int16) + n, 0, 255).astype(np.uint8)

    # vignette
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W/2, H/2
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2) / math.sqrt(cx**2 + cy**2)
    v = np.clip(1 - (0.65*s)*(r**1.7), 0.28, 1.0).astype(np.float32)[...,None]
    out = (out.astype(np.float32) * v).astype(np.uint8)

    # slight blur for tape softness
    if rng.random() < 0.25*s:
        pil = Image.fromarray(out).filter(ImageFilter.GaussianBlur(radius=0.6))
        out = np.array(pil, dtype=np.uint8)

    return out


def timecode_overlay(ctx: RenderContext, frame: np.ndarray, frame_idx: int) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    font = _font_try(["DejaVuSansMono.ttf", "Courier New.ttf"], 16)
    secs = frame_idx / ctx.FPS
    hh = int(secs//3600); mm = int((secs%3600)//60); ss = int(secs%60); ff = int((secs - int(secs))*ctx.FPS)
    tc = f"TC {hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}   CH{random.randint(1,12):02d}  SP"
    d.rectangle((10, ctx.H-36, 310, ctx.H-12), fill=(0,0,0))
    d.text((16, ctx.H-34), tc, fill=(255,255,255), font=font)
    return np.array(im, dtype=np.uint8)


def face_uncanny(ctx: RenderContext, rng: random.Random, im: Image.Image, t: float) -> np.ndarray:
    """Fullscreen face with lightweight uncanny warp."""
    base = np.array(cover_resize(im, ctx.W, ctx.H).convert("RGB"), dtype=np.uint8)
    # contrast + posterize
    pil = ImageOps.autocontrast(Image.fromarray(base))
    pil = ImageOps.posterize(pil, 4)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
    arr = np.array(pil, dtype=np.uint8)
    # warp: roll strips + eye band shift
    out = arr.copy()
    for _ in range(4):
        x0 = rng.randint(0, ctx.W-140)
        ww = rng.randint(60, 170)
        shift = int(10*math.sin(t*2.0 + x0*0.03))
        out[:, x0:x0+ww] = np.roll(out[:, x0:x0+ww], shift, axis=0)
    y0 = int(ctx.H*0.30) + rng.randint(-10,10)
    out[y0:y0+36] = np.roll(out[y0:y0+36], rng.randint(-22,22), axis=1)
    # vibrance skew
    scale = np.array([1.12, 0.92, 1.18], dtype=np.float32)
    out = np.clip(out.astype(np.float32)*scale, 0, 255).astype(np.uint8)
    return out


# ----------------------------
# Audio (music bed + pops + TTS)
# ----------------------------

def bitcrush(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    y = x[::factor]
    y = np.repeat(y, factor)[: len(x)]
    return y


def gen_music_and_sfx(rng: random.Random, sr: int, dur_s: float) -> np.ndarray:
    n = int(sr * dur_s)
    t = np.linspace(0, dur_s, n, False).astype(np.float32)

    # VHS hiss + mains hum + low drone
    audio = (rng.uniform(-1, 1) * 0.0)  # placeholder to keep type?
    audio = (np.random.uniform(-1,1,n).astype(np.float32) * 0.05)
    audio += 0.03*np.sin(2*np.pi*55*t) + 0.018*np.sin(2*np.pi*110*t)
    audio += 0.02*np.sin(2*np.pi*30*t)

    # slow wobble
    audio *= (0.82 + 0.18*np.sin(2*np.pi*0.18*t)).astype(np.float32)

    # abrupt bursts
    for _ in range(18):
        p0 = rng.randint(0, n-1)
        span = rng.randint(int(0.01*sr), int(0.06*sr))
        p1 = min(n, p0+span)
        burst = (np.random.uniform(-1,1,p1-p0).astype(np.float32) * rng.uniform(0.22, 0.55))
        audio[p0:p1] += burst

    # beeps
    for sec in [dur_s*0.25, dur_s*0.55, dur_s*0.8]:
        p0 = int(sec*sr)
        p1 = min(n, p0+int(0.16*sr))
        tt = np.linspace(0, (p1-p0)/sr, p1-p0, False).astype(np.float32)
        audio[p0:p1] += 0.18*np.sin(2*np.pi*880*tt).astype(np.float32)

    return audio


def tts_segment_espeak(
    text: str,
    out_wav: Path,
    voice: str,
    speed: int,
    pitch: int,
    amp: int,
) -> None:
    cmd = ["espeak", "-v", voice, "-s", str(speed), "-p", str(pitch), "-a", str(amp), "-w", str(out_wav), text]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tts_available() -> bool:
    return shutil_which("espeak") is not None


def ffmpeg_available() -> bool:
    return shutil_which("ffmpeg") is not None


def shutil_which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


def mix_in(dst: np.ndarray, src: np.ndarray, start_s: float, sr: int, gain: float) -> None:
    start = int(start_s * sr)
    end = min(len(dst), start + len(src))
    if end > start:
        dst[start:end] += src[: end-start] * gain


def build_tts_track(
    rng: random.Random,
    slides: List[Slide],
    cfg: Dict[str, Any],
    dur_s: float,
) -> np.ndarray:
    sr = int(cfg["audio"]["sr"])
    tcfg = cfg["audio"]
    speed = int(tcfg["tts_speed"])
    pitch = int(tcfg["tts_pitch"])
    amp = int(tcfg["tts_amp"])
    voices: List[str] = list(tcfg.get("voices") or [])

    track = np.zeros(int(sr * dur_s), dtype=np.float32)
    work = Path(cfg["workdir"]) / "tts"
    work.mkdir(parents=True, exist_ok=True)

    # Narration: corporate-normal voice -> slightly wrong, with occasional crunchy voice.
    t = 0.0
    for i, s in enumerate(slides):
        base = narration_for_slide(rng, s)
        voice = rng.choice(voices) if voices else "en-us"
        # variation per slide for uncanny
        speed_i = _clamp(speed + rng.randint(-12, 10), 115, 190)
        pitch_i = _clamp(pitch + rng.randint(-12, 14), 10, 80)
        amp_i = _clamp(amp + rng.randint(-10, 10), 80, 200)

        wav_path = work / f"tts_{i:02d}.wav"
        try:
            tts_segment_espeak(base, wav_path, voice=voice, speed=int(speed_i), pitch=int(pitch_i), amp=int(amp_i))
            sr2, data = read_wav(str(wav_path))
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            if sr2 != sr:
                # resample linear
                x = np.linspace(0, 1, len(data), False)
                x2 = np.linspace(0, 1, int(len(data)*sr/sr2), False)
                data = np.interp(x2, x, data).astype(np.float32)
            data /= (np.max(np.abs(data)) + 1e-6)

            # FX: mild saturation + occasional bitcrush
            data = np.tanh(data * 1.6) * 0.70
            if rng.random() < 0.55:
                data = bitcrush(data, rng.choice([4,5,6]))
            ring = np.sin(2*np.pi*220*np.linspace(0, len(data)/sr, len(data), False).astype(np.float32))
            data = (0.86*data + 0.14*data*ring).astype(np.float32)
            data += (np.random.uniform(-1,1,len(data)).astype(np.float32) * 0.006)

            mix_in(track, data, start_s=t + 0.25, sr=sr, gain=1.0)
        except Exception:
            # if espeak fails, keep going silently
            pass

        t += s.seconds

    return track


def narration_for_slide(rng: random.Random, s: Slide) -> str:
    # Keep intelligible but eerie. Emphasize "do not describe / do not repeat" motifs.
    if s.kind == "intro":
        return "Playback. Job training tape. Tracking is stable. Please do not duplicate."
    if s.kind == "outro":
        return "End of transmission. Thank you. Do not replay this tape."
    if s.kind == "fatal":
        return "Tracking lost. Fatal error. Do not restart. Do not rewind."
    if s.kind == "janedoe":
        return "Jane Doe intermission. Identity redacted. Do not attempt recognition."
    if s.kind == "normal":
        tail = rng.choice([
            "Please continue normally.",
            "If you feel watched, do not react.",
            "Do not read the notes aloud.",
        ])
        return f"{s.title}. {re.sub(r'[^A-Za-z0-9 ,\.\-\']+', ' ', s.body)}. {tail}"
    # protocol
    tail = rng.choice([
        "Description teaches it.",
        "Silence is a valid response.",
        "If you are watching, you are participating.",
    ])
    return f"{s.title}. {re.sub(r'[^A-Za-z0-9 ,\.\-\']+', ' ', s.body)}. {tail}"


# ----------------------------
# Transmission failure moments
# ----------------------------

@dataclass
class TransmissionPlan:
    enable: bool
    cut_frame: Optional[int]
    freeze_at: Optional[int]
    freeze_frames: int


def plan_transmission(rng: random.Random, cfg: Dict[str, Any], total_frames: int, fps: int) -> TransmissionPlan:
    tcfg = cfg["transmission"]
    if not tcfg.get("enable", True):
        return TransmissionPlan(False, None, None, 0)

    if rng.random() > float(tcfg["error_probability"]):
        return TransmissionPlan(True, None, None, 0)

    # pick an error region past 20%
    at = rng.randint(int(total_frames*0.20), max(int(total_frames*0.85), int(total_frames*0.20)+1))
    freeze_at = None
    freeze_frames = 0
    cut_frame = None

    if rng.random() < float(tcfg["freeze_probability"]):
        fs = rng.uniform(float(tcfg["freeze_seconds_min"]), float(tcfg["freeze_seconds_max"]))
        freeze_frames = max(1, int(fs * fps))
        freeze_at = at

    if rng.random() < float(tcfg["early_end_probability"]):
        # end shortly after the error
        cut_frame = min(total_frames-1, at + rng.randint(int(0.2*fps), int(1.2*fps)))

    return TransmissionPlan(True, cut_frame, freeze_at, freeze_frames)


# ----------------------------
# Pipeline: gather images
# ----------------------------

def load_local_images(dirpath: Path, limit: int = 24) -> List[Image.Image]:
    if not dirpath.exists():
        return []
    imgs = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(dirpath.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        try:
            im = Image.open(p).convert("RGB")
            imgs.append(im)
        except Exception:
            continue
        if len(imgs) >= limit:
            break
    return imgs


def split_faces_and_objects(imgs: List[Image.Image]) -> Tuple[List[Image.Image], List[Image.Image]]:
    # crude heuristic: treat taller images as "faces/portraits" more often
    faces, objs = [], []
    for im in imgs:
        w, h = im.size
        if h >= w and h > 220:
            faces.append(im)
        else:
            objs.append(im)
    if not faces:
        # fallback: allow any
        faces = imgs[:]
    return faces, objs


# ----------------------------
# Render + mux
# ----------------------------

def render_video(
    rng: random.Random,
    cfg: Dict[str, Any],
    slides: List[Slide],
    out_mp4: Path,
) -> Tuple[Path, float, int]:
    render_cfg = cfg["render"]
    W = int(render_cfg["width"])
    H = int(render_cfg["height"])
    FPS = int(render_cfg["fps"])

    ctx = RenderContext(W=W, H=H, FPS=FPS, vhs_strength=float(render_cfg["vhs_strength"]),
                        redaction_strength=float(render_cfg["redaction_strength"]))

    # schedule popups + flashes
    max_popups = int(render_cfg["max_popups"])
    popup_frames_total = max(1, int(float(render_cfg["popup_seconds"]) * FPS))
    flashes = int(render_cfg["flashes"])

    total_frames = sum(int(s.seconds * FPS) for s in slides)
    plan = plan_transmission(rng, cfg, total_frames, FPS)

    # precompute popup pool from all slide images
    all_imgs = []
    for s in slides:
        all_imgs.extend(s.bg_imgs)
        all_imgs.extend(s.face_imgs)
    all_imgs = [im for im in all_imgs if im is not None]
    pop_pool = [make_popup(ctx, rng, im) for im in rng.sample(all_imgs, k=min(len(all_imgs), 8))] if all_imgs else []

    # choose popup moments (max N, each ~popup_frames_total)
    popup_moments: List[int] = []
    if max_popups > 0 and total_frames > 10:
        for _ in range(max_popups):
            popup_moments.append(rng.randint(int(total_frames*0.15), total_frames-1))
        popup_moments = sorted(set(popup_moments))[:max_popups]

    flash_frames = set()
    for _ in range(max(0, flashes)):
        flash_frames.add(rng.randint(int(total_frames*0.10), max(int(total_frames*0.95), 1)))

    out_silent = out_mp4.with_name(out_mp4.stem + "_silent.mp4")
    writer = imageio.get_writer(str(out_silent), fps=FPS, codec="libx264", bitrate="2600k")

    frame_idx = 0
    popup_active_until = -1

    for si, slide in enumerate(slides):
        nF = max(1, int(slide.seconds * FPS))
        ui = make_ui_layer(ctx, rng, slide)

        # background base: combine gradient + images (uncorrelated montage)
        bg = vibrant_bg(rng, W, H)
        # stamp 1-2 random objects (from bg_imgs) to create "wrong collage"
        if slide.bg_imgs:
            for im in rng.sample(slide.bg_imgs, k=min(len(slide.bg_imgs), rng.randint(1, 2))):
                obj = np.array(cover_resize(im, rng.randint(220, 360), rng.randint(180, 320)).convert("RGB"))
                x = rng.randint(0, W-obj.shape[1])
                y = rng.randint(70, H-obj.shape[0]-70)
                bg[y:y+obj.shape[0], x:x+obj.shape[1]] = obj

        for fi in range(nF):
            if plan.cut_frame is not None and frame_idx >= plan.cut_frame:
                break

            t = frame_idx / FPS
            frame = bg.copy()

            # subtle drift
            if fi % 4 == 0:
                frame = np.roll(frame, rng.randint(-2, 2), axis=1)

            # face overlay for uncanny
            if slide.face_imgs and (slide.kind in ("protocol", "janedoe") or rng.random() < 0.25):
                face = rng.choice(slide.face_imgs)
                face_arr = face_uncanny(ctx, rng, face, t)
                alpha = 0.55 if slide.kind == "normal" else 0.75
                frame = np.clip(frame.astype(np.float32)*(1-alpha) + face_arr.astype(np.float32)*alpha, 0, 255).astype(np.uint8)

            # ui overlay
            frame = alpha_over(frame, ui)

            # redactions on scary slides
            if slide.kind in ("protocol", "janedoe", "fatal") and rng.random() < 0.35:
                frame = redact_image(ctx, rng, frame)

            # popup logic: activate when reaching popup moment, keep for popup_frames_total
            if frame_idx in popup_moments:
                popup_active_until = frame_idx + popup_frames_total
            if frame_idx < popup_active_until and pop_pool:
                pop = rng.choice(pop_pool)
                if rng.random() < 0.30:
                    pop = 255 - pop
                frame = stamp_popup(ctx, rng, frame, pop)

            # one-frame flash (distorted)
            if frame_idx in flash_frames and all_imgs:
                im = rng.choice(all_imgs)
                flash = np.array(cover_resize(im, W, H).convert("RGB"), dtype=np.uint8)
                flash = vhs_stack(ctx, rng, flash)
                # invert flash sometimes
                if rng.random() < 0.55:
                    flash = 255 - flash
                frame = flash

            # transmission freeze: hold last frame
            if plan.freeze_at is not None and frame_idx == plan.freeze_at:
                freeze = frame.copy()
                for _ in range(plan.freeze_frames):
                    fr = vhs_stack(ctx, rng, freeze.copy())
                    fr = timecode_overlay(ctx, fr, frame_idx)
                    writer.append_data(fr)
                    frame_idx += 1
                # continue after freeze
                continue

            # VHS look
            frame = vhs_stack(ctx, rng, frame)

            # edge transitions
            if fi < 3 or fi > nF-4:
                if rng.random() < 0.65:
                    frame = 255 - frame

            # timecode overlay
            frame = timecode_overlay(ctx, frame, frame_idx)

            writer.append_data(frame)
            frame_idx += 1

        if plan.cut_frame is not None and frame_idx >= plan.cut_frame:
            break

    writer.close()
    dur_s = frame_idx / FPS
    return out_silent, dur_s, frame_idx


def mux_av(video_path: Path, audio_path: Path, out_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml", help="YAML config file")
    ap.add_argument("--out", type=str, default="out.mp4", help="Output mp4 path")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    workdir = Path(cfg["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg["seed"])

    # brain chooses theme key if blank
    theme_key = cfg.get("theme_key", "").strip() or choose_theme_key(rng)

    # gather local images
    local_dir = Path(cfg["local_images_dir"])
    local_imgs = load_local_images(local_dir, limit=36)

    web_imgs: List[Image.Image] = []
    scraped_text = ""
    if cfg["web"]["enable"]:
        try:
            scraped_text = wiki_extract(theme_key, max_paragraphs=int(cfg["web"]["text_paragraphs"]), timeout_s=int(cfg["web"]["timeout_s"]))
        except Exception:
            scraped_text = ""
        try:
            urls = commons_images(theme_key, limit=int(cfg["web"]["image_limit"]), timeout_s=int(cfg["web"]["timeout_s"]))
            for u in urls:
                im = download_image(u, timeout_s=int(cfg["web"]["timeout_s"]))
                if im:
                    web_imgs.append(im)
        except Exception:
            web_imgs = []

    # keep manageable
    rng.shuffle(web_imgs)
    web_imgs = web_imgs[: max(6, min(18, len(web_imgs)))]

    # Split images into face-like and object-like pools to help uncanny composition
    face_pool, obj_pool = split_faces_and_objects(web_imgs + local_imgs)

    slides = build_story_slides(rng, theme_key, scraped_text, obj_pool, local_imgs, cfg)

    # post-process slides to ensure face availability on scary slides
    for s in slides:
        if s.kind in ("protocol", "janedoe") and not s.face_imgs:
            s.face_imgs = [rng.choice(face_pool)] if face_pool else []

    out_mp4 = Path(args.out)
    out_silent, dur_s, frames = render_video(rng, cfg, slides, out_mp4)

    # audio
    sr = int(cfg["audio"]["sr"])
    audio = np.zeros(int(sr * dur_s), dtype=np.float32)
    if cfg["audio"]["music"]:
        audio += gen_music_and_sfx(rng, sr=sr, dur_s=dur_s)

    if cfg["audio"]["tts"] and tts_available():
        audio += build_tts_track(rng, slides, cfg, dur_s=dur_s)

    # normalize
    if np.max(np.abs(audio)) > 1e-6:
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
    audio_i16 = (audio * 32767).astype(np.int16)
    audio_path = out_mp4.with_suffix(".wav")
    write_wav(str(audio_path), sr, audio_i16)

    if not ffmpeg_available():
        print("ffmpeg not found; leaving silent video + wav.", file=sys.stderr)
        out_silent.rename(out_mp4)
        return

    mux_av(out_silent, audio_path, out_mp4)

    print(f"Done: {out_mp4} (duration ~{dur_s:.2f}s, seed={cfg['seed']}, theme='{theme_key}')")


if __name__ == "__main__":
    main()
