#!/usr/bin/env python3
"""
generate.py — "Found VHS Job Training" analogue horror generator.

What it does:
- Builds an always-different horror video that blends "normal corporate wellness" slides with unsettling
  safety bulletins, "Jane Doe" intermissions, redactions, and VHS tracking errors.
- Scrapes theme text + images from the web (Wikipedia + Wikimedia Commons) when enabled.
- Mixes in local images from your repo (assets/images) into the slide collage.
- Adds pop-up distorted images, VHS artifacts (tracking, noise, bleed), and audio glitches.
- Generates a noisy synth soundtrack + TTS narration.

Usage:
  python generate.py --config config.yaml --out out.mp4

Dependencies:
  pip install requests pyyaml numpy scipy pillow imageio
  System requirements: ffmpeg, espeak (for TTS)
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
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import yaml
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav

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
        "error_probability": 0.35,
        "freeze_probability": 0.6,
        "early_end_probability": 0.45,
        "freeze_seconds_min": 0.8,
        "freeze_seconds_max": 2.2,
    },
    "story": {
        "slide_count": 10,
        "normal_ratio": 0.45,
        "include_intro_outro": True,
        "include_jane_doe": True,
        "include_fatal": True,
        "easter_egg_probability": 0.25,
    },
}

UA = "pptex-vhs-generator/1.1 (+github-actions; educational/art project)"


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_env_value(v: Any) -> Any:
    """Resolve ${ENV} placeholders."""
    if isinstance(v, str):
        m = re.match(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$", v.strip())
        if m:
            return os.environ.get(m.group(1), "")
    return v


def coerce_type(v: Any, default: Any, target_type: type) -> Any:
    v = resolve_env_value(v)
    if v is None:
        return default
    try:
        if target_type is bool:
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes", "y", "on")
            return bool(v)
        return target_type(v)
    except (ValueError, TypeError):
        return default


def load_config(path: Path) -> Dict[str, Any]:
    raw = {}
    if path and path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"Warning: Could not parse config file {path}: {e}")
            
    cfg = deep_merge(DEFAULTS, raw)

    # Normalize subsections
    for section in ["web", "render", "audio", "transmission", "story"]:
        if not isinstance(cfg[section], dict):
            cfg[section] = DEFAULTS[section].copy()

    # Coerce values
    cfg["seed"] = coerce_type(cfg.get("seed"), 1337, int)
    
    # Web
    cfg["web"]["enable"] = coerce_type(cfg["web"].get("enable"), True, bool)
    cfg["web"]["text_paragraphs"] = coerce_type(cfg["web"].get("text_paragraphs"), 10, int)
    cfg["web"]["image_limit"] = coerce_type(cfg["web"].get("image_limit"), 12, int)
    cfg["web"]["timeout_s"] = coerce_type(cfg["web"].get("timeout_s"), 15, int)

    # Render
    cfg["render"]["width"] = coerce_type(cfg["render"].get("width"), 640, int)
    cfg["render"]["height"] = coerce_type(cfg["render"].get("height"), 480, int)
    cfg["render"]["fps"] = coerce_type(cfg["render"].get("fps"), 15, int)
    cfg["render"]["slide_seconds"] = coerce_type(cfg["render"].get("slide_seconds"), 3.2, float)
    cfg["render"]["max_popups"] = coerce_type(cfg["render"].get("max_popups"), 3, int)
    cfg["render"]["popup_seconds"] = coerce_type(cfg["render"].get("popup_seconds"), 0.5, float)
    cfg["render"]["vhs_strength"] = coerce_type(cfg["render"].get("vhs_strength"), 1.0, float)
    cfg["render"]["redaction_strength"] = coerce_type(cfg["render"].get("redaction_strength"), 1.0, float)
    cfg["render"]["flashes"] = coerce_type(cfg["render"].get("flashes"), 12, int)

    # Audio
    cfg["audio"]["sr"] = coerce_type(cfg["audio"].get("sr"), 44100, int)
    cfg["audio"]["music"] = coerce_type(cfg["audio"].get("music"), True, bool)
    cfg["audio"]["tts"] = coerce_type(cfg["audio"].get("tts"), True, bool)
    cfg["audio"]["tts_speed"] = coerce_type(cfg["audio"].get("tts_speed"), 155, int)
    cfg["audio"]["tts_pitch"] = coerce_type(cfg["audio"].get("tts_pitch"), 32, int)
    cfg["audio"]["tts_amp"] = coerce_type(cfg["audio"].get("tts_amp"), 170, int)
    if not isinstance(cfg["audio"].get("voices"), list):
        cfg["audio"]["voices"] = DEFAULTS["audio"]["voices"]

    # Transmission
    cfg["transmission"]["enable"] = coerce_type(cfg["transmission"].get("enable"), True, bool)
    cfg["transmission"]["error_probability"] = coerce_type(cfg["transmission"].get("error_probability"), 0.35, float)
    cfg["transmission"]["freeze_probability"] = coerce_type(cfg["transmission"].get("freeze_probability"), 0.6, float)
    cfg["transmission"]["early_end_probability"] = coerce_type(cfg["transmission"].get("early_end_probability"), 0.45, float)
    cfg["transmission"]["freeze_seconds_min"] = coerce_type(cfg["transmission"].get("freeze_seconds_min"), 0.8, float)
    cfg["transmission"]["freeze_seconds_max"] = coerce_type(cfg["transmission"].get("freeze_seconds_max"), 2.2, float)

    # Story
    cfg["story"]["slide_count"] = coerce_type(cfg["story"].get("slide_count"), 10, int)
    cfg["story"]["normal_ratio"] = coerce_type(cfg["story"].get("normal_ratio"), 0.45, float)
    cfg["story"]["include_intro_outro"] = coerce_type(cfg["story"].get("include_intro_outro"), True, bool)
    cfg["story"]["include_jane_doe"] = coerce_type(cfg["story"].get("include_jane_doe"), True, bool)
    cfg["story"]["include_fatal"] = coerce_type(cfg["story"].get("include_fatal"), True, bool)
    cfg["story"]["easter_egg_probability"] = coerce_type(cfg["story"].get("easter_egg_probability"), 0.25, float)

    cfg["theme_key"] = str(resolve_env_value(cfg.get("theme_key", "")) or "").strip()
    cfg["workdir"] = str(resolve_env_value(cfg.get("workdir", ".work")) or ".work")
    cfg["local_images_dir"] = str(resolve_env_value(cfg.get("local_images_dir", "assets/images")) or "assets/images")
    
    return cfg


# ----------------------------
# Web scraping
# ----------------------------

def _http_get(url: str, timeout: int) -> requests.Response:
    return requests.get(url, headers={"User-Agent": UA}, timeout=timeout)


def choose_theme_key(rng: random.Random) -> str:
    pool = [
        "workplace safety", "industrial training", "office equipment", "employee handbook",
        "memory", "amnesia", "cognitive psychology", "identity", "surveillance camera",
        "public health poster", "telephone", "keys", "stairs", "hospital corridor", "portrait photograph",
        "street signage", "warning label", "VHS", "analog television", "computer error", "missing person",
    ]
    return rng.choice(pool)


def wiki_extract(query: str, max_paragraphs: int, timeout_s: int) -> str:
    q = query.strip()
    if not q:
        return ""
    try:
        # 1) Search
        s_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": q, "format": "json"}
        r = _http_get(s_url + "?" + requests.compat.urlencode(params), timeout_s)
        data = r.json()
        hits = (data.get("query", {}).get("search") or [])
        if not hits:
            return ""
        title = hits[0].get("title", q)

        # 2) Extract
        params2 = {
            "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain",
            "titles": title, "format": "json"
        }
        r2 = _http_get(s_url + "?" + requests.compat.urlencode(params2), timeout_s)
        data2 = r2.json()
        pages = (data2.get("query", {}).get("pages") or {})
        page = next(iter(pages.values()), {})
        txt = page.get("extract", "") or ""
        
        # Split and limit
        paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        return "\n\n".join(paras[:max(1, max_paragraphs)])
    except Exception:
        return ""


def commons_images(query: str, limit: int, timeout_s: int) -> List[str]:
    q = query.strip()
    if not q:
        return []
    api = "https://commons.wikimedia.org/w/api.php"
    try:
        params = {"action": "query", "list": "search", "srsearch": q, "srnamespace": 6, "format": "json"}
        r = _http_get(api + "?" + requests.compat.urlencode(params), timeout_s)
        data = r.json()
        hits = (data.get("query", {}).get("search") or [])[: max(5, limit * 2)]
        
        titles = [h.get("title") for h in hits if h.get("title")]
        urls: List[str] = []
        
        for t in titles:
            if len(urls) >= limit: 
                break
            params2 = {"action": "query", "titles": t, "prop": "imageinfo", "iiprop": "url", "format": "json"}
            r2 = _http_get(api + "?" + requests.compat.urlencode(params2), timeout_s)
            d2 = r2.json()
            pages = d2.get("query", {}).get("pages", {})
            p = next(iter(pages.values()), {})
            ii = (p.get("imageinfo") or [])
            if ii:
                u = ii[0].get("url", "")
                if re.search(r"\.(jpg|jpeg|png)$", u, re.I):
                    if u not in urls:
                        urls.append(u)
        return urls
    except Exception:
        return []


def download_image(url: str, timeout_s: int) -> Optional[Image.Image]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout_s)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content))
        im.load() # Force load
        return im.convert("RGB")
    except Exception:
        return None


# ----------------------------
# "Brain" — story + slide spec
# ----------------------------

@dataclass
class Slide:
    kind: str  # "intro", "normal", "protocol", "face", "janedoe", "fatal", "outro"
    title: str
    body: str
    bg_imgs: List[Image.Image]
    face_imgs: List[Image.Image]
    seconds: float


def _clamp(x, a, b):
    return max(a, min(b, x))


def redact_text(rng: random.Random, s: str, strength: float) -> str:
    """Redact parts of text with [REDACTED] and ███ bars."""
    if not s: return s
    strength = _clamp(strength, 0.0, 3.0)
    prob = 0.12 * strength
    words = s.split()
    for i in range(len(words)):
        if rng.random() < prob and len(words[i]) > 3:
            words[i] = "[REDACTED]" if rng.random() < 0.55 else "█" * rng.randint(4, 10)
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

    # Combine sources for random picking
    all_images = web_images + local_images

    slide_n = max(6, int(story_cfg["slide_count"]))
    normal_ratio = _clamp(float(story_cfg["normal_ratio"]), 0.1, 0.9)
    normal_n = max(2, int(round(slide_n * normal_ratio)))
    scary_n = max(2, slide_n - normal_n)

    anchors = [
        "keys", "telephone", "name badge", "calendar", "stairwell", "camera", "door handle", "ID card"
    ]
    rng.shuffle(anchors)

    # Process scraped text
    lines = []
    if scraped_text:
        txt = re.sub(r"\s+", " ", scraped_text).strip()
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

    def pick(pool: List[Image.Image], kmin: int, kmax: int) -> List[Image.Image]:
        if not pool: return []
        k = rng.randint(kmin, min(kmax, len(pool)))
        return [rng.choice(pool) for _ in range(k)]

    slides: List[Slide] = []

    # Intro
    if story_cfg.get("include_intro_outro", True):
        slides.append(Slide(
            kind="intro",
            title="PLAYBACK / TRAINING ARCHIVE",
            body=f"JOB TRAINING TAPE // {rng.randint(1985, 1996)}\nCHANNEL {rng.randint(1, 12):02d}  SP  TRACKING: OK\nTHEME KEY: {theme_key.upper()}",
            bg_imgs=pick(all_images, 1, 2),
            face_imgs=[],
            seconds=2.2
        ))

    normal_titles = [
        "WORKPLACE WELLNESS", "HAPPINESS HABITS", "STRESS MANAGEMENT",
        "PRODUCTIVITY TIP", "POSTURE CHECK", "SLEEP ROUTINE"
    ]
    normal_bullets = [
        "Hydrate every hour.", "Breathe in for 4, out for 6.", "Short walk after lunch.",
        "Write down 3 good things.", "Reduce distractions.", "Keep your desk tidy.", "Smile (optional)."
    ]

    protocol_titles = [
        "MEMORY SAFETY BULLETIN", "RECOGNITION HAZARD", "ANCHOR OBJECTS",
        "INCIDENT RESPONSE", "ENTITY AVOIDANCE PROTOCOL"
    ]
    protocol_lines = [
        "If a face looks unfamiliar: look away.", "Do not describe it. Description teaches it.",
        "Touch an anchor object you can name.", "Count backward from forty.",
        "Leave immediately. Do not run.", "If it speaks in your voice: do not answer."
    ]

    def mk_normal() -> Slide:
        t = rng.choice(normal_titles)
        bs = rng.sample(normal_bullets, k=rng.randint(3, 5))
        if rng.random() < 0.35:
            bs.append(rng.choice([
                "If the slide blinks, blink back once.", "If you feel watched, do not react.",
                "Do not read the notes out loud."
            ]))
        body = "• " + "\n• ".join(bs)
        body = redact_text(rng, body, render_cfg["redaction_strength"] * 0.4)
        return Slide("normal", t, body, pick(all_images, 1, 3), pick(all_images, 0, 1), render_cfg["slide_seconds"])

    def mk_protocol() -> Slide:
        t = rng.choice(protocol_titles)
        ls = rng.sample(protocol_lines, k=rng.randint(3, 5))
        if anchors:
            ls.append(f"Anchor object: {anchors.pop(0)}.")
        if rng.random() < story_cfg.get("easter_egg_probability", 0.25):
            code = hashlib.sha1(f"{theme_key}-{rng.random()}".encode()).hexdigest()[:10].upper()
            ls.append(f"CODE {code} (do not repeat).")
        body = "\n".join([f"{i+1}) {l}" for i, l in enumerate(ls)])
        body = redact_text(rng, body, render_cfg["redaction_strength"] * 1.1)
        return Slide("protocol", t, body, pick(all_images, 1, 2), pick(all_images, 1, 2), render_cfg["slide_seconds"])

    def mk_janedoe() -> Slide:
        body = "\n".join([
            "SUBJECT: JANE DOE", "IDENTITY: [REDACTED]",
            f"LAST STABLE MEMORY: {rng.randint(0,59):02d}:{rng.randint(0,59):02d}",
            f"WITNESS COUNT: {rng.randint(1,4)}",
            f"COMPLIANCE: {'PARTIAL' if rng.random()<0.7 else 'FAILED'}",
            "NEXT INSTRUCTION: DO NOT DESCRIBE."
        ])
        return Slide("janedoe", "JANE DOE INTERMISSION", body, pick(all_images, 0, 1), pick(all_images, 1, 3), render_cfg["slide_seconds"])

    def mk_fatal() -> Slide:
        stop = f"0x{rng.randint(0, 0xFFFFFF):06X}"
        body = "\n".join([
            "FATAL ERROR: MNEMONIC_LEAK.EXE", f"STOP: {stop}",
            "ESC DISABLED", "DO NOT RESTART", "DO NOT REWIND"
        ])
        return Slide("fatal", "TRACKING LOST", body, [], pick(all_images, 0, 1), render_cfg["slide_seconds"])

    normals = [mk_normal() for _ in range(normal_n)]
    protocols = [mk_protocol() for _ in range(scary_n)]
    rng.shuffle(normals)
    rng.shuffle(protocols)

    while normals or protocols:
        if normals and (not protocols or rng.random() < 0.55):
            slides.append(normals.pop())
        else:
            slides.append(protocols.pop())

        if story_cfg.get("include_jane_doe", True) and rng.random() < 0.18:
            slides.append(mk_janedoe())
        if story_cfg.get("include_fatal", True) and rng.random() < 0.12:
            slides.append(mk_fatal())

    if story_cfg.get("include_intro_outro", True):
        slides.append(Slide(
            kind="outro",
            title="END OF TRANSMISSION",
            body="END OF MODULE\nThank you.\nDo not replay this tape.\nThe tape will replay you.",
            bg_imgs=pick(all_images, 1, 2),
            face_imgs=[],
            seconds=2.0
        ))

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


def _font_try(names: List[str], size: int) -> ImageFont.ImageFont:
    """Try to load fonts from system, fallback to default."""
    # Common paths for linux/mac/windows
    paths = [
        "/usr/share/fonts/truetype", "/usr/share/fonts", 
        "/Library/Fonts", "C:\\Windows\\Fonts"
    ]
    
    for n in names:
        # Check direct load
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
            
        # Check common system paths
        for p in paths:
            fpath = Path(p).rglob(n)
            for found in fpath:
                try:
                    return ImageFont.truetype(str(found), size)
                except OSError:
                    continue
                    
    return ImageFont.load_default()


def cover_resize(im: Image.Image, w: int, h: int) -> Image.Image:
    iw, ih = im.size
    s = max(w/iw, h/ih)
    nw, nh = int(iw*s), int(ih*s)
    im2 = im.resize((nw, nh), Image.Resampling.BILINEAR)
    x0, y0 = (nw - w)//2, (nh - h)//2
    return im2.crop((x0, y0, x0+w, y0+h))


def vibrant_bg(rng: random.Random, W: int, H: int) -> np.ndarray:
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

    # Simple caching would be better, but re-loading is safe
    fontT = _font_try(["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"], 28)
    fontB = _font_try(["DejaVuSans.ttf", "Arial.ttf", "arial.ttf"], 18)
    fontM = _font_try(["DejaVuSansMono.ttf", "Courier New.ttf", "cour.ttf"], 16)

    # Theme colors
    if slide.kind == "fatal":
        hdr = (255, 50, 60, 235)
    elif slide.kind in ("protocol", "face", "janedoe"):
        hdr = (255, 220, 60, 235)
    else:
        hdr = (60, 210, 255, 235)

    d.rectangle((0,0,W,60), fill=hdr)
    d.text((16, 16), slide.title, fill=(10,10,10,255), font=fontT)

    px0, py0, px1, py1 = 20, 92, int(W*0.76), 92+290
    d.rounded_rectangle((px0, py0, px1, py1), radius=16, fill=(255,255,255,220), outline=(0,0,0,70), width=2)

    y = py0 + 16
    for raw_line in slide.body.split("\n"):
        line = raw_line.strip()
        if not line:
            y += 10
            continue
        # Use standard textwrap
        wrapped = textwrap.wrap(line, width=44)
        for wline in wrapped:
            d.text((px0+18, y), wline, fill=(10,10,10,255), font=fontB)
            y += 24
        y += 2

    d.rectangle((0, H-52, W, H), fill=(0,0,0,120))
    d.text((16, H-40), "VHS TRAINING ARCHIVE  //  DO NOT DUPLICATE", fill=(255,255,255,255), font=fontM)

    if slide.kind == "janedoe":
        d.rectangle((px0+18, py1-54, px1-18, py1-22), fill=(230,230,230,235))
        d.text((px0+24, py1-50), "FILE: JANE_DOE.TAPE  //  ACCESS: DENIED", fill=(0,0,0,255), font=fontM)

    return np.array(layer, dtype=np.uint8)


def alpha_over(bg: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    # Ensure BG is RGB
    if bg.ndim == 2: bg = np.stack((bg,)*3, axis=-1)
    if bg.shape[2] == 4: bg = bg[:,:,:3]
    
    a = fg_rgba[:,:,3:4].astype(np.float32)/255.0
    return np.clip(bg.astype(np.float32)*(1-a) + fg_rgba[:,:,:3].astype(np.float32)*a, 0, 255).astype(np.uint8)


def make_popup(ctx: RenderContext, rng: random.Random, im: Image.Image) -> np.ndarray:
    arr = np.array(cover_resize(im, 220, 220).convert("RGB"), dtype=np.uint8)
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
    x = rng.randint(0, max(1, ctx.W - pw))
    y = rng.randint(70, max(71, ctx.H - ph - 70))
    
    # Safe bounds
    ye, xe = min(out.shape[0], y+ph), min(out.shape[1], x+pw)
    ph, pw = ye-y, xe-x
    if ph <= 0 or pw <= 0: return out
    
    out[y:ye, x:xe] = popup[:ph, :pw]
    
    # Border
    out[y:min(ye,y+3), x:xe] = 0
    out[max(y,ye-3):ye, x:xe] = 0
    out[y:ye, x:min(xe,x+3)] = 0
    out[y:ye, max(x,xe-3):xe] = 0
    return out


def redact_image(ctx: RenderContext, rng: random.Random, frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    strength = _clamp(ctx.redaction_strength, 0.0, 3.0)
    bars = rng.randint(1, 2 + int(2*strength))
    for _ in range(bars):
        w = rng.randint(int(ctx.W*0.22), int(ctx.W*0.65))
        h = rng.randint(14, 30)
        x = rng.randint(18, max(19, ctx.W - w - 18))
        y = rng.randint(72, max(73, ctx.H - 90))
        out[y:min(ctx.H, y+h), x:min(ctx.W, x+w)] = 0
    
    if rng.random() < 0.45*strength:
        x = rng.randint(0, max(1, ctx.W-160))
        y = rng.randint(60, max(61, ctx.H-180))
        block = out[y:y+160, x:x+160]
        if block.size > 0:
            pil = Image.fromarray(block).resize((40,40), Image.Resampling.BILINEAR).resize(block.shape[:2][::-1], Image.Resampling.NEAREST)
            out[y:y+block.shape[0], x:x+block.shape[1]] = np.array(pil, dtype=np.uint8)
    return out


def vhs_stack(ctx: RenderContext, rng: random.Random, frame: np.ndarray) -> np.ndarray:
    W, H = ctx.W, ctx.H
    s = _clamp(ctx.vhs_strength, 0.0, 3.0)
    out = frame.copy()

    # Chroma bleed
    amt = 2 + int(rng.random()*4*s)
    out[:,:,0] = np.roll(out[:,:,0], -amt, axis=1)
    out[:,:,2] = np.roll(out[:,:,2],  amt, axis=1)

    # Scanlines
    if s > 0:
        # Use [:, None, None] to expand dimensions to (H, 1, 1) for broadcasting against (H, W, 3)
        scanlines = (0.86 + 0.14 * np.sin(np.arange(H)[:, None, None] * math.pi))
        out = (out.astype(np.float32) * scanlines).astype(np.uint8)

    # Tracking line
    if rng.random() < 0.35*s:
        y = rng.randint(int(H*0.55), H-12)
        hh = rng.randint(4, 10)
        out[y:y+hh] = np.clip(out[y:y+hh].astype(np.int16) + rng.randint(40, 90), 0, 255).astype(np.uint8)
        out[y:y+hh] = np.roll(out[y:y+hh], rng.randint(-80, 80), axis=1)

    # Noise
    level = int(10 + 10*s)
    # Replaced rng.randint with np.random.randint to support size/dtype arguments
    n = np.random.randint(-level, level+1, size=out.shape, dtype=np.int16)
    out = np.clip(out.astype(np.int16) + n, 0, 255).astype(np.uint8)

    # Vignette
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W/2, H/2
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2) / math.sqrt(cx**2 + cy**2)
    v = np.clip(1 - (0.65*s)*(r**1.7), 0.28, 1.0).astype(np.float32)[...,None]
    out = (out.astype(np.float32) * v).astype(np.uint8)

    if rng.random() < 0.25*s:
        pil = Image.fromarray(out).filter(ImageFilter.GaussianBlur(radius=0.6))
        out = np.array(pil, dtype=np.uint8)

    return out


def timecode_overlay(ctx: RenderContext, frame: np.ndarray, frame_idx: int) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    font = _font_try(["DejaVuSansMono.ttf", "Courier New.ttf", "cour.ttf"], 16)
    secs = frame_idx / ctx.FPS
    hh = int(secs//3600); mm = int((secs%3600)//60); ss = int(secs%60); ff = int((secs - int(secs))*ctx.FPS)
    tc = f"TC {hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}   CH{random.randint(1,12):02d}  SP"
    d.rectangle((10, ctx.H-36, 310, ctx.H-12), fill=(0,0,0))
    d.text((16, ctx.H-34), tc, fill=(255,255,255), font=font)
    return np.array(im, dtype=np.uint8)


def face_uncanny(ctx: RenderContext, rng: random.Random, im: Image.Image, t: float) -> np.ndarray:
    base = np.array(cover_resize(im, ctx.W, ctx.H).convert("RGB"), dtype=np.uint8)
    pil = ImageOps.autocontrast(Image.fromarray(base))
    pil = ImageOps.posterize(pil, 4)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
    arr = np.array(pil, dtype=np.uint8)
    
    out = arr.copy()
    for _ in range(4):
        x0 = rng.randint(0, max(1, ctx.W-140))
        ww = rng.randint(60, 170)
        shift = int(10*math.sin(t*2.0 + x0*0.03))
        # Clip roll range
        col = out[:, x0:x0+ww]
        if col.size > 0:
            out[:, x0:x0+ww] = np.roll(col, shift, axis=0)
            
    y0 = int(ctx.H*0.30) + rng.randint(-10,10)
    if 0 <= y0 < ctx.H-36:
        out[y0:y0+36] = np.roll(out[y0:y0+36], rng.randint(-22,22), axis=1)
        
    scale = np.array([1.12, 0.92, 1.18], dtype=np.float32)
    out = np.clip(out.astype(np.float32)*scale, 0, 255).astype(np.uint8)
    return out


# ----------------------------
# Audio (music bed + pops + TTS)
# ----------------------------

def bitcrush(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1: return x
    y = x[::factor]
    y = np.repeat(y, factor)[: len(x)]
    return y


def gen_music_and_sfx(rng: random.Random, sr: int, dur_s: float) -> np.ndarray:
    n = int(sr * dur_s)
    if n <= 0: return np.zeros(0, dtype=np.float32)
    
    t = np.linspace(0, dur_s, n, False).astype(np.float32)

    audio = (np.random.uniform(-1,1,n).astype(np.float32) * 0.05)
    audio += 0.03*np.sin(2*np.pi*55*t) + 0.018*np.sin(2*np.pi*110*t)
    audio += 0.02*np.sin(2*np.pi*30*t)
    audio *= (0.82 + 0.18*np.sin(2*np.pi*0.18*t)).astype(np.float32)

    for _ in range(18):
        p0 = rng.randint(0, max(1, n-1))
        span = rng.randint(int(0.01*sr), int(0.06*sr))
        p1 = min(n, p0+span)
        if p1 > p0:
            burst = (np.random.uniform(-1,1,p1-p0).astype(np.float32) * rng.uniform(0.22, 0.55))
            audio[p0:p1] += burst

    for sec in [dur_s*0.25, dur_s*0.55, dur_s*0.8]:
        p0 = int(sec*sr)
        p1 = min(n, p0+int(0.16*sr))
        if p1 > p0:
            tt = np.linspace(0, (p1-p0)/sr, p1-p0, False).astype(np.float32)
            audio[p0:p1] += 0.18*np.sin(2*np.pi*880*tt).astype(np.float32)

    return audio


def tts_segment_espeak(text: str, out_wav: Path, voice: str, speed: int, pitch: int, amp: int) -> None:
    cmd = ["espeak", "-v", voice, "-s", str(speed), "-p", str(pitch), "-a", str(amp), "-w", str(out_wav), text]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mix_in(dst: np.ndarray, src: np.ndarray, start_s: float, sr: int, gain: float) -> None:
    start = int(start_s * sr)
    if start >= len(dst): return
    
    end = min(len(dst), start + len(src))
    if end > start:
        dst[start:end] += src[: end-start] * gain


def build_tts_track(rng: random.Random, slides: List[Slide], cfg: Dict[str, Any], dur_s: float) -> np.ndarray:
    sr = int(cfg["audio"]["sr"])
    tcfg = cfg["audio"]
    voices = list(tcfg.get("voices") or [])
    track = np.zeros(int(sr * dur_s), dtype=np.float32)
    
    work = Path(cfg["workdir"]) / "tts"
    work.mkdir(parents=True, exist_ok=True)

    t = 0.0
    for i, s in enumerate(slides):
        base = narration_for_slide(rng, s)
        voice = rng.choice(voices) if voices else "en-us"
        
        speed_i = _clamp(tcfg["tts_speed"] + rng.randint(-12, 10), 115, 190)
        pitch_i = _clamp(tcfg["tts_pitch"] + rng.randint(-12, 14), 10, 80)
        amp_i = _clamp(tcfg["tts_amp"] + rng.randint(-10, 10), 80, 200)

        wav_path = work / f"tts_{i:02d}.wav"
        try:
            tts_segment_espeak(base, wav_path, voice, int(speed_i), int(pitch_i), int(amp_i))
            sr2, data = read_wav(str(wav_path))
            if data.ndim > 1: data = data.mean(axis=1)
            data = data.astype(np.float32)
            
            # Resample if needed
            if sr2 != sr:
                x = np.linspace(0, 1, len(data), False)
                x2 = np.linspace(0, 1, int(len(data)*sr/sr2), False)
                data = np.interp(x2, x, data).astype(np.float32)
            
            mx = np.max(np.abs(data))
            if mx > 0: data /= mx

            # Saturation / ring mod
            data = np.tanh(data * 1.6) * 0.70
            if rng.random() < 0.55:
                data = bitcrush(data, rng.choice([4,5,6]))
            ring = np.sin(2*np.pi*220*np.linspace(0, len(data)/sr, len(data), False).astype(np.float32))
            data = (0.86*data + 0.14*data*ring).astype(np.float32)

            mix_in(track, data, start_s=t + 0.25, sr=sr, gain=1.0)
        except Exception:
            pass
        
        t += s.seconds

    return track


def narration_for_slide(rng: random.Random, s: Slide) -> str:
    if s.kind == "intro":
        return "Playback. Job training tape. Tracking is stable. Please do not duplicate."
    if s.kind == "outro":
        return "End of transmission. Thank you. Do not replay this tape."
    if s.kind == "fatal":
        return "Tracking lost. Fatal error. Do not restart. Do not rewind."
    if s.kind == "janedoe":
        return "Jane Doe intermission. Identity redacted. Do not attempt recognition."
    
    clean_body = re.sub(r'[^A-Za-z0-9 ]+', ' ', s.body)
    if s.kind == "normal":
        tail = rng.choice([
            "Please continue normally.", "If you feel watched, do not react.",
            "Do not read the notes aloud."
        ])
        return f"{s.title}. {clean_body}. {tail}"
    
    tail = rng.choice([
        "Description teaches it.", "Silence is a valid response.",
        "If you are watching, you are participating."
    ])
    return f"{s.title}. {clean_body}. {tail}"


# ----------------------------
# Render Logic
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

    at = rng.randint(int(total_frames*0.20), max(int(total_frames*0.85), int(total_frames*0.20)+1))
    freeze_at = None
    freeze_frames = 0
    cut_frame = None

    if rng.random() < float(tcfg["freeze_probability"]):
        fs = rng.uniform(float(tcfg["freeze_seconds_min"]), float(tcfg["freeze_seconds_max"]))
        freeze_frames = max(1, int(fs * fps))
        freeze_at = at

    if rng.random() < float(tcfg["early_end_probability"]):
        cut_frame = min(total_frames-1, at + rng.randint(int(0.2*fps), int(1.2*fps)))

    return TransmissionPlan(True, cut_frame, freeze_at, freeze_frames)


def load_local_images(dirpath: Path, limit: int = 24) -> List[Image.Image]:
    if not dirpath.exists():
        return []
    imgs = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(dirpath.rglob("*")):
        if p.suffix.lower() not in exts: continue
        try:
            im = Image.open(p).convert("RGB")
            imgs.append(im)
        except Exception:
            continue
        if len(imgs) >= limit: break
    return imgs


def split_faces_and_objects(imgs: List[Image.Image]) -> Tuple[List[Image.Image], List[Image.Image]]:
    faces, objs = [], []
    for im in imgs:
        w, h = im.size
        if h >= w and h > 220:
            faces.append(im)
        else:
            objs.append(im)
    if not faces: faces = imgs[:]
    return faces, objs


def render_video(rng: random.Random, cfg: Dict[str, Any], slides: List[Slide], out_mp4: Path) -> Tuple[Path, float, int]:
    render_cfg = cfg["render"]
    W, H, FPS = int(render_cfg["width"]), int(render_cfg["height"]), int(render_cfg["fps"])

    ctx = RenderContext(W, H, FPS, float(render_cfg["vhs_strength"]), float(render_cfg["redaction_strength"]))
    
    total_frames = sum(int(s.seconds * FPS) for s in slides)
    plan = plan_transmission(rng, cfg, total_frames, FPS)

    all_imgs = [im for s in slides for im in (s.bg_imgs + s.face_imgs) if im]
    pop_pool = [make_popup(ctx, rng, im) for im in rng.sample(all_imgs, k=min(len(all_imgs), 8))] if all_imgs else []

    popup_moments: List[int] = []
    if render_cfg["max_popups"] > 0 and total_frames > 10:
        for _ in range(int(render_cfg["max_popups"])):
            popup_moments.append(rng.randint(int(total_frames*0.15), total_frames-1))
        popup_moments = sorted(set(popup_moments))

    flash_frames = {rng.randint(int(total_frames*0.10), max(int(total_frames*0.95), 1)) for _ in range(render_cfg["flashes"])}
    popup_dur = max(1, int(render_cfg["popup_seconds"] * FPS))

    out_silent = out_mp4.with_name(out_mp4.stem + "_silent.mp4")
    # Use imageio v2 explicit params
    writer = imageio.get_writer(str(out_silent), fps=FPS, codec="libx264", quality=7)

    frame_idx = 0
    popup_active_until = -1
    
    for slide in slides:
        nF = max(1, int(slide.seconds * FPS))
        ui = make_ui_layer(ctx, rng, slide)
        
        # Base BG
        bg = vibrant_bg(rng, W, H)
        if slide.bg_imgs:
            for im in rng.sample(slide.bg_imgs, k=min(len(slide.bg_imgs), rng.randint(1, 2))):
                obj = np.array(cover_resize(im, rng.randint(220, 360), rng.randint(180, 320)).convert("RGB"))
                x = rng.randint(0, max(0, W-obj.shape[1]))
                y = rng.randint(70, max(0, H-obj.shape[0]-70))
                bg[y:y+obj.shape[0], x:x+obj.shape[1]] = obj

        for fi in range(nF):
            if plan.cut_frame is not None and frame_idx >= plan.cut_frame: break

            t = frame_idx / FPS
            frame = bg.copy()
            
            if fi % 4 == 0:
                frame = np.roll(frame, rng.randint(-2, 2), axis=1)

            if slide.face_imgs and (slide.kind in ("protocol", "janedoe") or rng.random() < 0.25):
                face = rng.choice(slide.face_imgs)
                face_arr = face_uncanny(ctx, rng, face, t)
                alpha = 0.55 if slide.kind == "normal" else 0.75
                frame = np.clip(frame.astype(np.float32)*(1-alpha) + face_arr.astype(np.float32)*alpha, 0, 255).astype(np.uint8)

            frame = alpha_over(frame, ui)

            if slide.kind in ("protocol", "janedoe", "fatal") and rng.random() < 0.35:
                frame = redact_image(ctx, rng, frame)

            if frame_idx in popup_moments:
                popup_active_until = frame_idx + popup_dur
            if frame_idx < popup_active_until and pop_pool:
                pop = rng.choice(pop_pool)
                if rng.random() < 0.30: pop = 255 - pop
                frame = stamp_popup(ctx, rng, frame, pop)

            if frame_idx in flash_frames and all_imgs:
                im = rng.choice(all_imgs)
                flash = np.array(cover_resize(im, W, H).convert("RGB"), dtype=np.uint8)
                flash = vhs_stack(ctx, rng, flash)
                if rng.random() < 0.55: flash = 255 - flash
                frame = flash

            if plan.freeze_at is not None and frame_idx == plan.freeze_at:
                freeze = frame.copy()
                for _ in range(plan.freeze_frames):
                    fr = vhs_stack(ctx, rng, freeze.copy())
                    fr = timecode_overlay(ctx, fr, frame_idx)
                    writer.append_data(fr)
                    frame_idx += 1
                continue

            frame = vhs_stack(ctx, rng, frame)
            
            if fi < 3 or fi > nF-4:
                if rng.random() < 0.65: frame = 255 - frame

            frame = timecode_overlay(ctx, frame, frame_idx)
            writer.append_data(frame)
            frame_idx += 1

        if plan.cut_frame is not None and frame_idx >= plan.cut_frame: break

    writer.close()
    return out_silent, frame_idx / FPS, frame_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--out", type=str, default="out.mp4")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    workdir = Path(cfg["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    if seed == 1337:
        seed = random.randint(0, 1000000)
    rng = random.Random(seed)    # Seed numpy as well for array operations
        np.random.seed(cfg["seed"])

    # Check dependencies
    has_ffmpeg = shutil.which("ffmpeg") is not None
    has_espeak = shutil.which("espeak") is not None

    theme_key = cfg["theme_key"] or choose_theme_key(rng)
    
    # Image gathering
    local_imgs = load_local_images(Path(cfg["local_images_dir"]), limit=36)
    web_imgs: List[Image.Image] = []
    scraped_text = ""

    if cfg["web"]["enable"]:
        print(f"Scraping web for theme: '{theme_key}'...")
        scraped_text = wiki_extract(theme_key, cfg["web"]["text_paragraphs"], cfg["web"]["timeout_s"])
        
        urls = commons_images(theme_key, cfg["web"]["image_limit"], cfg["web"]["timeout_s"])
        for u in urls:
            im = download_image(u, cfg["web"]["timeout_s"])
            if im: web_imgs.append(im)
        rng.shuffle(web_imgs)
        web_imgs = web_imgs[:18]

    face_pool, obj_pool = split_faces_and_objects(web_imgs + local_imgs)
    
    # Story gen
    slides = build_story_slides(rng, theme_key, scraped_text, obj_pool, local_imgs, cfg)
    
    # Ensure scary slides have faces
    for s in slides:
        if s.kind in ("protocol", "janedoe") and not s.face_imgs and face_pool:
            s.face_imgs = [rng.choice(face_pool)]

    out_mp4 = Path(args.out)
    print("Rendering video...")
    out_silent, dur_s, _ = render_video(rng, cfg, slides, out_mp4)

    # Audio gen
    print("Generating audio...")
    sr = int(cfg["audio"]["sr"])
    audio = np.zeros(int(sr * dur_s), dtype=np.float32)
    
    if cfg["audio"]["music"]:
        audio += gen_music_and_sfx(rng, sr, dur_s)

    if cfg["audio"]["tts"]:
        if has_espeak:
            audio += build_tts_track(rng, slides, cfg, dur_s)
        else:
            print("Warning: 'espeak' not found. Skipping TTS generation.")

    # Normalize and write WAV
    mx = np.max(np.abs(audio))
    if mx > 1e-6: audio /= mx
    
    audio_i16 = (audio * 32767).astype(np.int16)
    audio_path = out_mp4.with_suffix(".wav")
    write_wav(str(audio_path), sr, audio_i16)

    # Mux
    if has_ffmpeg:
        print("Muxing audio/video...")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_silent), "-i", str(audio_path),
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_mp4)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        # Cleanup temp files
        out_silent.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
        print(f"Done: {out_mp4}")
    else:
        print(f"ffmpeg not found. Outputs saved separately:\nVideo: {out_silent}\nAudio: {audio_path}")

if __name__ == "__main__":
    main()
