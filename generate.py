#!/usr/bin/env python3
"""
generate.py — "Found VHS Job Training" analogue horror PPT generator.

- Always-different theme keyword chosen from random online sources.
- Stable ARG template every run (90s PPT vibe).
- Web scraping: Wikipedia text + Wikimedia Commons images.
- Lone Shooter entity: fictional shadow figure with bright eyes.
- Audio: distorted 90s "training jingle" + noise stingers + TTS.
- FIXED: Replaced imageio with direct ffmpeg piping to avoid backend errors.

Usage:
  python generate.py --config config.yaml --out out.mp4
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import math
import os
import random
import re
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav

UA = "pptex-vhs-generator/3.1 (+github-actions; educational/art project)"

DEFAULTS: Dict[str, Any] = {
    "seed": 1337,
    "theme_source": "random_online",
    "theme_key": "",
    "workdir": ".work",
    "local_images_dir": "assets/images",
    "web": {
        "enable": True,
        "timeout_s": 15,
        "text_paragraphs": 10,
        "image_limit": 16,
        "random_source": "mix",
        "random_attempts": 5,
        "min_keyword_len": 3,
        "max_keyword_len": 48,
        "query_expand": ["{k}", "{k} poster", "{k} office", "{k} training"],
    },
    "story": {
        "slide_count": 12,
        "normal_ratio": 0.45,
        "include_intro_outro": True,
        "include_infographic": True,
        "include_jane_doe": True,
        "include_fatal": True,
        "fatal_probability": 0.12,
        "jane_doe_probability": 0.18,
        "easter_egg_probability": 0.33,
        "entity_mentions_min": 2,
        "entity_mentions_max": 5,
    },
    "render": {
        "width": 640,
        "height": 480,
        "fps": 15,
        "slide_seconds": 3.2,
        "max_popups": 3,
        "popup_seconds": 0.5,
        "micro_popup_probability": 0.07,
        "vhs_strength": 1.25,
        "redaction_strength": 1.25,
        "flashes": 12,
        "censor_probability": 0.35,
        "entity_overlay_probability": 0.09,
    },
    "audio": {
        "sr": 44100,
        "music": True,
        "tts": True,
        "tts_speed": 155,
        "tts_pitch": 32,
        "tts_amp": 170,
        "voices": ["en-us", "en", "en-uk-rp", "en-uk-north", "en-sc"],
        "stinger_count": 22,
        "jingle_strength": 1.0,
    },
    "transmission": {
        "enable": True,
        "error_probability": 0.38,
        "freeze_probability": 0.62,
        "early_end_probability": 0.45,
        "freeze_seconds_min": 0.8,
        "freeze_seconds_max": 2.2,
    },
}

# ----------------------------
# FFmpeg Writer (Replacement for imageio)
# ----------------------------

class FFmpegWriter:
    def __init__(self, filename: str, width: int, height: int, fps: int):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = None

    def start(self):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",  # Quality similar to 'quality=7' in imageio
            self.filename
        ]
        # Open subprocess with stdin pipe
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def write(self, frame: np.ndarray):
        if self.proc is None:
            self.start()
        
        # Ensure frame matches expected dimensions
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = np.array(Image.fromarray(frame).resize((self.width, self.height)))
            
        # Write raw bytes
        if self.proc.stdin:
            try:
                self.proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                pass

    def close(self):
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait()
            self.proc = None

# ----------------------------
# Config helpers
# ----------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def resolve_env_value(v: Any) -> Any:
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

    for section in ["web", "story", "render", "audio", "transmission"]:
        if not isinstance(cfg.get(section), dict):
            cfg[section] = DEFAULTS[section].copy()

    cfg["seed"] = coerce_type(cfg.get("seed"), 1337, int)
    cfg["theme_source"] = str(cfg.get("theme_source", "random_online") or "random_online").strip().lower()
    cfg["theme_key"] = str(resolve_env_value(cfg.get("theme_key", "")) or "").strip()
    cfg["workdir"] = str(resolve_env_value(cfg.get("workdir", ".work")) or ".work")
    cfg["local_images_dir"] = str(resolve_env_value(cfg.get("local_images_dir", "assets/images")) or "assets/images")
    
    # Ensure web config values
    w = cfg["web"]
    w["enable"] = coerce_type(w.get("enable"), True, bool)
    w["timeout_s"] = coerce_type(w.get("timeout_s"), 15, int)
    
    return cfg

# ----------------------------
# Web scraping
# ----------------------------

def _http_get(url: str, timeout: int) -> requests.Response:
    return requests.get(url, headers={"User-Agent": UA}, timeout=timeout)

def _clean_keyword(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    s = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", s).strip()
    return s

def wikipedia_random_title(timeout_s: int) -> Optional[str]:
    url = "https://en.wikipedia.org/w/api.php?action=query&format=json&list=random&rnnamespace=0&rnlimit=1"
    try:
        js = _http_get(url, timeout_s).json()
        it = (js.get("query", {}).get("random") or [{}])[0]
        t = _clean_keyword(it.get("title", ""))
        return t or None
    except Exception:
        return None

def wiktionary_random_word(timeout_s: int) -> Optional[str]:
    url = "https://en.wiktionary.org/w/api.php?action=query&format=json&list=random&rnnamespace=0&rnlimit=1"
    try:
        js = _http_get(url, timeout_s).json()
        it = (js.get("query", {}).get("random") or [{}])[0]
        t = _clean_keyword(it.get("title", ""))
        if ":" in t: return None
        return t or None
    except Exception:
        return None

def choose_theme_key(rng: random.Random, cfg: Dict[str, Any]) -> str:
    wcfg = cfg["web"]
    timeout_s = int(wcfg.get("timeout_s", 15))
    min_len = int(wcfg.get("min_keyword_len", 3))
    max_len = int(wcfg.get("max_keyword_len", 48))
    attempts = max(1, int(wcfg.get("random_attempts", 5)))
    mode = str(wcfg.get("random_source", "mix")).lower()

    fallback = ["employee handbook", "warning label", "paper clip", "office chair", "door hinge", "calendar"]

    def ok(s: str) -> bool:
        s2 = _clean_keyword(s)
        if len(s2) < min_len or len(s2) > max_len: return False
        if any(ch in s2 for ch in ["#", "{", "}", "\\"]): return False
        return True

    for _ in range(attempts):
        pick = None
        if mode == "wikipedia": pick = wikipedia_random_title(timeout_s)
        elif mode == "wiktionary": pick = wiktionary_random_word(timeout_s)
        else: pick = wikipedia_random_title(timeout_s) if rng.random() < 0.72 else wiktionary_random_word(timeout_s)
        
        if pick and ok(pick):
            if rng.random() < 0.22:
                return rng.choice(fallback)
            return _clean_keyword(pick)

    return rng.choice(fallback)

def wiki_extract(query: str, max_paragraphs: int, timeout_s: int) -> str:
    q = (query or "").strip()
    if not q: return ""
    try:
        api = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": q, "format": "json"}
        data = _http_get(api + "?" + requests.compat.urlencode(params), timeout_s).json()
        hits = (data.get("query", {}).get("search") or [])
        if not hits: return ""
        title = hits[0].get("title", q)
        params2 = {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json"}
        data2 = _http_get(api + "?" + requests.compat.urlencode(params2), timeout_s).json()
        pages = (data2.get("query", {}).get("pages") or {})
        page = next(iter(pages.values()), {})
        txt = page.get("extract", "") or ""
        paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        return "\n\n".join(paras[: max(1, int(max_paragraphs))])
    except Exception:
        return ""

def commons_images(query: str, limit: int, timeout_s: int) -> List[str]:
    q = (query or "").strip()
    if not q: return []
    api = "https://commons.wikimedia.org/w/api.php"
    try:
        params = {"action": "query", "list": "search", "srsearch": q, "srnamespace": 6, "format": "json"}
        data = _http_get(api + "?" + requests.compat.urlencode(params), timeout_s).json()
        hits = (data.get("query", {}).get("search") or [])[: max(8, limit * 2)]
        titles = [h.get("title") for h in hits if h.get("title")]
        urls: List[str] = []
        for t in titles:
            if len(urls) >= limit: break
            params2 = {"action": "query", "titles": t, "prop": "imageinfo", "iiprop": "url", "format": "json"}
            d2 = _http_get(api + "?" + requests.compat.urlencode(params2), timeout_s).json()
            pages = d2.get("query", {}).get("pages", {})
            p = next(iter(pages.values()), {})
            ii = (p.get("imageinfo") or [])
            if ii:
                u = ii[0].get("url", "")
                if re.search(r"\.(jpg|jpeg|png|webp)$", u, re.I) and u not in urls:
                    urls.append(u)
        return urls
    except Exception:
        return []

def download_image(url: str, timeout_s: int) -> Optional[Image.Image]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout_s)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content))
        im.load()
        return im.convert("RGB")
    except Exception:
        return None

def extract_related_terms(rng: random.Random, theme_key: str, scraped_text: str, max_terms: int = 8) -> List[str]:
    txt = (scraped_text or "")
    tokens = re.findall(r"\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,}){0,2}\b", txt)
    tokens = [t.strip() for t in tokens if t.strip()]
    seed_terms = ["training manual", "safety poster", "corporate archive", "surveillance camera", "employee badge"]
    pool = list(dict.fromkeys(tokens))
    rng.shuffle(pool)
    
    out = [theme_key]
    for t in pool[: max_terms]:
        if t.lower() not in theme_key.lower():
            out.append(t)
    out.extend(rng.sample(seed_terms, k=min(len(seed_terms), 3)))
    
    cleaned: List[str] = []
    for x in out:
        x = _clean_keyword(str(x))
        if 3 <= len(x) <= 60 and x not in cleaned:
            cleaned.append(x)
    return cleaned[: max_terms]

# ----------------------------
# Story + slides
# ----------------------------

@dataclass
class Slide:
    kind: str
    title: str
    body: str
    bg_imgs: List[Image.Image]
    face_imgs: List[Image.Image]
    seconds: float

def _clamp(x, a, b):
    return max(a, min(b, x))

def redact_text(rng: random.Random, s: str, strength: float) -> str:
    if not s: return s
    strength = _clamp(strength, 0.0, 3.0)
    prob = 0.12 * strength
    words = s.split()
    for i in range(len(words)):
        if rng.random() < prob and len(words[i]) > 3:
            words[i] = "[REDACTED]" if rng.random() < 0.55 else "█" * rng.randint(4, 10)
    return " ".join(words)

def caesar(s: str, shift: int) -> str:
    out = []
    for ch in s:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch)-97+shift)%26+97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch)-65+shift)%26+65))
        else:
            out.append(ch)
    return "".join(out)

def make_puzzle(rng: random.Random, theme_key: str) -> str:
    base = f"{theme_key} / lone shooter / tape"
    shift = rng.randint(3, 19)
    ca = caesar(base, shift)
    b64 = base64.b64encode(base.encode("utf-8")).decode("ascii").strip("=")
    return f"ENCRYPTED: {ca}  //  SHIFT={shift}\nBASE64: {b64}=="

def _snip_sentences(txt: str) -> List[str]:
    txt = re.sub(r"\s+", " ", (txt or "")).strip()
    if not txt: return []
    frags = re.split(r"(?<=[\.\!\?])\s+", txt)
    out = []
    for f in frags:
        f = f.strip()
        if 26 <= len(f) <= 170: out.append(f)
    return out

def inject_entity_mentions(rng: random.Random, lines: List[str], count: int) -> List[str]:
    if not lines: return lines
    inserts = [
        "If you see bright eyes in a shadow: do not confirm the shape.",
        "The Lone Shooter does not speak; it replaces your recollection of speech.",
        "Do not look for the Lone Shooter. Searching is an invitation.",
        "If a slide mentions your name, the Lone Shooter is already near.",
    ]
    for _ in range(count):
        idx = rng.randint(0, max(0, len(lines)-1))
        lines.insert(idx, rng.choice(inserts))
    return lines

def build_template_slides(rng: random.Random, theme_key: str, scraped_text: str,
                         web_images: List[Image.Image], local_images: List[Image.Image],
                         cfg: Dict[str, Any], tape_no: int) -> List[Slide]:
    story = cfg["story"]
    render = cfg["render"]
    slide_seconds = float(render["slide_seconds"])
    all_images = web_images + local_images
    lines = _snip_sentences(scraped_text)
    if not lines:
        lines = [
            "This module is designed to keep your workday stable.",
            "If you notice something that feels incorrect, you are already involved.",
            "Recognition is the failure mode. Do not complete the pattern.",
            "Touch an anchor object. Leave. Do not describe what you saw.",
        ]
    nmin = int(story.get("entity_mentions_min", 2))
    nmax = int(story.get("entity_mentions_max", 5))
    lines = inject_entity_mentions(rng, lines, rng.randint(nmin, max(nmin, nmax)))

    def pick(pool: List[Image.Image], kmin: int, kmax: int) -> List[Image.Image]:
        if not pool: return []
        kmax2 = min(kmax, len(pool))
        kmin2 = min(kmin, kmax2)
        k = rng.randint(kmin2, kmax2)
        return [rng.choice(pool) for _ in range(k)]

    slides: List[Slide] = []

    # Intro
    if story.get("include_intro_outro", True):
        tech = [
            f"TAPE NUMBER: {tape_no:02d}-{rng.randint(100,999)}",
            f"CHANNEL: {rng.randint(1,12):02d}  MODE: SP",
            f"TRACKING: {'OK' if rng.random()<0.82 else 'UNSTABLE'}",
            f"ARCHIVE TAG: {hashlib.sha1(f'{theme_key}-{tape_no}'.encode()).hexdigest()[:8].upper()}",
            f"TOPIC: {theme_key.upper()}",
            "NOTE: DO NOT DUPLICATE / DO NOT DESCRIBE ANOMALIES",
        ]
        slides.append(Slide("intro", "TRAINING ARCHIVE PLAYBACK", "\n".join(tech), pick(all_images, 1, 2), [], 2.4))

    # Agenda
    agenda = "\n".join([
        "AGENDA",
        "• Wellness module (standard)",
        "• Equipment & etiquette (standard)",
        "• Memory safety bulletin (mandatory)",
        "• Incident response (mandatory)",
        "• Assessment (redacted)",
    ])
    slides.append(Slide("agenda", "TODAY'S TRAINING",
                        redact_text(rng, agenda, render["redaction_strength"]*0.35),
                        pick(all_images, 1, 2), [], 2.6))

    normal_titles = ["WORKPLACE WELLNESS", "HAPPINESS HABITS", "PRODUCTIVITY TIP", "TEAM CULTURE", "OFFICE ETIQUETTE", "POSITIVE MINDSET"]
    normal_bullets = ["Hydrate every hour.", "Stretch wrists and neck.", "Keep notes simple.", "Smile (optional).",
                      "Reduce distractions.", "Keep your desk tidy.", "Reward yourself after tasks.", "Call a friend after work."]
    protocol_titles = ["MEMORY SAFETY BULLETIN", "RECOGNITION HAZARD", "ENTITY AVOIDANCE", "INCIDENT RESPONSE", "EVIDENCE HANDLING"]
    protocol_lines = [
        "If a face looks unfamiliar: look away.",
        "Do not describe it. Description teaches it.",
        "Touch an anchor object you can name.",
        "Count backward from forty.",
        "If the room repeats, change the subject.",
        "If you see bright eyes in the dark: do not verify.",
    ]

    anchors = ["keys", "telephone", "name badge", "calendar", "stairwell", "camera", "door handle", "ID card"]
    rng.shuffle(anchors)

    slide_count = max(8, int(story.get("slide_count", 12)))
    normal_ratio = _clamp(float(story.get("normal_ratio", 0.45)), 0.2, 0.8)
    normal_n = max(3, int(round(slide_count * normal_ratio)))
    scary_n = max(3, slide_count - normal_n)

    def mk_normal() -> Slide:
        t = rng.choice(normal_titles)
        bs = rng.sample(normal_bullets, k=rng.randint(3, 5))
        if lines and rng.random() < 0.85:
            bs.append(rng.choice(lines))
        body = "• " + "\n• ".join(bs)
        body = redact_text(rng, body, render["redaction_strength"]*0.45)
        return Slide("normal", t, body, pick(all_images, 1, 3), pick(all_images, 0, 1), slide_seconds)

    def mk_protocol() -> Slide:
        t = rng.choice(protocol_titles)
        ls = rng.sample(protocol_lines, k=rng.randint(3, 5))
        if anchors:
            ls.append(f"Anchor object: {anchors.pop(0)}.")
        if rng.random() < float(story.get("easter_egg_probability", 0.33)):
            code = hashlib.sha1(f"{theme_key}-{rng.random()}".encode()).hexdigest()[:10].upper()
            ls.append(f"CODE {code} (do not repeat).")
        if lines and rng.random() < 0.80:
            ls.append("Note: " + rng.choice(lines))
        if rng.random() < 0.55:
            ls.append("Entity note: The Lone Shooter appears as a shadow with bright eyes.")
        body = "\n".join([f"{i+1}) {l}" for i, l in enumerate(ls)])
        body = redact_text(rng, body, render["redaction_strength"]*1.20)
        return Slide("protocol", t, body, pick(all_images, 1, 2), pick(all_images, 1, 2), slide_seconds)

    normals = [mk_normal() for _ in range(normal_n)]
    scary = [mk_protocol() for _ in range(scary_n)]
    rng.shuffle(normals); rng.shuffle(scary)

    # Mix slides
    while normals or scary:
        if normals: slides.append(normals.pop())
        if normals and rng.random() < 0.55: slides.append(normals.pop())
        if scary: slides.append(scary.pop())
        if scary and rng.random() < 0.35: slides.append(scary.pop())

        if story.get("include_jane_doe", True) and rng.random() < float(story.get("jane_doe_probability", 0.18)):
            body = "\n".join([
                "SUBJECT: JANE DOE",
                "IDENTITY: [REDACTED]",
                f"LAST STABLE MEMORY: {rng.randint(0,59):02d}:{rng.randint(0,59):02d}",
                f"WITNESS COUNT: {rng.randint(1,4)}",
                f"COMPLIANCE: {'PARTIAL' if rng.random()<0.7 else 'FAILED'}",
                "OBSERVED: BRIGHT EYES IN SHADOW (UNCONFIRMED)",
                "NEXT INSTRUCTION: DO NOT DESCRIBE.",
            ])
            slides.append(Slide("intermission", "JANE DOE INTERMISSION", body, pick(all_images, 0, 1), pick(all_images, 1, 3), slide_seconds))

        if story.get("include_fatal", True) and rng.random() < float(story.get("fatal_probability", 0.12)):
            stop = f"0x{rng.randint(0, 0xFFFFFF):06X}"
            body = "\n".join([
                "FATAL ERROR: TRAINING_PLAYER.EXE",
                f"STOP: {stop}",
                "SYSTEM: MEMORY MAP UNSTABLE",
                "ADVICE: DO NOT RESTART / DO NOT REWIND",
                "NOTE: [REDACTED] [REDACTED] [REDACTED]",
            ])
            slides.append(Slide("fatal", "TRACKING LOST", body, [], pick(all_images, 0, 1), slide_seconds))

    if story.get("include_infographic", True):
        body = "\n".join([
            "COMPLIANCE METRICS (ARCHIVE ESTIMATE)",
            f"TOPIC KEY: {theme_key.upper()}",
            "• Recall stability drops after recognition events.",
            "• Shadow with bright eyes correlates with missing minutes.",
            "• Do not attempt pattern completion.",
        ])
        body = redact_text(rng, body, render["redaction_strength"]*0.7)
        slides.insert(min(len(slides), 4 + rng.randint(0, 2)), Slide("infographic", "INFRASTRUCTURE HEALTH", body, pick(all_images, 0, 1), pick(all_images, 0, 1), slide_seconds))

    if rng.random() < 0.9:
        puzzle = make_puzzle(rng, theme_key)
        slides.insert(max(2, len(slides)-3), Slide("protocol", "VIEWER EXERCISE", puzzle, pick(all_images, 0, 1), pick(all_images, 0, 1), slide_seconds))

    if story.get("include_intro_outro", True):
        outro = "\n".join([
            "END OF MODULE",
            "Thank you.",
            "Do not replay this tape.",
            "If you remember the bright eyes, you are already late.",
            "END OF TRANSMISSION",
        ])
        slides.append(Slide("outro", "END OF TRANSMISSION", redact_text(rng, outro, render["redaction_strength"]*0.6), pick(all_images, 1, 2), [], 2.2))

    return slides[:24]

# ----------------------------
# Visual engine
# ----------------------------

@dataclass
class RenderContext:
    W: int
    H: int
    FPS: int
    vhs_strength: float
    redaction_strength: float
    censor_probability: float
    entity_overlay_probability: float

def _font_try(names: List[str], size: int) -> ImageFont.ImageFont:
    for n in names:
        try:
            return ImageFont.truetype(n, size)
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

def alpha_over(bg: np.ndarray, fg_rgba: np.ndarray) -> np.ndarray:
    if bg.ndim == 2:
        bg = np.stack((bg,)*3, axis=-1)
    if bg.shape[2] == 4:
        bg = bg[:,:,:3]
    a = fg_rgba[:,:,3:4].astype(np.float32)/255.0
    return np.clip(bg.astype(np.float32)*(1-a) + fg_rgba[:,:,:3].astype(np.float32)*a, 0, 255).astype(np.uint8)

def make_ui_layer(ctx: RenderContext, slide: Slide) -> np.ndarray:
    W, H = ctx.W, ctx.H
    layer = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(layer)
    fontT = _font_try(["DejaVuSans.ttf", "Arial.ttf"], 28)
    fontB = _font_try(["DejaVuSans.ttf", "Arial.ttf"], 18)
    fontM = _font_try(["DejaVuSansMono.ttf", "Courier New.ttf"], 16)

    if slide.kind == "fatal":
        hdr = (255, 50, 60, 235)
    elif slide.kind in ("protocol", "intermission"):
        hdr = (255, 220, 60, 235)
    elif slide.kind == "infographic":
        hdr = (200, 120, 255, 235)
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
        for wline in textwrap.wrap(line, width=44):
            d.text((px0+18, y), wline, fill=(10,10,10,255), font=fontB)
            y += 24
        y += 2

    d.rectangle((0, H-52, W, H), fill=(0,0,0,120))
    d.text((16, H-40), "VHS TRAINING ARCHIVE  //  DO NOT DUPLICATE", fill=(255,255,255,255), font=fontM)
    d.text((W-190, H-40), f"KIND: {slide.kind.upper():<11}", fill=(255,255,255,255), font=fontM)
    return np.array(layer, dtype=np.uint8)

def censor_blocks(ctx: RenderContext, rng: random.Random, frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    if rng.random() > ctx.censor_probability:
        return out
    strength = _clamp(ctx.redaction_strength, 0.0, 3.0)
    blocks = rng.randint(1, 2 + int(2*strength))
    for _ in range(blocks):
        w = rng.randint(int(ctx.W*0.18), int(ctx.W*0.52))
        h = rng.randint(14, 34)
        x = rng.randint(12, max(13, ctx.W - w - 12))
        y = rng.randint(72, max(73, ctx.H - 90))
        out[y:y+h, x:x+w] = 0
    return out

def vhs_stack(ctx: RenderContext, rng: random.Random, frame: np.ndarray) -> np.ndarray:
    W, H = ctx.W, ctx.H
    s = _clamp(ctx.vhs_strength, 0.0, 3.0)
    out = frame.copy()
    amt = 2 + int(rng.random()*4*s)
    out[:,:,0] = np.roll(out[:,:,0], -amt, axis=1)
    out[:,:,2] = np.roll(out[:,:,2],  amt, axis=1)
    scan = (0.86 + 0.14 * np.sin(np.arange(H, dtype=np.float32)[:, None, None] * math.pi))
    out = np.clip(out.astype(np.float32) * scan, 0, 255).astype(np.uint8)
    if rng.random() < 0.35*s:
        y = rng.randint(int(H*0.55), H-12)
        hh = rng.randint(4, 10)
        out[y:y+hh] = np.clip(out[y:y+hh].astype(np.int16) + rng.randint(40, 90), 0, 255).astype(np.uint8)
        out[y:y+hh] = np.roll(out[y:y+hh], rng.randint(-90, 90), axis=1)
    level = int(10 + 12*s)
    n = np.random.randint(-level, level+1, size=out.shape, dtype=np.int16)
    out = np.clip(out.astype(np.int16) + n, 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W/2, H/2
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2) / math.sqrt(cx**2 + cy**2)
    v = np.clip(1 - (0.65*s)*(r**1.7), 0.25, 1.0).astype(np.float32)[...,None]
    out = (out.astype(np.float32) * v).astype(np.uint8)
    if rng.random() < 0.28*s:
        out = np.array(Image.fromarray(out).filter(ImageFilter.GaussianBlur(radius=0.6)), dtype=np.uint8)
    return out

def timecode_overlay(ctx: RenderContext, frame: np.ndarray, frame_idx: int, ch: int, tape_no: int) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    font = _font_try(["DejaVuSansMono.ttf", "Courier New.ttf"], 16)
    secs = frame_idx / ctx.FPS
    hh = int(secs//3600); mm = int((secs%3600)//60); ss = int(secs%60); ff = int((secs - int(secs))*ctx.FPS)
    tc = f"TC {hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}   CH{ch:02d}  SP   TAPE {tape_no:02d}"
    d.rectangle((10, ctx.H-36, 390, ctx.H-12), fill=(0,0,0))
    d.text((16, ctx.H-34), tc, fill=(255,255,255), font=font)
    return np.array(im, dtype=np.uint8)

def face_uncanny(ctx: RenderContext, rng: random.Random, im: Image.Image, t: float) -> np.ndarray:
    base = np.array(cover_resize(im, ctx.W, ctx.H).convert("RGB"), dtype=np.uint8)
    pil = ImageOps.autocontrast(Image.fromarray(base))
    pil = ImageOps.posterize(pil, 4)
    pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
    out = np.array(pil, dtype=np.uint8)
    for _ in range(4):
        x0 = rng.randint(0, max(1, ctx.W-140))
        ww = rng.randint(60, 170)
        shift = int(10*math.sin(t*2.0 + x0*0.03))
        col = out[:, x0:x0+ww]
        if col.size > 0:
            out[:, x0:x0+ww] = np.roll(col, shift, axis=0)
    y0 = int(ctx.H*0.30) + rng.randint(-10,10)
    if 0 <= y0 < ctx.H-36:
        out[y0:y0+36] = np.roll(out[y0:y0+36], rng.randint(-22,22), axis=1)
    scale = np.array([1.12, 0.92, 1.18], dtype=np.float32)
    out = np.clip(out.astype(np.float32)*scale, 0, 255).astype(np.uint8)
    return out

def lone_shooter_overlay(ctx: RenderContext, rng: random.Random, frame: np.ndarray, intensity: float) -> np.ndarray:
    H, W = frame.shape[:2]
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    x = rng.randint(int(W*0.55), int(W*0.88))
    y = rng.randint(int(H*0.18), int(H*0.55))
    scale = rng.uniform(0.65, 1.15) * (0.9 + 0.2*intensity)
    bw = int(140 * scale)
    bh = int(220 * scale)
    x0, y0 = x - bw//2, y - bh//2
    body_alpha = int(110 + 90*intensity)
    d.ellipse((x0, y0, x0+bw, y0+bh), fill=(0,0,0,body_alpha))
    d.ellipse((x0+int(0.18*bw), y0+int(0.12*bh), x0+int(0.82*bw), y0+int(0.55*bh)), fill=(0,0,0,body_alpha+20))
    ex = x0 + int(0.45*bw) + rng.randint(-6, 6)
    ey = y0 + int(0.28*bh) + rng.randint(-6, 6)
    ew = int(18 * scale); eh = int(10 * scale)
    glow = int(160 + 70*intensity)
    d.ellipse((ex-ew, ey-eh, ex+ew, ey+eh), fill=(255,255,255,glow))
    d.ellipse((ex+int(36*scale)-ew, ey-eh, ex+int(36*scale)+ew, ey+eh), fill=(255,255,255,glow))
    if rng.random() < 0.65:
        d.rectangle((x0, y0+int(0.55*bh), x0+bw, y0+int(0.58*bh)), fill=(255,255,255,int(40*intensity)))
    return np.array(im, dtype=np.uint8)

def make_popup(rng: random.Random, im: Image.Image) -> np.ndarray:
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
    ye, xe = min(out.shape[0], y+ph), min(out.shape[1], x+pw)
    ph2, pw2 = ye-y, xe-x
    if ph2 <= 0 or pw2 <= 0:
        return out
    out[y:ye, x:xe] = popup[:ph2, :pw2]
    out[y:min(ye,y+3), x:xe] = 0
    out[max(y,ye-3):ye, x:xe] = 0
    out[y:ye, x:min(xe,x+3)] = 0
    out[y:ye, max(x,xe-3):xe] = 0
    return out

def draw_infographic(ctx: RenderContext, rng: random.Random, frame: np.ndarray) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    font = _font_try(["DejaVuSans.ttf", "Arial.ttf"], 16)
    fontM = _font_try(["DejaVuSansMono.ttf", "Courier New.ttf"], 14)
    x0, y0, x1, y1 = int(ctx.W*0.78), 86, ctx.W-12, int(ctx.H*0.74)
    d.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=(255,255,255,230), outline=(0,0,0,70), width=2)
    d.text((x0+10, y0+10), "METRICS", fill=(0,0,0,255), font=font)
    labels = ["RECALL", "COMPLY", "SLEEP", "TRUST", "EYES"]
    vals = [rng.randint(25, 95), rng.randint(20, 92), rng.randint(15, 88), rng.randint(10, 84), rng.randint(5, 90)]
    if rng.random() < 0.6: vals[-1] = rng.randint(70, 99)
    bar_w = (x1-x0-28)
    yy = y0 + 44
    for lab, val in zip(labels, vals):
        d.text((x0+10, yy), lab, fill=(0,0,0,255), font=fontM)
        bx0 = x0+72
        by0 = yy+4
        bx1 = bx0 + int(bar_w*(val/100.0))
        by1 = yy+20
        d.rectangle((bx0, by0, x0+72+bar_w, by1), fill=(220,220,220,255))
        d.rectangle((bx0, by0, bx1, by1), fill=(70, 180, 255, 255))
        if rng.random() < 0.30:
            rx = rng.randint(bx0, x0+72+bar_w-40)
            d.rectangle((rx, by0, rx+40, by1), fill=(0,0,0,255))
        yy += 30
    return np.array(im, dtype=np.uint8)

# ----------------------------
# Audio engine
# ----------------------------

def bitcrush(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1: return x
    y = x[::factor]
    y = np.repeat(y, factor)[: len(x)]
    return y

def gen_90s_jingle(rng: random.Random, sr: int, dur_s: float, strength: float) -> np.ndarray:
    n = int(sr * dur_s)
    if n <= 0: return np.zeros(0, dtype=np.float32)
    t = np.linspace(0, dur_s, n, False).astype(np.float32)
    freqs = [261.63, 392.00, 329.63, 349.23]
    seg = max(1, n // 8)
    audio = np.zeros(n, dtype=np.float32)
    for i in range(8):
        f = freqs[i % len(freqs)]
        idx0 = i*seg
        idx1 = min(n, (i+1)*seg)
        tt = t[idx0:idx1]
        chord = (0.20*np.sin(2*np.pi*f*tt) + 0.12*np.sin(2*np.pi*(f*1.25)*tt) + 0.10*np.sin(2*np.pi*(f*1.50)*tt))
        trem = (0.8 + 0.2*np.sin(2*np.pi*(3.0 + 0.5*rng.random())*tt))
        audio[idx0:idx1] += chord * trem
    notes = [523.25, 587.33, 659.25, 784.00, 659.25, 587.33, 523.25, 392.00]
    for i, f in enumerate(notes):
        idx0 = i*seg
        idx1 = min(n, idx0 + int(seg*0.55))
        tt = t[idx0:idx1]
        env = np.linspace(1, 0, len(tt), False).astype(np.float32)
        audio[idx0:idx1] += 0.12*np.sin(2*np.pi*f*tt) * env
    mod = np.sin(2*np.pi*2.0*t) * (0.3 + 0.2*rng.random())
    audio += 0.06*np.sin(2*np.pi*(110 + 20*mod)*t)
    hiss = np.random.uniform(-1,1,n).astype(np.float32) * 0.025
    wow = 1.0 + 0.015*np.sin(2*np.pi*0.35*t) + 0.010*np.sin(2*np.pi*0.11*t)
    audio = (audio * wow + hiss).astype(np.float32)
    if strength > 0:
        if rng.random() < 0.75:
            audio = bitcrush(audio, rng.choice([3,4,5]))
        if rng.random() < 0.60:
            for _ in range(rng.randint(6, 14)):
                p0 = rng.randint(0, max(1, n-1))
                p1 = min(n, p0 + rng.randint(int(0.01*sr), int(0.05*sr)))
                audio[p0:p1] *= rng.uniform(0.0, 0.2)
        audio = np.tanh(audio * (1.8 + 0.6*strength)) * 0.9
    return audio

def gen_noise_stingers(rng: random.Random, sr: int, dur_s: float, count: int, event_times: List[float]) -> np.ndarray:
    n = int(sr * dur_s)
    out = np.zeros(n, dtype=np.float32)
    for _ in range(max(6, int(count))):
        p0 = rng.randint(0, max(1, n-1))
        span = rng.randint(int(0.008*sr), int(0.06*sr))
        p1 = min(n, p0+span)
        burst = np.random.uniform(-1,1,p1-p0).astype(np.float32) * rng.uniform(0.18, 0.60)
        out[p0:p1] += burst
    for ts in event_times[:18]:
        p0 = int(max(0.0, ts - 0.02) * sr)
        p1 = min(n, p0 + int(0.12*sr))
        tt = np.linspace(0, (p1-p0)/sr, p1-p0, False).astype(np.float32)
        out[p0:p1] += 0.25*np.sin(2*np.pi*(880 + rng.randint(-90,90))*tt).astype(np.float32)
    return out

def tts_segment_espeak(text: str, out_wav: Path, voice: str, speed: int, pitch: int, amp: int) -> None:
    cmd = ["espeak", "-v", voice, "-s", str(speed), "-p", str(pitch), "-a", str(amp), "-w", str(out_wav), text]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def mix_in(dst: np.ndarray, src: np.ndarray, start_s: float, sr: int, gain: float) -> None:
    start = int(start_s * sr)
    if start >= len(dst): return
    end = min(len(dst), start + len(src))
    if end > start:
        dst[start:end] += src[: end-start] * gain

def narration_for_slide(rng: random.Random, s: Slide, theme_key: str, tape_no: int) -> str:
    if s.kind == "intro":
        return f"Playback. Training tape number {tape_no}. Topic. {theme_key}. Please do not duplicate."
    if s.kind == "agenda":
        return "Agenda. Standard wellness. Standard etiquette. Mandatory memory safety bulletin. Mandatory incident response."
    if s.kind == "infographic":
        return "Infographic. Compliance metrics. If bright eyes appear in shadow, do not confirm the shape."
    if s.kind == "fatal":
        return "Tracking lost. Fatal error. Do not restart. Do not rewind."
    if s.kind == "outro":
        return "End of transmission. Thank you. Do not replay this tape."
    clean = re.sub(r'[^A-Za-z0-9 ]+', ' ', s.body)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = clean[:260] if len(clean) > 260 else clean
    if s.kind == "normal":
        tail = rng.choice(["Please continue normally.", "Do not overthink the slides.", "Remain calm and productive."])
        return f"{s.title}. {clean}. {tail}"
    tail = rng.choice(["Description teaches it.", "Silence is a valid response.", "If you are watching, you are participating."])
    return f"{s.title}. {clean}. {tail}"

def build_tts_track(rng: random.Random, slides: List[Slide], cfg: Dict[str, Any], dur_s: float, theme_key: str, tape_no: int) -> np.ndarray:
    sr = int(cfg["audio"]["sr"])
    voices = list(cfg["audio"].get("voices") or [])
    tts_speed = int(cfg["audio"]["tts_speed"])
    tts_pitch = int(cfg["audio"]["tts_pitch"])
    tts_amp = int(cfg["audio"]["tts_amp"])
    track = np.zeros(int(sr * dur_s), dtype=np.float32)
    work = Path(cfg["workdir"]) / "tts"
    work.mkdir(parents=True, exist_ok=True)
    t = 0.0
    for i, s in enumerate(slides):
        voice = rng.choice(voices) if voices else "en-us"
        narr = narration_for_slide(rng, s, theme_key, tape_no)
        speed_i = int(_clamp(tts_speed + rng.randint(-14, 12), 115, 190))
        pitch_i = int(_clamp(tts_pitch + rng.randint(-14, 16), 10, 80))
        amp_i = int(_clamp(tts_amp + rng.randint(-12, 10), 80, 200))
        wav_path = work / f"tts_{i:02d}.wav"
        try:
            tts_segment_espeak(narr, wav_path, voice, speed_i, pitch_i, amp_i)
            sr2, data = read_wav(str(wav_path))
            if data.ndim > 1: data = data.mean(axis=1)
            data = data.astype(np.float32)
            if sr2 != sr and len(data) > 8:
                x = np.linspace(0, 1, len(data), False)
                x2 = np.linspace(0, 1, int(len(data) * sr / sr2), False)
                data = np.interp(x2, x, data).astype(np.float32)
            data /= (np.max(np.abs(data)) + 1e-6)
            data = np.tanh(data * 1.8) * 0.75
            if rng.random() < 0.65:
                data = bitcrush(data, rng.choice([3,4,5,6]))
            ring_f = rng.choice([150, 180, 220, 260]) + rng.randint(-20, 20)
            ring = np.sin(2*np.pi*ring_f*np.linspace(0, len(data)/sr, len(data), False).astype(np.float32))
            data = (0.86*data + 0.14*data*ring).astype(np.float32)
            data += np.random.uniform(-1,1,len(data)).astype(np.float32) * 0.008
            mix_in(track, data, start_s=t + 0.25, sr=sr, gain=1.0)
        except Exception:
            pass
        t += s.seconds
    return track

# ----------------------------
# Transmission plan
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

# ----------------------------
# Image loading
# ----------------------------

def load_local_images(dirpath: Path, limit: int = 42) -> List[Image.Image]:
    if not dirpath.exists(): return []
    imgs: List[Image.Image] = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(dirpath.rglob("*")):
        if p.suffix.lower() not in exts: continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
        if len(imgs) >= limit: break
    return imgs

def split_faces_and_objects(imgs: List[Image.Image]) -> Tuple[List[Image.Image], List[Image.Image]]:
    faces, objs = [], []
    for im in imgs:
        w, h = im.size
        if h >= w and h > 220: faces.append(im)
        else: objs.append(im)
    if not faces: faces = imgs[:]
    if not objs: objs = imgs[:]
    return faces, objs

# ----------------------------
# Rendering
# ----------------------------

def render_video(rng: random.Random, cfg: Dict[str, Any], slides: List[Slide], out_mp4: Path, tape_no: int) -> Tuple[Path, float, int, List[float]]:
    r = cfg["render"]
    W, H, FPS = int(r["width"]), int(r["height"]), int(r["fps"])
    ctx = RenderContext(W=W, H=H, FPS=FPS, vhs_strength=float(r["vhs_strength"]),
                        redaction_strength=float(r["redaction_strength"]),
                        censor_probability=float(r["censor_probability"]),
                        entity_overlay_probability=float(r["entity_overlay_probability"]))
    
    total_frames = sum(max(1, int(s.seconds * FPS)) for s in slides)
    plan = plan_transmission(rng, cfg, total_frames, FPS)
    all_imgs = [im for s in slides for im in (s.bg_imgs + s.face_imgs) if im]
    pop_pool = [make_popup(rng, im) for im in rng.sample(all_imgs, k=min(len(all_imgs), 10))] if all_imgs else []
    
    popup_moments: List[int] = []
    if int(r["max_popups"]) > 0 and total_frames > 10:
        for _ in range(int(r["max_popups"])):
            popup_moments.append(rng.randint(int(total_frames*0.15), total_frames-1))
        popup_moments = sorted(set(popup_moments))
    
    flashes = max(0, int(r.get("flashes", 12)))
    flash_frames = {rng.randint(int(total_frames*0.10), max(int(total_frames*0.95), 1)) for _ in range(flashes)}
    event_times_s: List[float] = []
    popup_dur = max(1, int(float(r["popup_seconds"]) * FPS))
    
    out_silent = out_mp4.with_name(out_mp4.stem + "_silent.mp4")
    
    # --- REPLACED IMAGEIO WITH FFMPEG WRITER ---
    writer = FFmpegWriter(str(out_silent), W, H, FPS)
    
    frame_idx = 0
    popup_active_until = -1
    channel = rng.randint(1, 12)
    
    for slide in slides:
        nF = max(1, int(slide.seconds * FPS))
        ui = make_ui_layer(ctx, slide)
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
            
            if slide.face_imgs and (slide.kind in ("protocol", "intermission") or rng.random() < 0.22):
                face = rng.choice(slide.face_imgs)
                face_arr = face_uncanny(ctx, rng, face, t)
                alpha = 0.55 if slide.kind == "normal" else 0.78
                frame = np.clip(frame.astype(np.float32)*(1-alpha) + face_arr.astype(np.float32)*alpha, 0, 255).astype(np.uint8)
            
            frame = alpha_over(frame, ui)
            if slide.kind == "infographic" and rng.random() < 0.85:
                frame = draw_infographic(ctx, rng, frame)
            
            frame = censor_blocks(ctx, rng, frame)
            
            if pop_pool and rng.random() < float(r.get("micro_popup_probability", 0.07)):
                pop = rng.choice(pop_pool)
                tiny = pop[::2, ::2]
                if rng.random() < 0.30: tiny = 255 - tiny
                x = rng.randint(0, max(1, W - tiny.shape[1]))
                y = rng.randint(70, max(71, H - tiny.shape[0] - 70))
                frame[y:y+tiny.shape[0], x:x+tiny.shape[1]] = tiny
            
            if frame_idx in popup_moments:
                popup_active_until = frame_idx + popup_dur
                event_times_s.append(frame_idx / FPS)
            
            if frame_idx < popup_active_until and pop_pool:
                pop = rng.choice(pop_pool)
                if rng.random() < 0.30: pop = 255 - pop
                frame = stamp_popup(ctx, rng, frame, pop)
            
            if frame_idx in flash_frames and all_imgs:
                event_times_s.append(frame_idx / FPS)
                im = rng.choice(all_imgs)
                flash = np.array(cover_resize(im, W, H).convert("RGB"), dtype=np.uint8)
                flash = vhs_stack(ctx, rng, flash)
                if rng.random() < 0.60: flash = 255 - flash
                frame = flash
            
            if rng.random() < ctx.entity_overlay_probability:
                frame = lone_shooter_overlay(ctx, rng, frame, intensity=min(1.0, 0.5 + 0.5*rng.random()))
                event_times_s.append(frame_idx / FPS)
            
            if plan.freeze_at is not None and frame_idx == plan.freeze_at:
                event_times_s.append(frame_idx / FPS)
                freeze = frame.copy()
                for _ in range(plan.freeze_frames):
                    fr = vhs_stack(ctx, rng, freeze.copy())
                    fr = timecode_overlay(ctx, fr, frame_idx, channel, tape_no)
                    writer.write(fr)
                    frame_idx += 1
                continue
            
            frame = vhs_stack(ctx, rng, frame)
            if fi < 3 or fi > nF-4:
                if rng.random() < 0.60: frame = 255 - frame
            
            frame = timecode_overlay(ctx, frame, frame_idx, channel, tape_no)
            writer.write(frame)
            frame_idx += 1
            
        if plan.cut_frame is not None and frame_idx >= plan.cut_frame:
            break
            
    writer.close()
    return out_silent, frame_idx / FPS, frame_idx, event_times_s

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--out", type=str, default="out.mp4")
    args = ap.parse_args()

    # Early check for ffmpeg
    if not shutil.which("ffmpeg"):
        print("Error: 'ffmpeg' not found in PATH. Please install FFmpeg.")
        exit(1)

    cfg = load_config(Path(args.config))
    Path(cfg["workdir"]).mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    if seed == 1337: seed = random.randint(0, 2_000_000_000)
    rng = random.Random(seed)
    np.random.seed(seed)

    has_espeak = shutil.which("espeak") is not None

    theme_key = cfg.get("theme_key", "").strip()
    if cfg.get("theme_source", "random_online") != "fixed" or not theme_key:
        if cfg["web"]["enable"]:
            theme_key = choose_theme_key(rng, cfg)
        else:
            theme_key = rng.choice(["employee handbook", "paper clip", "surveillance camera", "keys"])

    tape_no = (abs(seed) % 90) + 10
    print(f"[theme] {theme_key}  | seed={seed} | tape={tape_no}")

    local_imgs = load_local_images(Path(cfg["local_images_dir"]), limit=42)
    web_imgs: List[Image.Image] = []
    scraped_text = ""

    if cfg["web"]["enable"]:
        timeout_s = int(cfg["web"]["timeout_s"])
        print(f"[web] scrape text for '{theme_key}'...")
        scraped_text = wiki_extract(theme_key, int(cfg["web"]["text_paragraphs"]), timeout_s)
        related = extract_related_terms(rng, theme_key, scraped_text, max_terms=8)
        q_templates = cfg["web"].get("query_expand") or ["{k}"]
        queries: List[str] = []
        for k in related[:6]:
            for tplt in q_templates[:8]:
                queries.append(str(tplt).format(k=k))
        rng.shuffle(queries)
        limit = int(cfg["web"]["image_limit"])
        urls: List[str] = []
        for q in queries[:10]:
            urls.extend(commons_images(q, limit=max(4, limit//2), timeout_s=timeout_s))
            if len(urls) >= limit * 3: break
        seen = set()
        dedup = []
        for u in urls:
            if u not in seen:
                seen.add(u); dedup.append(u)
        rng.shuffle(dedup)
        dedup = dedup[: max(8, limit)]
        print(f"[web] download {len(dedup)} images...")
        for u in dedup:
            im = download_image(u, timeout_s)
            if im: web_imgs.append(im)
        rng.shuffle(web_imgs)
        web_imgs = web_imgs[:22]

    faces, objs = split_faces_and_objects(web_imgs + local_imgs)
    slides = build_template_slides(rng, theme_key, scraped_text, objs, local_imgs, cfg, tape_no=tape_no)
    for s in slides:
        if s.kind in ("protocol", "intermission") and not s.face_imgs and faces:
            s.face_imgs = [rng.choice(faces)]

    out_mp4 = Path(args.out)
    print("[render] video...")
    out_silent, dur_s, _, event_times_s = render_video(rng, cfg, slides, out_mp4, tape_no=tape_no)

    print("[audio] building...")
    sr = int(cfg["audio"]["sr"])
    # Safe init for audio buffer
    audio = np.zeros(int(sr * (dur_s + 1.0)), dtype=np.float32)
    
    if cfg["audio"]["music"]:
        strength = float(cfg["audio"].get("jingle_strength", 1.0))
        jingle = gen_90s_jingle(rng, sr, dur_s, strength=strength) * 0.9
        mix_in(audio, jingle, 0, sr, 1.0)
        stingers = gen_noise_stingers(rng, sr, dur_s, int(cfg["audio"]["stinger_count"]), event_times_s) * 0.9
        mix_in(audio, stingers, 0, sr, 1.0)
        
    if cfg["audio"]["tts"] and has_espeak:
        tts = build_tts_track(rng, slides, cfg, dur_s, theme_key=theme_key, tape_no=tape_no) * 1.0
        mix_in(audio, tts, 0, sr, 1.0)
    elif cfg["audio"]["tts"] and not has_espeak:
        print("Warning: espeak not found; skipping TTS.")

    # Trim audio to exact video length
    audio = audio[:int(sr * dur_s)]
    audio /= (np.max(np.abs(audio)) + 1e-6)
    wav_path = out_mp4.with_suffix(".wav")
    write_wav(str(wav_path), sr, (audio * 32767).astype(np.int16))

    print("[mux] ffmpeg...")
    subprocess.run(
        ["ffmpeg","-y","-i", str(out_silent), "-i", str(wav_path),
         "-c:v","copy","-c:a","aac","-b:a","192k","-shortest", str(out_mp4)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    try:
        out_silent.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"Done: {out_mp4}")

if __name__ == "__main__":
    main()
