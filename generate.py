#!/usr/bin/env python3
"""
generate.py — "Found VHS Job Training" analogue-horror PSA presentation video generator.

Design goals (GitHub Actions friendly):
- Keeps the 90s PSA / corporate training presentation structure (slides, bullet points, charts).
- Adds "incursion into daily secret life" analogue-horror vibe inspired by your HTML prototypes:
  - CRT/terminal overlays, unstable signal warnings, forensic-style examinations.
- Stronger scraping: Wikipedia text + Wikimedia Commons images + RSS headlines (optional).
- Forensic zoom slides: random zoomed crops from scraped images, arrows, ominous annotations.
- Improved "realistic" charts: pie, barrel (cylindrical) bars, and a world map with random locations.
- VHS tech-spec intro slide (random tape metadata each run).
- Hidden "Entity" flash (~0.1s) with masked stutter.
- Renders to MP4 via ffmpeg piping (no imageio backends).

Usage:
  python generate.py --config config.yaml --out out.mp4

Env overrides (optional):
  SEED=<int>            Deterministic runs
  THEME_FORCE=<string>  Force theme keyword
  WEB_ENABLE=true/false Enable/disable web scraping (default true)
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
import feedparser
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy.io.wavfile import write as write_wav

# ----------------------------
# Defaults / Config
# ----------------------------

DEFAULT_CFG: Dict[str, Any] = {
    "seed": None,
    "web": {"enabled": True, "timeout": 25, "max_images": 18, "min_w": 720, "min_h": 540},
    "theme": {
        "fallback_keywords": [
            "office safety", "weather systems", "cathode ray tube", "telephone etiquette",
            "elevator procedures", "data entry", "inventory control", "fire drills",
            "computer lab", "public access terminal", "municipal archive"
        ],
        "query_expand": ["{k}", "{k} 1990s", "{k} training", "{k} surveillance", "{k} signage"],
        "max_keyword_len": 48,
    },
    "story": {
        "slide_count": 12,
        "include_intro_outro": True,
        "normal_ratio": 0.55,
        "forensic_slides_min": 2,
        "forensic_slides_max": 4,
        "chart_slides_min": 2,
        "chart_slides_max": 3,
        "interrupt_probability": 0.18,
        "micro_text_probability": 0.10,
        "entity_flash_probability": 0.14,
    },
    "render": {
        "width": 640,
        "height": 480,
        "fps": 15,
        "slide_seconds": 3.2,
        "vhs_strength": 1.35,
        "flashes": 10,
        "dropout_probability": 0.018,
        "tracking_strength": 0.95,
        "grain_strength": 0.90,
        "scanline_strength": 0.85,
    },
    "audio": {
        "sr": 44100,
        "enabled": True,
        "tts": True,
        "tts_voice": "en-us",
        "tts_wpm": 155,
        "music": True,
        "music_level": 0.22,
        "stingers": True,
    },
}

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
UA = "AnalogHorrorPPT/2.0 (GitHub Actions)"

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.reuters.com/rssFeed/topNews",
]

# ----------------------------
# Utilities
# ----------------------------

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x

def safe_int(x: Any, d: int) -> int:
    try:
        return int(x)
    except Exception:
        return d

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def now_like_90s(rng: random.Random) -> str:
    # random "archive" date between 1989-2003
    y = rng.randint(1989, 2003)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"

def pick_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

# ----------------------------
# FFmpeg Writer (raw frames -> mp4)
# ----------------------------

class FFmpegWriter:
    def __init__(self, filename: str, width: int, height: int, fps: int):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            self.filename,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, frame_rgb: np.ndarray):
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("FFmpegWriter not started.")
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        self.proc.stdin.write(frame_rgb.tobytes())

    def close(self):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        if self.proc:
            self.proc.wait(timeout=120)
        self.proc = None

# ----------------------------
# Web scraping (stronger)
# ----------------------------

def http_get(url: str, timeout: int) -> Optional[requests.Response]:
    try:
        return requests.get(url, timeout=timeout, headers={"User-Agent": UA})
    except Exception:
        return None

def pick_theme_keyword(cfg: Dict[str, Any], rng: random.Random) -> str:
    forced = os.getenv("THEME_FORCE", "").strip()
    if forced:
        return forced[: cfg["theme"]["max_keyword_len"]]

    # If web enabled, try Wikipedia "random" title via REST redirect trick:
    # https://en.wikipedia.org/api/rest_v1/page/random/summary
    if cfg["web"]["enabled"]:
        r = http_get("https://en.wikipedia.org/api/rest_v1/page/random/summary", cfg["web"]["timeout"])
        if r and r.ok:
            try:
                j = r.json()
                title = (j.get("title") or "").strip()
                if title:
                    title = re.sub(r"\s+", " ", title)
                    # keep it plausible
                    title = re.sub(r"\(.*?\)", "", title).strip()
                    if 3 <= len(title) <= cfg["theme"]["max_keyword_len"]:
                        return title
            except Exception:
                pass

    return rng.choice(cfg["theme"]["fallback_keywords"])

def fetch_wikipedia_summary(title: str, timeout: int) -> str:
    safe = requests.utils.quote(title.replace(" ", "_"))
    r = http_get(WIKI_API + safe, timeout)
    if not r or not r.ok:
        return ""
    try:
        j = r.json()
        txt = (j.get("extract") or "").strip()
        txt = re.sub(r"\s+", " ", txt)
        return txt
    except Exception:
        return ""

def commons_search_images(query: str, timeout: int, limit: int, thumb_w: int = 1600) -> List[Dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "thumburl|url|size|mime",
        "iiurlwidth": str(thumb_w),
        "origin": "*",
    }
    try:
        r = requests.get(COMMONS_API, params=params, timeout=timeout, headers={"User-Agent": UA})
        r.raise_for_status()
        j = r.json()
        pages = (j.get("query") or {}).get("pages") or {}
        out = []
        for _, p in pages.items():
            iis = p.get("imageinfo") or []
            if not iis:
                continue
            ii = iis[0]
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            out.append({
                "title": p.get("title") or "",
                "url": url,
                "width": safe_int(ii.get("width"), 0),
                "height": safe_int(ii.get("height"), 0),
                "mime": ii.get("mime") or "",
            })
        return out
    except Exception:
        return []

def download_images(
    queries: List[str],
    out_dir: Path,
    timeout: int,
    max_images: int,
    min_w: int,
    min_h: int,
    rng: random.Random
) -> List[Path]:
    mkdirp(out_dir)
    images: List[Path] = []
    seen = set()

    # Progressive relax of min size if needed
    relax_steps = [(min_w, min_h), (640, 480), (0, 0)]
    for (mw, mh) in relax_steps:
        for q in queries:
            if len(images) >= max_images:
                return images
            results = commons_search_images(q, timeout=timeout, limit=max_images * 4, thumb_w=1600)
            rng.shuffle(results)

            for it in results:
                if len(images) >= max_images:
                    return images
                url = it["url"]
                if url in seen:
                    continue
                seen.add(url)

                w, h = it.get("width", 0), it.get("height", 0)
                if w and h and (w < mw or h < mh):
                    continue

                ext = os.path.splitext(url.split("?")[0])[1].lower()
                if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                    ext = ".jpg"

                fn = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", (it.get("title") or sha1(url)))[:120].strip("_")
                if not fn:
                    fn = sha1(url)
                path = out_dir / (fn + ext)

                try:
                    rr = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
                    rr.raise_for_status()
                    if len(rr.content) < 12_000:
                        continue
                    path.write_bytes(rr.content)
                    images.append(path)
                except Exception:
                    continue

                time.sleep(0.08 + rng.random() * 0.12)

        if len(images) >= max(6, max_images // 3):
            break

    return images

def fetch_rss_headline(rng: random.Random) -> str:
    feeds = RSS_FEEDS[:]
    rng.shuffle(feeds)
    for f in feeds:
        try:
            d = feedparser.parse(f)
            if d.entries:
                e = rng.choice(d.entries[: min(25, len(d.entries))])
                t = (getattr(e, "title", "") or "").strip()
                if t:
                    t = re.sub(r"\s+", " ", t)
                    return t[:140]
        except Exception:
            continue
    return "Local services resume after minor disruption."

# ----------------------------
# Visual style: 90s PSA + incursion overlays
# ----------------------------

def make_slide_canvas(w: int, h: int, bg: Tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (w, h), bg)
    return img

def draw_psa_header(d: ImageDraw.ImageDraw, w: int, title: str, rng: random.Random):
    font = pick_font(26)
    d.rectangle([0, 0, w, 46], fill=(8, 10, 28))
    d.text((14, 10), title.upper()[:42], font=font, fill=(235, 235, 235))
    # right-side "VTR" badge
    d.rectangle([w - 120, 8, w - 12, 38], fill=(20, 24, 64), outline=(120, 120, 160))
    d.text((w - 108, 12), rng.choice(["VTR", "PSA", "HR", "OPS", "SAFETY"]), font=pick_font(18), fill=(220, 220, 220))

def draw_footer(d: ImageDraw.ImageDraw, w: int, h: int, rng: random.Random, tape_id: str, date: str):
    d.rectangle([0, h - 32, w, h], fill=(8, 10, 28))
    font = pick_font(16)
    left = f"TAPE: {tape_id}   DATE: {date}"
    right = rng.choice(["SP", "LP", "EP"]) + "  TRACKING: " + str(rng.randint(-3, 3))
    d.text((12, h - 25), left, font=font, fill=(210, 210, 210))
    tw = d.textlength(right, font=font)
    d.text((w - tw - 12, h - 25), right, font=font, fill=(210, 210, 210))

def draw_terminal_incursion(d: ImageDraw.ImageDraw, w: int, h: int, rng: random.Random, intensity: float):
    """
    Subtle overlay like the HTML 'TERMINAL_ERROR // NULL_SKY' vibe:
    green monospace lines + red shadow artifacts.
    """
    if rng.random() > intensity:
        return
    font = pick_font(14)
    lines = rng.randint(3, 7)
    x0, y0 = rng.randint(10, 30), rng.randint(60, h - 120)
    for i in range(lines):
        msg = rng.choice([
            "SIGNAL_LEAK :: 0xDEAD",
            "PROTOCOL_9 :: ACTIVE",
            "EYES_SHUT :: TRUE",
            "GLASS_WARNING :: PENDING",
            "TRACKING :: ADJUST",
            "NULL_SKY :: DETECTED",
            "CACHE :: SPOILED",
        ])
        msg = msg + f"  [{rng.randint(100,999)}]"
        y = y0 + i * 16
        # red shadow
        d.text((x0 + 2, y + 1), msg, font=font, fill=(180, 0, 0))
        d.text((x0, y), msg, font=font, fill=(0, 230, 120))

def vignette(img: Image.Image, amount: float) -> Image.Image:
    if amount <= 0:
        return img
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    d = ((xx - cx) ** 2 + (yy - cy) ** 2) / (max(w, h) ** 2)
    v = 1.0 - amount * (d * 3.0)
    v = np.clip(v, 0.55, 1.0)
    arr *= v[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

# ----------------------------
# Forensic zoom slide
# ----------------------------

def draw_arrow(d: ImageDraw.ImageDraw, p0: Tuple[int,int], p1: Tuple[int,int], color=(255,255,255)):
    d.line([p0, p1], fill=color, width=3)
    # arrow head
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    ang = math.atan2(dy, dx)
    a1 = ang + math.radians(160)
    a2 = ang - math.radians(160)
    L = 14
    h1 = (p1[0] + int(math.cos(a1)*L), p1[1] + int(math.sin(a1)*L))
    h2 = (p1[0] + int(math.cos(a2)*L), p1[1] + int(math.sin(a2)*L))
    d.polygon([p1, h1, h2], fill=color)

def make_forensic_slide(
    w: int, h: int,
    img: Image.Image,
    theme: str,
    rng: random.Random,
    tape_id: str,
    date: str
) -> Image.Image:
    bg = make_slide_canvas(w, h, (8, 10, 28))
    d = ImageDraw.Draw(bg)

    draw_psa_header(d, w, f"FIELD REVIEW // {theme}", rng)

    # Fit image
    main_rect = (22, 64, w - 22, h - 54)
    mw = main_rect[2] - main_rect[0]
    mh = main_rect[3] - main_rect[1]

    img2 = ImageOps.fit(img.convert("RGB"), (mw, mh), method=Image.Resampling.LANCZOS)
    bg.paste(img2, (main_rect[0], main_rect[1]))

    # Choose zoom region
    zw, zh = int(mw * 0.28), int(mh * 0.28)
    zw = max(80, min(zw, mw - 2))
    zh = max(80, min(zh, mh - 2))

    zx0 = rng.randint(0, max(0, mw - zw - 1))
    zy0 = rng.randint(0, max(0, mh - zh - 1))

    crop = img2.crop((zx0, zy0, zx0 + zw, zy0 + zh))
    crop = crop.resize((zw * 2, zh * 2), Image.Resampling.NEAREST).filter(
        ImageFilter.UnsharpMask(radius=2, percent=140)
    )

    # Place inset (robust: clamp ranges so we never call randint on an empty range)
    inset_w, inset_h = crop.size

    # Prefer the right side, but fall back to anywhere that fits
    pref_min_x = int(w * 0.52)
    max_x = w - inset_w - 28
    if max_x < 28:
        # If the inset is too wide, shrink it to fit
        inset_w = max(120, w - 56)
        inset_h = max(90, int(inset_w * (crop.size[1] / max(1, crop.size[0]))))
        inset_h = min(inset_h, h - 120)
        crop = crop.resize((inset_w, inset_h), Image.Resampling.NEAREST)
        max_x = w - inset_w - 28

    min_x = pref_min_x if max_x >= pref_min_x else 28
    if max_x < min_x:
        inset_x = max(28, max_x)
    else:
        inset_x = rng.randint(min_x, max_x)

    # Y range also must fit
    y_min = int(h * 0.22)
    y_max = min(int(h * 0.55), h - inset_h - 64)
    if y_max < y_min:
        y_min = max(64, h - inset_h - 64)
        y_max = max(y_min, y_max)
    inset_y = rng.randint(y_min, y_max) if y_max >= y_min else y_min

    bg.paste(crop, (inset_x, inset_y))

    # Inset border + label
    d.rectangle([inset_x - 2, inset_y - 2, inset_x + inset_w + 2, inset_y + inset_h + 2], outline=(255,255,255), width=2)
    lbl = rng.choice(["ANOMALOUS DETAIL", "UNREGISTERED MOTION", "REFLECTION ARTIFACT", "SUBJECT TRACE", "INCONSISTENT SHADOW"])
    d.text((inset_x, inset_y - 22), lbl, font=pick_font(16), fill=(255,255,255))

    # Arrow from inset to original location
    src_pt = (main_rect[0] + zx0 + zw//2, main_rect[1] + zy0 + zh//2)
    dst_pt = (inset_x + inset_w//2, inset_y + inset_h//2)
    draw_arrow(d, dst_pt, src_pt, color=(255,255,255))

    # Ominous explanatory text
    note = rng.choice([
        "The weather is fine. Do not address the reflection.",
        "Unexplained occlusion persists across frames.",
        "Subject appears only during tracking instability.",
        "This is routine. Continue training.",
        "Evidence suggests an observer near the glass.",
    ])
    d.rectangle([28, int(h*0.66), w-28, h-64], fill=(0,0,0), outline=(110,110,140))
    d.text((40, int(h*0.675)), note, font=pick_font(18), fill=(230,230,230))

    draw_terminal_incursion(d, w, h, rng, intensity=0.85)
    draw_footer(d, w, h, rng, tape_id, date)
    return vignette(bg, 0.35)

# ----------------------------
# Charts (pie, barrel, map)
# ----------------------------

def draw_axes(d: ImageDraw.ImageDraw, rect: Tuple[int,int,int,int], ticks: int = 5):
    x0,y0,x1,y1 = rect
    d.rectangle([x0,y0,x1,y1], outline=(200,200,200))
    # grid + ticks
    for i in range(1, ticks):
        yy = y0 + int((y1-y0)*i/ticks)
        d.line([(x0, yy), (x1, yy)], fill=(70,70,90))
    for i in range(1, ticks):
        xx = x0 + int((x1-x0)*i/ticks)
        d.line([(xx, y0), (xx, y1)], fill=(50,50,70))

def chart_pie(img: Image.Image, rect: Tuple[int,int,int,int], rng: random.Random, labels: List[str]):
    d = ImageDraw.Draw(img)
    x0,y0,x1,y1=rect
    cx, cy = (x0+x1)//2, (y0+y1)//2
    R = min(x1-x0, y1-y0)//2 - 10

    vals = np.array([rng.random() for _ in range(len(labels))], dtype=float)
    vals = vals / vals.sum()
    angles = (vals * 360.0).tolist()

    start = rng.uniform(0, 360)
    colors = [(220,220,220),(160,160,190),(120,140,170),(180,120,120),(140,180,140)]
    colors = [colors[i % len(colors)] for i in range(len(labels))]

    # draw pseudo 3D depth (shadow layers)
    depth = 10
    for dz in range(depth, 0, -1):
        a = start
        for ang, col in zip(angles, colors):
            d.pieslice([cx-R, cy-R+dz, cx+R, cy+R+dz], a, a+ang, fill=tuple(int(c*0.55) for c in col))
            a += ang

    a = start
    for ang, col in zip(angles, colors):
        d.pieslice([cx-R, cy-R, cx+R, cy+R], a, a+ang, fill=col, outline=(10,10,20))
        a += ang

    # legend
    font = pick_font(16)
    lx, ly = x1 + 16, y0 + 10
    for i, (lab, v) in enumerate(zip(labels, vals)):
        box = [lx, ly + i*22, lx+14, ly + i*22 + 14]
        d.rectangle(box, fill=colors[i], outline=(0,0,0))
        d.text((lx+20, ly + i*22 - 1), f"{lab[:18]}  {int(v*100):02d}%", font=font, fill=(235,235,235))

def chart_barrel(img: Image.Image, rect: Tuple[int,int,int,int], rng: random.Random, labels: List[str]):
    """
    "Barrel chart": cylindrical bars with top ellipse shading.
    """
    d = ImageDraw.Draw(img)
    x0,y0,x1,y1=rect
    n = len(labels)
    vals = [rng.uniform(0.25, 1.0) for _ in range(n)]
    maxv = max(vals)
    bar_w = int((x1-x0) / (n*1.4))
    gap = int(bar_w*0.4)
    base_y = y1

    font = pick_font(14)
    for i,(lab,v) in enumerate(zip(labels, vals)):
        h = int((y1-y0) * (v/maxv))
        bx = x0 + i*(bar_w+gap) + 10
        by0 = base_y - h

        col = rng.choice([(120,150,190),(160,160,200),(140,120,160),(170,140,120)])
        dark = tuple(int(c*0.55) for c in col)

        # body
        d.rectangle([bx, by0, bx+bar_w, base_y], fill=dark, outline=(0,0,0))
        # highlight strip
        d.rectangle([bx+2, by0+2, bx+int(bar_w*0.35), base_y-2], fill=col)

        # top ellipse
        ell_h = max(8, int(bar_w*0.35))
        d.ellipse([bx, by0-ell_h//2, bx+bar_w, by0+ell_h//2], fill=col, outline=(0,0,0))
        # bottom ellipse shadow
        d.ellipse([bx, base_y-ell_h//2, bx+bar_w, base_y+ell_h//2], fill=dark, outline=(0,0,0))

        # label
        d.text((bx, base_y+8), lab[:8], font=font, fill=(235,235,235))

    # axes / grid (behind the bars feel)
    d.rectangle([x0,y0,x1,y1], outline=(200,200,200))

def chart_map(img: Image.Image, rect: Tuple[int,int,int,int], rng: random.Random, points: int = 9):
    """
    Simple world map: equirect projection with random "incursion" points and lines.
    No external geo libs; draws a stylized world silhouette + dots.
    """
    d = ImageDraw.Draw(img)
    x0,y0,x1,y1=rect
    W = x1-x0
    H = y1-y0

    # background
    d.rectangle([x0,y0,x1,y1], fill=(10,12,24), outline=(200,200,200))
    # pseudo landmasses (stylized blobs)
    def blob(cx, cy, rx, ry, col):
        d.ellipse([x0+cx-rx, y0+cy-ry, x0+cx+rx, y0+cy+ry], fill=col)

    land = (28, 35, 55)
    blob(int(W*0.26), int(H*0.45), int(W*0.18), int(H*0.22), land)  # americas-ish
    blob(int(W*0.62), int(H*0.42), int(W*0.22), int(H*0.18), land)  # eurasia-ish
    blob(int(W*0.70), int(H*0.70), int(W*0.10), int(H*0.12), land)  # africa-ish
    blob(int(W*0.82), int(H*0.78), int(W*0.08), int(H*0.06), land)  # aus-ish

    # grid lines
    for i in range(1, 6):
        yy = y0 + int(H*i/6)
        d.line([(x0, yy),(x1, yy)], fill=(20,24,40))
    for i in range(1, 10):
        xx = x0 + int(W*i/10)
        d.line([(xx, y0),(xx, y1)], fill=(18,22,36))

    # points and connecting lines
    pts=[]
    for _ in range(points):
        px = x0 + rng.randint(10, W-10)
        py = y0 + rng.randint(10, H-10)
        pts.append((px,py))

    # connect a few
    rng.shuffle(pts)
    for i in range(min(len(pts)-1, rng.randint(4, 7))):
        d.line([pts[i], pts[i+1]], fill=(140, 0, 0), width=2)

    for (px,py) in pts:
        d.ellipse([px-3,py-3,px+3,py+3], fill=(220, 30, 30))
        if rng.random() < 0.35:
            d.ellipse([px-8,py-8,px+8,py+8], outline=(0, 230, 120), width=1)

    font = pick_font(16)
    d.text((x0+10, y1-24), "INCIDENT PINGS (UNVERIFIED)", font=font, fill=(235,235,235))

def make_chart_slide(
    w: int, h: int,
    theme: str,
    rng: random.Random,
    tape_id: str,
    date: str
) -> Image.Image:
    img = make_slide_canvas(w, h, (8, 10, 28))
    d = ImageDraw.Draw(img)
    draw_psa_header(d, w, f"ANALYSIS REPORT // {theme}", rng)

    # frame for chart
    chart_rect = (40, 76, int(w*0.70), int(h*0.78))
    draw_axes(d, chart_rect, ticks=6)

    # legend / notes area
    notes_rect = (int(w*0.72), 76, w-40, int(h*0.78))
    d.rectangle(notes_rect, fill=(0,0,0), outline=(110,110,140))

    chart_type = rng.choice(["pie", "barrel", "map"])
    labels = [
        rng.choice(["COMPLIANCE", "ANOMALY", "LATENCY", "EXPOSURE", "REDACTION", "ECHO"]),
        rng.choice(["HUMIDITY", "VOLTAGE", "STAFF", "ACCESS", "NOISE", "FOG"]),
        rng.choice(["WITNESS", "CAMERA", "GLASS", "SIGNAL", "HALLWAY", "TAPE"]),
        rng.choice(["SHIFT A", "SHIFT B", "SHIFT C", "UNKNOWN", "ARCHIVE"]),
    ]
    labels = list(dict.fromkeys(labels))[: rng.randint(3, 5)]

    if chart_type == "pie":
        chart_pie(img, chart_rect, rng, labels)
        insight = rng.choice([
            "Distribution suggests routine behavior with a residual anomaly tail.",
            "Spike indicates observer-pattern clustering near reflective surfaces.",
            "Variance aligns with unplanned hallway repetition events.",
        ])
    elif chart_type == "barrel":
        chart_barrel(img, chart_rect, rng, labels)
        insight = rng.choice([
            "Cylinder bars reflect cyclical recurrence across shifts and rooms.",
            "A single category persists beyond expected tape degradation.",
            "Nonlinear uplift observed during TRACKING instability.",
        ])
    else:
        chart_map(img, chart_rect, rng, points=rng.randint(7, 12))
        insight = rng.choice([
            "Locations are unrelated. The pattern is the point.",
            "Signals appear where documentation is thin.",
            "The map is not a map. It's a memory index.",
        ])

    # creepy, realistic notes
    font = pick_font(16)
    d.text((notes_rect[0]+10, notes_rect[1]+10), "SUMMARY:", font=font, fill=(235,235,235))
    wrapped = textwrap.fill(insight, width=18)
    d.text((notes_rect[0]+10, notes_rect[1]+34), wrapped, font=pick_font(14), fill=(220,220,220))

    d.text((notes_rect[0]+10, notes_rect[1]+130), "RECOMMENDATION:", font=font, fill=(235,235,235))
    rec = rng.choice([
        "Continue training. Do not report side-channel audio.",
        "Increase TRACKING by +2 and avoid glass surfaces.",
        "Mark tape as SPARE. File under 'WEATHER'.",
        "Do not rewind past the first warning tone.",
    ])
    d.text((notes_rect[0]+10, notes_rect[1]+154), textwrap.fill(rec, width=18), font=pick_font(14), fill=(0, 230, 120))

    draw_terminal_incursion(d, w, h, rng, intensity=0.90)
    draw_footer(d, w, h, rng, tape_id, date)
    return vignette(img, 0.35)

# ----------------------------
# VHS tech spec intro slide
# ----------------------------

def make_vhs_spec_slide(w: int, h: int, theme: str, rng: random.Random, tape_id: str, date: str) -> Image.Image:
    img = make_slide_canvas(w, h, (0, 0, 0))
    d = ImageDraw.Draw(img)

    # big title like HTML "UNSTABLE SIGNAL"
    d.text((w//2, int(h*0.22)), "UNSTABLE SIGNAL", font=pick_font(40), fill=(255,255,255), anchor="mm")
    d.text((w//2, int(h*0.30)), "[ DAILY TRAINING FEED ]", font=pick_font(18), fill=(200,200,200), anchor="mm")

    # tech blocks
    font = pick_font(16)
    blocks = [
        ("FORMAT", rng.choice(["VHS", "S-VHS", "Hi8", "U-MATIC"])),
        ("MODE", rng.choice(["SP", "LP", "EP"])),
        ("TRACKING", str(rng.randint(-4, 4))),
        ("HEAD DRIFT", f"{rng.uniform(0.2, 3.9):.1f} ms"),
        ("CHROMA", rng.choice(["LOW", "MED", "UNSTABLE"])),
        ("AUDIO", rng.choice(["MONO", "HI-FI", "MUTED"])),
        ("CRC", rng.choice(["PASS", "WARN", "FAIL"])),
        ("CASE ID", tape_id),
        ("DATE", date),
        ("TOPIC", theme[:32].upper()),
    ]
    x0, y0 = int(w*0.18), int(h*0.40)
    for i, (k, v) in enumerate(blocks):
        y = y0 + i * 22
        d.text((x0, y), f"{k:10s}: {v}", font=font, fill=(0, 230, 120))
        # red glitch shadow
        if rng.random() < 0.35:
            d.text((x0+1, y+1), f"{k:10s}: {v}", font=font, fill=(180, 0, 0))

    # bottom hint
    d.text((w//2, int(h*0.92)), "CLICK TO TUNE FREQUENCY (ARCHIVED)", font=pick_font(14), fill=(120,120,120), anchor="mm")
    return vignette(img, 0.55)

# ----------------------------
# Basic PSA slide (bullets + image)
# ----------------------------

def make_normal_slide(w: int, h: int, theme: str, rng: random.Random, tape_id: str, date: str, img: Optional[Image.Image], body: str) -> Image.Image:
    bg = make_slide_canvas(w, h, (8, 10, 28))
    d = ImageDraw.Draw(bg)
    draw_psa_header(d, w, f"TRAINING MODULE // {theme}", rng)

    # image panel
    panel = (26, 70, int(w*0.48), h - 54)
    d.rectangle(panel, fill=(0,0,0), outline=(110,110,140))
    if img is not None:
        pic = ImageOps.fit(img.convert("RGB"), (panel[2]-panel[0]-6, panel[3]-panel[1]-6), method=Image.Resampling.LANCZOS)
        bg.paste(pic, (panel[0]+3, panel[1]+3))
        # subtle label
        d.text((panel[0]+10, panel[1]+10), rng.choice(["FIELD", "ARCHIVE", "TRAINING STILL", "REFERENCE"]), font=pick_font(14), fill=(255,255,255))

    # text panel
    tx = int(w*0.52)
    d.rectangle([tx, 70, w-26, h-54], fill=(0,0,0), outline=(110,110,140))
    font = pick_font(18)
    lines = textwrap.wrap(body, width=28)
    y = 84
    for ln in lines[:10]:
        d.text((tx+12, y), "• " + ln, font=font, fill=(235,235,235))
        y += 24

    # occasional "interrupt" red line
    if rng.random() < 0.22:
        warn = rng.choice(["DO NOT LOOK AT GLASS", "PROTOCOL 9 ACTIVE", "EYES SHUT", "TRACKING LOST"])
        d.text((tx+12, h-86), warn, font=pick_font(18), fill=(220,30,30))

    draw_terminal_incursion(d, w, h, rng, intensity=0.75)
    draw_footer(d, w, h, rng, tape_id, date)
    return vignette(bg, 0.35)

def make_interrupt_slide(w: int, h: int, rng: random.Random, tape_id: str, date: str) -> Image.Image:
    img = make_slide_canvas(w, h, (0,0,0))
    d = ImageDraw.Draw(img)
    d.text((w//2, int(h*0.30)), "PSA INTERRUPT", font=pick_font(44), fill=(220,30,30), anchor="mm")
    msg = rng.choice([
        "THE WEATHER IS FINE. [REDACTED] IS WATCHING.",
        "STAY AWAY FROM GLASS SURFACES.",
        "REPORT ANY SMELL OF OZONE TO MANAGEMENT.",
        "DO NOT REWIND BEYOND THE TONE.",
        "PROTOCOL 9 HAS BEEN ENABLED.",
    ])
    d.text((w//2, int(h*0.44)), msg, font=pick_font(18), fill=(235,235,235), anchor="mm")
    # fake scanline bands
    for y in range(0, h, 3):
        if rng.random() < 0.55:
            d.line([(0,y),(w,y)], fill=(10,10,10))
    draw_footer(d, w, h, rng, tape_id, date)
    return vignette(img, 0.6)

# ----------------------------
# VHS post-processing per frame
# ----------------------------

def vhs_process(frame_rgb: np.ndarray, rng: random.Random, t: float, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Apply analog degradation: chroma shift, scanlines, grain, tracking bands, dropouts, jitter.
    """
    h, w = frame_rgb.shape[:2]
    arr = frame_rgb.astype(np.int16)

    strength = float(cfg["render"]["vhs_strength"])
    scan = float(cfg["render"]["scanline_strength"]) * strength
    grain = float(cfg["render"]["grain_strength"]) * strength
    tracking = float(cfg["render"]["tracking_strength"]) * strength
    dropout_p = float(cfg["render"]["dropout_probability"])

    # jitter (whole frame)
    if rng.random() < 0.12 * strength:
        dx = rng.randint(-3, 3)
        dy = rng.randint(-1, 1)
        arr = np.roll(arr, dx, axis=1)
        arr = np.roll(arr, dy, axis=0)

    # chroma shift (RGB channel offsets)
    if rng.random() < 0.85:
        sx = rng.randint(1, 2 + int(2*strength))
        r = np.roll(arr[:,:,0], sx, axis=1)
        b = np.roll(arr[:,:,2], -sx, axis=1)
        arr[:,:,0] = r
        arr[:,:,2] = b

    # tracking bands (horizontal slips)
    if rng.random() < 0.85:
        bands = 1 + int(tracking * 4)
        for _ in range(bands):
            bh = rng.randint(max(4, int(h*0.02)), max(8, int(h*0.10)))
            y0 = rng.randint(0, max(0, h-bh-1))
            off = int(math.sin(t*(1.5+rng.random()*2.5)+rng.random()*6.28) * (8 + 40*tracking) + rng.randint(-4,4))
            for yy in range(y0, y0+bh):
                row_off = int(off*(0.4 + 0.6*(yy-y0)/max(1,bh-1)))
                arr[yy,:,:] = np.roll(arr[yy,:,:], row_off, axis=0)
            # bright tracking line
            if rng.random() < 0.55:
                ly = min(h-1, y0 + rng.randint(0, bh-1))
                arr[ly:ly+1,:,:] = np.clip(arr[ly:ly+1,:,:] + 60, 0, 255)

    # scanlines
    if scan > 0:
        mask = ((np.arange(h) % 2) == 0).astype(np.int16)[:, None, None]
        arr = arr - (mask * int(18 * scan))

    # grain
    if grain > 0:
        n = np.random.normal(0, 10 + 20*grain, (h,w,1)).astype(np.int16)
        arr = arr + n

    # dropouts: black/green rectangles
    if rng.random() < dropout_p:
        out = arr.copy()
        green = rng.random() < 0.55
        col = np.array([0,255,0], dtype=np.int16) if green else np.array([0,0,0], dtype=np.int16)
        for _ in range(rng.randint(1,3)):
            x0 = rng.randint(0, w-1)
            y0 = rng.randint(0, h-1)
            ww = rng.randint(int(w*0.05), int(w*0.35))
            hh = rng.randint(int(h*0.02), int(h*0.20))
            x1 = min(w, x0+ww); y1=min(h,y0+hh)
            out[y0:y1, x0:x1, :] = col
        arr = out

    return np.clip(arr, 0, 255).astype(np.uint8)

def entity_flash(frame_rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Add imperceptible 'entity' eyes with masked stutter.
    """
    h,w = frame_rgb.shape[:2]
    arr = frame_rgb.astype(np.float32)
    # random mask
    mask = np.zeros((h,w), dtype=np.float32)
    for _ in range(rng.randint(3,7)):
        cx = rng.randint(0,w-1)
        cy = rng.randint(0,h-1)
        rx = rng.randint(int(w*0.03), int(w*0.12))
        ry = rng.randint(int(h*0.03), int(h*0.12))
        yy,xx=np.ogrid[:h,:w]
        blob = ((xx-cx)**2/(rx*rx+1e-6) + (yy-cy)**2/(ry*ry+1e-6)) <= 1.0
        mask[blob] = np.maximum(mask[blob], rng.uniform(0.5, 1.0))
    # blur mask
    mask = cv2_gaussian(mask, k=rng.choice([7,9,11]))
    mask = (mask / (mask.max()+1e-6)) * rng.uniform(0.10, 0.18)

    # draw eyes on a dark overlay
    overlay = np.zeros_like(arr)
    ex = w//2 + rng.randint(-40,40)
    ey = int(h*0.45) + rng.randint(-20,20)
    overlay[ey:ey+2, ex-28:ex-26, :] = 255
    overlay[ey:ey+2, ex+26:ex+28, :] = 255

    out = arr*(1-mask[:,:,None]) + overlay*mask[:,:,None]
    return np.clip(out,0,255).astype(np.uint8)

def cv2_gaussian(mask: np.ndarray, k: int) -> np.ndarray:
    # small gaussian blur without importing cv2 (keep deps minimal)
    # separable kernel
    k = int(k)
    if k < 3:
        return mask
    # approximate gaussian kernel
    ax = np.linspace(-(k-1)/2., (k-1)/2., k)
    kernel = np.exp(-0.5*np.square(ax)/(0.35*k)**2)
    kernel = kernel / kernel.sum()
    # convolve
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=mask)
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=tmp)
    return out

# ----------------------------
# Audio assembly (simple)
# ----------------------------

def synth_audio(cfg: Dict[str, Any], rng: random.Random, duration_s: float, out_wav_path: Path, narration: str):
    sr = int(cfg["audio"]["sr"])
    n = int(duration_s * sr)

    t = np.arange(n, dtype=np.float32) / sr

    # drone base
    base_f = rng.choice([33.0, 36.0, 40.0, 44.0])
    drone = 0.12*np.sin(2*np.pi*base_f*t) + 0.05*np.sin(2*np.pi*(base_f*2.0)*t)
    drone += 0.03*np.sin(2*np.pi*(base_f*0.5)*t)

    # noise bed
    noise = (np.random.normal(0, 1, n).astype(np.float32))
    # crude low-pass via moving average
    k = 900
    noise_lp = np.convolve(noise, np.ones(k, dtype=np.float32)/k, mode="same")
    bed = 0.04*noise_lp

    # "training jingle" blips
    j = np.zeros_like(t)
    if cfg["audio"]["music"]:
        for beat in range(0, int(duration_s*2)):
            if rng.random() < 0.65:
                f = rng.choice([220, 330, 440, 550])
                start = int((beat*0.5 + rng.uniform(0,0.1))*sr)
                end = min(n, start + int(0.12*sr))
                tt = np.arange(end-start)/sr
                env = np.exp(-tt*18.0)
                j[start:end] += 0.12*np.sin(2*np.pi*f*tt)*env

    audio = drone + bed + j

    # stingers
    if cfg["audio"]["stingers"]:
        for _ in range(rng.randint(2, 5)):
            st = rng.uniform(5, duration_s-2)
            start=int(st*sr)
            end=min(n, start+int(rng.uniform(0.08,0.20)*sr))
            tt=np.arange(end-start)/sr
            sweep = np.sin(2*np.pi*(rng.uniform(30,80))*tt)
            audio[start:end] += 0.35*sweep*np.exp(-tt*30)

    # Normalize
    mx = float(np.max(np.abs(audio)) + 1e-6)
    audio = (audio / mx) * 0.5

    # Optional TTS via espeak-ng -> mix in
    if cfg["audio"]["tts"]:
        tts_wav = out_wav_path.with_name("tts.wav")
        ok = espeak_tts(narration, str(tts_wav), voice=str(cfg["audio"]["tts_voice"]), wpm=int(cfg["audio"]["tts_wpm"]))
        if ok:
            tts_audio, tts_sr = read_wav_mono(str(tts_wav))
            if tts_sr != sr:
                tts_audio = resample_linear(tts_audio, tts_sr, sr)
            # place at ~8s
            off = int(8*sr)
            L = min(len(tts_audio), n-off)
            if L > 0:
                audio[off:off+L] = np.clip(audio[off:off+L] + 0.35*tts_audio[:L], -1.0, 1.0)

    write_wav(str(out_wav_path), sr, (audio * 32767).astype(np.int16))

def espeak_tts(text: str, out_wav: str, voice: str, wpm: int) -> bool:
    try:
        # sanitize
        text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\:\;\!\?\[\]\(\)]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return False
        cmd = ["espeak-ng", "-v", voice, "-s", str(wpm), "-w", out_wav, text]
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        return p.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 1000
    except Exception:
        return False

def read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    import wave
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        data = wf.readframes(n)
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            arr = arr.reshape(-1, ch).mean(axis=1)
        return arr, sr

def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) < 2:
        return x
    ratio = sr_out / sr_in
    n_out = int(len(x) * ratio)
    xp = np.linspace(0, 1, len(x))
    xq = np.linspace(0, 1, n_out)
    return np.interp(xq, xp, x).astype(np.float32)

# ----------------------------
# Slide plan / rendering
# ----------------------------

def build_slide_bodies(theme: str, wiki_text: str, headline: str, rng: random.Random, count: int) -> List[str]:
    """
    Produce bullet-ish bodies, mixing Wikipedia, headline, and procedural text.
    """
    bodies: List[str] = []
    facts = []
    if wiki_text:
        # split sentences
        sents = re.split(r"(?<=[\.\!\?])\s+", wiki_text)
        sents = [s.strip() for s in sents if 20 <= len(s.strip()) <= 160]
        rng.shuffle(sents)
        facts = sents[:8]

    procedural = [
        "Maintain eye contact with signage only.",
        "Do not enter maintenance corridors during TRACKING drift.",
        "If the screen flickers, continue the module as normal.",
        "Report no anomalies unless asked directly.",
        "The weather is fine. Complete the checklist.",
        "Avoid reflective surfaces after 19:00.",
        "If you hear a second voice, lower volume by 2 steps.",
    ]

    for i in range(count):
        parts = []
        if facts and rng.random() < 0.7:
            parts.append(rng.choice(facts))
        if headline and rng.random() < 0.55:
            parts.append("Daily note: " + headline)
        parts.append(rng.choice(procedural))
        # redacted flourish
        if rng.random() < 0.35:
            parts.append("[" + "REDACTED" + ("] [" + "REDACTED" if rng.random() < 0.45 else "]"))
        bodies.append(" ".join(parts))
    return bodies

def render_video(cfg: Dict[str, Any], out_path: Path) -> None:
    rng = random.Random(cfg["seed"])
    w, h = int(cfg["render"]["width"]), int(cfg["render"]["height"])
    fps = int(cfg["render"]["fps"])
    slide_seconds = float(cfg["render"]["slide_seconds"])
    slide_frames = max(1, int(slide_seconds * fps))

    work = Path("work")
    if work.exists():
        shutil.rmtree(work)
    mkdirp(work)
    mkdirp(work / "images")

    theme = pick_theme_keyword(cfg, rng)
    queries = [q.format(k=theme) for q in cfg["theme"]["query_expand"]]

    wiki_text = ""
    images: List[Path] = []
    headline = ""
    if cfg["web"]["enabled"]:
        wiki_text = fetch_wikipedia_summary(theme, timeout=int(cfg["web"]["timeout"]))
        images = download_images(
            queries=queries,
            out_dir=work / "images",
            timeout=int(cfg["web"]["timeout"]),
            max_images=int(cfg["web"]["max_images"]),
            min_w=int(cfg["web"]["min_w"]),
            min_h=int(cfg["web"]["min_h"]),
            rng=rng,
        )
        headline = fetch_rss_headline(rng)

    # Hard fallback images if scraping is thin
    if len(images) < 4:
        # Create simple synthetic images from existing assets (if any) or noise
        for i in range(8):
            synth = make_slide_canvas(w, h, (0,0,0))
            dd = ImageDraw.Draw(synth)
            dd.rectangle([0,0,w,h], fill=(rng.randint(0,20), rng.randint(0,20), rng.randint(0,20)))
            dd.text((20, 20), "NO SIGNAL", font=pick_font(32), fill=(220,220,220))
            arr = np.array(synth).astype(np.int16) + np.random.normal(0, 22, (h,w,3)).astype(np.int16)
            synth = Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
            p = work / "images" / f"fallback_{i:02d}.jpg"
            synth.save(p, quality=92)
            images.append(p)

    # Build slide plan
    story = cfg["story"]
    total_slides = int(story["slide_count"])
    tape_id = f"{rng.choice(['A','B','C','D'])}{rng.randint(1000,9999)}-{rng.choice(['HR','OPS','CIV','WX'])}"
    date = now_like_90s(rng)

    forensic_n = rng.randint(int(story["forensic_slides_min"]), int(story["forensic_slides_max"]))
    chart_n = rng.randint(int(story["chart_slides_min"]), int(story["chart_slides_max"]))

    # slide kinds pool (excluding forced intro/outro)
    kinds = ["normal"] * max(1, int(total_slides * float(story["normal_ratio"])))
    kinds += ["forensic"] * forensic_n
    kinds += ["chart"] * chart_n
    while len(kinds) < max(1, total_slides - 2):
        kinds.append(rng.choice(["normal", "forensic", "chart"]))
    rng.shuffle(kinds)
    kinds = kinds[: max(1, total_slides - 2)]

    # bodies for normal slides
    bodies = build_slide_bodies(theme, wiki_text, headline, rng, count=max(6, total_slides))
    body_idx = 0

    # Prepare slides as PIL images
    slides: List[Image.Image] = []

    if story["include_intro_outro"]:
        slides.append(make_vhs_spec_slide(w, h, theme, rng, tape_id, date))

    for k in kinds:
        # random interrupt slide sometimes
        if rng.random() < float(story["interrupt_probability"]):
            slides.append(make_interrupt_slide(w, h, rng, tape_id, date))
            continue

        img_path = rng.choice(images)
        try:
            pic = Image.open(img_path).convert("RGB")
        except Exception:
            pic = None

        if k == "forensic" and pic is not None:
            slides.append(make_forensic_slide(w, h, pic, theme, rng, tape_id, date))
        elif k == "chart":
            slides.append(make_chart_slide(w, h, theme, rng, tape_id, date))
        else:
            body = bodies[body_idx % len(bodies)]
            body_idx += 1
            slides.append(make_normal_slide(w, h, theme, rng, tape_id, date, pic, body))

    if story["include_intro_outro"]:
        # outro: "end of module" but unsettling
        out = make_slide_canvas(w, h, (0,0,0))
        d = ImageDraw.Draw(out)
        d.text((w//2, int(h*0.30)), "END OF TRAINING", font=pick_font(46), fill=(235,235,235), anchor="mm")
        d.text((w//2, int(h*0.42)), "PLEASE CONTINUE NORMAL DUTIES.", font=pick_font(18), fill=(200,200,200), anchor="mm")
        d.text((w//2, int(h*0.58)), "…", font=pick_font(64), fill=(220,30,30), anchor="mm")
        draw_footer(d, w, h, rng, tape_id, date)
        slides.append(vignette(out, 0.6))

    # Render frames
    writer = FFmpegWriter(str(out_path), w, h, fps)
    writer.start()

    # Entity flash scheduling
    entity_enabled = rng.random() < float(story["entity_flash_probability"])
    entity_at_frame = rng.randint(fps*2, max(fps*2+1, len(slides)*slide_frames - fps*2))
    entity_len = max(2, int(0.10 * fps))
    # stutter by duplicating
    entity_dup = rng.random() < 0.85

    global_frame = 0
    for si, slide in enumerate(slides):
        base = np.array(slide).astype(np.uint8)

        for f in range(slide_frames):
            t = global_frame / float(fps)

            frame = base.copy()

            # micro random text overlay
            if rng.random() < float(story["micro_text_probability"]):
                frame = micro_overlay(frame, rng)

            # apply VHS
            frame = vhs_process(frame, rng, t, cfg)

            # entity flash
            if entity_enabled and (entity_at_frame <= global_frame < entity_at_frame + entity_len):
                frame = entity_flash(frame, rng)
                if entity_dup and rng.random() < 0.35:
                    # stutter repeats
                    pass

            writer.write(frame)
            global_frame += 1

    writer.close()

    # mux audio
    if cfg["audio"]["enabled"]:
        narration = (f"Training module: {theme}. " +
                     (headline + ". " if headline else "") +
                     "Please remain calm. Do not look at the glass.")
        wav_path = work / "audio.wav"
        synth_audio(cfg, rng, duration_s=(len(slides)*slide_frames)/fps, out_wav_path=wav_path, narration=narration)
        mux_audio(str(out_path), str(wav_path))

def micro_overlay(frame_rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    img = Image.fromarray(frame_rgb)
    d = ImageDraw.Draw(img)
    w, h = img.size
    font = pick_font(14)
    msg = rng.choice([
        "SIGNAL_LEAK", "CACHE SPOILED", "UNSTABLE", "NULL_SKY", "TRACKING", "PROTOCOL_9", "EYES SHUT"
    ])
    msg = msg + f" :: 0x{rng.choice(['DEAD','BEEF','000F','404','666'])}"
    x = rng.randint(8, w-220)
    y = rng.randint(50, h-60)
    d.text((x+1, y+1), msg, font=font, fill=(180,0,0))
    d.text((x, y), msg, font=font, fill=(0,230,120))
    return np.array(img).astype(np.uint8)

def mux_audio(video_path: str, wav_path: str):
    tmp = video_path + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", wav_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        tmp,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
        os.replace(tmp, video_path)

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="out.mp4")
    args = ap.parse_args()

    cfg_file = load_yaml(Path(args.config))
    cfg = deep_merge(DEFAULT_CFG, cfg_file)

    # env overrides
    seed_env = os.getenv("SEED", "").strip()
    if seed_env:
        cfg["seed"] = safe_int(seed_env, None) or cfg.get("seed")
    if cfg.get("seed") is None:
        cfg["seed"] = int.from_bytes(os.urandom(4), "big")
    cfg["web"]["enabled"] = env_bool("WEB_ENABLE", bool(cfg["web"]["enabled"]))

    out = Path(args.out)
    render_video(cfg, out)

    print(f"OK: {out}  (seed={cfg['seed']})")

if __name__ == "__main__":
    main()

