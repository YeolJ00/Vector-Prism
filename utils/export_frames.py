"""
This script exports frames from an SVG+CSS animation in an HTML file 
to per-frame PDFs (vector) or PNGs (raster) using Selenium and Chrome/Edge in headless mode. 
It can also compile the frames into a video.
    
Usage:
    python export_frames.py --input_file path/to/animation.html
        [--selector 'svg']
        [--fps 24]
        [--duration 5 | --frames N]
        [--format pdf|png]
        [--scale 1.0]
        [--save_video]
Arguments:
    --input_file: Path to your HTML file or a URL (file:///... or http[s]://...)
    --selector: CSS selector for the target SVG (default: 'svg')
    --fps: Frames per second (default: 24)
    --duration: Total duration in seconds (optional)
    --frames: Number of frames (alternative to --duration)
    --format: Output format, 'pdf' for vector or 'png' for raster (default: 'pdf')
    --scale: Scale factor for PNG output (ignored for PDF)
    --save_video: Whether to compile the frames into a video file 
        (requires ffmpeg to be installed)
"""

import argparse
import base64
import sys
import time
from pathlib import Path
from urllib.request import pathname2url
import fitz  # PyMuPDF

import ffmpeg
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
import logging

logger = logging.getLogger(__name__)

JS_SCRIPT = r"""
const selector = arguments[0];
const svg = document.querySelector(selector);
if (!svg) { throw new Error("No element matches selector: " + selector); }
svg.setAttribute('id', '__FRAME_TARGET__');

// Use RENDERED size, not viewBox
const r = svg.getBoundingClientRect();
let w = 128;
let h = 128;

// If you want to respect explicit width/height attributes instead:
const wAttr = svg.getAttribute('width');
const hAttr = svg.getAttribute('height');
if (wAttr && hAttr) { 
  w = parseFloat(wAttr); 
  h = parseFloat(hAttr); 
}

// Hide everything except the target SVG and pin it to (0,0)
const style = document.createElement('style');
style.id = '__FRAME_EXPORT_STYLE__';
style.textContent = `
  @page { size: ${w}px ${h}px; margin: 0; }
  html, body { width:${w}px; height:${h}px; margin:0; padding:0; background:transparent; }
  body > * { visibility: hidden !important; }
  svg#__FRAME_TARGET__ {
    visibility: visible !important;
    position: fixed !important;
    top: 0; left: 0;
    width: ${w}px !important;
    height: ${h}px !important;
  }`;
document.head.appendChild(style);

// Build a seek function for CSS/WAAPI + SMIL animations
window.__seekFrame__ = (tSec) => {
  try {
    const anims = document.getAnimations({ subtree: true });
    for (const a of anims) { try { a.pause(); a.currentTime = tSec * 1000; } catch(e){} }
  } catch(e){}
  document.querySelectorAll('svg').forEach(s => {
    try { if (typeof s.pauseAnimations==='function') s.pauseAnimations();
          if (typeof s.setCurrentTime==='function') s.setCurrentTime(tSec); } catch(e){}
  });
};

// Try to infer overall duration from active animations
let inferredDuration = 0;
try {
  const anims = document.getAnimations({ subtree: true });
  if (anims.length) {
    inferredDuration = Math.max(0, ...anims.map(a => {
      const t = a.effect && a.effect.getComputedTiming ? a.effect.getComputedTiming() : null;
      return t && Number.isFinite(t.endTime) ? t.endTime / 1000 : 0;
    }));
  }
} catch(e){}

return { width: w, height: h, inferredDuration: inferredDuration };
"""

def init_webdriver():
    options = ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--force-device-scale-factor=1")
    options.add_argument("--hide-scrollbars")

    # Prevent caching issues between runs
    options.add_argument("--disable-cache")
    options.add_argument("--disk-cache-size=0")

    # if linux, add no-sandbox
    if sys.platform.startswith("linux"):
        options.add_argument("--no-sandbox")  # Bypass OS security model
        options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
        options.add_argument("--disable-software-rasterizer")

    driver = webdriver.Chrome(options=options)

    return driver

def resolve_input_path(path_str):
    path = Path(path_str)
    if path.exists() or path.is_absolute():
        return path.resolve()
    if any(path_str.startswith(p) for p in ['home/', 'usr/', 'var/', 'tmp/', 'opt/', 'etc/']):
        fixed_path = Path('/') / path
        if fixed_path.exists():
            return fixed_path.resolve()
    return path.resolve()

def normalize_url(s):
    if s.startswith(("http://", "https://", "file://")):
        return s
    return "file://" + pathname2url(str(Path(s).resolve()))

def save_video_frames(args):
    input_path = resolve_input_path(args.input_file)
    outdir = Path(input_path).parent
    frame_dir = outdir / 'frames' # input pdf frames are here
    outdir.mkdir(parents=True, exist_ok=True)
    video_path = outdir / "animation.mp4"

    if args.format == "pdf":
        # Convert PDFs to PNGs first for video encoding
        png_dir = outdir / 'png_frames'
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = sorted(frame_dir.glob("frame-*.pdf"))
        for pdf_file in pdf_files:
            doc = fitz.open(pdf_file)
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=150)
            png_path = png_dir / (pdf_file.stem + ".png")
            pix.save(png_path)
        frame_dir = png_dir
    (
        ffmpeg
        .input(str(frame_dir / f"frame-%03d.png"), framerate=args.fps)
        .filter('scale', 'ceil(iw/2)*2', 'ceil(ih/2)*2')  # Force even dimensions
        .output(str(video_path), vcodec='libx264', pix_fmt='yuv420p', crf=18)
        .overwrite_output()
        .run(quiet=True)
    )

    if args.format == "pdf": # Clean up temporary PNGs
        for png_file in frame_dir.glob("frame-*.png"):
            png_file.unlink()
        frame_dir.rmdir()

    return

def main(args):
    input_path = resolve_input_path(args.input_file)
    input_url = normalize_url(str(input_path))
    outdir = Path(input_path).parent / 'frames'
    outdir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    driver = init_webdriver()
    driver.get(input_url)

    WebDriverWait(driver, 30).until(lambda d: d.execute_script("return document.readyState") == "complete")
    driver.execute_cdp_cmd("Emulation.setEmulatedMedia", {"media": "screen"})
    page_info = driver.execute_script(JS_SCRIPT, args.selector)
    
    try:
        page_info = driver.execute_script(JS_SCRIPT, args.selector)
    except Exception as e:
        raise RuntimeError("Failed to execute JS script to prepare the page / find SVG.") from e
    if not page_info:
        raise RuntimeError("Failed to prepare the page / find SVG.")
    width, height = float(page_info["width"]), float(page_info["height"])

    if args.frames is not None:
        frames = max(1, int(args.frames))
    elif args.duration is not None: 
        dur = args.duration
        frames = max(1, int(round(dur * args.fps)))
    else:
        raise RuntimeError("Either --duration or --frames must be specified.")
    
    step = 1.0 / args.fps
    for i in range(frames):
        fname = f"frame-{i:03d}"
        outpath = str(outdir / f"{fname}.{args.format}")

        driver.execute_script("window.__seekFrame__(arguments[0]);", i * step)
        time.sleep(0.02)

        if args.format == "pdf": # Vector PDF via DevTools printToPDF
            shot = driver.execute_cdp_cmd("Page.printToPDF", {
                "printBackground": False,
                "preferCSSPageSize": True,
                "pageRanges": "1"
            })
        else: # PNG fallback
            shot = driver.execute_cdp_cmd("Page.captureScreenshot", {
                "format": "png",
                "fromSurface": True,
                "clip": {"x": 0, "y": 0, 
                    "width": width, "height": height, 
                    "scale": float(args.scale)
                  }
            })
        with open(outpath, "wb") as f:
            f.write(base64.b64decode(shot["data"]))

        sys.stdout.write(f"\r|> Exported {i+1}/{frames}")
        sys.stdout.flush()
    logger.info(f"Exported to pdf in {time.time() - start_time:.1f}s.")
    driver.quit()
    
    if args.save_video:
        start_time = time.time()
        save_video_frames(args)
        logger.info(f"Saved video in {time.time() - start_time:.1f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SVG+CSS animation to per-frame PDFs (vector) using Selenium + Chrome/Edge.")
    parser.add_argument("--input_file", required=True, help="Path to your HTML or a URL (file:///... or http[s]://...)")
    parser.add_argument("--selector", default="svg", help="CSS selector for the target SVG (default: 'svg')")
    parser.add_argument("--fps", type=float, default=24.0, help="Frames per second (default: 24)")
    parser.add_argument("--duration", type=float, default=5, help="Total duration in seconds (optional)")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames (alternative to --duration)")
    parser.add_argument("--format", choices=["pdf","png"], default="pdf", help="Output format (pdf=vector, png=raster)")
    parser.add_argument("--scale", type=float, default=1.0, help="PNG scale factor (ignored for PDF)")
    parser.add_argument("--save_video", action='store_true', help="Whether to save video frames")
    args = parser.parse_args()
    main(args)
