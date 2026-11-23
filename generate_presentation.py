import os
import sys
import json
import re
import math
import asyncio
import subprocess
import urllib.parse
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# ---------------- CONFIGURATION ----------------
WIDTH, HEIGHT = 1280, 720
FPS = 24

# Importance threshold: ONLY words with score >= threshold appear in center
MIN_SCORE_THRESHOLD = 8

# Natural context-aware voice
TTS_VOICE = "en-US-JennyNeural"
PAUSE_BETWEEN_SENTENCES = 0.35

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

SCORE_STYLES = {
    10: {"color": (244, 63, 94), "base_size": 100, "pulse": 0.12},  # Fiery Coral
    9:  {"color": (251, 191, 36), "base_size": 86,  "pulse": 0.09},  # Radiant Gold
    8:  {"color": (56, 189, 248), "base_size": 74,  "pulse": 0.06},  # Sky Cyan
}

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def get_audio_duration(file_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = res.stdout.strip()
    return float(out) if out else 1.0

async def synthesize_speech(text, audio_out):
    """Synthesizes speech and captures word boundary timestamps."""
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate="-4%")
    audio_data = bytearray()
    words = []

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            start_s = chunk["offset"] / 10_000_000
            dur_s = chunk["duration"] / 10_000_000
            words.append({
                "word": chunk["text"],
                "start": start_s,
                "end": start_s + dur_s
            })

    with open(audio_out, "wb") as f:
        f.write(audio_data)

    duration = get_audio_duration(audio_out)

    if not words:
        raw_words = [w for w in text.split() if w.strip()]
        w_dur = duration / max(1, len(raw_words))
        for i, w in enumerate(raw_words):
            words.append({"word": w, "start": i * w_dur, "end": (i + 1) * w_dur})

    return duration, words

def analyze_script(input_path):
    """Analyzes script to extract sentence scores and visual scene descriptions."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    prompt = f"""
You are an expert AI video director. Analyze this narration script.

For EVERY sentence:
1. "visual_prompt": A 10-15 word descriptive visual scene prompt suitable for generating a background image illustrating the meaning of this sentence (e.g., "cinematic digital art of massive neural network data clusters, glowing nodes, dark atmosphere, 8k").
2. "important_words": Extract 1 to 3 key words from the sentence scored 1 to 10 (10 being most important).

Return ONLY a raw JSON array matching this schema:
[
  {{
    "sentence": "The revolutionary expansion of artificial intelligence is fundamentally reshaping civilization.",
    "visual_prompt": "cinematic futuristic cityscape with glowing neural network lines, digital art, moody lighting",
    "important_words": [
      {{"word": "intelligence", "score": 9}},
      {{"word": "civilization", "score": 10}}
    ]
  }}
]

Script:
\"\"\"
{content}
\"\"\"
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": "You return strictly raw JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    print("Analyzing script and generating scene concepts via OpenRouter...")
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        print(f"API Error ({res.status_code}): {res.text}", file=sys.stderr)
        sys.exit(1)

    raw = res.json()["choices"][0]["message"]["content"].strip()
    clean = re.sub(r"^```(?:json)?\s*", "", raw)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except Exception:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
        return [{"sentence": s, "visual_prompt": "abstract dark technology digital art", "important_words": []} for s in sentences]

def fetch_scene_image(prompt, save_path, seed=42):
    """Downloads a relevant AI-generated scene image via free Pollinations API."""
    encoded = urllib.parse.quote(prompt.strip())
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&nologo=true&seed={seed}"
    try:
        r = requests.get(url, timeout=35)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
            img = Image.open(save_path).convert("RGB")
            # Apply dark vignette / contrast so text overlays stand out cleanly
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(0.45)
    except Exception as e:
        print(f"  Note: Background download fallback ({e}). Using atmospheric backdrop.")

    # Fallback atmospheric dark canvas
    fallback = Image.new("RGB", (WIDTH, HEIGHT), (15, 23, 42))
    return fallback

def precompute_caption_layout(spoken_words, font, max_w=1120, line_h=38):
    lines, curr_line, curr_w = [], [], 0
    for item in spoken_words:
        bbox = font.getbbox(item["word"] + " ")
        w_width = bbox[2] - bbox[0]
        if curr_w + w_width > max_w and curr_line:
            lines.append(curr_line)
            curr_line, curr_w = [], 0
        curr_line.append((item, w_width))
        curr_w += w_width
    if curr_line:
        lines.append(curr_line)

    total_h = len(lines) * line_h
    start_y = HEIGHT - total_h - 40

    positioned = []
    for l_idx, line in enumerate(lines):
        line_w = sum(w for _, w in line)
        cur_x = (WIDTH - line_w) // 2
        y = start_y + l_idx * line_h
        for word_info, w_width in line:
            positioned.append({**word_info, "x": cur_x, "y": y})
            cur_x += w_width

    return positioned, start_y - 12, HEIGHT - 22

def render_scene_frame(slide_meta, current_time, total_slide_dur, caption_font):
    # 1. KEN BURNS CAMERA ANIMATION (Smooth zoom & pan across sentence duration)
    base_img = slide_meta["bg_image"]
    progress = min(1.0, max(0.0, current_time / max(0.1, total_slide_dur)))
    zoom = 1.0 + (0.12 * progress)  # Zooms in from 100% to 112%

    crop_w = int(WIDTH / zoom)
    crop_h = int(HEIGHT / zoom)
    # Drift camera slightly horizontally
    shift_x = int((WIDTH - crop_w) * (0.3 + 0.4 * progress))
    shift_y = int((HEIGHT - crop_h) * 0.5)

    cropped = base_img.crop((shift_x, shift_y, shift_x + crop_w, shift_y + crop_h))
    frame = cropped.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(frame)

    # 2. CENTER: ANIMATED FOCAL WORD (Score >= Threshold)
    active_item = None
    for item in slide_meta["spoken_words"]:
        if item["start"] <= current_time <= item["end"] + 0.08:
            active_item = item
            break

    if active_item:
        clean_w = re.sub(r"[^\w]", "", active_item["word"]).lower()
        score = slide_meta["score_map"].get(clean_w, 0)

        if score >= MIN_SCORE_THRESHOLD:
            style = SCORE_STYLES.get(min(10, score), SCORE_STYLES[9])
            word_time = max(0.0, current_time - active_item["start"])
            pop = min(1.0, word_time / 0.05)
            pulse = 1.0 + style["pulse"] * math.sin(word_time * 12)
            font_size = max(24, int(style["base_size"] * pop * pulse))
            center_font = get_font(FONT_BOLD, font_size)

            display_text = active_item["word"].upper()
            bbox = center_font.getbbox(display_text)
            w_width = bbox[2] - bbox[0]
            w_height = bbox[3] - bbox[1]

            cx = (WIDTH - w_width) // 2
            cy = int(HEIGHT * 0.42) - (w_height // 2)

            # Translucent backdrop pill behind keyword for crisp readability over video
            pad_x, pad_y = 28, 14
            draw.rounded_rectangle(
                [cx - pad_x, cy - pad_y, cx + w_width + pad_x, cy + w_height + pad_y],
                radius=14,
                fill=(10, 15, 26)
            )

            # Drop shadow + Main glowing keyword
            draw.text((cx + 3, cy + 3), display_text, fill=(0, 0, 0, 200), font=center_font)
            draw.text((cx, cy), display_text, fill=style["color"], font=center_font)

    # 3. LOWER EDGE: SYNCED CAPTION BAR
    pos_words = slide_meta["pos_words"]
    y1, y2 = slide_meta["bar_y1"], slide_meta["bar_y2"]

    draw.rounded_rectangle([50, y1, WIDTH - 50, y2], radius=10, fill=(10, 15, 26), outline=(71, 85, 105), width=1)

    for item in pos_words:
        if current_time >= item["start"]:
            is_speaking_now = (item["start"] <= current_time <= item["end"] + 0.08)
            clean_item_w = re.sub(r"[^\w]", "", item["word"]).lower()
            item_score = slide_meta["score_map"].get(clean_item_w, 0)

            if is_speaking_now and item_score >= MIN_SCORE_THRESHOLD:
                c = SCORE_STYLES.get(min(10, item_score), SCORE_STYLES[9])["color"]
            elif is_speaking_now:
                c = (251, 191, 36)
            else:
                c = (226, 232, 240)

            draw.text((item["x"], item["y"]), item["word"], fill=c, font=caption_font)

    return frame

def build_score_map(slide):
    score_map = {}
    items = slide.get("important_words", [])
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                w = re.sub(r"[^\w]", "", str(it.get("word", ""))).lower()
                try:
                    score_map[w] = int(it.get("score", 0))
                except ValueError:
                    pass
    return score_map

def main(input_file, output_mp4):
    slides = analyze_script(input_file)
    with open("presentation_data.json", "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2)

    os.makedirs("temp_build", exist_ok=True)
    audio_concat_list = []
    slide_metadata = []
    caption_font = get_font(FONT_REGULAR, 24)

    total_scenes = len(slides)
    print(f"\n--- Generating {total_scenes} Synced Animated Scenes ---")

    for idx, slide in enumerate(slides):
        sentence = slide.get("sentence", "").strip()
        visual_prompt = slide.get("visual_prompt", "cinematic technology digital art")
        print(f"\n[Scene {idx+1}/{total_scenes}]")
        print(f"  Sentence: {sentence[:60]}...")
        print(f"  Visual Concept: {visual_prompt}")

        raw_audio = f"temp_build/raw_{idx}.mp3"
        padded_audio = f"temp_build/padded_{idx}.mp3"
        bg_image_path = f"temp_build/scene_{idx}.jpg"

        # 1. Synthesize audio & timestamps
        duration, spoken_words = asyncio.run(synthesize_speech(sentence, raw_audio))
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_audio,
            "-af", f"apad=pad_dur={PAUSE_BETWEEN_SENTENCES}",
            padded_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        full_dur = get_audio_duration(padded_audio)
        audio_concat_list.append(f"file '{os.path.abspath(padded_audio)}'")

        # 2. Fetch AI background illustration for this scene
        bg_image = fetch_scene_image(visual_prompt, bg_image_path, seed=100 + idx)

        # 3. Precompute caption layout
        pos_words, y1, y2 = precompute_caption_layout(spoken_words, caption_font)

        slide_metadata.append({
            "sentence": sentence,
            "duration": full_dur,
            "spoken_words": spoken_words,
            "score_map": build_score_map(slide),
            "bg_image": bg_image,
            "pos_words": pos_words,
            "bar_y1": y1,
            "bar_y2": y2
        })

    # Combine narration audio
    list_path = "temp_build/audio_list.txt"
    with open(list_path, "w") as f:
        f.write("\n".join(audio_concat_list))

    merged_audio = "temp_build/narration.mp3"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", merged_audio],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Render video stream
    raw_video = "temp_build/video_raw.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        raw_video
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print("\n--- Rendering Synced Animated Scenes ---")
    for idx, meta in enumerate(slide_metadata):
        total_frames = int(round(meta["duration"] * FPS))
        print(f"Rendering animated scene {idx+1}/{total_scenes} ({total_frames} frames)...")
        for f_idx in range(total_frames):
            t = f_idx / FPS
            frame = render_scene_frame(meta, t, meta["duration"], caption_font)
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    # Mux final video
    print("\nFinalizing presentation video with synchronized audio...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", raw_video,
        "-i", merged_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_mp4
    ], check=True)

    subprocess.run(["rm", "-rf", "temp_build"])
    print(f"\nDone! Animated presentation saved to: {output_mp4}")

if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else "my_script.txt"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "my_presentation.mp4"
    main(in_file, out_file)
