import os
import sys
import json
import re
import math
import subprocess
import requests
import soundfile as sf
from kokoro_onnx import Kokoro
from PIL import Image, ImageDraw, ImageFont

# ---------------- CONFIGURATION ----------------
WIDTH, HEIGHT = 1280, 720
FPS = 24

# Importance threshold: ONLY words with score >= threshold appear in center
MIN_SCORE_THRESHOLD = 8

# Kokoro Voice Options:
# "af_heart" (Warm, human, popular), "af_bella", "af_sarah"
# "am_adam" (Deep, authoritative), "am_michael"
# "bf_emma" (British female), "bm_george" (British male)
KOKORO_VOICE = "am_adam"
KOKORO_SPEED = 0.85            # Slower pace for clear documentary delivery
PAUSE_BETWEEN_SENTENCES = 0.5  # Pause after each sentence in seconds

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

SCORE_STYLES = {
    10: {"color": (244, 63, 94), "base_size": 100, "pulse": 0.12},  # Fiery Coral
    9:  {"color": (251, 191, 36), "base_size": 86,  "pulse": 0.09},  # Radiant Gold
    8:  {"color": (56, 189, 248), "base_size": 74,  "pulse": 0.06},  # Sky Cyan
}

# Initialize Kokoro Engine
print("Loading Kokoro-82M ONNX model...")
kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")

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

def synthesize_sentence_kokoro(sentence, audio_out):
    """Synthesizes speech using local Kokoro and calculates character-weighted word timing."""
    samples, sample_rate = kokoro.create(
        sentence,
        voice=KOKORO_VOICE,
        speed=KOKORO_SPEED,
        lang="en-us"
    )
    sf.write(audio_out, samples, sample_rate)
    duration = len(samples) / float(sample_rate)

    # Calculate word timestamps weighted by character length for natural rhythm
    raw_words = [w for w in sentence.split() if w.strip()]
    weights = [max(1, len(re.sub(r"[^\w]", "", w))) for w in raw_words]
    total_weight = sum(weights)

    words = []
    curr_time = 0.0
    for w, wt in zip(raw_words, weights):
        w_dur = (wt / total_weight) * duration
        words.append({
            "word": w,
            "start": curr_time,
            "end": curr_time + w_dur
        })
        curr_time += w_dur

    return duration, words

def analyze_script(input_path):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    prompt = f"""
Analyze the narration script below. Break down into sentences and assign importance scores (1 to 10) to words in each sentence.

CRITICAL RULES:
- The "word" attribute MUST be the literal word copied from that sentence.
- NEVER output placeholder words like "word" or "keyword".
- Score each word from 1 to 10 (10 being most important).
- Return ONLY valid JSON array with no markdown backticks.

Schema:
[
  {{
    "sentence": "The revolutionary expansion of artificial intelligence is fundamentally reshaping civilization.",
    "important_words": [
      {{"word": "intelligence", "score": 9}},
      {{"word": "reshaping", "score": 8}},
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
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    print("Calling OpenRouter (openrouter/free) for word importance analysis...")
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        print(f"API Error ({res.status_code}): {res.text}", file=sys.stderr)
        sys.exit(1)

    raw = res.json()["choices"][0]["message"]["content"].strip()
    clean = re.sub(r"^```(?:json)?\s*", "", raw)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        data = json.loads(clean)
        if isinstance(data, list) and len(data) > 0:
            return data
    except Exception:
        pass

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
    return [{"sentence": s, "important_words": []} for s in sentences]

def build_score_map(slide):
    score_map = {}
    items = []
    if "important_words" in slide and isinstance(slide["important_words"], list):
        items = slide["important_words"]
    elif "words" in slide:
        if isinstance(slide["words"], dict):
            items = [{"word": k, "score": v} for k, v in slide["words"].items()]
        elif isinstance(slide["words"], list):
            items = slide["words"]

    for it in items:
        if isinstance(it, dict):
            w = re.sub(r"[^\w]", "", str(it.get("word", ""))).lower()
            try:
                s = int(it.get("score", 0))
                if w:
                    score_map[w] = s
            except ValueError:
                pass
    return score_map

def draw_background(draw):
    top_color = (15, 23, 42)
    bottom_color = (30, 41, 59)
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

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

def render_frame(slide_data, current_time, caption_font):
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    draw_background(draw)

    # 1. FIND CURRENT WORD
    active_item = None
    for item in slide_data["spoken_words"]:
        if item["start"] <= current_time <= item["end"] + 0.08:
            active_item = item
            break

    # 2. CENTER: SHOW ONLY IF SCORE >= MIN_SCORE_THRESHOLD
    if active_item:
        clean_w = re.sub(r"[^\w]", "", active_item["word"]).lower()
        score = slide_data["score_map"].get(clean_w, 0)

        if score >= MIN_SCORE_THRESHOLD:
            style = SCORE_STYLES.get(min(10, score), SCORE_STYLES[10])

            word_time = max(0.0, current_time - active_item["start"])
            pop = min(1.0, word_time / 0.05)
            pulse = 1.0 + style["pulse"] * math.sin(word_time * 12)
            font_size = max(24, int(style["base_size"] * pop * pulse))
            center_font = get_font(FONT_BOLD, font_size)

            display_text = active_item["word"].upper()
            bbox = center_font.getbbox(display_text)
            w_width = bbox[2] - bbox[0]
            w_height = bbox[3] - bbox[1]

            center_x = (WIDTH - w_width) // 2
            center_y = int(HEIGHT * 0.42) - (w_height // 2)

            draw.text((center_x + 3, center_y + 3), display_text, fill=(0, 0, 0, 180), font=center_font)
            draw.text((center_x, center_y), display_text, fill=style["color"], font=center_font)

    # 3. BOTTOM CAPTION
    pos_words = slide_data["pos_words"]
    y1, y2 = slide_data["bar_y1"], slide_data["bar_y2"]

    draw.rounded_rectangle([50, y1, WIDTH - 50, y2], radius=10, fill=(10, 15, 26), outline=(51, 65, 85), width=1)

    for item in pos_words:
        if current_time >= item["start"]:
            is_speaking_now = (item["start"] <= current_time <= item["end"] + 0.08)
            clean_item_w = re.sub(r"[^\w]", "", item["word"]).lower()
            item_score = slide_data["score_map"].get(clean_item_w, 0)

            if is_speaking_now and item_score >= MIN_SCORE_THRESHOLD:
                c = SCORE_STYLES.get(min(10, item_score), SCORE_STYLES[10])["color"]
            elif is_speaking_now:
                c = (251, 191, 36)
            else:
                c = (226, 232, 240)

            draw.text((item["x"], item["y"]), item["word"], fill=c, font=caption_font)

    return img

def main(input_file, output_mp4):
    slides = analyze_script(input_file)
    with open("presentation_data.json", "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2)

    os.makedirs("temp_build", exist_ok=True)
    audio_concat_list = []
    slide_metadata = []
    caption_font = get_font(FONT_REGULAR, 24)

    print(f"\n--- Generating Speech with Kokoro TTS (Voice: {KOKORO_VOICE}) ---")
    for idx, slide in enumerate(slides):
        sentence = slide.get("sentence", "").strip()
        print(f"Synthesizing sentence {idx+1}/{len(slides)} with Kokoro...")
        raw_audio = f"temp_build/raw_{idx}.wav"
        padded_audio = f"temp_build/padded_{idx}.wav"

        duration, spoken_words = synthesize_sentence_kokoro(sentence, raw_audio)

        subprocess.run([
            "ffmpeg", "-y", "-i", raw_audio,
            "-af", f"apad=pad_dur={PAUSE_BETWEEN_SENTENCES}",
            padded_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        full_dur = get_audio_duration(padded_audio)
        audio_concat_list.append(f"file '{os.path.abspath(padded_audio)}'")

        pos_words, y1, y2 = precompute_caption_layout(spoken_words, caption_font)

        slide_metadata.append({
            "sentence": sentence,
            "duration": full_dur,
            "spoken_words": spoken_words,
            "score_map": build_score_map(slide),
            "pos_words": pos_words,
            "bar_y1": y1,
            "bar_y2": y2
        })

    list_path = "temp_build/audio_list.txt"
    with open(list_path, "w") as f:
        f.write("\n".join(audio_concat_list))

    merged_audio = "temp_build/narration.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", merged_audio],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

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

    print("\n--- Rendering Video Frames (Kokoro Synced) ---")
    for idx, meta in enumerate(slide_metadata):
        total_frames = int(round(meta["duration"] * FPS))
        print(f"Rendering slide {idx+1}/{len(slide_metadata)}...")
        for f_idx in range(total_frames):
            t = f_idx / FPS
            frame = render_frame(meta, t, caption_font)
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    print("\nMuxing synchronized audio and video...")
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
    print(f"\nDone! Kokoro video presentation saved to: {output_mp4}")

if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else "my_script.txt"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "my_presentation.mp4"
    main(in_file, out_file)
