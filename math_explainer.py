import os
import sys
import json
import re
import math
import textwrap
import argparse
import asyncio
import subprocess
import requests
import edge_tts
from pypdf import PdfReader
from PIL import Image, ImageDraw, ImageFont

# ---------------- CONFIGURATION ----------------
WIDTH, HEIGHT = 1280, 720
FPS = 24
PAUSE_BETWEEN_STEPS = 0.55

# Energetic, friendly tutor voice
TTS_VOICE = "en-SG-LunaNeural"  # Clear Singapore tutor voice

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def extract_question_from_pdf(pdf_path, question_num=1):
    """Accurately extracts question N using standard exam paper numbering patterns."""
    reader = PdfReader(pdf_path)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

    q_curr = str(question_num)
    q_next = str(question_num + 1)

    patterns = [
        rf"(?:^|\n)\s*{q_curr}[\.\)]\s+([\s\S]*?)(?=(?:^|\n)\s*{q_next}[\.\)]\s+|$)",
        rf"\b{q_curr}[\.\)]\s+([\s\S]*?)(?=\b{q_next}[\.\)]\s+|$)",
        rf"(?:^|\n)\s*{q_curr}\s*\n([\s\S]*?)(?=(?:^|\n)\s*{q_next}\s*\n|$)"
    ]

    for pat in patterns:
        m = re.search(pat, full_text)
        if m and len(m.group(1).strip()) > 15:
            raw_q = m.group(1).strip()
            clean_q = re.sub(r"Page\s+\d+.*", "", raw_q, flags=re.IGNORECASE).strip()
            return clean_q

    lines = full_text.splitlines()
    captured = []
    collecting = False
    for line in lines:
        s = line.strip()
        if re.match(rf"^{q_curr}[\.\)]\s+", s) or s == q_curr:
            collecting = True
            captured.append(line)
            continue
        if collecting:
            if re.match(rf"^{q_next}[\.\)]\s+", s) or (s == q_next and len(captured) > 2):
                break
            captured.append(line)

    if captured:
        return "\n".join(captured).strip()

    return full_text[:350]

def plan_math_solution(question_text, question_num):
    """Generates a detailed, humorous, child-friendly walkthrough tailored for a 10-year-old."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    prompt = f"""
You are an energetic, fun, and warm Singapore primary school Math tutor teaching a 10-year-old kid (Primary 4/Primary 5).
Analyze this exam question and build a crystal-clear, detailed visual walkthrough.

Question raw text:
\"\"\"
{question_text}
\"\"\"

CRITICAL TEACHING GUIDELINES:
1. Tone: Super engaging, lighthearted, and funny! Use playful analogies that a 10-year-old loves (e.g. slicing pizzas, chocolate bars, lego bricks, or sharing snacks with hungry friends).
2. Detail: Don't just jump to the formula! Explain the "WHY". For example:
   - If turning mixed numbers into fractions, explain that each whole pizza has 7 slices, so 5 whole pizzas gives 5 times 7 = 35 yummy slices!
   - Walk through the problem slowly and clearly so any 10-year-old can follow without getting confused.
3. "intro_narration": MUST start EXACTLY with:
   "Let's have a look at the question we are solving. " followed by reading the question stem in a friendly voice. DO NOT read the options.
4. Provide 3 to 4 detailed step-by-step visual stages:
   - "title": Kid-friendly catchy title (e.g. "Step 1: The Pizza Box Secret!", "Step 2: Counting All The Slices")
   - "narration": 3-4 natural conversational sentences packed with humor, encouragement ("No panic, this is easy-peasy!"), and clear breakdown.
   - "models": Visual bar models/unit blocks:
       - "label": Clear kid-friendly label (e.g., "5 Wholes (Pizzas)", "Extra 6 Slices", "1 Unit")
       - "units": Integer from 1 to 7 equal visual blocks
       - "color": "cyan", "amber", "emerald", or "coral"
       - "extra_value": Brief helper note (e.g., "5 × 7 = 35 slices" or "6 slices")
   - "equations": Simple, friendly math workings (e.g. ["1 whole pizza = 7 slices", "5 whole pizzas = 5 × 7 = 35 slices", "Total slices = 35 + 6 = 41 slices!"])
   - "answer_highlight": Cheerful confirmation of the final answer (e.g. "Bingo! Option (4) is our winner: 41/7").

Return ONLY a raw JSON object matching this schema:
{{
  "question_stem": "Which of the following fractions is equal to 5 6/7?",
  "options": ["30/7", "35/7", "37/7", "41/7"],
  "intro_narration": "Let's have a look at the question we are solving. Which of the following fractions is equal to 5 and 6 over 7?",
  "steps": [
    {{
      "title": "Step 1: Imagine 5 Whole Pizzas!",
      "narration": "Don't panic! Think of this like a pizza party. We have 5 whole pizzas, plus 6 extra slices. Because the bottom number is 7, every single pizza is cut into 7 yummy slices. Let's find out how many slices are in the 5 whole boxes!",
      "models": [
        {{"label": "5 Whole Pizzas", "units": 5, "color": "cyan", "extra_value": "5 × 7 slices"}}
      ],
      "equations": ["Each whole pizza = 7/7 slices", "5 whole pizzas = 5 × 7 = 35 slices"],
      "answer_highlight": null
    }},
    {{
      "title": "Step 2: Add The Leftover Slices",
      "narration": "We already have 35 slices from the whole boxes. Now, let's not forget the 6 leftover slices on the plate! 35 slices plus 6 more slices gives us exactly 41 slices in total. That means we have 41 over 7!",
      "models": [
        {{"label": "5 Boxes (35 slices)", "units": 5, "color": "cyan", "extra_value": "35/7"}},
        {{"label": "Plate (6 slices)", "units": 1, "color": "amber", "extra_value": "+ 6/7"}}
      ],
      "equations": ["35 slices + 6 slices = 41 slices", "Total = 41/7"],
      "answer_highlight": null
    }},
    {{
      "title": "Step 3: Spot The Winning Option!",
      "narration": "Look at the options! 41 over 7 is sitting right there at Option 4. Give yourself a high five, we solved it without breaking a sweat!",
      "models": [
        {{"label": "Final Total", "units": 6, "color": "emerald", "extra_value": "41/7"}}
      ],
      "equations": ["5 6/7 = (5 × 7 + 6) / 7 = 41/7"],
      "answer_highlight": "Bingo! Option (4) is our winner: 41/7"
    }}
  ]
}}
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "openrouter/free",
        "messages": [
            {"role": "system", "content": "You are a fun primary school Singapore math teacher. Return raw JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.25
    }

    print("Generating kid-friendly humorous walkthrough via OpenRouter...")
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        print(f"API Error ({res.status_code}): {res.text}", file=sys.stderr)
        sys.exit(1)

    raw = res.json()["choices"][0]["message"]["content"].strip()
    clean = re.sub(r"^```(?:json)?\s*", "", raw)
    clean = re.sub(r"\s*```$", "", clean)
    return json.loads(clean)

async def synthesize_speech(text, audio_path):
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate="-2%")
    await communicate.save(audio_path)
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = res.stdout.strip()
    return float(out) if out else 1.0

def draw_canvas_grid(draw):
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=(15, 23, 42))
    for x in range(0, WIDTH, 40):
        draw.line([(x, 0), (x, HEIGHT)], fill=(30, 41, 59), width=1)
    for y in range(0, HEIGHT, 40):
        draw.line([(0, y), (WIDTH, y)], fill=(30, 41, 59), width=1)

def render_intro_frame(plan, q_num, progress):
    """Renders the friendly Question intro screen."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    draw_canvas_grid(draw)

    # 1. HEADER
    draw.rounded_rectangle([60, 30, WIDTH - 60, 100], radius=12, fill=(30, 41, 59), outline=(56, 189, 248), width=2)
    h_font = get_font(FONT_BOLD, 26)
    draw.text((85, 48), f"QUESTION {q_num}  •  LET'S CRACK THIS TOGETHER!", fill=(56, 189, 248), font=h_font)

    # 2. QUESTION CARD
    draw.rounded_rectangle([60, 120, WIDTH - 60, 420], radius=14, fill=(19, 31, 51), outline=(71, 85, 105), width=1)

    q_font = get_font(FONT_BOLD, 28)
    q_stem = plan.get("question_stem", "")
    wrapped_lines = textwrap.wrap(q_stem, width=54)

    cur_y = 150
    for line in wrapped_lines:
        draw.text((95, cur_y), line, fill=(248, 250, 252), font=q_font)
        cur_y += 42

    # 3. OPTIONS
    options = plan.get("options", [])
    if options:
        opt_start_y = 445
        opt_w = 260
        opt_h = 58
        spacing = 20
        total_w = len(options) * opt_w + (len(options) - 1) * spacing
        start_x = (WIDTH - total_w) // 2

        opt_font = get_font(FONT_BOLD, 20)
        for i, opt in enumerate(options[:4]):
            bx1 = start_x + i * (opt_w + spacing)
            bx2 = bx1 + opt_w
            by1 = opt_start_y
            by2 = by1 + opt_h
            draw.rounded_rectangle([bx1, by1, bx2, by2], radius=8, fill=(30, 41, 59), outline=(100, 116, 139), width=1)
            draw.text((bx1 + 18, by1 + 16), f"({i+1})  {opt}", fill=(226, 232, 240), font=opt_font)

    # 4. CAPTION BAR
    draw.rounded_rectangle([60, 590, WIDTH - 60, 690], radius=10, fill=(10, 15, 26), outline=(71, 85, 105), width=1)
    caption_font = get_font(FONT_REGULAR, 22)
    intro_spoken = plan.get("intro_narration", "")
    draw.text((85, 625), f"Tutor: \"{intro_spoken[:110]}...\"", fill=(251, 191, 36), font=caption_font)

    return img

def render_step_frame(plan, step_idx, q_num, progress):
    """Renders kid-friendly visual unit bars and colorful math breakdown."""
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    draw_canvas_grid(draw)

    step = plan["steps"][step_idx]

    # TOP HEADER
    draw.rounded_rectangle([40, 20, WIDTH - 40, 90], radius=10, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
    draw.text((60, 32), f"QUESTION {q_num}: {plan.get('question_stem', '')[:70]}...", fill=(241, 245, 249), font=get_font(FONT_BOLD, 20))
    draw.text((60, 60), f"★ {step.get('title', '')}", fill=(56, 189, 248), font=get_font(FONT_BOLD, 20))

    COLOR_MAP = {
        "cyan": ((56, 189, 248), (14, 116, 144)),
        "amber": ((251, 191, 36), (180, 83, 9)),
        "emerald": ((52, 211, 153), (4, 120, 87)),
        "coral": ((244, 63, 94), (190, 18, 60)),
    }

    # LEFT PANEL: KID-FRIENDLY VISUAL BLOCKS
    models = step.get("models", [])
    model_start_y = 150
    unit_box_w = 64
    unit_box_h = 48
    label_font = get_font(FONT_BOLD, 18)
    num_font = get_font(FONT_BOLD, 17)

    anim_progress = min(1.0, progress / 0.30)

    for m_idx, m in enumerate(models):
        y = model_start_y + m_idx * 85
        label = m.get("label", "")
        units = max(1, min(7, int(m.get("units", 1))))
        c_fill, c_border = COLOR_MAP.get(m.get("color", "cyan"), COLOR_MAP["cyan"])

        draw.text((60, y + 12), label, fill=(226, 232, 240), font=label_font)

        bar_x = 280
        for u in range(units):
            bx1 = bar_x + u * unit_box_w
            bx2 = bx1 + int(unit_box_w * anim_progress)
            by1 = y
            by2 = y + unit_box_h

            if anim_progress > 0.05:
                draw.rectangle([bx1, by1, bx2, by2], fill=c_border, outline=c_fill, width=2)
                draw.text((bx1 + 18, by1 + 12), "1 u", fill=(241, 245, 249), font=num_font)

        if m.get("extra_value"):
            ex_x = bar_x + units * unit_box_w + 12
            draw.text((ex_x, y + 12), f"({m['extra_value']})", fill=(251, 191, 36), font=label_font)

    # RIGHT PANEL: SIMPLE VISUAL STEPS & JOKES/TIPS
    eq_start_x = 730
    draw.rounded_rectangle([eq_start_x - 20, 120, WIDTH - 40, 560], radius=12, fill=(19, 31, 51), outline=(51, 65, 85), width=1)
    draw.text((eq_start_x, 140), "Easy Peasy Steps:", fill=(148, 163, 184), font=get_font(FONT_BOLD, 22))

    equations = step.get("equations", [])
    eq_font = get_font(FONT_REGULAR, 23)
    for e_idx, eq in enumerate(equations):
        ey = 195 + e_idx * 55
        draw.text((eq_start_x + 10, ey), f"➜  {eq}", fill=(241, 245, 249), font=eq_font)

    # WINNER / ANSWER HIGHLIGHT BOX
    if step.get("answer_highlight"):
        ans_box_y = 480
        draw.rounded_rectangle([eq_start_x, ans_box_y, WIDTH - 60, ans_box_y + 60], radius=8, fill=(4, 120, 87), outline=(52, 211, 153), width=2)
        draw.text((eq_start_x + 15, ans_box_y + 16), step["answer_highlight"], fill=(255, 255, 255), font=get_font(FONT_BOLD, 22))

    # BOTTOM CAPTION
    draw.rounded_rectangle([40, 590, WIDTH - 40, 690], radius=10, fill=(10, 15, 26), outline=(71, 85, 105), width=1)
    narration_text = step.get("narration", "")
    draw.text((60, 625), f"Tutor: \"{narration_text[:110]}...\"", fill=(251, 191, 36), font=get_font(FONT_REGULAR, 22))

    return img

def main():
    parser = argparse.ArgumentParser(description="Fun PSLE Math Video Walkthrough Generator for Kids")
    parser.add_argument("--pdf", default="psle_paper.pdf", help="Path to PDF paper")
    parser.add_argument("--q", type=int, default=2, help="Question number to solve")
    parser.add_argument("-o", "--output", default="math_solution.mp4", help="Output MP4 filename")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  EXTRACTING QUESTION {args.q} FROM: {args.pdf}")
    print(f"=======================================================")
    raw_question = extract_question_from_pdf(args.pdf, args.q)
    print(f"Extracted Question Text:\n{raw_question}\n")

    plan = plan_math_solution(raw_question, args.q)
    with open("math_solution_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    os.makedirs("temp_math", exist_ok=True)
    audio_concat = []
    video_timeline = []

    # 1. INTRO: READ QUESTION ONLY
    print("\n--- 1. Recording Tutor Voiceover (Kid-Friendly & Detailed) ---")
    intro_audio = "temp_math/intro_padded.mp3"
    raw_intro = "temp_math/intro_raw.mp3"
    intro_dur = asyncio.run(synthesize_speech(plan["intro_narration"], raw_intro))
    subprocess.run([
        "ffmpeg", "-y", "-i", raw_intro,
        "-af", f"apad=pad_dur={PAUSE_BETWEEN_STEPS}",
        intro_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    full_intro_dur = intro_dur + PAUSE_BETWEEN_STEPS
    audio_concat.append(f"file '{os.path.abspath(intro_audio)}'")
    video_timeline.append(("intro", 0, full_intro_dur))
    print(f"  [Intro Scene] Question Reading: {full_intro_dur:.2f}s")

    # 2. DETAILED STEP BY STEP SOLUTION
    for idx, step in enumerate(plan["steps"]):
        raw_audio = f"temp_math/step_{idx}_raw.mp3"
        padded_audio = f"temp_math/step_{idx}_padded.mp3"
        dur = asyncio.run(synthesize_speech(step["narration"], raw_audio))

        subprocess.run([
            "ffmpeg", "-y", "-i", raw_audio,
            "-af", f"apad=pad_dur={PAUSE_BETWEEN_STEPS}",
            padded_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        full_step_dur = dur + PAUSE_BETWEEN_STEPS
        audio_concat.append(f"file '{os.path.abspath(padded_audio)}'")
        video_timeline.append(("step", idx, full_step_dur))
        print(f"  [Step {idx+1}] {step.get('title')}: {full_step_dur:.2f}s")

    with open("temp_math/audio_list.txt", "w") as f:
        f.write("\n".join(audio_concat))
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "temp_math/audio_list.txt", "-c", "copy", "temp_math/full_audio.mp3"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # 3. RENDER VISUAL WALKTHROUGH FRAMES
    raw_video = "temp_math/video_raw.mp4"
    proc = subprocess.Popen([
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
    ], stdin=subprocess.PIPE)

    print("\n--- 2. Rendering Visual Video Walkthrough ---")
    for stage_type, idx_val, dur in video_timeline:
        total_frames = int(round(dur * FPS))
        if stage_type == "intro":
            print(f"  Rendering Question Intro ({total_frames} frames)...")
            for f in range(total_frames):
                frame = render_intro_frame(plan, args.q, f / float(total_frames))
                proc.stdin.write(frame.tobytes())
        else:
            print(f"  Rendering Step {idx_val+1} ({total_frames} frames)...")
            for f in range(total_frames):
                frame = render_step_frame(plan, idx_val, args.q, f / float(total_frames))
                proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    # 4. MUX INTO FINAL VIDEO
    print(f"\n--- 3. Finalizing Walkthrough Video: {args.output} ---")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", raw_video,
        "-i", "temp_math/full_audio.mp3",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        args.output
    ], check=True)

    subprocess.run(["rm", "-rf", "temp_math"])
    print(f"\nDone! Fun 10-year-old math video generated: {args.output}")

if __name__ == "__main__":
    main()
