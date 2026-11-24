import os
import sys
import json
import re
import base64
import argparse
import subprocess
import requests
import soundfile as sf
import fitz  # PyMuPDF
from kokoro_onnx import Kokoro

# ---------------- CONFIGURATION ----------------
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = 0.95

print("Loading Kokoro-82M ONNX model...")
kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")

def extract_question_and_render_page(pdf_path, question_num=1, output_img="temp_manim/page_preview.png"):
    """
    Extracts question text and renders the PDF page containing the question
    as a high-res image for vision models to inspect diagrams/shapes.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    target_page_idx = 0
    q_curr = str(question_num)
    q_next = str(question_num + 1)

    # Search for page index containing question
    for p_idx, page in enumerate(doc):
        p_text = page.get_text()
        full_text += p_text + "\n"
        if re.search(rf"(?:^|\n)\s*{q_curr}[\.\)]\s+", p_text):
            target_page_idx = p_idx

    # Render page to image for vision processing
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    page = doc[target_page_idx]
    pix = page.get_pixmap(dpi=150)
    pix.save(output_img)
    doc.close()

    # Extract text block
    patterns = [
        rf"(?:^|\n)\s*{q_curr}[\.\)]\s+([\s\S]*?)(?=(?:^|\n)\s*{q_next}[\.\)]\s+|$)",
        rf"\b{q_curr}[\.\)]\s+([\s\S]*?)(?=\b{q_next}[\.\)]\s+|$)",
        rf"(?:^|\n)\s*{q_curr}\s*\n([\s\S]*?)(?=(?:^|\n)\s*{q_next}\s*\n|$)"
    ]

    for pat in patterns:
        m = re.search(pat, full_text)
        if m and len(m.group(1).strip()) > 15:
            raw_q = m.group(1).strip()
            return re.sub(r"Page\s+\d+.*", "", raw_q, flags=re.IGNORECASE).strip(), output_img

    return full_text[:400], output_img

def query_llm(prompt, image_path, provider, model_name, host="http://localhost:11434"):
    """Dispatches the prompt and base64-encoded page screenshot to Ollama or OpenRouter."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    if provider == "ollama":
        print(f"Connecting to local Ollama ({host}) using vision model: '{model_name}'...")
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [img_b64],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2}
        }
        res = requests.post(f"{host}/api/generate", json=payload)
        if res.status_code != 200:
            print(f"Ollama Error ({res.status_code}): {res.text}", file=sys.stderr)
            sys.exit(1)
        raw_reply = res.json().get("response", "")

    else:  # OpenRouter
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
            sys.exit(1)

        print(f"Connecting to OpenRouter using model: '{model_name}'...")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }
            ],
            "temperature": 0.2
        }
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        if res.status_code != 200:
            print(f"OpenRouter Error ({res.status_code}): {res.text}", file=sys.stderr)
            sys.exit(1)
        raw_reply = res.json()["choices"][0]["message"]["content"]

    clean = re.sub(r"^```(?:json)?\s*", "", raw_reply.strip())
    clean = re.sub(r"\s*```$", "", clean)
    return json.loads(clean)

def plan_lesson(question_text, image_path, question_num, provider, model_name, host):
    prompt = f"""
You are an energetic primary school Math tutor explaining a problem to a 10-year-old child (Primary 4/5).
Look at the attached image of the exam paper page containing Question {question_num} and analyze the diagrams, tables, angles, or shapes.

Question text extracted:
\"\"\"
{question_text}
\"\"\"

Requirements:
1. "question_stem": Question statement text WITHOUT options (1)-(4).
2. "intro_narration": MUST start with:
   "Let's have a look at the question we are solving. " followed by reading the question stem cleanly (DO NOT read options).
3. "visual_type": Select the best visual presentation for this problem based on the diagram:
   - "bars" (if the question uses ratio, quantities, bar models, or tape diagrams)
   - "cards" (if the question uses unit conversions, measurements, or calculation facts)
   - "fractions" (if the question uses fractions, mixed numbers, or pizza/circle slices)
4. "steps": An array of 3 pedagogical steps for this question:
   - "title": Short catchy title.
   - "narration": Humorous, clear explanation for a 10-year-old.
   - "visual_data":
       - If visual_type is "bars": {{"bars": [{{"label": "Part A", "units": 3, "color": "TEAL"}}, {{"label": "Part B", "units": 2, "color": "ORANGE"}}]}}
       - If visual_type is "cards": {{"cards": [{{"heading": "Fact", "value": "1 m = 100 cm", "color": "BLUE"}}]}}
       - If visual_type is "fractions": {{"wholes": 5, "denom": 7, "numer": 6}}
   - "equations": List of 1-3 arithmetic equations for this step.
   - "highlight": Final answer string on step 3 (e.g. "Answer: Option (4) 60 m 70 cm"), else null.

Return ONLY valid JSON matching this schema:
{{
  "question_stem": "The question text without options",
  "intro_narration": "Let's have a look at the question we are solving. ...",
  "visual_type": "cards",
  "steps": [
    {{
      "title": "Step 1: Heading",
      "narration": "Tutor explanation for step 1...",
      "visual_data": {{"cards": [{{"heading": "Rule", "value": "1 m = 100 cm", "color": "BLUE"}}]}},
      "equations": ["100 cm = 1 m"],
      "highlight": null
    }}
  ]
}}
"""
    return query_llm(prompt, image_path, provider, model_name, host)

def synthesize_speech_kokoro(text, audio_path):
    samples, sample_rate = kokoro.create(
        text,
        voice=KOKORO_VOICE,
        speed=KOKORO_SPEED,
        lang="en-us"
    )
    sf.write(audio_path, samples, sample_rate)
    return len(samples) / float(sample_rate)

def build_manim_script(plan, durations):
    clean_stem = plan.get("question_stem", "").replace('"', '\\"').replace("\n", " ")
    visual_type = plan.get("visual_type", "cards")
    intro_dur = durations[0]

    script = f'''from manim import *
import math

class MathWalkthrough(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"

        # ==================== INTRO SCENE ====================
        intro_hdr = Text("Question Examination", font_size=28, color=TEAL).to_edge(UP, buff=0.4)
        card = RoundedRectangle(corner_radius=0.15, width=11.2, height=3.6, color=BLUE_D, fill_color="#1e293b", fill_opacity=0.95).next_to(intro_hdr, DOWN, buff=0.3)
        q_label = Paragraph(
            "{clean_stem}",
            width=10.2,
            font_size=24,
            color=WHITE,
            alignment="center"
        ).move_to(card.get_center())

        self.play(FadeIn(intro_hdr), Create(card), run_time=1.0)
        self.play(Write(q_label), run_time=2.0)
        self.wait(max(0.5, {intro_dur} - 3.8))
        self.play(FadeOut(intro_hdr), FadeOut(card), FadeOut(q_label), run_time=0.6)
'''

    for s_idx, step in enumerate(plan["steps"]):
        step_dur = durations[s_idx + 1]
        title = step.get("title", f"Step {s_idx+1}").replace('"', '\\"')
        eqs = step.get("equations", [])
        v_data = step.get("visual_data", {})
        highlight = step.get("highlight")

        script += f'''
        # ==================== STEP {s_idx + 1} ====================
        hdr = Text("{title}", font_size=24, color=YELLOW).to_edge(UP, buff=0.35)
        self.play(FadeIn(hdr), run_time=0.5)

        vis_group = VGroup()
'''

        if visual_type == "fractions" or "wholes" in v_data:
            wholes = int(v_data.get("wholes", 5))
            denom = int(v_data.get("denom", 7))
            script += f'''
        pizza_row = VGroup()
        for i in range({min(5, wholes)}):
            c = Circle(radius=0.55, color=ORANGE, fill_color=MAROON_E, fill_opacity=0.85)
            spokes = VGroup()
            for k in range({denom}):
                ang = k * (TAU / {denom})
                line = Line(c.get_center(), c.get_center() + 0.55 * np.array([math.cos(ang), math.sin(ang), 0]), stroke_width=1.5, color=GOLD_A)
                spokes.add(line)
            tag = Text("{denom}/{denom}", font_size=14, color=WHITE).move_to(c.get_center())
            pizza_row.add(VGroup(c, spokes, tag))
        pizza_row.arrange(RIGHT, buff=0.3).shift(UP * 0.9)
        vis_group.add(pizza_row)
        self.play(Create(pizza_row), run_time=1.8)
'''
        elif visual_type == "bars" or "bars" in v_data:
            bars_list = v_data.get("bars", [{"label": "Units", "units": 3, "color": "TEAL"}])
            script += f'''
        bar_group = VGroup()
        for b_info in {bars_list}:
            row = VGroup()
            u_count = min(6, int(b_info.get("units", 1)))
            lbl = Text(str(b_info.get("label", "")), font_size=18, color=WHITE)
            blocks = VGroup()
            for _ in range(u_count):
                sq = Rectangle(width=0.9, height=0.55, color=TEAL, fill_color=BLUE_E, fill_opacity=0.8)
                t = Text("1u", font_size=14, color=WHITE).move_to(sq.get_center())
                blocks.add(VGroup(sq, t))
            blocks.arrange(RIGHT, buff=0.08)
            row.add(lbl, blocks)
            row.arrange(RIGHT, buff=0.4)
            bar_group.add(row)
        bar_group.arrange(DOWN, buff=0.4).shift(UP * 0.8)
        vis_group.add(bar_group)
        self.play(Create(bar_group), run_time=1.8)
'''
        else:
            cards_list = v_data.get("cards", [{"heading": "Fact", "value": eqs[0] if eqs else "Working", "color": "BLUE"}])
            script += f'''
        card_row = VGroup()
        for c_info in {cards_list}:
            c_box = RoundedRectangle(corner_radius=0.12, width=3.4, height=1.6, color=TEAL, fill_color="#1e293b", fill_opacity=0.9)
            h = Text(str(c_info.get("heading", "")), font_size=17, color=YELLOW_B).move_to(c_box.get_top() + DOWN * 0.35)
            v = Text(str(c_info.get("value", "")), font_size=20, color=WHITE).move_to(c_box.get_center() + DOWN * 0.15)
            card_row.add(VGroup(c_box, h, v))
        card_row.arrange(RIGHT, buff=0.5).shift(UP * 0.9)
        vis_group.add(card_row)
        self.play(FadeIn(card_row), run_time=1.5)
'''

        script += f'''
        eq_group = VGroup()
'''
        for eq_idx, eq in enumerate(eqs[:3]):
            clean_eq = eq.replace('"', '\\"').replace("\n", " ")
            color = "GREEN_B" if eq_idx == len(eqs)-1 else "WHITE"
            script += f'''
        eq_group.add(Text("{clean_eq}", font_size=24, color={color}))
'''
        script += f'''
        eq_group.arrange(DOWN, buff=0.3).shift(DOWN * 0.9)
        self.play(Write(eq_group), run_time=1.5)
'''

        if highlight:
            clean_hl = highlight.replace('"', '\\"')
            script += f'''
        ans_box = RoundedRectangle(corner_radius=0.15, width=8.5, height=1.1, color=GREEN_A, fill_color="#047857", fill_opacity=1.0).to_edge(DOWN, buff=0.4)
        ans_txt = Text("{clean_hl}", font_size=24, color=YELLOW_A).move_to(ans_box.get_center())
        self.play(FadeIn(ans_box), Write(ans_txt), run_time=1.0)
        self.wait(max(0.5, {step_dur} - 4.8))
        self.play(FadeOut(hdr), FadeOut(vis_group), FadeOut(eq_group), FadeOut(ans_box), FadeOut(ans_txt), run_time=0.6)
'''
        else:
            script += f'''
        self.wait(max(0.5, {step_dur} - 3.8))
        self.play(FadeOut(hdr), FadeOut(vis_group), FadeOut(eq_group), run_time=0.6)
'''

    return script

def main():
    parser = argparse.ArgumentParser(description="Multimodal PSLE Math Explainer")
    parser.add_argument("--pdf", default="psle_paper.pdf", help="Path to PDF paper")
    parser.add_argument("--q", type=int, default=1, help="Question number")
    parser.add_argument("-o", "--output", default="math_solution.mp4", help="Output video")
    
    # Provider & Model arguments (No hardcoded models)
    parser.add_argument("--provider", choices=["ollama", "openrouter"], default=os.environ.get("LLM_PROVIDER", "openrouter"), help="LLM backend: ollama or openrouter")
    parser.add_argument("-m", "--model", default=os.environ.get("OLLAMA_MODEL") or os.environ.get("OPENROUTER_MODEL"), help="Model name (e.g., gemma4 for Ollama, or openrouter/free)")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"), help="Ollama server URL")

    args = parser.parse_args()

    # Dynamic model defaults based on provider
    if not args.model:
        args.model = "gemma4" if args.provider == "ollama" else "openrouter/free"

    print(f"\n=== SOLVING QUESTION {args.q} USING {args.provider.upper()} ({args.model}) ===")
    os.makedirs("temp_manim", exist_ok=True)
    raw_q, preview_img = extract_question_and_render_page(args.pdf, args.q, "temp_manim/page_preview.png")
    print(f"Question Isolated:\n{raw_q}\n")
    print(f"Rendered Page Snapshot for Vision Inspection: {preview_img}")

    print("\n--- Generating Multimodal Math Solution Plan ---")
    plan = plan_lesson(raw_q, preview_img, args.q, args.provider, args.model, args.ollama_host)
    with open("lesson_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    audio_files = []
    durations = []

    print(f"\n--- Synthesizing Kokoro Voiceover ({KOKORO_VOICE}) ---")
    intro_audio = "temp_manim/intro.wav"
    intro_dur = synthesize_speech_kokoro(plan["intro_narration"], intro_audio)
    durations.append(intro_dur)
    audio_files.append(f"file '{os.path.abspath(intro_audio)}'")
    print(f"  [Intro Audio] {intro_dur:.2f}s")

    for idx, s in enumerate(plan["steps"]):
        s_audio = f"temp_manim/step_{idx}.wav"
        s_dur = synthesize_speech_kokoro(s["narration"], s_audio)
        durations.append(s_dur)
        audio_files.append(f"file '{os.path.abspath(s_audio)}'")
        print(f"  [Step {idx+1} Audio] {s_dur:.2f}s")

    with open("temp_manim/audio_list.txt", "w") as f:
        f.write("\n".join(audio_files))
    combined_audio = "temp_manim/narration.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "temp_manim/audio_list.txt", "-c", "copy", combined_audio],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    print("\n--- Rendering Manim Animation ---")
    script_content = build_manim_script(plan, durations)
    with open("temp_manim/scene.py", "w", encoding="utf-8") as f:
        f.write(script_content)

    subprocess.run([
        "manim", "-qm", "--disable_caching",
        "-o", "manim_raw.mp4",
        "--media_dir", "temp_manim/media",
        "temp_manim/scene.py", "MathWalkthrough"
    ], check=True)

    rendered_raw = "temp_manim/media/videos/scene/720p30/manim_raw.mp4"

    print(f"\n--- Finalizing Video: {args.output} ---")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", rendered_raw,
        "-i", combined_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        args.output
    ], check=True)

    subprocess.run(["rm", "-rf", "temp_manim"])
    print(f"\nSUCCESS! Video walkthrough complete: {args.output}")

if __name__ == "__main__":
    main()
