import os
import sys
import json
import re
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY is not set.", file=sys.stderr)
    print("Export it using: echo 'export OPENROUTER_API_KEY=\"your_key\"' >> ~/.bashrc && source ~/.bashrc", file=sys.stderr)
    sys.exit(1)

input_file = sys.argv[1] if len(sys.argv) > 1 else "sample_script.txt"
output_file = sys.argv[2] if len(sys.argv) > 2 else "output.json"

try:
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
except Exception as e:
    print(f"Error reading {input_file}: {e}", file=sys.stderr)
    sys.exit(1)

prompt = f"""
You are a vetran theatre actor know for your powerful dialogu delivery .Analyze the narration script below.

Requirements:
1. Break down the script into every individual sentence.
2. For each sentence, identify words that are most important to that specific sentence and to the entire document.
3. Assign an importance score (integer from 1 to 10, where 10 is the most important) to each identified key word. Multiple words can share the same score.
4. Output MUST be ONLY valid JSON matching this schema:

[
  {{
    "sentence": "The full sentence text here.",
    "words": {{
      "word1": 10,
      "word2": 8
    }}
  }}
]

Important:
- Return ONLY raw JSON without markdown formatting (no ```json).

Script:
\"\"\"
{content}
\"\"\"
"""

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/drshariqalikhan/explainer",
    "X-Title": "Explainer Script Analyzer"
}

# Uses the openrouter/free router endpoint
payload = {
    "model": "openrouter/free",
    "messages": [
        {"role": "system", "content": "You are a JSON-only response analyzer. Output only valid JSON."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.2
}

print(f"Sending request to OpenRouter using 'openrouter/free'...")

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    data=json.dumps(payload)
)

if response.status_code != 200:
    print(f"API Error ({response.status_code}): {response.text}", file=sys.stderr)
    sys.exit(1)

res_json = response.json()

# OpenRouter reports which underlying model was chosen
resolved_model = res_json.get("model", "openrouter/free")
print(f"Served by: {resolved_model}")

raw_reply = res_json["choices"][0]["message"]["content"].strip()
clean_reply = re.sub(r"^```(?:json)?\s*", "", raw_reply)
clean_reply = re.sub(r"\s*```$", "", clean_reply)

try:
    parsed_json = json.loads(clean_reply)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=2)
    print(f"\nSuccess! Results saved to: {output_file}\n")
    print(json.dumps(parsed_json, indent=2))
except json.JSONDecodeError:
    print("Warning: Response was not clean JSON. Output saved to raw_output.txt", file=sys.stderr)
    with open("raw_output.txt", "w", encoding="utf-8") as f:
        f.write(clean_reply)
    print(clean_reply)
