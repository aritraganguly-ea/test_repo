# !pip install easyocr openai

import json
import os
import warnings

import easyocr
import openai
import pandas as pd

# Suppress all warnings.
warnings.filterwarnings("ignore")

openai.api_key = os.getenv("OPENAI_API_KEY")

LLM_PROMPT = """
Extract Flight Metrics for the Latest Year.

1. Identify the latest reporting period (Month, YYYY).
2. Flight Type, such as Departure or Arrival.
3. Extract these four values:
   - international_flights
   - domestic_flights
   - international_passengers
   - domestic_passengers

Respond with ONLY the JSON object exactly in this format, without any markdown or code fences:

{
  "report_period": "Month, YYYY",
  "flight_type": "",
  "metrics": {
     "international_flights": 0,
     "domestic_flights": 0,
     "international_passengers": 0,
     "domestic_passengers": 0
  }
}
"""


def extract_text_with_easyocr(path: str) -> str:
    reader = easyocr.Reader(["en"], gpu=False)
    return "\n".join(reader.readtext(path, detail=0))


def clean_json_response(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response: {content!r}")
    return content[start:end]


def extract_metrics_with_llm(text: str) -> dict:
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract structured data only."},
            {"role": "user", "content": LLM_PROMPT + "\n\n" + text},
        ],
        temperature=0,
        max_tokens=500,
    )
    raw = resp.choices[0].message.content or ""
    print("🔍 Raw LLM returned:", repr(raw))

    cleaned = clean_json_response(raw)
    print("🔍 Cleaned JSON:", cleaned)

    return json.loads(cleaned)


def build_table(llm_output: dict) -> pd.DataFrame:
    date = llm_output["report_period"]
    type = llm_output["flight_type"]
    m = llm_output["metrics"]
    rows = [
        (
            date,
            "Total number of international flights",
            type,
            m["international_flights"],
        ),
        (date, "Total number of internal flights", type, m["domestic_flights"]),
        (
            date,
            "Total number of passengers on international flights",
            type,
            m["international_passengers"],
        ),
        (
            date,
            "Total number of passengers on internal flights",
            type,
            m["domestic_passengers"],
        ),
    ]
    return pd.DataFrame(rows, columns=["Date", "Metrics", "Type of Flight", "Values"])


if __name__ == "__main__":
    img_path = "path/to/your/image.png"

    # 1. OCR the Image.
    text = extract_text_with_easyocr(img_path)

    # 2. Extract via LLM.
    llm_data = extract_metrics_with_llm(text)

    # 3. Build and Print DataFrame.
    df = build_table(llm_data)
    print(df.to_markdown(index=False))
