# !pip install openai pdfminer.six

import json
import os
import re
import warnings

import openai
import pandas as pd
from pdfminer.high_level import extract_text

# Suppress all warnings.
warnings.filterwarnings("ignore")

openai.api_key = os.getenv("OPENAI_API_KEY")

LLM_PROMPT = """
Extract Flight Metrics for the Latest Year.

1. Identify the latest reporting period (Month, YYYY).
2. For each flight direction (Departure and Arrival), extract these four values:
   - international_flights
   - domestic_flights
   - international_passengers
   - domestic_passengers

Respond with ONLY a JSON array of two objects in this exact format, without any markdown or code fences:

[
  {
    "report_period": "Month, YYYY",
    "flight_type": "Departure",
    "metrics": {
       "international_flights": 0,
       "domestic_flights": 0,
       "international_passengers": 0,
       "domestic_passengers": 0
    }
  },
  {
    "report_period": "Month, YYYY",
    "flight_type": "Arrival",
    "metrics": {
       "international_flights": 0,
       "domestic_flights": 0,
       "international_passengers": 0,
       "domestic_passengers": 0
    }
  }
]
"""


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from PDF using pdfminer.six."""
    return extract_text(pdf_path)


def clean_json_response(content: str) -> str:
    """Slice out the JSON array from the LLM's reply."""
    start = content.find("[")
    end = content.rfind("]") + 1
    if start < 0 or end < 0:
        raise ValueError(f"No JSON array found:\n{content!r}")
    return content[start:end]


def normalize_number_commas(json_str: str) -> str:
    """
    Remove commas used as thousands separators in numbers,
    e.g. turns '15,907' into '15907'.
    """
    return re.sub(r"(?<=\d),(?=\d)", "", json_str)


def extract_metrics_with_llm(raw_text: str) -> list[dict]:
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract structured data only."},
            {"role": "user", "content": LLM_PROMPT + "\n\n" + raw_text},
        ],
        temperature=0,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content or ""
    print("🔍 Raw LLM response:", repr(raw))

    cleaned = clean_json_response(raw)
    print("🔍 Cleaned JSON array:", cleaned)

    normalized = normalize_number_commas(cleaned)
    print("🔍 Normalized JSON array:", normalized)

    return json.loads(normalized)


def build_table(metrics_list: list[dict]) -> pd.DataFrame:
    rows = []
    for obj in metrics_list:
        date = obj["report_period"]
        ft = obj["flight_type"]
        m = obj["metrics"]
        rows += [
            (
                date,
                "Total number of international flights",
                ft,
                m["international_flights"],
            ),
            (date, "Total number of internal flights", ft, m["domestic_flights"]),
            (
                date,
                "Total number of passengers on international flights",
                ft,
                m["international_passengers"],
            ),
            (
                date,
                "Total number of passengers on internal flights",
                ft,
                m["domestic_passengers"],
            ),
        ]
    return pd.DataFrame(rows, columns=["Date", "Metrics", "Type of Flight", "Values"])


if __name__ == "__main__":
    pdf_path = "path/to/your/pdf_file.pdf"

    # 1. Extract text.
    raw_text = extract_text_from_pdf(pdf_path)

    # 2. LLM Extraction.
    metrics_list = extract_metrics_with_llm(raw_text)

    # 3. Build and Display DataFrame.
    df = build_table(metrics_list)
    print(df.to_markdown(index=False))
