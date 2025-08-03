# !pip install openai pdfminer.six pandas boto3

import io
import json
import logging
import os
import re
import warnings
from datetime import datetime

import boto3
import openai
import pandas as pd
from dotenv import load_dotenv
from pdfminer.high_level import extract_text

load_dotenv()

# Suppress all warnings.
warnings.filterwarnings("ignore")

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("OmanTourismETL")


# Load environment variables.
openai.api_key = os.environ.get("OPENAI_API_KEY")
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "oman-tourism")
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw_data/")
CLEAN_PREFIX = os.environ.get("CLEAN_PREFIX", "clean_data/")


# LLM prompt for extracting flight metrics.
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
    cleaned = clean_json_response(raw)
    normalized = normalize_number_commas(cleaned)
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


def run_etl() -> None:
    """
    Extract all raw PDFs from S3 under raw_prefix, process each to extract flight metrics,
    combine into a single DataFrame, and upload as CSV to clean_prefix in S3.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    output_filename = f"flight_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    dfs = []

    # Iterate through all objects in the raw prefix.
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith(".pdf"):
                continue

            # Download the PDF into memory.
            logger.info(f"Processing {key} ...")
            pdf_body = s3.get_object(Bucket=BUCKET_NAME, Key=key)["Body"].read()

            # Extract text from the PDF.
            raw_text = extract_text_from_pdf(io.BytesIO(pdf_body))

            # Use LLM to extract metrics.
            try:
                metrics_list = extract_metrics_with_llm(raw_text)
            except ValueError as ve:
                logger.warning(f"Failed to parse JSON from LLM for {key}: {ve}")
                continue

            # Build the DataFrame.
            df = build_table(metrics_list)
            df["SourceFile"] = key
            dfs.append(df)

    # Combine DataFrames.
    result_df = (
        pd.concat(dfs, ignore_index=True)
        if dfs
        else (_ for _ in ()).throw(Exception("No valid PDF files found in S3"))
    )

    # Upload the combined DataFrame as a CSV to S3.
    logger.info(f"Upload CSV to S3 bucket: {BUCKET_NAME} under {CLEAN_PREFIX}")

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=os.path.join(CLEAN_PREFIX, output_filename),
        Body=csv_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )

    logger.info("ETL process completed successfully.")


if __name__ == "__main__":
    run_etl()
