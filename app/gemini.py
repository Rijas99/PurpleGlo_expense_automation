from __future__ import annotations

import json
import re
from io import BytesIO

from PIL import Image

from app.config import settings
from app.constants import CATEGORIES

_JSON_OBJECT = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)

ANALYZE_PROMPT = """Analyze this receipt image and extract the following information.

Return ONLY a valid JSON object with these exact keys:
- "date": YYYY-MM-DD format (required)
- "time": HH:MM format (24-hour) or null if not available
- "amount": numeric value only (required)
- "description": short description text (required)
- "category": must be one of these exact strings:
  "Hotel Booking", "Food & Beverages", "Visa & Ticket", "Parking", "R & D Expenses", "Subscriptions",
  "Office - Tools & Consumables", "Project - Consumables", "Transportation",
  "Project Expenses - Miscellaneous", "Office Expenses - Miscellaneous", "Can't classify"

Example:
{"date": "2024-01-15", "time": "14:30", "amount": 45.50, "description": "Restaurant meal", "category": "Food & Beverages"}
"""


def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty response from Gemini.")
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            pass
    matches = _JSON_OBJECT.findall(text)
    matches.sort(key=len, reverse=True)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in Gemini response.")


def analyze_receipt_bytes(image_bytes: bytes) -> dict:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(settings.gemini_model)
    img = Image.open(BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    response = model.generate_content(
        [ANALYZE_PROMPT, img],
        generation_config={"temperature": 0, "max_output_tokens": 1000},
    )
    if not response or not getattr(response, "text", None):
        raise RuntimeError("Empty response from Gemini.")
    data = _extract_json(response.text)
    category = data.get("category")
    if category not in CATEGORIES:
        data["category"] = "Can't classify"
    return data
