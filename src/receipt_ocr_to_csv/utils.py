import re
from typing import Iterable, List, Optional, Tuple
import unicodedata
import dateparser

# Normalize currency symbols and common OCR noise
CURRENCY_MAP = {
    "₹": "INR", "Rs": "INR", "rs": "INR", "INR": "INR",
    "$": "USD", "usd": "USD",
    "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
}

# Amounts like ₹199, 199.00, 1,299.50, $3.5
MONEY_RE = re.compile(
    r"(?P<cur>[₹$€£]|INR|Rs|rs|USD|EUR|GBP)?\s*"
    r"(?P<amt>\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
)

DATE_HINT = re.compile(r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}")

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def strip_ocr_noise(s: str) -> str:
    # remove control chars and normalize unicode
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch if (32 <= ord(ch) <= 126) or ch in "₹€£" else " " for ch in s)
    s = re.sub(r"[^\w₹$€£.,:/\-\(\) xX]", " ", s)
    return normalize_spaces(s)

def clean_text(s: str) -> str:
    return normalize_spaces(strip_ocr_noise(s))

def detect_currency(token: Optional[str]) -> Optional[str]:
    if not token: return None
    t = token.strip()
    return CURRENCY_MAP.get(t, CURRENCY_MAP.get(t.upper()))

def parse_amount(s: str) -> Tuple[Optional[str], Optional[float]]:
    m = MONEY_RE.search(s)
    if not m:
        return None, None
    cur = detect_currency(m.group("cur"))
    amt_raw = m.group("amt").replace(",", "").replace(" ", "")
    try:
        return cur, float(amt_raw)
    except Exception:
        return cur, None

def find_all_amounts(s: str) -> List[Tuple[Optional[str], float]]:
    out: List[Tuple[Optional[str], float]] = []
    for m in MONEY_RE.finditer(s):
        cur = detect_currency(m.group("cur"))
        try:
            amt = float(m.group("amt").replace(",", "").replace(" ", ""))
            out.append((cur, amt))
        except Exception:
            continue
    return out

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def safe_dateparse(s: str) -> Optional[str]:
    if not s:
        return None
    if not (DATE_HINT.search(s) or any(ch.isdigit() for ch in s)):
        return None
    # Keep settings minimal for broad compatibility across dateparser versions
    dt = dateparser.parse(
        s,
        settings={
            "PREFER_DAY_OF_MONTH": "first",
            "DATE_ORDER": "DMY",
            # Do NOT set RELATIVE_BASE or REQUIRE_PARTS here to avoid TypeError
        },
    )
    return dt.date().isoformat() if dt else None


def head(lines: Iterable[str], n: int = 5) -> List[str]:
    out = []
    for i, x in enumerate(lines):
        if i >= n: break
        out.append(x)
    return out
