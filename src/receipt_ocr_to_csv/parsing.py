import re
from typing import List, Dict, Tuple
import pandas as pd

from .utils import (
    MONEY_RE,
    clean_text,
    parse_amount,
    find_all_amounts,
    safe_dateparse,
)

TOTAL_TOKENS = (
    "total amt", "total amount", "grand total", "net amount",
    "total", "amount", "amt", "payable", "balance"
)


def extract_date(candidates: List[str]):
    # Prefer lines containing date keywords
    keys = ("bill date", "date", "dt")
    prioritized = [x for x in candidates if any(k in x.lower() for k in keys)] + candidates
    seen = set()
    for line in prioritized:
        if line in seen: continue
        seen.add(line)
        line = clean_text(line)
        dt = safe_dateparse(line)
        if dt:
            return dt
    return None


def extract_merchant(candidates: List[str]):
    head = [clean_text(x) for x in candidates[:8]]
    head = [h for h in head if sum(ch.isalpha() for ch in h) >= 5]
    if not head:
        return None
    # score: letters + bonus for all-caps words
    def score(s):
        letters = sum(ch.isalpha() for ch in s)
        caps_bonus = sum(1 for w in s.split() if len(w)>=3 and w.isupper())
        return letters + 2*caps_bonus
    return max(head, key=score)


def extract_total(candidates: List[str]) -> Tuple[str, float]:
    best_cur, best_amt = None, None
    # Prioritize lines containing total-like tokens
    for line in candidates:
        cl = clean_text(line).lower()
        if any(tok in cl for tok in TOTAL_TOKENS):
            cur, amt = parse_amount(cl)
            if amt is not None and (best_amt is None or amt > best_amt):
                best_cur, best_amt = cur, amt
    # Fallback: global max amount in all lines
    if best_amt is None:
        for line in candidates:
            for cur, amt in find_all_amounts(line):
                if amt is not None and (best_amt is None or amt > best_amt):
                    best_cur, best_amt = cur, amt
    return best_cur, best_amt

def extract_items(lines: List[str]) -> pd.DataFrame:
    items = []
    for raw in lines:
        line = clean_text(raw)
        # Must contain letters (item name) and end with a money-like number
        if not any(c.isalpha() for c in line):
            continue
        matches = list(MONEY_RE.finditer(line))
        if not matches:
            continue
        last = matches[-1]
        # Require the amount to be near end of line (within last 6 chars)
        if len(line) - last.end() > 6:
            continue

        # price
        _, price = parse_amount(line[last.start():])
        if price is None:
            continue

        # left side as item text (must have >= 3 letters)
        left = line[:last.start()].strip(" .-xX")
        if sum(ch.isalpha() for ch in left) < 3:
            continue

        # qty heuristic: "... x2" or has a standalone integer token near end-left
        import re
        q = re.search(r"(?:^|\s)(?:x|qty)\s*(\d+(?:\.\d+)?)\b", left, re.IGNORECASE)
        qty = float(q.group(1)) if q else 1.0
        if q:
            left = re.sub(r"(?:^|\s)(?:x|qty)\s*\d+(?:\.\d+)?", "", left, flags=re.IGNORECASE).strip()

        unit = round(price/qty, 2) if qty else round(price, 2)
        items.append({"item": left, "qty": qty, "unit_price": unit, "line_total": round(price,2)})

    if not items:
        return pd.DataFrame(columns=["item","qty","unit_price","line_total"])
    return pd.DataFrame(items)


def parse_receipt(lines_dicts: List[Dict]):
    texts = [d.get("text","") for d in lines_dicts if d.get("text")]
    merchant = extract_merchant(texts)
    date_str = extract_date(texts)
    currency, total = extract_total(texts)
    items_df = extract_items(texts)
    meta = {"merchant": merchant, "date": date_str, "currency": currency, "total": total}
    return meta, items_df
