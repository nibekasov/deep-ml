from typing import List, Tuple

def run_etl(csv_text: str) -> List[Tuple[str, float]]:
    """ETL over CSV with columns user_id,event_type,value in any order (header-driven)."""
    if not csv_text or not csv_text.strip():
        return []

    # ignore blank lines
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return []

    # --- Extract: parse header -> column indices
    header = [h.strip().lower() for h in lines[0].split(",")]
    try:
        idx_user = header.index("user_id")
        idx_type = header.index("event_type")
        idx_val  = header.index("value")
    except ValueError:
        # required columns not found
        return []

    data_lines = lines[1:]

    # --- Transform: filter purchases, cast value, drop invalid, aggregate per user
    agg = {}
    for ln in data_lines:
        parts = [p.strip() for p in ln.split(",")]
        # skip rows with wrong column count
        if len(parts) < max(