def safe_ratio(num, den):
    try:
        return num/den if den else None
    except Exception:
        return None
