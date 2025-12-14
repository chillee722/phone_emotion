import time
import json
import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path

def parse_div(soup: BeautifulSoup, div_id: str):
    div = soup.find("div", id=div_id)
    if div is None:
        raise RuntimeError(f"Cannot find div#{div_id}")

    available = div.get("data-available", "false") == "true"
    if not available:
        return None

    def f(attr: str) -> float:
        return float(div[attr])

    return {
        "p10": f("data-p10"),
        "p25": f("data-p25"),
        "p50": f("data-p50"),
        "p75": f("data-p75"),
        "p90": f("data-p90"),
        "mean": f("data-mean"),
        "std": f("data-std"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="backend public stats url, e.g. https://xxx.onrender.com/public-stats")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "backend" / "reference_stats.json"))
    ap.add_argument("--window_days", type=int, default=0)
    args = ap.parse_args()

    url = args.url
    if "?" in url:
        fetch_url = url
    else:
        fetch_url = f"{url}?window_days={args.window_days}"

    # Render free는 sleep 후 첫 응답이 느릴 수 있어서 timeout↑ + retry
    last_err = None
    for attempt in range(5):
        try:
            r = requests.get(fetch_url, timeout=60)  # 10초 -> 60초
            r.raise_for_status()
            break
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))  # 2s,4s,6s,8s,10s
    else:
        raise last_err

    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    updated_at_span = soup.find("span", id="updated-at")
    updated_at = float(updated_at_span["data-ts"]) if updated_at_span and updated_at_span.get("data-ts") else time.time()

    sample = soup.find("span", id="sample-size")
    n_anx = int(sample.get("data-anxiety", "0")) if sample else 0
    n_fat = int(sample.get("data-fatigue", "0")) if sample else 0
    n_foc = int(sample.get("data-focus", "0")) if sample else 0

    anxiety = parse_div(soup, "anxiety")
    fatigue = parse_div(soup, "fatigue")
    focus = parse_div(soup, "focus")

    data = {
        "updated_at": updated_at,
        "source": fetch_url,
        "n": {"anxiety": n_anx, "fatigue": n_fat, "focus": n_foc},
        "anxiety": anxiety,
        "fatigue": fatigue,
        "focus": focus,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote: {out_path}")
    print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
