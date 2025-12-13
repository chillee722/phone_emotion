from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import sqlite3, json, time, os
import numpy as np

APP_TITLE = "Fidget Emotion Backend"
DB_PATH = os.environ.get("DB_PATH", "events.db")  # Render에서는 기본으로 로컬 파일
REFERENCE_JSON_PATH = os.environ.get("REFERENCE_JSON_PATH", "reference_stats.json")

MIN_N_FOR_PUBLIC_STATS = 1  # 과제/데모 용도로 낮춰둠(운영이면 30 추천)

app = FastAPI(title=APP_TITLE)

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        user_id TEXT,
        payload_json TEXT NOT NULL
    )
    """)
    con.commit()
    con.close()

init_db()

class EventIn(BaseModel):
    ts: float = Field(default_factory=lambda: time.time())
    user_id: Optional[str] = None
    consent: bool
    payload: Dict[str, Any]

@app.get("/health")
def health():
    return {"ok": True, "title": APP_TITLE}

@app.post("/events")
def ingest_event(e: EventIn):
    if not e.consent:
        raise HTTPException(status_code=400, detail="consent=false; not storing")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO events (ts, user_id, payload_json) VALUES (?, ?, ?)",
        (e.ts, e.user_id, json.dumps(e.payload, ensure_ascii=False))
    )
    con.commit()
    con.close()
    return {"stored": True}

def extract_scores(metric: str, window_days: int = 0) -> np.ndarray:
    since_ts = None
    if window_days and window_days > 0:
        since_ts = time.time() - window_days * 86400

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    if since_ts is None:
        cur.execute("SELECT payload_json FROM events")
    else:
        cur.execute("SELECT payload_json FROM events WHERE ts >= ?", (since_ts,))
    rows = cur.fetchall()
    con.close()

    vals = []
    for (payload_json,) in rows:
        try:
            payload = json.loads(payload_json)
            state = payload.get("state_scores", {})
            v = state.get(metric, None)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        except Exception:
            pass

    return np.array(vals, dtype=float)

def compute_percentiles(arr: np.ndarray) -> dict:
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90]).tolist()
    return {
        "p10": float(p10), "p25": float(p25), "p50": float(p50), "p75": float(p75), "p90": float(p90),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }

@app.get("/public-stats", response_class=HTMLResponse)
def public_stats(window_days: int = Query(0, description="0이면 전체, 아니면 최근 N일")):
    # DB에서 퍼센타일 계산
    metrics = ["anxiety_score", "fatigue_score", "focus_score"]
    results = {}
    ns = {}

    for m in metrics:
        arr = extract_scores(m, window_days=window_days)
        ns[m] = int(arr.size)
        if arr.size >= MIN_N_FOR_PUBLIC_STATS:
            results[m] = compute_percentiles(arr)
        else:
            results[m] = None

    updated_at = time.time()

    def div_block(metric_id: str, stats: Optional[dict]) -> str:
        if stats is None:
            return f'<div id="{metric_id}" data-available="false"></div>'
        return (
            f'<div id="{metric_id}" data-available="true" '
            f'data-p10="{stats["p10"]:.4f}" data-p25="{stats["p25"]:.4f}" data-p50="{stats["p50"]:.4f}" '
            f'data-p75="{stats["p75"]:.4f}" data-p90="{stats["p90"]:.4f}" '
            f'data-mean="{stats["mean"]:.4f}" data-std="{stats["std"]:.4f}"></div>'
        )

    html = f"""
    <html>
      <head><meta charset="utf-8"><title>Public Stats</title></head>
      <body>
        <h1>Public Emotion Statistics</h1>
        <p>window_days={window_days}</p>
        <span id="updated-at" data-ts="{updated_at:.4f}">{updated_at:.4f}</span><br/>
        <span id="sample-size"
              data-anxiety="{ns["anxiety_score"]}"
              data-fatigue="{ns["fatigue_score"]}"
              data-focus="{ns["focus_score"]}">
          n(anxiety)={ns["anxiety_score"]}, n(fatigue)={ns["fatigue_score"]}, n(focus)={ns["focus_score"]}
        </span>

        <h2>Anxiety</h2>
        {div_block("anxiety", results["anxiety_score"])}

        <h2>Fatigue</h2>
        {div_block("fatigue", results["fatigue_score"])}

        <h2>Focus</h2>
        {div_block("focus", results["focus_score"])}

      </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/reference-stats.json")
def reference_stats_json():
    # 크롤러가 만든 JSON을 그대로 제공(없으면 404)
    if not os.path.exists(REFERENCE_JSON_PATH):
        raise HTTPException(status_code=404, detail="reference_stats.json not found. Run crawler.")
    with open(REFERENCE_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)
