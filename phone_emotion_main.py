import time
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import uuid

# ✅ Render 백엔드 주소 (배포 후 여기를 Render URL로 바꿔주세요)
API_BASE = "https://phone-emotion.onrender.com"  # 예: "https://your-backend.onrender.com"

def ensure_anon_user_id():
    """사용자 식별(익명)용. 브라우저 세션 동안 유지."""
    if "anon_user_id" not in st.session_state:
        st.session_state["anon_user_id"] = f"anon_{int(time.time())}_{np.random.randint(1000,9999)}"

def post_event_to_api(payload: Dict[str, Any], consent: bool):
    """Render 백엔드(/events)에 이벤트 저장."""
    if not consent:
        return

    ensure_anon_user_id()

    try:
        r = requests.post(
            f"{API_BASE}/events",
            json={
                "ts": time.time(),
                "user_id": st.session_state["anon_user_id"],
                "consent": True,
                "payload": payload,
            },
            timeout=8,
        )
        # 디버깅용(원하면 지워도 됨)
        if r.status_code >= 400:
            st.warning(f"서버 저장 실패: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.warning(f"서버 저장 중 오류: {e}")
# ===============================
# 0. 한글 폰트 설정
# ===============================
font_path = Path(__file__).parent / "NanumGothic-Regular.ttf"
if font_path.exists():
    fontprop = fm.FontProperties(fname=str(font_path))
    matplotlib.rcParams["font.family"] = fontprop.get_name()
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

matplotlib.rcParams["axes.unicode_minus"] = False

# ===============================
# 1. 기본 설정 & 화면 스타일
# ===============================

st.set_page_config(
    page_title="피젯 기반 감정·상태 탐색",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .block-container {
            /* 🚨 수정: 상단 헤더 텍스트가 잘리지 않도록 padding-top 값을 늘립니다. */
            padding-top: 1.5rem !important; 
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===============================
# 2. 잠금화면 패턴 도안 및 점 배치
# ===============================

LOCK_PATTERNS: List[List[int]] = [
    [1, 2, 3, 6, 9], [1, 4, 7, 8, 9], [2, 5, 8], [1, 5, 9], [3, 5, 7],
    [1, 2, 5, 8], [4, 5, 6, 9], [7, 8, 5, 2], [3, 2, 1, 4, 7], [9, 6, 3, 2, 1],
]
def describe_pattern(pattern: List[int]) -> str:
    return " → ".join(str(p) for p in pattern)
def get_lock_points(width: int = 400, height: int = 400) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    margin_x = width * 0.18
    margin_y = height * 0.18
    cell_w = (width - 2 * margin_x) / 2
    cell_h = (height - 2 * margin_y) / 2
    idx = 1
    for row in range(3):
        for col in range(3):
            cx = margin_x + col * cell_w
            cy = margin_y + row * cell_h
            objects.append({
                "type": "circle", "radius": 12, "fill": "#4A90E2", "stroke": "#FFFFFF", "strokeWidth": 2,
                "left": float(cx - 12), "top": float(cy - 12), "originX": "left", "originY": "top",
            })
            objects.append({
                "type": "textbox", "text": str(idx), "left": float(cx - 4), "top": float(cy - 30),
                "fontSize": 16, "fill": "#DDDDDD", "editable": False
            })
            idx += 1
    return objects


# ===============================
# 3. 패턴 그리기 특징 추출
# ===============================

def compute_pattern_metrics(
    canvas_json: Dict[str, Any],
    duration: float | None,
    pattern_id: int,
) -> Dict[str, float]:
    """패턴 그리기 특징 계산. pattern_speed 특징 포함."""
    if not canvas_json or "objects" not in canvas_json:
        return {}

    xs: List[float] = []
    ys: List[float] = []

    # canvas 데이터에서 path 좌표만 추출
    for obj in canvas_json["objects"]:
        if obj.get("type") == "path":
            path = obj.get("path", [])
            for seg in path:
                # M, L, Q, C 등 어떤 세그먼트든 끝 좌표만 추출하면 됩니다.
                # (M, L은 [type, x, y], Q는 [type, c1x, c1y, x, y], C는 [type, c1x, c1y, c2x, c2y, x, y])
                # 모든 세그먼트의 마지막 두 요소는 x, y 좌표입니다.
                if len(seg) >= 3 and isinstance(seg[-2], (int, float)): 
                    xs.append(seg[-2])
                    ys.append(seg[-1])

    # 점이 너무 적으면 분석 불가
    if len(xs) < 3: 
        return {}

    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)

    # 직선에 대한 최소제곱 회귀 → 선에서 얼마나 벗어났는지(RMSE)
    A = np.vstack([xs_arr, np.ones(len(xs_arr))]).T
    a, b = np.linalg.lstsq(A, ys_arr, rcond=None)[0]
    residuals = ys_arr - (a * xs_arr + b)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # 길이 & jerkiness
    diffs = np.sqrt(np.diff(xs_arr) ** 2 + np.diff(ys_arr) ** 2)
    total_length = float(np.sum(diffs))
    jerkiness = float(np.std(diffs))

    metrics: Dict[str, float] = {
        "pattern_rmse": rmse,
        "pattern_length": total_length,
        "pattern_jerkiness": jerkiness,
    }

    if duration is not None and duration > 0:
        metrics["pattern_duration"] = float(duration)
        metrics["pattern_speed"] = total_length / duration
    else:
        metrics["pattern_duration"] = 0.0
        metrics["pattern_speed"] = 0.0

    metrics["pattern_id"] = float(pattern_id)
    return metrics


def aggregate_pattern_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    """여러 패턴 시도에 대한 metrics 리스트를 받아 각 특성의 평균값으로 요약."""
    if not records:
        return {}

    df = pd.DataFrame(records)
    agg: Dict[str, float] = {}

    for col in [
        "pattern_rmse",
        "pattern_length",
        "pattern_jerkiness",
        "pattern_duration",
        "pattern_speed",
    ]:
        if col in df.columns:
            agg[col] = float(df[col].mean())

    agg["pattern_trials"] = float(len(df))
    return agg


# ===============================
# 4. 키보드 / 스크롤 특징 추출 (ITD 기반)
# ===============================

def compute_typing_metrics(timing_records: List[Dict[str, float]]) -> Dict[str, float]:
    """
    ITD 기반으로 특징 계산 (Duration은 0으로 가정).
    """
    # 키 누름 이벤트가 5개 미만인 경우 분석 불가능하다고 가정
    if len(timing_records) < 5:
        return {}

    # ITD (Inter-Tap Duration) 계산: timestamp 간의 시간 간격
    timestamps = [rec["timestamp"] for rec in timing_records]
    itds = np.diff(np.array(timestamps)) 
    itds = itds[itds > 0]
    
    if len(itds) < 3:
        return {}

    # ITD 특징 (논문의 Q2 및 Variance 중요성 반영)
    q1_itd, q2_itd, q3_itd = np.percentile(itds, [25, 50, 75])
    var_itd = float(np.var(itds))
    mean_itd = float(np.mean(itds))
    
    # Duration 특징 (st.text_input 콜백 기반에서는 측정이 불가능함)
    mean_dur = 0.0
    var_dur = 0.0

    return {
        "typing_itd_q2": float(q2_itd),
        "typing_itd_var": var_itd,
        "typing_itd_mean": mean_itd,
        "typing_duration_mean": mean_dur, 
        "typing_duration_var": var_dur,   
        "typing_total_taps": float(len(timestamps)),
    }


def compute_scroll_metrics(start: float | None, click_times: List[float]) -> Dict[str, float]:
    """스크롤 버튼 클릭 시간 기반 특징."""
    if start is None or not click_times: return {}
    total_time = max(0.0, max(click_times) - start)
    if len(click_times) >= 2:
        itds = np.diff(sorted(click_times))
        itds = itds[itds > 0]
        if len(itds) > 0:
            mean_itd = float(np.mean(itds))
            var_itd = float(np.var(itds))
        else:
            mean_itd = 0.0
            var_itd = 0.0
    else:
        mean_itd = 0.0
        var_itd = 0.0
    return {
        "scroll_total_time": float(total_time),
        "scroll_click_count": float(len(click_times)),
        "scroll_click_mean": mean_itd,
        "scroll_click_var": var_itd,
    }


# ===============================
# 5. 상태 분석 heuristic (Duration 특징은 0으로 처리됨)
# ===============================
def analyze_state(
    pattern_metrics_agg: Dict[str, float],
    typing_metrics: Dict[str, float],
    scroll_metrics: Dict[str, float],
) -> Dict[str, float]:
    """불안(Anxiety), 피로(Fatigue), 집중/안정(Focus) 추정"""
    anxiety, fatigue, focus = 0.0, 0.0, 50.0

    # 패턴
    if pattern_metrics_agg:
        rmse = pattern_metrics_agg.get("pattern_rmse", 0.0)
        jerk = pattern_metrics_agg.get("pattern_jerkiness", 0.0)
        dur = pattern_metrics_agg.get("pattern_duration", 0.0)
        speed = pattern_metrics_agg.get("pattern_speed", 0.0)
        anxiety += min(35, rmse * 3 + jerk * 2)
        fatigue += min(20, dur * 0.4 + max(0, 1.0 - speed) * 10)
        focus += max(-20, 20 - rmse * 2 - jerk * 2)

    # 키보드 (Duration 특징은 0이므로 ITD 특징만 기여)
    if typing_metrics:
        var_itd = typing_metrics.get("typing_itd_var", 0.0)
        q2_itd = typing_metrics.get("typing_itd_q2", 0.0)
        
        # Duration 특징은 0으로 가정되므로, 해당 부분의 기여는 0
        # mean_dur = typing_metrics.get("typing_duration_mean", 0.0)
        # var_dur = typing_metrics.get("typing_duration_var", 0.0)
        
        # 불안: ITD 변동성(리듬 불안정)
        anxiety += min(30, math.log1p(var_itd) * 15)
        # 피로: 긴 ITD 중앙값(느린 속도)
        fatigue += min(25, q2_itd * 30)
        # 집중/안정: 낮은 ITD 변동성
        focus += max(-20, 20 - math.log1p(var_itd) * 10)

    # 스크롤
    if scroll_metrics:
        total_time = scroll_metrics.get("scroll_total_time", 0.0)
        click_var = scroll_metrics.get("scroll_click_var", 0.0)
        click_mean = scroll_metrics.get("scroll_click_mean", 0.0)
        scroll_speed = 1.0 / click_mean if click_mean > 0 else 0.0
        anxiety += min(20, math.log1p(scroll_speed) * 10 + math.log1p(click_var + 1) * 5)
        fatigue += min(15, total_time * 0.05)
        focus -= min(15, math.log1p(click_var + 1) * 5)

    anxiety = float(max(0, min(100, anxiety)))
    fatigue = float(max(0, min(100, fatigue)))
    focus = float(max(0, min(100, focus)))

    return {
        "anxiety_score": anxiety,
        "fatigue_score": fatigue,
        "focus_score": focus,
    }


# ===============================
# 6. 크롤링 예시 (기존 코드 유지)
# ===============================
AVERAGE_STATS_URL = "https://example.com/phone_emotion_stats.html"
COPING_TIP_URL = "https://example.com/phone_emotion_tips.html"

def fetch_reference_stats() -> Dict[str, float]:
    try:
        # (생략)
        return { "avg_anxiety": 40.0, "avg_fatigue": 35.0, "avg_focus": 55.0, }
    except Exception:
        return { "avg_anxiety": 40.0, "avg_fatigue": 35.0, "avg_focus": 55.0, }

def fetch_coping_tips(topic: str) -> List[str]:
    try:
        # (생략)
        return []
    except Exception:
        if topic == "anxiety":
            return ["천천히 깊게 숨을 들이쉬고 내쉬는 호흡을 몇 번 반복해 보세요.", "잠깐 의자에서 일어나 주변을 걸어보세요.",]
        elif topic == "fatigue":
            return ["눈을 감고 20~30초 정도 휴식을 취해 보세요.", "목·어깨를 가볍게 돌리며 스트레칭해 보세요.",]
        elif topic == "focus":
            return ["5~10분 정도 한 가지 일에만 집중해 보는 짧은 타이머를 설정해 보세요.", "잠깐 동안 알림을 꺼두고 화면에만 집중해 보세요.",]
        else: return []

def fetch_reference_stats_from_backend() -> Dict[str, Any]:
    """크롤러가 만든 reference-stats.json을 백엔드에서 가져옴"""
    try:
        r = requests.get(f"{API_BASE}/reference-stats.json", timeout=5)
        if r.status_code == 200:
            return r.json()
        return {}
    except Exception:
        return {}

def band_from_percentiles(value: float, stats: Dict[str, float], higher_is_better: bool = False) -> str:
    """
    stats: {"p10","p25","p50","p75","p90",...}
    higher_is_better=False: 불안/피로
    higher_is_better=True: 집중(높을수록 좋음) -> 문구만 반대로
    """
    if not stats:
        return "표본 부족(기준 없음)"

    p10, p25, p50, p75, p90 = stats["p10"], stats["p25"], stats["p50"], stats["p75"], stats["p90"]

    # 불안/피로(높을수록 나쁨) 기준
    if not higher_is_better:
        if value >= p90: return "상위 10% (매우 높음)"
        if value >= p75: return "상위 25% (높은 편)"
        if value >= p25: return "중간 구간 (보통)"
        if value >= p10: return "하위 25% (낮은 편)"
        return "하위 10% (매우 낮음)"

    # 집중(높을수록 좋음) 기준 -> “상위”가 좋은 의미
    if value >= p90: return "상위 10% (매우 좋음)"
    if value >= p75: return "상위 25% (좋은 편)"
    if value >= p25: return "중간 구간 (보통)"
    if value >= p10: return "하위 25% (낮은 편)"
    return "하위 10% (매우 낮음)"

# ===============================
# 7. 세션 상태 초기화 (ITD 기반)
# ===============================
if "anon_user_id" not in st.session_state:
    st.session_state["anon_user_id"] = str(uuid.uuid4())

if "pattern_index" not in st.session_state: st.session_state["pattern_index"] = 0
if "pattern_start_time" not in st.session_state: st.session_state["pattern_start_time"] = None
if "pattern_canvas_key" not in st.session_state: st.session_state["pattern_canvas_key"] = 0
if "pattern_records" not in st.session_state: st.session_state["pattern_records"] = []

# --- 키보드 상태 변경: ITD 측정용 세션 ---
if "typing_timing_records" not in st.session_state:
    # [{"timestamp": t, "key": "N/A", "duration": 0.0}, ...]
    st.session_state["typing_timing_records"] = [] 
if "last_typing_time" not in st.session_state:
    # 마지막 키 입력 시각 기록 (ITD 계산용)
    st.session_state["last_typing_time"] = None 

if "scroll_start_time" not in st.session_state: st.session_state["scroll_start_time"] = None
if "scroll_click_times" not in st.session_state: st.session_state["scroll_click_times"] = []

if "self_reports" not in st.session_state: st.session_state["self_reports"] = []


# ===============================
# 8. 사이드바 네비게이션
# ===============================

st.sidebar.title("📱 피젯 감정 탐색 앱")
page = st.sidebar.radio(
    "메뉴",
    [
        "1. 잠금화면 패턴 그리기",
        "2. 키보드 타이핑 분석",
        "3. 스크롤 테스트",
        "4. 사용자 활동 분석",
        "5. 데이터 관리 및 내보내기",
    ],
)

# ===============================
# Helper: 자가 보고 입력 및 저장
# ===============================

def collect_self_report(source: str):
    st.markdown("### 😊 지금 나의 감정·상태 자가 보고")
    st.caption("측정된 특징을 학습시키기 위한 **Ground Truth**로 사용됩니다.")

    consent = st.checkbox(
        "연구 및 통계 생성을 위해 내 데이터를 익명으로 저장하는 것에 동의합니다.",
        key=f"consent_{source}"
    )

    col_a, col_f, col_c = st.columns(3)

    with col_a:
        anxiety = st.slider(
            "현재 **불안** 수준 (1=매우 낮음, 5=매우 높음)",
            1, 5, 3,
            key=f"sr_anxiety_{source}"
        )

    with col_f:
        fatigue = st.slider(
            "현재 **피로** 수준 (1=매우 낮음, 5=매우 높음)",
            1, 5, 3,
            key=f"sr_fatigue_{source}"
        )

    with col_c:
        focus = st.slider(
            "현재 **집중** 수준 (1=매우 낮음, 5=매우 높음)",
            1, 5, 3,
            key=f"sr_focus_{source}"
        )

    if st.button("현재 상태 저장", key=f"save_sr_{source}"):
        if not consent:
            st.warning("데이터 저장에 동의해야 저장할 수 있습니다.")
            return

        report = {
            "anxiety": float(anxiety),
            "fatigue": float(fatigue),
            "focus": float(focus),
            "timestamp": time.time(),
            "source": source,
        }

        # 1) Streamlit 로컬(세션) 저장
        st.session_state["self_reports"].append(report)

        # 2) 지금까지 수집된 특징들로 종합 점수 계산(백엔드에 같이 저장)
        pattern_metrics_agg = aggregate_pattern_metrics(st.session_state.get("pattern_records", []))

        typing_records = st.session_state.get("typing_timing_records", [])
        typing_metrics = compute_typing_metrics(typing_records) if typing_records else {}

        scroll_start = st.session_state.get("scroll_start_time")
        scroll_clicks = st.session_state.get("scroll_click_times", [])
        scroll_metrics = compute_scroll_metrics(scroll_start, scroll_clicks) if scroll_clicks else {}

        state_scores_now = analyze_state(pattern_metrics_agg, typing_metrics, scroll_metrics)

        payload = {
            "source": source,
            "self_report": report,
            "pattern_metrics_agg": pattern_metrics_agg,
            "typing_metrics": typing_metrics,
            "scroll_metrics": scroll_metrics,
            "state_scores": state_scores_now,
            "app_version": "v1",
        }

        # 3) Render 백엔드 저장
        post_event_to_api(payload, consent=consent)

        st.success(f"현재 자가 보고 상태를 저장했습니다. (총 {len(st.session_state['self_reports'])}개)")




# ===============================
# 9-1. 잠금화면 패턴 그리기
# ===============================

if page.startswith("1"):
    st.header("🔐 1. 잠금화면 패턴 그리기")

    current_idx = st.session_state["pattern_index"]
    current_pattern = LOCK_PATTERNS[current_idx]
    st.markdown(
        f"""
        **잠금화면을 풀 듯이**, 아래 3×3 점들을 이용해  
        아래 도안을 따라 한 번 쭉 선을 그려보세요.
        - 이번 도안: **{describe_pattern(current_pattern)}**  
        """
    )
    st.markdown("---")
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.write(f"지금까지 저장된 패턴 시도 수: **{len(st.session_state['pattern_records'])}** 개")
    with col_btn:
        if st.button("다른 도안으로 바꾸기"):
            st.session_state["pattern_index"] = (current_idx + 1) % len(LOCK_PATTERNS)
            st.session_state["pattern_start_time"] = None
            st.session_state["pattern_canvas_key"] += 1

    if st.session_state["pattern_start_time"] is None:
        if st.button("패턴 그리기 시작"):
            st.session_state["pattern_start_time"] = time.time()

    initial_objects = get_lock_points()
    initial_json = { "version": "4.4.0", "objects": initial_objects }

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)", stroke_width=4, stroke_color="white", background_color="#111111",
        height=400, width=400, drawing_mode="freedraw", point_display_radius=0,
        key=f"pattern_canvas_{st.session_state['pattern_canvas_key']}", initial_drawing=initial_json,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("이 패턴 시도 저장하기"):
            if canvas_result.json_data:
                if st.session_state["pattern_start_time"] is not None:
                    duration = time.time() - st.session_state["pattern_start_time"]
                else:
                    duration = None

                metrics = compute_pattern_metrics(
                    canvas_result.json_data,
                    duration,
                    pattern_id=current_idx + 1,
                )
                if metrics:
                    st.session_state["pattern_records"].append(metrics)
                    st.success("이번 패턴 시도를 저장했습니다. (4번 탭에서 통계에 반영됩니다.)")
                else:
                    st.warning("선 데이터가 부족합니다. 패턴을 **시작부터 끝까지 한 번에** 이어서 그려주세요.")
            else:
                st.warning("아직 그려진 내용이 없습니다.")
    with col2:
        if st.button("화면 비우고 다시 그리기"):
            st.session_state["pattern_start_time"] = None
            st.session_state["pattern_canvas_key"] += 1
            
    st.markdown("---")
    collect_self_report("pattern")


# ===============================
# Helper: 타이핑 입력 시 콜백 함수 (ITD 측정 로직)
# ===============================

def record_typing_callback():
    """st.text_input 값이 변경될 때마다(키 입력 시) 실행되는 콜백."""
    current_time = time.time()
    
    if st.session_state["last_typing_time"] is not None:
        # ITD (Inter-Tap Duration) 계산을 위한 시점 기록
        st.session_state["typing_timing_records"].append({
            "timestamp": current_time,
            "key": "N/A", 
            "duration": 0.0 # Duration 측정 불가능
        })
    
    # 마지막 타이핑 시간 업데이트
    st.session_state["last_typing_time"] = current_time


# ===============================
# 9-2. 키보드 타이핑 분석 (st.text_input 기반)
# ===============================

if page.startswith("2"):
    st.header("⌨️ 2. 키보드 타이핑 분석")

    st.markdown(
        """
        아래 입력창에 **평소처럼** 문장을 입력해주세요. (띄어쓰기, 지우기 모두 분석에 포함됩니다.)
        
        - 이 분석은 **키와 키 사이의 간격(ITD)**을 분석하여 타이핑 리듬의 불안정성을 파악합니다.
        - **(참고)** 안정적인 배포 환경을 위해 **키를 누르고 있는 시간(Duration)** 분석은 제외되었습니다.
        """
    )
    
    col_input, col_status = st.columns([3, 1])

    with col_input:
        # st.text_input의 on_change 콜백을 활용하여 입력 시점을 기록합니다.
        user_input = st.text_input(
            "여기에 자유롭게 입력하세요:", 
            key="typing_area",
            on_change=record_typing_callback # 키 입력 시 콜백 실행
        )
        st.markdown(f"입력된 키 이벤트 수: **{len(st.session_state['typing_timing_records'])}**")

    # --- 분석 및 상태 표시 ---
    current_metrics = compute_typing_metrics(st.session_state["typing_timing_records"])
    
    with col_status:
        if current_metrics:
            st.success("데이터 수집 충분!")
            st.caption(f"평균 ITD: {current_metrics.get('typing_itd_mean', 0.0):.3f}초")
            st.caption(f"ITD 변동성: {current_metrics.get('typing_itd_var', 0.0):.4f}")
        else:
            st.warning(f"최소 5번 타이핑 필요 (현재 {len(st.session_state['typing_timing_records'])} / 5)")

    st.markdown("---")
    
    # 데이터 초기화
    if st.button("타이핑 기록 초기화", key="reset_typing_data"):
        st.session_state["typing_timing_records"] = []
        st.session_state["last_typing_time"] = None
        st.success("타이핑 기록을 초기화했습니다.")

    collect_self_report("typing") # 자가 보고 기능 추가


# ===============================
# 9-3. 스크롤 테스트
# ===============================


elif page.startswith("3"):
    st.header("🧷 3. 스크롤 테스트")

    st.markdown(
        """
        이번 화면에서는 **스크롤 행위의 리듬**을 가볍게 살펴봅니다.

        1. 아래 긴 텍스트를 천천히 내려가면서 읽어보거나  
        2. 텍스트 중간중간 나타나는 **'스크롤 기록' 버튼**을 눌러, 특정 지점까지 화면을 내린 시점을 기록해주세요.
        
        평소처럼 자연스럽게 화면을 내려본다고 생각하면 됩니다.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("스크롤 테스트 시작 / 재시작", key="scroll_start_btn"):
            st.session_state["scroll_start_time"] = time.time()
            st.session_state["scroll_click_times"] = []
            st.success("스크롤 테스트를 시작했습니다. 아래 내용을 읽거나 스크롤 기록 버튼을 눌러보세요.")
    with col_b:
        if st.button("스크롤 기록 초기화", key="scroll_reset_btn"):
            st.session_state["scroll_start_time"] = None
            st.session_state["scroll_click_times"] = []
            st.info("스크롤 관련 기록을 모두 지웠습니다.")

    st.markdown("---")

    # --- 스크롤 체크포인트 함수 정의 ---
    def scroll_checkpoint(checkpoint_id: int):
        if st.button(f"⬇️ 스크롤 기록: 체크포인트 {checkpoint_id}", key=f"cp_btn_{checkpoint_id}"):
            if st.session_state["scroll_start_time"] is None:
                st.warning("테스트를 시작 버튼을 먼저 눌러주세요.")
                st.session_state["scroll_start_time"] = time.time() # 시작 시간 자동 기록
            else:
                st.session_state["scroll_click_times"].append(time.time())
                st.info(f"체크포인트 {checkpoint_id} 시각을 기록했습니다.")


    # --- 긴 텍스트 및 체크포인트 배치 ---
    
    st.markdown("### 섹션 1: 시작 지점")
    st.write("이 부분은 스크롤을 만들기 위한 예시 텍스트입니다. 화면을 내리는 리듬을 측정합니다. 너무 의식하지 말고, 평소처럼 내려주세요." * 2)

    scroll_checkpoint(1) # 첫 번째 체크포인트

    st.markdown("### 섹션 2: 중간 탐색")
    st.write("스크롤을 내리는 동안 사용자의 리듬이 일정하거나, 혹은 갑자기 빨라지거나 느려지는 경향이 감정 상태를 반영합니다. 예를 들어, 조급한 상태에서는 리듬이 불규칙해지기 쉽습니다." * 4)

    scroll_checkpoint(2) # 두 번째 체크포인트

    st.markdown("### 섹션 3: 심층 탐색")
    st.write("긴 텍스트를 읽어야 할 때, 화면을 톡톡 내리는 패턴(짧은 ITD, 낮은 분산)은 집중하고 있는 상태일 수 있습니다. 반면, 길게 멈춘 후 한 번에 많이 내리는 패턴(긴 ITD)은 피로도를 나타낼 수 있습니다." * 6)

    scroll_checkpoint(3) # 세 번째 체크포인트
    
    st.markdown("---")

    st.write(f"총 기록된 스크롤 이벤트 수: **{len(st.session_state['scroll_click_times'])}**")
    
    st.markdown("---")
    collect_self_report("scroll")

# ===============================
# 9-4. 사용자 활동 분석
# ===============================


elif page.startswith("4"):
    st.header("📊 4. 사용자 활동 분석")

    # 1. 모든 특징 계산
    pattern_metrics_agg = aggregate_pattern_metrics(st.session_state["pattern_records"])
    typing_metrics = compute_typing_metrics(st.session_state["typing_timing_records"]) \
        if st.session_state["typing_timing_records"] else {}
    scroll_metrics = compute_scroll_metrics(
        st.session_state["scroll_start_time"],
        st.session_state["scroll_click_times"],
    ) if st.session_state["scroll_click_times"] else {}

    # 종합 상태 점수 계산 (모든 특징 사용)
    state_scores = analyze_state(pattern_metrics_agg, typing_metrics, scroll_metrics)
    ref_stats = fetch_reference_stats()

    if not (pattern_metrics_agg or typing_metrics or scroll_metrics):
        st.info("아직 수집된 데이터가 충분하지 않습니다. 1~3번 화면을 먼저 사용해 본 뒤 다시 와 주세요.")
    else:
        st.subheader("① 활동별로 정리된 특징")

        # --- 잠금 패턴 분석 표시 (정교화) ---
        if pattern_metrics_agg:
            st.markdown("### 🔐 잠금화면 패턴 분석 (Touch Dynamics)")

            explanation_map = {
                "pattern_rmse": ("RMSE (흔들림)", "궤적이 이상적인 직선에서 벗어난 정도. 손의 미세한 흔들림/부정확성 측정."),
                "pattern_length": ("총 길이 (픽셀)", "그려진 선의 총 길이."),
                "pattern_jerkiness": ("Jerkiness (불규칙성)", "연속된 움직임 구간의 길이 변화 불규칙성. 속도/압력 변화의 변동성 측정."),
                "pattern_duration": ("총 시간 (초)", "패턴 완성에 걸린 시간. 움직임 지연/피로도 측정."),
                "pattern_speed": ("속도 (픽셀/초)", "총 길이 / 총 시간. 패턴 그리기 속도 측정."),
                "pattern_trials": ("시도 횟수", "총 패턴 시도 횟수."),
            }

            data_list = []
            for feature, value in pattern_metrics_agg.items():
                if feature in explanation_map:
                    formatted_value = int(value) if feature == "pattern_trials" else f"{value:.3f}"
                    data_list.append({
                        "특징 이름": explanation_map[feature][0],
                        "나의 평균값": formatted_value,
                        "분석 목적": explanation_map[feature][1],
                    })

            df_pattern = pd.DataFrame(data_list)
            st.dataframe(df_pattern.set_index("특징 이름"), use_container_width=True)

            st.markdown("#### 패턴 움직임 특징이 상태 점수에 미치는 영향")
            st.markdown("""
                - **불안/초조 기여:** 높은 **RMSE** (흔들림) 및 **Jerkiness** (불규칙성)은 불안 점수를 높입니다.
                - **피로 기여:** 긴 **Duration** (총 시간) 또는 낮은 **Speed** (속도)는 피로 점수를 높입니다.
                - **집중/안정 기여:** 낮은 **RMSE** 및 **Jerkiness**는 집중 점수를 높입니다.
            """)
            st.markdown("---")

        # --- 키보드 타이핑 분석 ---
        if typing_metrics:
            st.markdown("### ⌨️ 키보드 타이핑 분석 (ITD 특징)")

            typing_explanation_map = {
                "typing_itd_q2": ("ITD 중앙값 (Q2)", "키와 키 사이 간격(ITD)의 중간값. 평균적인 타이핑 속도 측정."),
                "typing_itd_var": ("ITD 변동성 (분산)", "ITD의 분산. 타이핑 리듬의 불안정성 측정 (높을수록 불규칙)."),
                "typing_itd_mean": ("ITD 평균", "ITD의 평균."),
                "typing_duration_mean": ("Duration 평균", "키 누름 시간 평균 (현재 0으로 처리됨)."),
                "typing_duration_var": ("Duration 변동성", "키 누름 시간 변동성 (현재 0으로 처리됨)."),
                "typing_total_taps": ("총 키 입력 수", "총 기록된 키 입력 이벤트 횟수."),
            }

            typing_data_list = []
            for feature, value in typing_metrics.items():
                if feature in typing_explanation_map:
                    formatted_value = int(value) if feature == "typing_total_taps" else f"{value:.4f}"
                    typing_data_list.append({
                        "특징 이름": typing_explanation_map[feature][0],
                        "나의 평균값": formatted_value,
                        "분석 목적": typing_explanation_map[feature][1],
                    })

            df_typing = pd.DataFrame(typing_data_list)
            st.dataframe(df_typing.set_index("특징 이름"), use_container_width=True)

            st.markdown("""
                - **불안/초조:** 높은 **ITD 변동성**은 리듬 불안정으로 이어져 불안 점수를 높입니다.
                - **피로:** 긴 **ITD 중앙값/평균**은 느린 타이핑 속도를 의미하며 피로 점수를 높입니다.
            """)
            st.markdown("---")

        # --- 스크롤 버튼 사용 특징 ---
        if scroll_metrics:
            st.markdown("### 🧷 스크롤 테스트 분석 (체크포인트 ITD)")

            scroll_explanation_map = {
                "scroll_total_time": ("총 시간 (초)", "테스트 시작부터 마지막 클릭까지의 총 소요 시간."),
                "scroll_click_count": ("클릭 횟수", "기록된 스크롤 체크포인트 클릭 횟수."),
                "scroll_click_mean": ("ITD 평균 (초)", "연속된 클릭 간격(ITD)의 평균."),
                "scroll_click_var": ("ITD 변동성 (분산)", "ITD의 분산. 스크롤 리듬의 불규칙성 측정."),
            }

            scroll_data_list = []
            for feature, value in scroll_metrics.items():
                if feature in scroll_explanation_map:
                    formatted_value = int(value) if feature == "scroll_click_count" else f"{value:.3f}"
                    scroll_data_list.append({
                        "특징 이름": scroll_explanation_map[feature][0],
                        "나의 평균값": formatted_value,
                        "분석 목적": scroll_explanation_map[feature][1],
                    })

            df_scroll = pd.DataFrame(scroll_data_list)
            st.dataframe(df_scroll.set_index("특징 이름"), use_container_width=True)

            st.markdown("""
                - **불안/초조:** 높은 **ITD 변동성** 및 빠른 클릭 리듬은 불안 점수에 기여합니다.
                - **피로:** 긴 **총 시간**은 피로 점수에 기여합니다.
            """)
            st.markdown("---")

        # ---- 종합 점수 ----
        st.subheader("② 이 앱이 추정한 나의 종합 상태 점수 (0~100)")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric(label="불안 점수", value=f"{state_scores['anxiety_score']:.1f}점", delta=None)
        with col_s2:
            st.metric(label="피로 점수", value=f"{state_scores['fatigue_score']:.1f}점", delta=None)
        with col_s3:
            st.metric(label="집중/안정 점수", value=f"{state_scores['focus_score']:.1f}점", delta=None)

        # ✅ 백엔드가 제공하는 기준(reference-stats.json) 가져오기
        ref = fetch_reference_stats_from_backend()

        st.subheader("③ 전체 사용자 분포 기준(퍼센타일)에서의 나의 위치")
        st.caption("이 기준은 백엔드의 공개 통계 HTML을 BeautifulSoup으로 수집해 만든 reference-stats.json을 통해 제공됩니다.")

        if ref and (ref.get("anxiety") or ref.get("fatigue") or ref.get("focus")):
            anx_band = band_from_percentiles(state_scores["anxiety_score"], ref.get("anxiety") or {}, higher_is_better=False)
            fat_band = band_from_percentiles(state_scores["fatigue_score"], ref.get("fatigue") or {}, higher_is_better=False)
            foc_band = band_from_percentiles(state_scores["focus_score"], ref.get("focus") or {}, higher_is_better=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("불안 퍼센타일 위치", anx_band)
                st.caption(f"표본 수: {ref.get('n', {}).get('anxiety', 0)}")
            with c2:
                st.metric("피로 퍼센타일 위치", fat_band)
                st.caption(f"표본 수: {ref.get('n', {}).get('fatigue', 0)}")
            with c3:
                st.metric("집중 퍼센타일 위치", foc_band)
                st.caption(f"표본 수: {ref.get('n', {}).get('focus', 0)}")
        else:
            st.info("퍼센타일 기준 데이터를 아직 불러올 수 없습니다. (크롤러 실행 및 백엔드 reference-stats.json 확인 필요)")

        st.markdown("#### 💡 상태별 조언")
        if state_scores["anxiety_score"] >= 60:
            st.warning("현재 불안 점수가 높습니다. 마음이 급하고 초조할 수 있습니다.")
            st.caption("권장 조언: " + " ".join(fetch_coping_tips("anxiety")))
        elif state_scores["fatigue_score"] >= 60:
            st.warning("현재 피로 점수가 높습니다. 움직임이 느려지고 집중하기 어려울 수 있습니다.")
            st.caption("권장 조언: " + " ".join(fetch_coping_tips("fatigue")))
        elif state_scores["focus_score"] >= 75:
            st.success("현재 집중/안정 점수가 매우 높습니다. 안정된 상태를 유지하고 있습니다.")
        else:
            st.info("현재 상태는 비교적 안정적입니다.")

        st.markdown("---")

        # ✅ (삭제 반영) ③ "평균(예시)과 비교" 섹션은 제거됨

        st.subheader("④ 수집된 나의 자가 보고 데이터 요약")
        if st.session_state["self_reports"]:
            df_reports = pd.DataFrame(st.session_state["self_reports"])
            df_reports["source"] = df_reports["source"].apply(lambda x: {"pattern": "패턴", "typing": "키보드", "scroll": "스크롤"}.get(x, x))
            df_summary = df_reports.groupby("source")[["anxiety", "fatigue", "focus"]].mean().reset_index()
            df_summary.columns = ["활동", "불안 평균", "피로 평균", "집중 평균"]
            st.markdown(f"**총 {len(st.session_state['self_reports'])}개**의 자가 보고가 저장되었습니다.")
            st.dataframe(df_summary.set_index("활동"))
        else:
            st.info("저장된 자가 보고 데이터가 없습니다.")



# ===============================
# 9-5. 데이터 관리 및 내보내기 (통합 데이터 내보내기 기능 추가)
# ===============================

if page.startswith("5"):
    st.header("💾 5. 데이터 관리 및 내보내기")

    def create_aggregated_dataframe(pattern_records, typing_records, scroll_times, self_reports) -> pd.DataFrame:
        """모든 활동 특징과 자가 보고 점수를 시간 기준으로 통합된 데이터프레임으로 생성"""
        
        # 1. 활동별 특징 요약 (단일 행 특징)
        pattern_agg = aggregate_pattern_metrics(pattern_records)
        typing_features = compute_typing_metrics(typing_records)
        scroll_features = compute_scroll_metrics(st.session_state.get("scroll_start_time"), scroll_times)
        
        pattern_features = {f'pat_{k}': v for k, v in pattern_agg.items()}
        typing_features = {f'typ_{k}': v for k, v in typing_features.items()}
        scroll_features = {f'scr_{k}': v for k, v in scroll_features.items()}

        all_features = {**pattern_features, **typing_features, **scroll_features}
        
        # 2. 자가 보고서 (GT) 데이터 프레임 생성
        if not self_reports:
            return pd.DataFrame()

        df_reports = pd.DataFrame(self_reports)
        
        # 3. 모든 self_report 행에 계산된 종합 특징을 복사하여 붙입니다.
        if all_features:
            df_final = df_reports.assign(**all_features)
        else:
            df_final = df_reports
            
        df_final['timestamp_readable'] = pd.to_datetime(df_final['timestamp'], unit='s')
        
        return df_final.set_index('timestamp_readable').sort_index()


    df_full_export = create_aggregated_dataframe(
        st.session_state['pattern_records'],
        st.session_state['typing_timing_records'],
        st.session_state['scroll_click_times'],
        st.session_state['self_reports']
    )

    if df_full_export.empty:
        st.info("내보낼 데이터가 없습니다. 1~3번 탭을 이용하고 자가 보고를 저장해 주세요.")
    else:
        st.subheader("통합 데이터 (특징 + 자가 보고 라벨)")
        st.caption("이 데이터를 활용하여 머신러닝 모델을 학습시킬 수 있습니다.")
        st.dataframe(df_full_export)
        
        # CSV 다운로드 버튼
        csv = df_full_export.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="⬇️ 통합 데이터 CSV 다운로드",
            data=csv,
            file_name='fidget_emotion_data_integrated.csv',
            mime='text/csv',
        )

    st.markdown("---")
    
    if st.button("모든 데이터 초기화", help="초기화하면 모든 기록이 사라집니다."):
        st.session_state["pattern_index"] = 0; st.session_state["pattern_start_time"] = None; st.session_state["pattern_canvas_key"] = 0
        st.session_state["pattern_records"] = []; st.session_state["typing_timing_records"] = []; st.session_state["scroll_start_time"] = None
        st.session_state["scroll_click_times"] = []; st.session_state["self_reports"] = []
        st.session_state["last_typing_time"] = None
        st.rerun()
