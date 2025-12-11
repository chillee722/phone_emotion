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
            padding-top: 0.6rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ===============================
# 2. 잠금화면 패턴 도안 및 점 배치
# ===============================
def collect_self_report(source: str):
    st.markdown("### 😊 지금 나의 감정·상태 자가 보고")
    st.caption("측정된 특징을 학습시키기 위한 **Ground Truth**로 사용됩니다.")
    
    col_a, col_f, col_c = st.columns(3)
    
    with col_a:
        anxiety = st.slider("현재 **불안** 수준 (1=매우 낮음, 5=매우 높음)", 1, 5, 3, key=f"sr_anxiety_{source}")
    with col_f:
        fatigue = st.slider("현재 **피로** 수준 (1=매우 낮음, 5=매우 높음)", 1, 5, 3, key=f"sr_fatigue_{source}")
    with col_c:
        focus = st.slider("현재 **집중** 수준 (1=매우 낮음, 5=매우 높음)", 1, 5, 3, key=f"sr_focus_{source}")
        
    if st.button("현재 상태 저장", key=f"save_sr_{source}"):
        report = {
            "anxiety": float(anxiety), "fatigue": float(fatigue), "focus": float(focus),
            "timestamp": time.time(), "source": source
        }
        st.session_state["self_reports"].append(report)
        st.success(f"현재 자가 보고 상태를 저장했습니다. (총 {len(st.session_state['self_reports'])}개)")
        
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

def compute_pattern_metrics(canvas_json: Dict[str, Any],
                            duration: float | None,
                            pattern_id: int) -> Dict[str, float]:
    """패턴 그리기 특징 계산. pattern_speed 특징 포함."""
    if not canvas_json or "objects" not in canvas_json: return {}
    xs, ys = [], []
    for obj in canvas_json["objects"]:
        if obj.get("type") == "path":
            path = obj.get("path", [])
            for seg in path:
                if len(seg) >= 3 and seg[0] in ("M", "L"):
                    xs.append(seg[1])
                    ys.append(seg[2])
    if len(xs) < 5: return {}
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    
    A = np.vstack([xs_arr, np.ones(len(xs_arr))]).T
    a, b = np.linalg.lstsq(A, ys_arr, rcond=None)[0]
    residuals = ys_arr - (a * xs_arr + b)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    diffs = np.sqrt(np.diff(xs_arr) ** 2 + np.diff(ys_arr) ** 2)
    total_length = float(np.sum(diffs))
    jerkiness = float(np.std(diffs))

    metrics = {
        "pattern_rmse": rmse, "pattern_length": total_length, "pattern_jerkiness": jerkiness,
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
    """여러 패턴 시도에 대한 metrics 리스트를 받아 각 특성의 평균값을 하나로 요약합니다."""
    if not records: return {}
    df = pd.DataFrame(records)
    agg: Dict[str, float] = {}
    for col in ["pattern_rmse", "pattern_length", "pattern_jerkiness", "pattern_duration", "pattern_speed"]:
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


# ===============================
# 7. 세션 상태 초기화 (ITD 기반)
# ===============================

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
                duration = time.time() - st.session_state["pattern_start_time"] if st.session_state["pattern_start_time"] is not None else None
                metrics = compute_pattern_metrics(canvas_result.json_data, duration, pattern_id=current_idx + 1)
                if metrics:
                    st.session_state["pattern_records"].append(metrics)
                    st.success("이번 패턴 시도를 저장했습니다. (4번 탭에서 통계에 반영됩니다.)")
                else: st.warning("선 데이터가 부족해서 이번 시도는 저장되지 않았습니다.")
            else: st.warning("아직 그려진 내용이 없습니다.")
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

elif page.startswith("2"):
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
        이번 화면에서는 **스크롤하는 방식**을 가볍게 살펴봅니다.

        1. 아래 긴 텍스트를 천천히 내려가면서 읽어보거나  
        2. 아래쪽 버튼을 눌러 **화면을 내리는 느낌**으로 사용해 보세요.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("스크롤 테스트 시작 / 재시작"):
            st.session_state["scroll_start_time"] = time.time()
            st.session_state["scroll_click_times"] = []
            st.success("스크롤 테스트를 시작했습니다. 아래 내용을 읽거나 스크롤 버튼을 눌러보세요.")
    with col_b:
        if st.button("스크롤 기록 초기화"):
            st.session_state["scroll_start_time"] = None
            st.session_state["scroll_click_times"] = []
            st.info("스크롤 관련 기록을 모두 지웠습니다.")

    st.markdown("---")

    long_text = """
    이 부분은 스크롤을 만들기 위한 예시 텍스트입니다.  
    ... (중략)
    """ * 6

    st.write(long_text)

    st.markdown("**버튼을 눌러서 '스크롤했다'는 표시를 남길 수도 있습니다.**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬇️ 아래로 더 내려가기 느낌"):
            if st.session_state["scroll_start_time"] is None: st.session_state["scroll_start_time"] = time.time()
            st.session_state["scroll_click_times"].append(time.time())
    with col2:
        if st.button("⬇️ 다음 부분 보기 느낌"):
            if st.session_state["scroll_start_time"] is None: st.session_state["scroll_start_time"] = time.time()
            st.session_state["scroll_click_times"].append(time.time())

    st.write(f"스크롤 버튼을 누른 횟수: **{len(st.session_state['scroll_click_times'])}**")
    
    st.markdown("---")
    collect_self_report("scroll")


# ===============================
# 9-4. 사용자 활동 분석
# ===============================

elif page.startswith("4"):
    st.header("📊 4. 사용자 활동 분석")

    pattern_metrics_agg = aggregate_pattern_metrics(st.session_state["pattern_records"])
    typing_metrics = compute_typing_metrics(st.session_state["typing_timing_records"]) \
        if st.session_state["typing_timing_records"] else {}
    scroll_metrics = compute_scroll_metrics(
        st.session_state["scroll_start_time"],
        st.session_state["scroll_click_times"],
    ) if st.session_state["scroll_click_times"] else {}

    if not (pattern_metrics_agg or typing_metrics or scroll_metrics):
        st.info("아직 수집된 데이터가 충분하지 않습니다. 1~3번 화면을 먼저 사용해 본 뒤 다시 와 주세요.")
    else:
        st.subheader("① 활동별로 정리된 특징")

        if pattern_metrics_agg:
            st.markdown("#### 잠금화면 패턴 (여러 도안·시도 평균)")
            st.write(pd.DataFrame([pattern_metrics_agg]).T.rename(columns={0: "값"}))
            st.markdown("*(설명 유지)*")

        if typing_metrics:
            st.markdown("#### 키보드 타이핑 (ITD 특징)")
            st.write(pd.DataFrame([typing_metrics]).T.rename(columns={0: "값"}))
            st.markdown(
                """
                - `typing_itd_q2/mean/var`: 키 사이 간격(ITD) 중앙값, 평균, 변동성 (리듬 불안정성)  
                - `typing_total_taps`: 총 키 입력 횟수  
                - (Duration 특징은 안정성 문제로 제외되었습니다.)
                """
            )

        if scroll_metrics:
            st.markdown("#### 스크롤 버튼 사용 특징")
            st.write(pd.DataFrame([scroll_metrics]).T.rename(columns={0: "값"}))
            st.markdown("*(설명 유지)*")

        # ---- 종합 점수 ----
        state_scores = analyze_state(pattern_metrics_agg, typing_metrics, scroll_metrics)
        ref_stats = fetch_reference_stats()

        st.subheader("② 이 앱이 추정한 나의 상태 점수 (0~100)")
        st.write(pd.DataFrame([state_scores], index=["나"]).T)
        st.markdown("*(설명 유지)*")

        st.subheader("③ 다른 사람들의 평균(예시 값)과 비교")
        compare_df = pd.DataFrame({
            "나": [state_scores["anxiety_score"], state_scores["fatigue_score"], state_scores["focus_score"],],
            "평균(예시)": [ref_stats["avg_anxiety"], ref_stats["avg_fatigue"], ref_stats["avg_focus"],],
        }, index=["불안", "피로", "집중/안정"])
        st.write(compare_df)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        x, width = np.arange(len(compare_df.index)), 0.35
        ax.bar(x - width/2, compare_df["나"], width, label="나")
        ax.bar(x + width/2, compare_df["평균(예시)"], width, label="평균(예시)")
        ax.set_xticks(x); ax.set_xticklabels(compare_df.index); ax.set_ylabel("점수 (0~100)"); ax.set_title("나와 평균 상태 비교"); ax.legend()
        st.pyplot(fig)
        
        st.subheader("⑤ 수집된 나의 자가 보고 데이터 요약")
        if st.session_state["self_reports"]:
            df_reports = pd.DataFrame(st.session_state["self_reports"])
            df_reports['source'] = df_reports['source'].apply(lambda x: {"pattern": "패턴", "typing": "키보드", "scroll": "스크롤"}.get(x, x))
            df_summary = df_reports.groupby('source')[['anxiety', 'fatigue', 'focus']].mean().reset_index()
            df_summary.columns = ['활동', '불안 평균', '피로 평균', '집중 평균']
            st.markdown(f"**총 {len(st.session_state['self_reports'])}개**의 자가 보고가 저장되었습니다.")
            st.dataframe(df_summary.set_index('활동'))


# ===============================
# 9-5. 데이터 관리 및 내보내기 (통합 데이터 내보내기 기능 추가)
# ===============================

elif page.startswith("5"):
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
