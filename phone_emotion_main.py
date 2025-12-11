import time
import math
from pathlib import Path
from typing import Dict, Any, List

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
# 0. 한글 폰트 설정 (루트에 NanumGothic-Regular.ttf 있어야 함)
# ===============================

font_path = Path(__file__).parent / "NanumGothic-Regular.ttf"
if font_path.exists():
    fontprop = fm.FontProperties(fname=str(font_path))
    matplotlib.rcParams["font.family"] = fontprop.get_name()
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"  # 그래도 안 깨지게 fallback

matplotlib.rcParams["axes.unicode_minus"] = False


# ===============================
# 1. 기본 설정 & 화면 스타일
# ===============================

st.set_page_config(
    page_title="피젯 기반 감정·상태 탐색",
    layout="wide"
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 0.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ===============================
# 2. 잠금화면 패턴 도안 생성
# ===============================

def get_lock_points(width: int = 400, height: int = 400) -> List[Dict[str, Any]]:
    """
    3x3 잠금화면 점(원) 9개를 fabric.js 객체 리스트로 생성.
    좌상단부터 1~9 번호를 붙임.
    """
    objects = []
    margin_x = width * 0.15
    margin_y = height * 0.15
    cell_w = (width - 2 * margin_x) / 2
    cell_h = (height - 2 * margin_y) / 2

    idx = 1
    for row in range(3):
        for col in range(3):
            cx = margin_x + col * cell_w
            cy = margin_y + row * cell_h
            objects.append({
                "type": "circle",
                "radius": 12,
                "fill": "#4A90E2",
                "stroke": "#FFFFFF",
                "strokeWidth": 2,
                "left": float(cx - 12),
                "top": float(cy - 12),
                "originX": "left",
                "originY": "top",
            })
            # 숫자 라벨
            objects.append({
                "type": "textbox",
                "text": str(idx),
                "left": float(cx - 4),
                "top": float(cy - 30),
                "fontSize": 16,
                "fill": "#333333",
                "editable": False
            })
            idx += 1

    return objects


# 10개 정도의 잠금화면 패턴 도안 (1~9 인덱스)
LOCK_PATTERNS = [
    [1, 2, 3, 6, 9],
    [1, 4, 7, 8, 9],
    [2, 5, 8],
    [1, 5, 9],
    [3, 5, 7],
    [1, 2, 5, 8],
    [4, 5, 6, 9],
    [7, 8, 5, 2],
    [3, 2, 1, 4, 7],
    [9, 6, 3, 2, 1],
]


def describe_pattern(pattern: List[int]) -> str:
    """패턴 [1,5,9] → '1 → 5 → 9' 이런 식 텍스트로 보여주기."""
    return " → ".join(str(p) for p in pattern)


# ===============================
# 3. 패턴 그리기 데이터에서 특징 추출
# ===============================

def compute_pattern_metrics(canvas_json: Dict[str, Any], duration: float | None) -> Dict[str, float]:
    """
    canvas JSON에서 사용자가 그린 부분(path만)을 모아서:
      - 직선에서의 편차(RMSE)
      - 길이의 변동성(jerkiness)
      - 총 길이
      - 그리는 데 걸린 시간(duration)
    등을 계산.
    """
    if not canvas_json or "objects" not in canvas_json:
        return {}

    xs, ys = [], []

    for obj in canvas_json["objects"]:
        if obj.get("type") == "path":
            path = obj.get("path", [])
            for seg in path:
                if len(seg) >= 3 and seg[0] in ("M", "L"):
                    x, y = seg[1], seg[2]
                    xs.append(x)
                    ys.append(y)

    if len(xs) < 5:
        metrics = {}
    else:
        xs = np.array(xs)
        ys = np.array(ys)

        # 직선 근사 y = ax + b
        A = np.vstack([xs, np.ones(len(xs))]).T
        a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        y_hat = a * xs + b
        residuals = ys - y_hat
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        total_length = float(np.sum(diffs))
        jerkiness = float(np.std(diffs))

        metrics = {
            "pattern_rmse": rmse,
            "pattern_length": total_length,
            "pattern_jerkiness": jerkiness,
        }

    if duration is not None:
        metrics["pattern_duration"] = float(duration)

    return metrics


# ===============================
# 4. 키보드 눌림 특징 (ITD 기반)
# ===============================

def compute_typing_metrics(timestamps: List[float]) -> Dict[str, float]:
    """
    버튼 누른 시각 리스트 → Inter-Tap Duration → 분위수/변동성.
    """
    if len(timestamps) < 5:
        return {}

    itds = np.diff(sorted(timestamps))
    itds = itds[itds > 0]

    if len(itds) < 3:
        return {}

    q1, q2, q3 = np.percentile(itds, [25, 50, 75])
    var = float(np.var(itds))
    mean_itd = float(np.mean(itds))

    return {
        "typing_q1": float(q1),
        "typing_q2": float(q2),
        "typing_q3": float(q3),
        "typing_var": var,
        "typing_mean": mean_itd,
        "typing_count": float(len(itds)),
    }


# ===============================
# 5. 스크롤 특징 추출
# ===============================

def compute_scroll_metrics(start: float | None, click_times: List[float]) -> Dict[str, float]:
    """
    스크롤 테스트:
      - 시작 시각(start)
      - '더 보기/다음' 버튼 클릭 시각들(click_times)
    로부터
      - 전체 테스트 시간
      - 클릭 횟수
      - 클릭 간 간격의 평균/변동성
    계산.
    """
    if start is None or not click_times:
        return {}

    total_time = max(0.0, max(click_times) - start)
    if len(click_times) >= 2:
        itds = np.diff(sorted(click_times))
        itds = itds[itds > 0]
        if len(itds) >= 1:
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
# 6. 상태 분석 heuristic (패턴 + 키보드 + 스크롤)
# ===============================

def analyze_state(
    pattern_metrics: Dict[str, float],
    typing_metrics: Dict[str, float],
    scroll_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    - 불안(Anxiety)
    - 피로(Fatigue)
    - 집중/안정(Focus)
    점수(0~100)를 단순 heuristic으로 계산.
    """
    anxiety = 0.0
    fatigue = 0.0
    focus = 50.0  # 중간값에서 시작

    # ---- 패턴 그리기 ----
    if pattern_metrics:
        rmse = pattern_metrics.get("pattern_rmse", 0.0)
        jerk = pattern_metrics.get("pattern_jerkiness", 0.0)
        duration = pattern_metrics.get("pattern_duration", 0.0)

        # 패턴이 많이 흔들리거나 오래 걸릴수록 불안·피로 쪽으로 가중
        anxiety += min(35, rmse * 3 + jerk * 2)
        fatigue += min(20, duration * 0.3)
        focus -= min(20, rmse * 2 + jerk * 2)

    # ---- 키보드 리듬 ----
    if typing_metrics:
        var = typing_metrics.get("typing_var", 0.0)
        q2 = typing_metrics.get("typing_q2", 0.0)

        anxiety += min(30, math.log1p(var) * 18)
        fatigue += min(25, q2 * 40)
        focus += max(-20, 20 - math.log1p(var) * 12)

    # ---- 스크롤 ----
    if scroll_metrics:
        total_time = scroll_metrics.get("scroll_total_time", 0.0)
        click_var = scroll_metrics.get("scroll_click_var", 0.0)
        click_mean = scroll_metrics.get("scroll_click_mean", 0.0)

        # 너무 빠른·많은 스크롤(짧은 mean, 큰 var) → 안절부절/산만 쪽 가중
        if click_mean > 0:
            scroll_speed = 1.0 / click_mean
        else:
            scroll_speed = 0.0

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
# 7. 크롤링 예시 (평균값 & 상태별 팁)
# ===============================

AVERAGE_STATS_URL = "https://example.com/phone_emotion_stats.html"
COPING_TIP_URL = "https://example.com/phone_emotion_tips.html"


def fetch_reference_stats() -> Dict[str, float]:
    """
    외부 웹에서 평균적인 상태 값 가져오는 예시.
    BeautifulSoup 사용 (과제 요구사항용).
    """
    try:
        resp = requests.get(AVERAGE_STATS_URL, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        def get_span_float(span_id: str, default: float) -> float:
            tag = soup.find("span", id=span_id)
            if tag and tag.text.strip():
                try:
                    return float(tag.text.strip())
                except ValueError:
                    return default
            return default

        return {
            "avg_anxiety": get_span_float("avg_anxiety", 40.0),
            "avg_fatigue": get_span_float("avg_fatigue", 35.0),
            "avg_focus": get_span_float("avg_focus", 55.0),
        }
    except Exception:
        # 데모용 기본값
        return {
            "avg_anxiety": 40.0,
            "avg_fatigue": 35.0,
            "avg_focus": 55.0,
        }


def fetch_coping_tips(topic: str) -> List[str]:
    """
    상태별 간단 팁을 외부 사이트에서 가져오는 예시.
    실패 시 기본 문구 반환.
    """
    try:
        resp = requests.get(COPING_TIP_URL, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        class_map = {
            "anxiety": "tip-anxiety",
            "fatigue": "tip-fatigue",
            "focus": "tip-focus",
        }
        css_class = class_map.get(topic, "")
        if not css_class:
            return []

        container = soup.find("div", class_=css_class)
        if not container:
            return []

        tips = []
        for li in container.find_all("li"):
            text = li.get_text(strip=True)
            if text:
                tips.append(text)
        return tips[:5]
    except Exception:
        if topic == "anxiety":
            return [
                "천천히 깊게 숨을 들이쉬고 내쉬는 호흡을 몇 번 반복해 보세요.",
                "잠깐 의자에서 일어나 주변을 걸어보세요.",
            ]
        elif topic == "fatigue":
            return [
                "눈을 감고 20~30초 정도 휴식을 취해 보세요.",
                "목·어깨를 가볍게 돌리며 스트레칭해 보세요.",
            ]
        elif topic == "focus":
            return [
                "5~10분 정도 한 가지 일에만 집중해 보는 짧은 타이머를 설정해 보세요.",
                "잠깐 동안 알림을 꺼두고 화면에만 집중해 보세요.",
            ]
        else:
            return []


# ===============================
# 8. 세션 상태 초기화
# ===============================

if "pattern_canvas_json" not in st.session_state:
    st.session_state["pattern_canvas_json"] = None

if "pattern_start_time" not in st.session_state:
    st.session_state["pattern_start_time"] = None

if "pattern_duration" not in st.session_state:
    st.session_state["pattern_duration"] = None

if "pattern_index" not in st.session_state:
    st.session_state["pattern_index"] = 0

if "typing_timestamps" not in st.session_state:
    st.session_state["typing_timestamps"] = []

if "scroll_start_time" not in st.session_state:
    st.session_state["scroll_start_time"] = None

if "scroll_click_times" not in st.session_state:
    st.session_state["scroll_click_times"] = []

if "pattern_canvas_key" not in st.session_state:
    st.session_state["pattern_canvas_key"] = 0


# ===============================
# 9. 사이드바 네비게이션
# ===============================

st.sidebar.title("📱 피젯 감정 탐색 앱")
page = st.sidebar.radio(
    "메뉴 선택",
    [
        "1. 잠금화면 패턴 그리기",
        "2. 키보드 누르기",
        "3. 스크롤 테스트",
        "4. 사용자 활동 분석",
    ],
)


# ===============================
# 10-1. 잠금화면 패턴 그리기
# ===============================

if page.startswith("1"):
    st.header("🔐 1. 잠금화면 패턴 그리기")

    current_pattern = LOCK_PATTERNS[st.session_state["pattern_index"]]
    st.markdown(
        f"""
        **잠금화면을 풀 듯이**, 아래에 보이는 3×3 점들을 이용해서  
        아래 도안을 따라 선을 그려보세요.

        - 이번 도안: **{describe_pattern(current_pattern)}**  
        - 점 위를 자연스럽게 지나가면서, 한 번에 쭉 그려보면 됩니다.  
        - 정확하게 똑같이 안 맞아도 괜찮습니다.
        """
    )

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption("※ 다음 도안을 보고 싶으면 오른쪽의 '다른 도안 보기' 버튼을 눌러주세요.")
    with col_btn:
        if st.button("다른 도안 보기"):
            st.session_state["pattern_index"] = (st.session_state["pattern_index"] + 1) % len(LOCK_PATTERNS)
            # 캔버스 리셋
            st.session_state["pattern_canvas_json"] = None
            st.session_state["pattern_canvas_key"] += 1
            st.session_state["pattern_start_time"] = None
            st.session_state["pattern_duration"] = None
            st.rerun()

    st.markdown("---")
    st.markdown("**아래 상자 안에서 패턴을 그려보세요.**")

    # 패턴 시작 버튼
    if st.session_state["pattern_start_time"] is None:
        if st.button("패턴 그리기 시작"):
            st.session_state["pattern_start_time"] = time.time()

    # 캔버스
    initial_objects = get_lock_points()
    initial_json = {
        "version": "4.4.0",
        "objects": initial_objects
    }

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=4,
        stroke_color="black",
        background_color="#FFFFFF",
        height=400,
        width=400,
        drawing_mode="freedraw",
        point_display_radius=0,
        key=f"pattern_canvas_{st.session_state['pattern_canvas_key']}",
        initial_drawing=initial_json,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("이 패턴 저장하기"):
            st.session_state["pattern_canvas_json"] = canvas_result.json_data
            if st.session_state["pattern_start_time"] is not None:
                st.session_state["pattern_duration"] = time.time() - st.session_state["pattern_start_time"]
            st.success("패턴 그리기 데이터를 저장했습니다. (4번 탭에서 분석에 사용됩니다.)")
    with col2:
        if st.button("지우고 다시 그리기"):
            st.session_state["pattern_canvas_json"] = None
            st.session_state["pattern_duration"] = None
            st.session_state["pattern_start_time"] = None
            st.session_state["pattern_canvas_key"] += 1
            st.rerun()


# ===============================
# 10-2. 키보드 누르기
# ===============================

elif page.startswith("2"):
    st.header("⌨️ 2. 키보드 누르기")

    st.markdown(
        """
        아래 가상의 키보드를 **여러 번** 눌러보세요.

        - 일정한 속도로 눌러도 좋고,  
        - 생각나는 대로 톡톡 두드려도 괜찮습니다.  

        단어를 치려는 느낌보다는,  
        **손가락으로 리듬을 만든다**는 느낌으로 눌러보면 됩니다.
        """
    )

    if st.button("기록 초기화하고 다시 시작하기"):
        st.session_state["typing_timestamps"] = []
        st.success("지금까지의 키보드 누른 기록을 모두 지웠습니다.")

    st.markdown("---")
    st.text("가상 키보드 (아무 버튼이나 눌러보세요)")

    rows = [
        ["Q", "W", "E", "R", "T", "Y", "U"],
        ["A", "S", "D", "F", "G", "H", "J"],
        ["Z", "X", "C", "V", "B", "N", "M"],
    ]

    for r_idx, row_keys in enumerate(rows):
        cols = st.columns(len(row_keys))
        for i, key_label in enumerate(row_keys):
            with cols[i]:
                if st.button(key_label, key=f"kb_{r_idx}_{key_label}"):
                    st.session_state["typing_timestamps"].append(time.time())

    st.write(f"지금까지 누른 횟수: {len(st.session_state['typing_timestamps'])}")
    st.caption("※ 분석은 4번 탭에서 종합해서 보여줍니다.")


# ===============================
# 10-3. 스크롤 테스트
# ===============================

elif page.startswith("3"):
    st.header("🧷 3. 스크롤 테스트")

    st.markdown(
        """
        이번에는 **스크롤하는 습관**을 가볍게 살펴보는 화면입니다.

        1. 아래 긴 글을 천천히 내려가면서 읽어보거나,  
        2. 아래쪽에 있는 버튼을 이용해 **페이지를 내려가는 느낌으로** 눌러보세요.  

        너무 의식하지 말고,  
        평소처럼 화면을 내리는 느낌으로 사용하면 됩니다.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("스크롤 테스트 시작"):
            st.session_state["scroll_start_time"] = time.time()
            st.session_state["scroll_click_times"] = []
            st.success("스크롤 테스트를 시작했습니다. 아래 내용을 읽거나 스크롤 버튼을 눌러보세요.")
    with col_b:
        if st.button("스크롤 기록 초기화"):
            st.session_state["scroll_start_time"] = None
            st.session_state["scroll_click_times"] = []
            st.info("스크롤 관련 기록을 모두 지웠습니다.")

    st.markdown("---")

    # 스크롤할 수 있도록 긴 텍스트 제공
    st.subheader("스크롤용 텍스트")

    long_text = """
    이 부분은 스크롤을 만들기 위한 예시 텍스트입니다.  
    천천히 내려가면서 읽어보아도 좋고, 그냥 화면을 위아래로 움직여보아도 괜찮습니다.  

    - 지금 내가 어느 정도 집중해 있는지  
    - 마음이 조급한지, 여유로운지  
    - 손이 얼마나 자주 화면을 내리고 있는지  

    이런 것들이 자연스럽게 드러날 수 있습니다.  
    아래로 내려가다 보면 같은 문장이 여러 번 반복됩니다.

    ---
    """ * 6

    st.write(long_text)

    st.markdown("**버튼을 눌러서 '스크롤했다'는 표시를 남길 수도 있습니다.**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬇️ 아래로 더 내리기 (스크롤 느낌)"):
            if st.session_state["scroll_start_time"] is None:
                st.session_state["scroll_start_time"] = time.time()
            st.session_state["scroll_click_times"].append(time.time())
    with col2:
        if st.button("⬇️ 다음 부분 보기 (스크롤 느낌)"):
            if st.session_state["scroll_start_time"] is None:
                st.session_state["scroll_start_time"] = time.time()
            st.session_state["scroll_click_times"].append(time.time())

    st.write(f"스크롤 버튼을 누른 횟수: {len(st.session_state['scroll_click_times'])}")
    st.caption("※ 실제 마우스/트랙패드 스크롤도 가능하지만, 분석에는 위 버튼의 기록이 사용됩니다.")


# ===============================
# 10-4. 사용자 활동 분석
# ===============================

elif page.startswith("4"):
    st.header("📊 4. 사용자 활동 분석")

    pattern_metrics = compute_pattern_metrics(
        st.session_state["pattern_canvas_json"],
        st.session_state["pattern_duration"],
    ) if st.session_state["pattern_canvas_json"] is not None else {}

    typing_metrics = compute_typing_metrics(st.session_state["typing_timestamps"]) \
        if st.session_state["typing_timestamps"] else {}

    scroll_metrics = compute_scroll_metrics(
        st.session_state["scroll_start_time"],
        st.session_state["scroll_click_times"],
    ) if st.session_state["scroll_click_times"] else {}

    if not (pattern_metrics or typing_metrics or scroll_metrics):
        st.info("아직 수집된 데이터가 충분하지 않습니다. 1~3번 화면을 먼저 사용해 본 뒤 다시 와 주세요.")
    else:
        st.subheader("① 각각의 활동에서 추출된 특징")

        if pattern_metrics:
            st.markdown("#### 잠금화면 패턴 그리기")
            st.write(pd.DataFrame([pattern_metrics]).T.rename(columns={0: "값"}))
            st.markdown(
                """
                - `pattern_rmse`: 전체 선이 하나의 직선이라고 가정했을 때, 그 직선에서 얼마나 벗어나 있는지  
                - `pattern_jerkiness`: 선을 따라 움직일 때, 선 분 길이가 얼마나 들쭉날쭉했는지  
                - `pattern_length`: 전체 선의 길이  
                - `pattern_duration`: 패턴을 그리는 데 걸린 시간(초)  
                """
            )

        if typing_metrics:
            st.markdown("#### 키보드 누르기 (버튼 사이 시간 간격)")
            st.write(pd.DataFrame([typing_metrics]).T.rename(columns={0: "값"}))
            st.markdown(
                """
                - `typing_q1/Q2/Q3`: 버튼 사이 시간 간격의 분포(아래쪽, 중앙, 위쪽 분위수)  
                - `typing_var`: 간격의 변동성(리듬이 일정한지, 많이 흔들리는지)  
                - `typing_mean`: 평균 간격  
                - `typing_count`: 분석에 사용된 간격 개수  
                """
            )

        if scroll_metrics:
            st.markdown("#### 스크롤 테스트 (버튼 기준)")
            st.write(pd.DataFrame([scroll_metrics]).T.rename(columns={0: "값"}))
            st.markdown(
                """
                - `scroll_total_time`: 스크롤 테스트를 진행한 총 시간(초)  
                - `scroll_click_count`: 스크롤 버튼을 누른 횟수  
                - `scroll_click_mean`: 버튼 사이 평균 간격  
                - `scroll_click_var`: 버튼 사이 간격의 변동성  
                """
            )

        # ---- 종합 점수 ----
        state_scores = analyze_state(pattern_metrics, typing_metrics, scroll_metrics)
        ref_stats = fetch_reference_stats()

        st.subheader("② 이 앱이 추정한 나의 상태 점수 (0~100)")

        df_scores = pd.DataFrame([state_scores], index=["나"]).T
        st.write(df_scores)

        st.markdown(
            """
            - **불안 점수**: 손 움직임이 들쭉날쭉하거나, 패턴·스크롤이 급하게 진행될수록 높은 쪽으로 움직입니다.  
            - **피로 점수**: 전반적으로 움직임이 느려지고(버튼 간 간격이 길어지고), 시간이 오래 걸릴수록 올라갑니다.  
            - **집중/안정 점수**: 패턴·리듬·스크롤이 비교적 일정하고 과하게 흔들리지 않을수록 높게 나타납니다.  

            이 점수는 **진단 결과가 아니라**,  
            잠깐 동안의 손 움직임을 기반으로 한 작은 거울처럼 이해하면 좋습니다.
            """
        )

        # ---- 평균과 비교 ----
        st.subheader("③ 다른 사람들의 평균(예시 값)과 비교")

        compare_df = pd.DataFrame({
            "나": [
                state_scores["anxiety_score"],
                state_scores["fatigue_score"],
                state_scores["focus_score"],
            ],
            "평균(예시)": [
                ref_stats["avg_anxiety"],
                ref_stats["avg_fatigue"],
                ref_stats["avg_focus"],
            ],
        }, index=["불안", "피로", "집중/안정"])

        st.write(compare_df)

        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(compare_df.index))
        width = 0.35

        ax.bar(x - width/2, compare_df["나"], width, label="나")
        ax.bar(x + width/2, compare_df["평균(예시)"], width, label="평균(예시)")

        ax.set_xticks(x)
        ax.set_xticklabels(compare_df.index)
        ax.set_ylabel("점수 (0~100)")
        ax.set_title("나와 평균 상태 비교")
        ax.legend()

        st.pyplot(fig)

        # ---- 상태별 팁 ----
        st.subheader("④ 상태별로 참고해볼 만한 제안")

        col_a, col_f, col_c = st.columns(3)

        with col_a:
            st.markdown("##### 불안 점수가 높게 나왔을 때")
            for t in fetch_coping_tips("anxiety"):
                st.markdown(f"- {t}")

        with col_f:
            st.markdown("##### 피로 점수가 높게 나왔을 때")
            for t in fetch_coping_tips("fatigue"):
                st.markdown(f"- {t}")

        with col_c:
            st.markdown("##### 집중/안정 점수를 올려보고 싶을 때")
            for t in fetch_coping_tips("focus"):
                st.markdown(f"- {t}")

        st.caption(
            "※ 평균 값과 제안 문구는 데모용입니다. 실제 프로젝트에서는 신뢰할 수 있는 사이트를 골라 "
            "BeautifulSoup으로 데이터를 가져오도록 수정할 수 있습니다."
        )
