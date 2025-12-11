import time
import io
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
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas


# ===============================
# 0. 한글 폰트 설정 (NanumGothic-Regular.ttf 사용)
# ===============================

font_path = Path(__file__).parent / "NanumGothic-Regular.ttf"
if font_path.exists():
    fontprop = fm.FontProperties(fname=str(font_path))
    matplotlib.rcParams["font.family"] = fontprop.get_name()
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"  # fallback

matplotlib.rcParams["axes.unicode_minus"] = False


# ===============================
# 1. 기본 설정 & 화면 스타일
# ===============================

st.set_page_config(
    page_title="터치·타자 기반 피젯 감정 탐색",
    layout="wide"
)

# 여백 줄이고 헤더/풋터 숨기기
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
# 2. 점 가이드용 초기 캔버스 JSON 만들기
# ===============================

def make_dot_guides(width: int = 600, height: int = 300, n_points: int = 6) -> Dict[str, Any]:
    """
    캔버스 위에 미리 '점' 몇 개를 찍어두는 fabric.js 스타일 JSON 생성.
    사용자는 이 점들을 자연스럽게 이어서 선을 그리게 된다.
    """
    xs = np.linspace(60, width - 60, n_points)
    # 약간의 위/아래 변화를 줘서 너무 완전한 직선이 되지 않게
    ys = height / 2 + np.sin(np.linspace(0, math.pi, n_points)) * 40

    objects = []
    for x, y in zip(xs, ys):
        objects.append({
            "type": "circle",
            "radius": 6,
            "fill": "#4A90E2",
            "stroke": "#FFFFFF",
            "strokeWidth": 2,
            "left": float(x - 6),
            "top": float(y - 6),
            "originX": "left",
            "originY": "top"
        })

    return {
        "version": "4.4.0",
        "objects": objects
    }


# ===============================
# 3. 선 따라 그리기 특징 추출
# ===============================

def compute_line_metrics(canvas_json: Dict[str, Any]) -> Dict[str, float]:
    """
    캔버스 JSON에서 path 타입만 추출하여
    - 직선에서의 편차(RMSE)
    - 길이의 변동성(jerkiness)
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
        return {}

    xs = np.array(xs)
    ys = np.array(ys)

    # y = ax + b 직선 근사
    A = np.vstack([xs, np.ones(len(xs))]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    y_hat = a * xs + b
    residuals = ys - y_hat
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # 연속 구간 길이와 그 변동성
    diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    total_length = float(np.sum(diffs))
    jerkiness = float(np.std(diffs))

    return {
        "line_rmse": rmse,
        "line_length": total_length,
        "line_jerkiness": jerkiness,
    }


# ===============================
# 4. 타자 리듬 특징 추출 (ITD 기반)
# ===============================

def compute_typing_metrics(timestamps: List[float]) -> Dict[str, float]:
    """
    버튼을 누른 시각 리스트 → Inter-Tap Duration(ITD) → 분위수/변동성.
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
# 5. 상태 분석 heuristic
# ===============================

def analyze_state(
    line_metrics: Dict[str, float],
    typing_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    선 따라 그리기 + 타자 리듬에서 얻은 특징으로
    - 불안(Anxiety)
    - 피로(Fatigue)
    - 집중/안정(Focus)
    간단 점수(0~100)를 만드는 heuristic.
    """
    anxiety = 0.0
    fatigue = 0.0
    focus = 50.0  # 중간값에서 시작

    if line_metrics:
        rmse = line_metrics["line_rmse"]
        jerk = line_metrics["line_jerkiness"]

        anxiety += min(50, rmse * 4 + jerk * 3)
        focus -= min(25, rmse * 2 + jerk * 2)

    if typing_metrics:
        var = typing_metrics["typing_var"]
        q2 = typing_metrics["typing_q2"]

        anxiety += min(30, math.log1p(var) * 18)
        fatigue += min(30, q2 * 40)
        focus += max(-20, 20 - math.log1p(var) * 12)

    anxiety = float(max(0, min(100, anxiety)))
    fatigue = float(max(0, min(100, fatigue)))
    focus = float(max(0, min(100, focus)))

    return {
        "anxiety_score": anxiety,
        "fatigue_score": fatigue,
        "focus_score": focus,
    }


# ===============================
# 6. 크롤링 예시 (평균값 & 상태별 팁)
# ===============================

AVERAGE_STATS_URL = "https://example.com/phone_emotion_stats.html"
COPING_TIP_URL = "https://example.com/phone_emotion_tips.html"


def fetch_reference_stats() -> Dict[str, float]:
    """
    외부 웹에서 평균적인 상태 값을 긁어오는 예시.
    (실제 프로젝트에서는 URL과 span id를 수정)
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
    특정 주제(anxiety/fatigue/focus)에 대한 간단한 팁을
    외부 웹에서 긁어오는 예시. 실패 시 기본 팁 반환.
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
# 7. 세션 상태 초기화
# ===============================

if "line_json" not in st.session_state:
    st.session_state["line_json"] = None

if "typing_taps" not in st.session_state:
    st.session_state["typing_taps"] = []

if "line_canvas_key" not in st.session_state:
    st.session_state["line_canvas_key"] = 0

if "line_guides" not in st.session_state:
    # 점 가이드는 한 번 생성해서 계속 재사용
    st.session_state["line_guides"] = make_dot_guides()


# ===============================
# 8. 사이드바 네비게이션
# ===============================

st.sidebar.title("📱 피젯 감정 탐색 앱")
page = st.sidebar.radio(
    "메뉴",
    ["1. 점 이어 그리기", "2. 타자 리듬 테스트", "3. 종합 결과 보기"],
)


# ===============================
# 9-1. 점 이어 그리기 (Line tracing with dots)
# ===============================

if page.startswith("1"):
    st.header("✏️ 1. 점을 이어 선 그리기")

    st.markdown(
        """
        아래 상자 안에 보이는 **파란 점들을 순서대로 이어서**  
        한 번 쭉 선을 그려보세요.  

        - 점을 꼭 정확히 맞추지 않아도 괜찮습니다.  
        - 힘을 빼고, 자연스럽게 움직이는 느낌으로 그려보면 됩니다.
        """
    )

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="black",
        background_color="#FFFFFF",
        height=300,
        width=600,
        drawing_mode="freedraw",
        point_display_radius=0,
        key=f"line_canvas_{st.session_state['line_canvas_key']}",
        initial_drawing=st.session_state["line_guides"],  # 점 가이드
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("지금 그림을 저장하기"):
            st.session_state["line_json"] = canvas_result.json_data
            st.success("그려진 선을 저장했습니다. (종합 결과 보기에서 사용됩니다.)")
    with col2:
        if st.button("지우고 다시 그리기"):
            st.session_state["line_json"] = None
            st.session_state["line_canvas_key"] += 1  # 캔버스 리셋


# ===============================
# 9-2. 타자 리듬 테스트
# ===============================

elif page.startswith("2"):
    st.header("⌨️ 2. 타자 리듬 테스트")

    st.markdown(
        """
        아래 버튼들을 **여러 번** 눌러보세요.  

        - 일정한 속도로 눌러도 좋고,  
        - 생각나는 대로 두드려도 괜찮습니다.  

        그냥 손이 가는 대로 눌러보면 됩니다.
        """
    )

    if st.button("기록 초기화하고 다시 시작하기"):
        st.session_state["typing_taps"] = []
        st.success("지금까지의 버튼 누른 기록을 지웠습니다.")

    st.text("가상 키보드 (아무 버튼이나 눌러보세요)")

    cols = st.columns(6)
    keys = ["A", "S", "D", "F", "J", "K"]
    for i, key_label in enumerate(keys):
        with cols[i]:
            if st.button(key_label, key=f"kb_{key_label}"):
                st.session_state["typing_taps"].append(time.time())

    st.write(f"지금까지 누른 횟수: {len(st.session_state['typing_taps'])}")


# ===============================
# 9-3. 종합 결과 보기 (해석 + 시각화)
# ===============================

elif page.startswith("3"):
    st.header("📊 3. 종합 결과 보기")

    line_metrics = compute_line_metrics(st.session_state["line_json"]) if st.session_state["line_json"] else {}
    typing_metrics = compute_typing_metrics(st.session_state["typing_taps"]) if st.session_state["typing_taps"] else {}

    if not (line_metrics or typing_metrics):
        st.info("아직 저장된 데이터가 없습니다. 1, 2번 화면에서 먼저 해본 뒤 다시 와 주세요.")
    else:
        st.subheader("① 만지작거림에서 추출된 특징들")

        if line_metrics:
            st.markdown("#### 점 이어 그리기")
            st.write(pd.DataFrame([line_metrics]).T.rename(columns={0: "값"}))
            st.markdown(
                """
                - `line_rmse`: 전체적으로 볼 때, 한 줄로 이어졌다고 가정했을 때 그 직선에서 얼마나 벗어나 있는지  
                - `line_jerkiness`: 선을 따라 움직일 때, 선 분 길이가 얼마나 들쭉날쭉했는지  
                """
            )

        if typing_metrics:
            st.markdown("#### 타자 리듬 (버튼 사이 시간 간격)")
            st.write(pd.DataFrame([typing_metrics]).T.rename(columns={0: "값"}))
            st.markmarkdown(
                """
                - `typing_q1/Q2/Q3`: 버튼 사이 시간 간격 분포의 위치(중앙값과 범위)  
                - `typing_var`: 간격의 변동성(리듬이 일정한지, 많이 흔들리는지)  
                """
            )

        state_scores = analyze_state(line_metrics, typing_metrics)
        ref_stats = fetch_reference_stats()

        st.subheader("② 이 앱이 추정한 나의 상태 점수 (0~100)")

        df_scores = pd.DataFrame([state_scores], index=["나"]).T
        st.write(df_scores)

        st.markdown(
            """
            - **불안 점수**: 선의 흔들림과 리듬의 들쭉날쭉함이 클수록 높은 쪽으로 움직입니다.  
            - **피로 점수**: 버튼 간 간격이 전반적으로 길어질수록(느려질수록) 올라갑니다.  
            - **집중/안정 점수**: 선이 비교적 일정하고, 리듬이 너무 흔들리지 않을수록 높게 나옵니다.  

            이 점수는 진단 결과라기보다는,  
            **지금 내 손이 어떤 패턴으로 움직이고 있는지**를 숫자로 요약해서 보여주는 지표라고 보면 좋습니다.
            """
        )

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
            st.markdown("##### 집중/안정 점수를 조금 더 올려보고 싶을 때")
            for t in fetch_coping_tips("focus"):
                st.markdown(f"- {t}")

        st.caption(
            "※ 평균 값과 제안 문구는 데모용입니다. 실제 프로젝트에서는 신뢰할 수 있는 사이트를 골라 "
            "BeautifulSoup으로 데이터를 가져오도록 수정할 수 있습니다."
        )
