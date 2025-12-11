import time
import io
import math
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas


# ===============================
# 0. 기본 설정
# ===============================

st.set_page_config(
    page_title="터치·타자 기반 상태 분석 앱",
    layout="wide"
)


# ===============================
# 1. 선 따라 그리기용 배경 이미지 만들기
# ===============================

@st.cache_data
def make_line_background(width=600, height=300) -> Image.Image:
    """
    흰 배경에 가운데를 가로지르는 얇은 회색 직선을 그린 이미지 생성.
    사용자가 이 선을 따라 그리도록 안내.
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = height // 2
    margin = 40
    draw.line((margin, y, width - margin, y), fill=(200, 200, 200), width=3)
    return img


# ===============================
# 2. 선 따라 그리기 특징 추출
# ===============================

def compute_line_metrics(canvas_json: Dict[str, Any]) -> Dict[str, float]:
    """
    선 따라 그리기 데이터에서 특징 추출.
    - path 중 'L' 커맨드의 (x, y)들을 이용해 선형성/떨림 정도 계산
    """
    if not canvas_json or "objects" not in canvas_json:
        return {}

    xs = []
    ys = []

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

    # 1) 선형성: y = ax + b로 회귀 → 잔차(RMSE)
    A = np.vstack([xs, np.ones(len(xs))]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    y_hat = a * xs + b
    residuals = ys - y_hat
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # 2) 길이 & jerkiness
    diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    total_length = float(np.sum(diffs))
    jerkiness = float(np.std(diffs))  # 세그먼트 길이 변동성

    return {
        "line_rmse": rmse,          # 직선에서 얼마나 벗어났는지
        "line_length": total_length,
        "line_jerkiness": jerkiness
    }


# ===============================
# 3. 타자 리듬 특징 추출 (Mandi et al. 스타일)
# ===============================

def compute_typing_metrics(timestamps: List[float]) -> Dict[str, float]:
    """
    타이핑 타임스탬프 → ITD(Inter-Tap Duration) → 분위수/변동성 계산.
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
# 4. 상태 분석 heuristic
# ===============================

def analyze_state(
    line_metrics: Dict[str, float],
    typing_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    지금까지 본 논문들 패턴을 바탕으로
    - 불안(Anxiety)
    - 피로(Fatigue)
    - 집중/안정(Focus)
    간단 점수(0~100)로 환산.
    """
    anxiety = 0.0
    fatigue = 0.0
    focus = 50.0  # 중간에서 시작

    # --- 선 따라 그리기: RMSE, jerkiness ---
    if line_metrics:
        rmse = line_metrics["line_rmse"]
        jerk = line_metrics["line_jerkiness"]

        # rmse, jerkiness가 클수록 불안/스트레스↑, 집중↓
        anxiety += min(50, rmse * 4 + jerk * 3)
        focus -= min(25, rmse * 2 + jerk * 2)

    # --- 타자 리듬: 변동성 & 중앙값 ---
    if typing_metrics:
        var = typing_metrics["typing_var"]
        q2 = typing_metrics["typing_q2"]

        # 변동성↑ → 불안↑
        anxiety += min(30, math.log1p(var) * 18)
        # 중앙값 ITD가 커질수록(전반적으로 느리면) 피로↑
        fatigue += min(30, q2 * 40)
        # 안정적인 리듬이면 집중↑
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
# 5. 크롤링 예시 (평균 상태 & 해결책)
# ===============================

AVERAGE_STATS_URL = "https://example.com/phone_emotion_stats.html"
COPING_TIP_URL = "https://example.com/phone_emotion_tips.html"


def fetch_reference_stats() -> Dict[str, float]:
    """
    웹에서 '평균적인 상태' 정보를 긁어오는 예시.
    실제로 쓸 땐 URL과 span id만 바꿔주면 됨.
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
        # 실패 시 기본값 사용
        return {
            "avg_anxiety": 40.0,
            "avg_fatigue": 35.0,
            "avg_focus": 55.0,
        }


def fetch_coping_tips(topic: str) -> List[str]:
    """
    'anxiety' / 'fatigue' / 'focus'별 해결책 텍스트 크롤링 예시.
    실패 시 기본 팁 반환.
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
                "천천히 4-7-8 호흡을 1분간 반복해보세요.",
                "짧게라도 몸을 움직이거나 스트레칭을 해보세요.",
                "머릿속 걱정을 종이에 적고, 지금 할 수 있는 한 가지만 골라보세요.",
            ]
        elif topic == "fatigue":
            return [
                "화면에서 눈을 떼고 20~30초 동안 먼 곳을 바라보세요.",
                "목·어깨를 가볍게 돌리면서 스트레칭해보세요.",
                "가능하다면 5분 정도 자리에서 일어나 걸어보세요.",
            ]
        elif topic == "focus":
            return [
                "타이머를 10~15분으로 맞추고, 한 가지 일에만 집중해보세요.",
                "핸드폰 알림을 잠시 꺼두면 집중 유지에 도움이 됩니다.",
            ]
        else:
            return []


# ===============================
# 6. 세션 상태 초기화
# ===============================

if "line_json" not in st.session_state:
    st.session_state["line_json"] = None

if "typing_taps" not in st.session_state:
    st.session_state["typing_taps"] = []

# 캔버스 리셋용 key
if "line_canvas_key" not in st.session_state:
    st.session_state["line_canvas_key"] = 0


# ===============================
# 7. 사이드바 네비게이션
# ===============================

st.sidebar.title("📱 감정·상태 피젯 앱")
page = st.sidebar.radio(
    "메뉴를 선택하세요",
    ["1. 선 따라 그리기", "2. 타자 리듬 테스트", "3. 종합 결과 보기"],
)


# ===============================
# 8-1. 선 따라 그리기
# ===============================

if page.startswith("1"):
    st.header("✏️ 1. 선 따라 그리기 (Line Tracing)")

    st.markdown(
        """
        아래 회색 선을 **손가락(또는 마우스)**으로 최대한 따라 그려보세요.  
        - 선에서 많이 벗어나거나, 떨리면서 그려지면  
          → 논문에서 이야기한 것처럼 **불안·긴장·피로**가 반영될 수 있습니다.
        """
    )

    bg_img = make_line_background()

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="black",
        background_image=bg_img,
        height=300,
        width=600,
        drawing_mode="freedraw",
        point_display_radius=0,
        key=f"line_canvas_{st.session_state['line_canvas_key']}",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("이 그림으로 분석하기"):
            st.session_state["line_json"] = canvas_result.json_data
            st.success("선 따라 그리기 데이터를 저장했습니다. (종합 결과 탭에서 사용됩니다.)")
    with col2:
        if st.button("화면 지우기"):
            st.session_state["line_json"] = None
            st.session_state["line_canvas_key"] += 1  # key를 바꿔서 캔버스 초기화

    if st.session_state["line_json"]:
        metrics = compute_line_metrics(st.session_state["line_json"])
        if metrics:
            st.subheader("현재 그림에 대한 기본 지표")
            st.write(pd.DataFrame([metrics]).T.rename(columns={0: "값"}))
        else:
            st.info("그려진 선이 너무 적어서 분석이 어렵습니다. 선 전체를 한 번 이상 따라 그려보세요.")


# ===============================
# 8-2. 타자 리듬 테스트
# ===============================

elif page.startswith("2"):
    st.header("⌨️ 2. 타자 리듬 테스트 (Typing Rhythm)")

    st.markdown(
        """
        아래 **가상 키보드 버튼을 20번 이상** 눌러보세요.  
        - 무엇을 치는지는 중요하지 않고,  
        - **얼마나 일정한 리듬으로 누르는지**가 중요합니다.  
        논문에서는 **Inter-Tap Duration(ITD)의 분위수(Q1/Q2/Q3)와 변동성**이  
        감정 상태를 잘 반영한다고 보고합니다.
        """
    )

    if st.button("테스트 시작 (기록 초기화)"):
        st.session_state["typing_taps"] = []
        st.success("타이핑 타임스탬프를 초기화했습니다.")

    st.text("가상 키보드 (아무 버튼이나 눌러도 됩니다)")

    cols = st.columns(6)
    keys = ["A", "S", "D", "F", "J", "K"]
    for i, key_label in enumerate(keys):
        with cols[i]:
            if st.button(key_label, key=f"kb_{key_label}"):
                st.session_state["typing_taps"].append(time.time())

    st.write(f"현재 눌린 횟수: {len(st.session_state['typing_taps'])}")

    if len(st.session_state["typing_taps"]) >= 5:
        metrics = compute_typing_metrics(st.session_state["typing_taps"])
        st.subheader("타자 리듬 지표 (ITD 기반)")
        st.write(pd.DataFrame([metrics]).T.rename(columns={0: "값"}))
    else:
        st.info("5번 이상 눌러야 기본적인 분석이 가능합니다.")


# ===============================
# 8-3. 종합 결과 보기
# ===============================

elif page.startswith("3"):
    st.header("📊 3. 종합 상태 분석 & 시각화")

    line_metrics = compute_line_metrics(st.session_state["line_json"]) if st.session_state["line_json"] else {}
    typing_metrics = compute_typing_metrics(st.session_state["typing_taps"]) if st.session_state["typing_taps"] else {}

    if not (line_metrics or typing_metrics):
        st.info("아직 수집된 데이터가 없습니다. 1, 2번 테스트를 먼저 진행해 주세요.")
    else:
        state_scores = analyze_state(line_metrics, typing_metrics)
        ref_stats = fetch_reference_stats()

        st.subheader("① 나의 상태 점수")
        df_scores = pd.DataFrame(
            [state_scores],
            index=["나"]
        ).T
        st.write(df_scores)

        st.subheader("② '평균적인 값'과 비교 (예시용 크롤링 결과)")
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

        # 막대그래프 시각화
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(compare_df.index))
        width = 0.35

        ax.bar(x - width/2, compare_df["나"], width, label="나")
        ax.bar(x + width/2, compare_df["평균(예시)"], width, label="평균(예시)")

        ax.set_xticks(x)
        ax.set_xticklabels(compare_df.index)
        ax.set_ylabel("점수 (0~100)")
        ax.set_title("나 vs 평균 비교")
        ax.legend()

        st.pyplot(fig)

        # 해결책 제안
        st.subheader("③ 상태별 해결책 제안")

        col_a, col_f, col_c = st.columns(3)

        with col_a:
            st.markdown("### 불안(Anxiety) 관련 제안")
            for t in fetch_coping_tips("anxiety"):
                st.markdown(f"- {t}")

        with col_f:
            st.markdown("### 피로(Fatigue) 관련 제안")
            for t in fetch_coping_tips("fatigue"):
                st.markdown(f"- {t}")

        with col_c:
            st.markdown("### 집중/안정(Focus) 관련 제안")
            for t in fetch_coping_tips("focus"):
                st.markdown(f"- {t}")

        st.caption(
            "※ 해결책 텍스트와 평균값은 데모용입니다. 실제 프로젝트에서는 신뢰할 만한 정신건강/웰빙 사이트를 골라 BeautifulSoup으로 가져오도록 수정하세요."
        )
