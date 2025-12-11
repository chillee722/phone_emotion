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
# 0. 기본 설정 & 화면 스타일
# ===============================

st.set_page_config(
    page_title="터치·타자 기반 피젯 감정 탐색",
    layout="wide"
)

# 화면 여백 줄이고 헤더/풋터 숨기기 (노트북 화면 꽉 차게)
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
# 1. 선 따라 그리기용 배경 이미지
# ===============================

@st.cache_data
def make_line_background(width=600, height=300) -> Image.Image:
    """
    흰 배경에 가운데를 가로지르는 얇은 회색 직선을 그린 이미지.
    사용자는 이 선을 따라 자연스럽게 그리게 됨.
    """
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = height // 2
    margin = 40
    draw.line((margin, y, width - margin, y), fill=(200, 200, 200), width=3)
    return img


def pil_to_bytes(img: Image.Image) -> bytes:
    """PIL 이미지를 PNG bytes로 변환 (canvas background_image용)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===============================
# 2. 선 따라 그리기 특징 추출
# ===============================

def compute_line_metrics(canvas_json: Dict[str, Any]) -> Dict[str, float]:
    """
    선 따라 그리기 데이터에서 특징 추출.
    - path 중 'M'/'L' 커맨드의 (x, y)를 모아서
      선형성(RMSE) + jerkiness(세그먼트 길이 변동) 계산
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

    # 직선 근사 y = ax + b
    A = np.vstack([xs, np.ones(len(xs))]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    y_hat = a * xs + b
    residuals = ys - y_hat
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    total_length = float(np.sum(diffs))
    jerkiness = float(np.std(diffs))

    return {
        "line_rmse": rmse,
        "line_length": total_length,
        "line_jerkiness": jerkiness,
    }


# ===============================
# 3. 타자 리듬 특징 (ITD 기반)
# ===============================

def compute_typing_metrics(timestamps: List[float]) -> Dict[str, float]:
    """
    타이핑 타임스탬프 → Inter-Tap Duration(ITD) → 분위수/변동성 계산.
    논문(Mandi et al.)에서 쓴 구조와 유사한 형태.
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
    논문에서 나온 경향을 참고해서
    - 불안(Anxiety)
    - 피로(Fatigue)
    - 집중/안정(Focus)
    점수(0~100)로 환산하는 간단한 heuristic.
    """
    anxiety = 0.0
    fatigue = 0.0
    focus = 50.0  # 중간값에서 시작

    # 선 따라 그리기: 직선에서 많이 벗어나고 떨릴수록 점수 변화
    if line_metrics:
        rmse = line_metrics["line_rmse"]
        jerk = line_metrics["line_jerkiness"]

        anxiety += min(50, rmse * 4 + jerk * 3)
        focus -= min(25, rmse * 2 + jerk * 2)

    # 타자 리듬: ITD 변동성과 중앙값
    if typing_metrics:
        var = typing_metrics["typing_var"]
        q2 = typing_metrics["typing_q2"]

        # 변동성↑ → 리듬이 불규칙 → 불안/긴장 쪽 가중
        anxiety += min(30, math.log1p(var) * 18)
        # ITD 전체가 길어짐(q2↑) → 전반적으로 느림 → 피로 가중
        fatigue += min(30, q2 * 40)
        # 변동성이 낮으면 집중/안정↑
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
# 5. 크롤링 예시 (평균값 & 해결책)
# ===============================

AVERAGE_STATS_URL = "https://example.com/phone_emotion_stats.html"
COPING_TIP_URL = "https://example.com/phone_emotion_tips.html"


def fetch_reference_stats() -> Dict[str, float]:
    """
    BeautifulSoup으로 외부 페이지에서 평균적인 상태 값을 가져오는 예시.
    실제 사용할 때는 URL과 span id를 수정하면 됨.
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
    'anxiety' / 'fatigue' / 'focus'에 해당하는 해결책 텍스트를 가져오는 예시.
    실패 시 기본 팁들을 반환.
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
# 6. 세션 상태
# ===============================

if "line_json" not in st.session_state:
    st.session_state["line_json"] = None

if "typing_taps" not in st.session_state:
    st.session_state["typing_taps"] = []

if "line_canvas_key" not in st.session_state:
    st.session_state["line_canvas_key"] = 0


# ===============================
# 7. 사이드바 네비게이션
# ===============================

st.sidebar.title("📱 피젯 감정 탐색 앱")
page = st.sidebar.radio(
    "메뉴",
    ["1. 선 따라 그리기", "2. 타자 리듬 테스트", "3. 종합 결과 보기"],
)


# ===============================
# 8-1. 선 따라 그리기
# ===============================

if page.startswith("1"):
    st.header("✏️ 1. 선 따라 그리기")

    st.markdown(
        """
        아래 회색 선을 **손가락(또는 마우스)**으로 한 번 쭉 따라 그려보세요.  
        어떻게 그리는지는 신경 쓰지 말고, 그냥 자연스럽게 그려보면 됩니다.
        """
    )

    bg_img = make_line_background()
    bg_bytes = pil_to_bytes(bg_img)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="black",
        background_image=bg_bytes,
        height=300,
        width=600,
        drawing_mode="freedraw",
        point_display_radius=0,
        key=f"line_canvas_{st.session_state['line_canvas_key']}",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("이 그림으로 저장하기"):
            st.session_state["line_json"] = canvas_result.json_data
            st.success("선 따라 그리기 데이터를 저장했습니다. (종합 결과 탭에서 사용됩니다.)")
    with col2:
        if st.button("화면 지우기"):
            st.session_state["line_json"] = None
            st.session_state["line_canvas_key"] += 1  # 캔버스 리셋

    if st.session_state["line_json"]:
        metrics = compute_line_metrics(st.session_state["line_json"])
        if metrics:
            st.subheader("기본 수치 (참고용)")
            st.write(pd.DataFrame([metrics]).T.rename(columns={0: "값"}))
            st.caption("※ 이 수치가 어떤 의미인지는 '종합 결과 보기' 탭에서 설명합니다.")
        else:
            st.info("선이 너무 짧으면 분석이 어렵습니다. 선 전체를 한 번 이상 따라 그려보세요.")


# ===============================
# 8-2. 타자 리듬 테스트
# ===============================

elif page.startswith("2"):
    st.header("⌨️ 2. 타자 리듬 테스트")

    st.markdown(
        """
        아래 **가상 키보드 버튼을 20번 이상** 원하는 대로 눌러보세요.  
        어떤 버튼을 누르는지는 중요하지 않습니다.  
        그냥 손이 가는 대로 두드려보면 됩니다.
        """
    )

    if st.button("테스트 시작 (기록 초기화)"):
        st.session_state["typing_taps"] = []
        st.success("타이핑 기록을 초기화했습니다.")

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
        st.subheader("기본 수치 (참고용)")
        st.write(pd.DataFrame([metrics]).T.rename(columns={0: "값"}))
        st.caption("※ 이 수치가 어떤 의미인지는 '종합 결과 보기' 탭에서 설명합니다.")
    else:
        st.info("5번 이상 눌러야 기본적인 분석이 가능합니다.")


# ===============================
# 8-3. 종합 결과 보기
# ===============================

elif page.startswith("3"):
    st.header("📊 3. 종합 결과 보기")

    line_metrics = compute_line_metrics(st.session_state["line_json"]) if st.session_state["line_json"] else {}
    typing_metrics = compute_typing_metrics(st.session_state["typing_taps"]) if st.session_state["typing_taps"] else {}

    if not (line_metrics or typing_metrics):
        st.info("아직 수집된 데이터가 없습니다. 1, 2번 테스트를 먼저 진행해 주세요.")
    else:
        st.subheader("① 개별 특징이 의미하는 것")

        if line_metrics:
            st.markdown("#### 선 따라 그리기")
            st.write(pd.DataFrame([line_metrics]).T.rename(columns={0: "값"}))
            st.markdown(
                """
                - `line_rmse`: 회색 기준선에서 얼마나 벗어나 있는지 (선형성 편차)\n
                - `line_jerkiness`: 선이 부드럽게 이어졌는지, 중간에 덜컥거리는 구간이 많은지(길이 변동성)\n
                """
            )

        if typing_metrics:
            st.markdown("#### 타자 리듬 (Inter-Tap Duration)")
            st.write(pd.DataFrame([typing_metrics]).T.rename(columns={0: "값"}))
            st.markdown(
                """
                - `typing_q1/Q2/Q3`: 두 번 누르는 사이 시간 간격의 분위수 (리듬의 중심과 범위)\n
                - `typing_var`: 간격의 변동성 (리듬이 일정한지·들쭉날쭉한지)\n
                """
            )

        state_scores = analyze_state(line_metrics, typing_metrics)
        ref_stats = fetch_reference_stats()

        st.subheader("② 나의 상태 점수 (0~100)")

        df_scores = pd.DataFrame([state_scores], index=["나"]).T
        st.write(df_scores)

        st.markdown(
            """
            - **불안 점수**: 기준선에서 많이 벗어나거나, 리듬 변동성이 큰 방향으로 올라갑니다.\n
            - **피로 점수**: 전반적으로 타자 속도가 느려지는 방향(q2↑)으로 올라갑니다.\n
            - **집중/안정 점수**: 선이 비교적 일정하고, 리듬 변동성이 낮을수록 높게 나옵니다.\n
            ※ 점수는 연구용 정확한 진단이 아니라, **행동 패턴을 시각화한 지표**로 이해하면 좋습니다.
            """
        )

        st.subheader("③ '평균적인 값'과 비교 (예시)")

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

        # ---- 한글 폰트 설정 (환경에 따라 조정 필요) ----
        try:
            plt.rcParams["font.family"] = "NanumGothic"  # 서버에 설치된 한글 폰트 이름으로 변경 가능
        except Exception:
            plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["axes.unicode_minus"] = False

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

        st.subheader("④ 상태별 간단한 제안")

        col_a, col_f, col_c = st.columns(3)

        with col_a:
            st.markdown("##### 불안(Anxiety) 관련")
            for t in fetch_coping_tips("anxiety"):
                st.markdown(f"- {t}")

        with col_f:
            st.markdown("##### 피로(Fatigue) 관련")
            for t in fetch_coping_tips("fatigue"):
                st.markdown(f"- {t}")

        with col_c:
            st.markdown("##### 집중/안정(Focus) 관련")
            for t in fetch_coping_tips("focus"):
                st.markdown(f"- {t}")

        st.caption(
            "※ 그래프 한글이 네 환경에서 여전히 깨지면, 서버에 한글 폰트(NanumGothic 등)를 설치하고 "
            "위의 `plt.rcParams['font.family']`를 해당 폰트 이름으로 바꿔 주세요."
        )
