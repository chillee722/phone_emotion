import time
import math
import random
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas


# ===============================
# 0. 기본 설정
# ===============================

st.set_page_config(
    page_title="터치·타자·Go/No-Go 기반 상태 분석 앱",
    layout="wide"
)


# ===============================
# 1. 유틸 함수들
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
                # seg 예: ['M', x, y] 또는 ['L', x, y, ...]
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

    # 2) 길이 & 점 개수 → 손 떨림 지표
    diffs = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    total_length = float(np.sum(diffs))
    jerkiness = float(np.std(diffs))  # 세그먼트 길이 변동성

    return {
        "line_rmse": rmse,          # 선에서 얼마나 벗어났는지
        "line_length": total_length,
        "line_jerkiness": jerkiness
    }


def compute_typing_metrics(timestamps: List[float]) -> Dict[str, float]:
    """
    타자 리듬 데이터에서 ITD 기반 특징 추출 (Mandi et al. 논문 구조).
    """
    if len(timestamps) < 5:
        return {}

    itds = np.diff(sorted(timestamps))  # 안전하게 시간 순 정렬
    itds = itds[itds > 0]  # 0, 음수 제거

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


def compute_gng_metrics(trials: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Go/No-Go 과제 결과에서 기본 지표 계산.
    trials 예:
    {
      "ttype": "go"/"nogo",
      "stim_time": float,
      "resp_time": float or None,
      "responded": bool
    }
    """
    if not trials:
        return {}

    go_rts = []
    go_hits = 0
    go_total = 0

    nogo_fa = 0
    nogo_total = 0

    for t in trials:
        if t["ttype"] == "go":
            go_total += 1
            if t["responded"] and t["resp_time"] is not None:
                go_hits += 1
                go_rts.append(t["resp_time"] - t["stim_time"])
        else:
            nogo_total += 1
            if t["responded"]:
                nogo_fa += 1

    if go_rts:
        go_rts = np.array(go_rts)
        go_rt_mean = float(np.mean(go_rts))
        go_rt_std = float(np.std(go_rts))
    else:
        go_rt_mean, go_rt_std = float("nan"), float("nan")

    hit_rate = go_hits / go_total if go_total > 0 else float("nan")
    fa_rate = nogo_fa / nogo_total if nogo_total > 0 else float("nan")

    return {
        "gng_go_rt_mean": go_rt_mean,
        "gng_go_rt_std": go_rt_std,
        "gng_hit_rate": hit_rate,
        "gng_fa_rate": fa_rate,
        "gng_go_total": float(go_total),
        "gng_nogo_total": float(nogo_total),
    }


# ===============================
# 2. 상태 분석 로직 (간단 heuristic)
# ===============================

def analyze_state(
    line_metrics: Dict[str, float],
    typing_metrics: Dict[str, float],
    gng_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    논문들에서 나온 패턴을 바탕으로
    - 불안(Anxiety)
    - 피로(Fatigue)
    - 집중/안정(Focus/Calm)
    간단 점수로 환산 (0~100, 높을수록 강함).
    아주 러프한 heuristic이라 “연구결과를 바탕으로 한 데모” 정도로 이해하면 됨.
    """

    anxiety = 0.0
    fatigue = 0.0
    focus = 50.0  # 중간에서 시작한 뒤 가중치로 조정

    # --- 1) 선 따라 그리기: RMSE, jerkiness ---
    if line_metrics:
        rmse = line_metrics["line_rmse"]
        jerk = line_metrics["line_jerkiness"]

        # RMSE, jerkiness가 크면 불안/스트레스↑, 집중↓
        anxiety += min(40, rmse * 4 + jerk * 3)  # 적당히 scale
        focus -= min(20, rmse * 2 + jerk * 2)

    # --- 2) 타자 리듬: 변동성/중앙값 (Mandi et al.) ---
    if typing_metrics:
        var = typing_metrics["typing_var"]
        q2 = typing_metrics["typing_q2"]
        # 분산↑ → 불안↑, 피로↑
        anxiety += min(25, math.log1p(var) * 15)
        fatigue += min(25, q2 * 40)  # 전반적으로 느리면 피로↑
        # 안정적인 리듬이면 집중↑
        focus += max(-15, 15 - math.log1p(var) * 10)

    # --- 3) Go/No-Go: fa_rate & rt_mean 기반 (억제/주의) ---
    if gng_metrics:
        fa = gng_metrics["gng_fa_rate"]
        rt = gng_metrics["gng_go_rt_mean"]

        if not math.isnan(fa):
            anxiety += min(25, fa * 100)  # false alarm 많으면 불안/충동↑
            focus -= min(20, fa * 80)

        if not math.isnan(rt):
            # 너무 짧거나 너무 길면 피로/주의분산 의심
            if rt < 0.25:
                anxiety += 10  # 과도하게 급함
            elif rt > 0.7:
                fatigue += 15  # 전반적으로 느림
                focus -= 10

    # 값 범위 clipping
    anxiety = float(max(0, min(100, anxiety)))
    fatigue = float(max(0, min(100, fatigue)))
    focus = float(max(0, min(100, focus)))

    return {
        "anxiety_score": anxiety,
        "fatigue_score": fatigue,
        "focus_score": focus,
    }


# ===============================
# 3. BeautifulSoup을 이용한 크롤링 예시
# ===============================

AVERAGE_STATS_URL = "https://example.com/phone_emotion_stats.html"
COPING_TIP_URL = "https://example.com/phone_emotion_tips.html"


def fetch_reference_stats() -> Dict[str, float]:
    """
    웹에서 '평균적인 상태' 정보를 긁어오는 예시.
    HTML 예시 구조 (가정):

    <span id="avg_anxiety">42.3</span>
    <span id="avg_fatigue">38.1</span>
    <span id="avg_focus">55.0</span>

    실제 쓸 땐 위 URL과 id 이름만 바꾸면 됨.
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
        # 실패 시 예시값 사용
        return {
            "avg_anxiety": 40.0,
            "avg_fatigue": 35.0,
            "avg_focus": 55.0,
        }


def fetch_coping_tips(topic: str) -> List[str]:
    """
    '불안', '피로', '집중력' 같은 키워드로
    해결책/팁을 웹에서 크롤링해 오는 예시.

    예시 HTML 구조(가정):
    <div class="tip anxiety">
        <li>호흡 운동 4-7-8로 1분간 숨쉬기</li>
        <li>...</li>
    </div>

    실제 적용 시에는 topic별 CSS class나 id를 맞춰주면 됨.
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
        return tips[:5]  # 상위 5개만
    except Exception:
        # 크롤링 실패 시 기본 팁 반환
        if topic == "anxiety":
            return [
                "천천히 4-7-8 호흡을 1분간 반복해보세요.",
                "짧게라도 몸을 움직이거나 스트레칭을 해보세요.",
                "해야 할 일을 종이에 써서 '지금 할 수 있는 1가지'만 골라보세요.",
            ]
        elif topic == "fatigue":
            return [
                "20~30초 동안 화면에서 눈을 떼고 먼 곳을 바라보세요.",
                "가벼운 목/어깨 스트레칭을 해보세요.",
                "가능하다면 5분 정도 자리에서 일어나 몸을 움직여 보세요.",
            ]
        elif topic == "focus":
            return [
                "타이머를 10~15분으로 맞추고 그 시간 동안 한 가지 일에만 집중해보세요.",
                "핸드폰 알림을 잠시 꺼두는 것도 도움이 됩니다.",
            ]
        else:
            return []


# ===============================
# 4. Streamlit 세션 상태 초기화
# ===============================

if "line_json" not in st.session_state:
    st.session_state["line_json"] = None

if "typing_taps" not in st.session_state:
    st.session_state["typing_taps"] = []

if "gng_trials" not in st.session_state:
    st.session_state["gng_trials"] = []
    st.session_state["gng_index"] = 0
    st.session_state["gng_running"] = False


# ===============================
# 5. UI: 사이드바 네비게이션
# ===============================

st.sidebar.title("📱 감정·상태 탐색 피젯 앱")
page = st.sidebar.radio(
    "메뉴를 선택하세요",
    ["1. 선 따라 그리기", "2. 타자 리듬 테스트", "3. Go/No-Go 테스트", "4. 종합 결과 보기"],
)


# ===============================
# 6-1. 선 따라 그리기 페이지
# ===============================

if page.startswith("1"):
    st.header("✏️ 1. 선 따라 그리기 (Line Tracing)")

    st.markdown(
        """
        화면 안에서 **한 번에 쭉 직선을 그려보세요.**  
        - 손이 많이 떨리거나, 선이 크게 비틀어지면  
          → 논문에서 말하는 것처럼 **불안/긴장/피로**가 반영될 수 있습니다.
        """
    )

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="black",
        background_color="white",
        height=300,
        width=600,
        drawing_mode="freedraw",
        point_display_radius=0,
        key="line_canvas",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("이 그림으로 분석하기"):
            st.session_state["line_json"] = canvas_result.json_data
            st.success("선 따라 그리기 데이터를 저장했습니다. 종합 결과 탭에서 분석에 사용됩니다.")
    with col2:
        if st.button("화면 지우기"):
            st.session_state["line_json"] = None
            st.experimental_rerun()  # 캔버스 리셋용 (여기는 rerun 써도 안전)

    if st.session_state["line_json"]:
        metrics = compute_line_metrics(st.session_state["line_json"])
        if metrics:
            st.subheader("현재 그림에 대한 기본 지표")
            st.write(pd.DataFrame([metrics]).T.rename(columns={0: "값"}))
        else:
            st.info("그려진 선이 너무 적어서 분석이 어렵습니다. 조금 더 길게 그려주세요.")


# ===============================
# 6-2. 타자 리듬 테스트
# ===============================

elif page.startswith("2"):
    st.header("⌨️ 2. 타자 리듬 테스트 (Typing Rhythm)")

    st.markdown(
        """
        **가상 키보드 버튼을 20번 이상** 눌러보세요.  
        - 내용은 중요하지 않고, **어떤 리듬으로 누르는지**가 중요합니다.  
        - 논문에서처럼 **Inter-Tap Duration(ITD)**의 분위수/변동성을 분석합니다.
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
# 6-3. Go/No-Go 테스트
# ===============================

elif page.startswith("3"):
    st.header("🚦 3. Go/No-Go (반응 억제) 테스트")

    st.markdown(
        """
        **지금부터 자극이 12번** 제시됩니다.  
        - 화면에 **🟢 GO**가 나오면 → 아래 **반응 버튼**을 최대한 빠르게 눌러주세요.  
        - 화면에 **🔴 NO GO**가 나오면 → **아무 것도 누르지 말고, 다음 자극으로 넘어가세요.**  

        Go/No-Go 과제는 논문들에서 **불안/충동성/주의력**과 관련된 행동 지표로 쓰입니다.
        """
    )

    def init_gng():
        trials = []
        for _ in range(12):
            ttype = "go" if random.random() < 0.7 else "nogo"
            trials.append({
                "ttype": ttype,
                "stim_time": None,
                "resp_time": None,
                "responded": False,
            })
        st.session_state["gng_trials"] = trials
        st.session_state["gng_index"] = 0
        st.session_state["gng_running"] = True

    if st.button("테스트 새로 시작하기"):
        init_gng()
        st.success("Go/No-Go 테스트를 시작합니다.")

    if st.session_state["gng_running"] and st.session_state["gng_trials"]:
        idx = st.session_state["gng_index"]
        trials = st.session_state["gng_trials"]

        if idx >= len(trials):
            st.session_state["gng_running"] = False
        else:
            trial = trials[idx]

            # 자극 제시 시점 기록
            if trial["stim_time"] is None:
                trial["stim_time"] = time.time()
                st.session_state["gng_trials"][idx] = trial

            # 자극 표시
            if trial["ttype"] == "go":
                st.subheader("🟢 GO (지금 눌러야 합니다!)")
            else:
                st.subheader("🔴 NO GO (누르지 마세요)")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("반응 버튼"):
                    if not trial["responded"]:
                        trial["responded"] = True
                        trial["resp_time"] = time.time()
                        st.session_state["gng_trials"][idx] = trial
            with col2:
                if st.button("다음 자극으로"):
                    st.session_state["gng_index"] += 1

            st.write(f"진행 상황: {idx+1} / {len(trials)}")

    if (not st.session_state["gng_running"]) and st.session_state["gng_trials"]:
        st.success("Go/No-Go 테스트가 끝났습니다. 아래에서 요약 지표를 볼 수 있습니다.")
        metrics = compute_gng_metrics(st.session_state["gng_trials"])
        st.write(pd.DataFrame([metrics]).T.rename(columns={0: "값"}))


# ===============================
# 6-4. 종합 결과 보기
# ===============================

elif page.startswith("4"):
    st.header("📊 4. 종합 상태 분석 & 시각화")

    st.markdown(
        """
        앞의 세 가지 테스트 결과를 **통합하여**  
        - 불안(Anxiety)  
        - 피로(Fatigue)  
        - 집중/안정(Focus)  
        간단한 점수로 환산하고,  
        웹에서 가져온 **'평균적인 값'**과 비교해 보여줍니다.
        """
    )

    line_metrics = compute_line_metrics(st.session_state["line_json"]) if st.session_state["line_json"] else {}
    typing_metrics = compute_typing_metrics(st.session_state["typing_taps"]) if st.session_state["typing_taps"] else {}
    gng_metrics = compute_gng_metrics(st.session_state["gng_trials"]) if st.session_state["gng_trials"] else {}

    if not (line_metrics or typing_metrics or gng_metrics):
        st.info("아직 수집된 데이터가 없습니다. 위의 1~3번 테스트를 먼저 진행해 주세요.")
    else:
        state_scores = analyze_state(line_metrics, typing_metrics, gng_metrics)
        ref_stats = fetch_reference_stats()

        st.subheader("① 나의 상태 점수")
        df_scores = pd.DataFrame(
            [state_scores],
            index=["나"]
        ).T
        st.write(df_scores)

        st.subheader("② '평균적인 값'과 비교 (예시용 크롤링)")
        compare_df = pd.DataFrame({
            "나": [
                state_scores["anxiety_score"],
                state_scores["fatigue_score"],
                state_scores["focus_score"],
            ],
            "평균(크롤링/기본값)": [
                ref_stats["avg_anxiety"],
                ref_stats["avg_fatigue"],
                ref_stats["avg_focus"],
            ],
        }, index=["불안", "피로", "집중"])

        st.write(compare_df)

        # 간단한 막대그래프 시각화
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(compare_df.index))
        width = 0.35

        ax.bar(x - width/2, compare_df["나"], width, label="나")
        ax.bar(x + width/2, compare_df["평균(크롤링/기본값)"], width, label="평균")

        ax.set_xticks(x)
        ax.set_xticklabels(compare_df.index)
        ax.set_ylabel("점수 (0~100)")
        ax.set_title("나 vs 평균 비교")
        ax.legend()

        st.pyplot(fig)

        # ------------------------
        # ③ 상태별 해결책 제안 (크롤링 + 기본값)
        # ------------------------
        st.subheader("③ 상태별 해결책 제안")

        col_a, col_f, col_c = st.columns(3)

        with col_a:
            st.markdown("### 불안(Anxiety) 관련 제안")
            tips_anx = fetch_coping_tips("anxiety")
            for t in tips_anx:
                st.markdown(f"- {t}")

        with col_f:
            st.markdown("### 피로(Fatigue) 관련 제안")
            tips_fat = fetch_coping_tips("fatigue")
            for t in tips_fat:
                st.markdown(f"- {t}")

        with col_c:
            st.markdown("### 집중/안정(Focus) 관련 제안")
            tips_focus = fetch_coping_tips("focus")
            for t in tips_focus:
                st.markdown(f"- {t}")

        st.caption(
            "※ 해결책 텍스트는 예시이며, 실제 프로젝트에서는 신뢰할 만한 정신건강/웰빙 관련 사이트에서 BeautifulSoup을 이용해 가져오도록 수정하면 됩니다."
        )
