# app.py
import time
import random
import statistics
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------
# 0. 세션 상태 초기화
# ---------------------------
def init_session_state():
    defaults = {
        "tap_times": [],              # 탭이 눌린 시각들
        "tap_start_time": None,
        "reaction_phase": "idle",     # 'idle' | 'waiting' | 'ready'
        "reaction_target_time": None,
        "reaction_prompt_time": None,
        "reaction_times": [],         # 반응 속도 기록
        "last_choice": None,
        "switch_count": 0,            # 화면 전환(선택 변화) 횟수
        "experiment_logs": [],        # 개별 이벤트 로그 (dict 리스트)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------
# 1. 유틸 함수들
# ---------------------------
def log_event(event_type: str, extra: Dict = None):
    """행동 로그 기록용 유틸리티."""
    if extra is None:
        extra = {}
    st.session_state["experiment_logs"].append(
        {
            "timestamp": time.time(),
            "event_type": event_type,
            **extra,
        }
    )


def compute_tap_metrics(tap_times: List[float]):
    if len(tap_times) < 2:
        return None

    intervals = [
        t2 - t1 for t1, t2 in zip(tap_times[:-1], tap_times[1:])
    ]
    avg_interval = statistics.mean(intervals)
    std_interval = statistics.pstdev(intervals) if len(intervals) > 1 else 0.0
    tap_rate = 1.0 / avg_interval if avg_interval > 0 else 0.0

    return {
        "tap_count": len(tap_times),
        "avg_interval": avg_interval,
        "std_interval": std_interval,
        "tap_rate": tap_rate,
    }


def compute_reaction_metrics(reaction_times: List[float]):
    if not reaction_times:
        return None

    avg_rt = statistics.mean(reaction_times)
    std_rt = statistics.pstdev(reaction_times) if len(reaction_times) > 1 else 0.0

    return {
        "trial_count": len(reaction_times),
        "avg_reaction_time": avg_rt,
        "std_reaction_time": std_rt,
    }


def infer_emotion(tap_metrics, reaction_metrics, switch_count: int):
    """
    아주 단순한 규칙 기반 감정 추론.
    - tap_rate 높고 switch 많고 반응 빠름 → 긴장/불안
    - tap_rate 낮고 반응 느리고 switch 적음 → 피로/무기력
    - switch 많지만 tap_rate·reaction 중간 → 지루함
    - 나머지 → 비교적 안정
    """
    if tap_metrics is None or reaction_metrics is None:
        return {
            "label": "데이터 부족",
            "description": "조금 더 오래 실험에 참여하면 감정 상태를 추정할 수 있어요.",
        }

    tap_rate = tap_metrics["tap_rate"]
    avg_rt = reaction_metrics["avg_reaction_time"]

    # 기본 임계값 (실험하면서 조절 가능)
    FAST_TAP = 3.0       # 초당 3회 이상
    SLOW_TAP = 1.0       # 초당 1회 이하
    FAST_RT = 0.4        # 0.4초 이하면 빠른 반응
    SLOW_RT = 0.9        # 0.9초 이상이면 느린 반응
    MANY_SWITCH = 8

    if tap_rate >= FAST_TAP and avg_rt <= FAST_RT and switch_count >= MANY_SWITCH:
        return {
            "label": "긴장/불안 상태 가능성",
            "description": "탭 속도와 화면 전환이 매우 빠르고, 반응 시간이 전반적으로 짧게 나타납니다. "
                           "초조하거나 불안한 상태에서 보이는 패턴과 유사합니다. "
                           "짧은 호흡 조절이나 스트레칭으로 긴장을 풀어보는 것이 좋습니다.",
        }

    if tap_rate <= SLOW_TAP and avg_rt >= SLOW_RT and switch_count <= 3:
        return {
            "label": "피로/무기력 상태 가능성",
            "description": "탭 빈도가 낮고 반응 시간이 상대적으로 길게 나타납니다. "
                           "피로하거나 집중이 잘 되지 않을 때 관찰되는 패턴과 비슷합니다. "
                           "잠깐 눈을 감고 쉬거나, 기지개를 켜는 등의 휴식이 도움이 될 수 있습니다.",
        }

    if switch_count >= MANY_SWITCH and SLOW_TAP < tap_rate < FAST_TAP:
        return {
            "label": "지루함/산만 상태 가능성",
            "description": "화면 전환이 잦은 반면, 탭 속도와 반응 속도는 중간 수준입니다. "
                           "현재 상황에 집중하기 어렵거나, 다른 것을 찾고 싶은 지루한 상태일 가능성이 있습니다. "
                           "해야 할 일을 짧게 쪼개서 하나씩 처리해 보거나, 잠시 다른 활동으로 전환해 보세요.",
        }

    return {
        "label": "비교적 안정된 상태",
        "description": "탭 속도, 반응 속도, 화면 전환 패턴이 전반적으로 극단적이지 않습니다. "
                       "현재는 비교적 안정된 정서 상태로 보입니다. "
                       "이 상태를 유지하기 위해 짧은 휴식과 규칙적인 호흡을 이어 가는 것이 좋습니다.",
    }


@st.cache_data
def load_relax_tips():
    """
    BeautifulSoup를 활용한 간단한 크롤링 예시.
    인터넷/사이트 구조에 따라 실패할 수 있으므로,
    실패 시 기본 하드코딩 팁을 반환한다.
    """
    url = "https://www.psychologytoday.com/us/basics/stress/relaxation-techniques"
    tips = []

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 이 부분은 실제 사이트 구조에 맞게 조정 필요
        for li in soup.select("li"):
            text = li.get_text(strip=True)
            if 15 <= len(text) <= 120:
                tips.append(text)

    except Exception:
        # 실패 시 기본값
        tips = [
            "3분 동안 천천히 복식호흡을 하며 들숨보다 날숨을 약간 더 길게 유지해 보세요.",
            "창문을 열고 바깥 풍경을 30초 이상 바라보며 눈을 쉬게 해 주세요.",
            "지금 잡고 있는 핸드폰을 내려두고, 손가락과 손목을 가볍게 스트레칭해 보세요.",
            "오늘 있었던 '고마웠던 일' 한 가지를 떠올리며 10초 동안 그 장면을 떠올려 보세요.",
        ]

    # 중복 제거
    unique = []
    for t in tips:
        if t not in unique:
            unique.append(t)
    return unique[:15]


# ---------------------------
# 2. 페이지: 디지털 피젯 실험
# ---------------------------
def page_experiment():
    st.title("디지털 피젯 행동 실험")
    st.write(
        "이 화면에서는 특별한 목표 없이, 그냥 마음 가는 대로 버튼을 눌러 보고, "
        "메뉴를 옮겨 다니며, 반응 테스트를 해 보면서 **스마트폰을 만지작거리는 행동**을 기록합니다."
    )

    st.info("※ 이 앱은 연구/학습 목적의 데모입니다. 개별 사용자의 감정 상태를 정확히 진단하는 도구가 아닙니다.")

    col_left, col_right = st.columns([2, 1])

    # ---- 왼쪽: 인터랙션 영역 ----
    with col_left:
        st.subheader("1) 탭(연타) 테스트")

        st.caption("지금 기분 내키는 대로 버튼을 여러 번 눌러 보세요. 빠르게 눌러도, 느리게 눌러도 됩니다.")

        if st.button("여기를 탭! (Tap)", key="tap_button"):
            now = time.time()
            st.session_state["tap_times"].append(now)
            log_event("tap", {"time": now})

        st.write(f"지금까지 탭한 횟수: **{len(st.session_state['tap_times'])}회**")

        st.markdown("---")
        st.subheader("2) 반응 속도 테스트")

        st.caption(
            "아래 버튼을 누르면 **2~5초 사이 랜덤한 시간**이 지난 뒤에 신호가 나타납니다. "
            "신호가 보이면 가능한 한 빨리 버튼을 눌러 보세요."
        )

        # 상태에 따라 다른 버튼/텍스트 표시
        phase = st.session_state["reaction_phase"]

        if phase == "idle":
            if st.button("반응 테스트 시작"):
                st.session_state["reaction_phase"] = "waiting"
                st.session_state["reaction_target_time"] = time.time() + random.uniform(2, 5)
                log_event("reaction_start")
                st.experimental_rerun()

        elif phase == "waiting":
            # 아직 신호 안 뜸
            if time.time() >= st.session_state["reaction_target_time"]:
                st.session_state["reaction_phase"] = "ready"
                st.session_state["reaction_prompt_time"] = time.time()
                st.experimental_rerun()
            else:
                st.warning("잠시만 기다리세요... (신호가 곧 나타납니다)")
                if st.button("취소"):
                    st.session_state["reaction_phase"] = "idle"
                    st.experimental_rerun()

        elif phase == "ready":
            st.success("지금! 아래 버튼을 가능한 한 빨리 눌러 보세요!")

            if st.button("지금 클릭! (반응)"):
                rt = time.time() - st.session_state["reaction_prompt_time"]
                st.session_state["reaction_times"].append(rt)
                log_event("reaction_click", {"reaction_time": rt})
                st.session_state["reaction_phase"] = "idle"
                st.success(f"이번 반응 속도: **{rt:.3f}초**")
                st.balloons()

        st.markdown("---")
        st.subheader("3) 화면 전환/선택 행동")

        st.caption(
            "지금 느낌에 따라 아래 메뉴 중 아무거나 골라 보세요. "
            "마음이 바뀌면 여러 번 바꿔도 됩니다."
        )

        choice = st.radio(
            "지금 가장 끌리는 항목은 무엇인가요?",
            ["할 일 생각하기", "아무것도 하기 싫음", "딴생각 하기", "그냥 멍 때리기"],
            index=0 if st.session_state["last_choice"] is None else
            ["할 일 생각하기", "아무것도 하기 싫음", "딴생각 하기", "그냥 멍 때리기"].index(
                st.session_state["last_choice"]
            ),
        )

        if st.session_state["last_choice"] is None:
            st.session_state["last_choice"] = choice
        elif choice != st.session_state["last_choice"]:
            st.session_state["switch_count"] += 1
            st.session_state["last_choice"] = choice
            log_event("choice_switch", {"choice": choice})

        st.write(f"지금까지 선택을 바꾼 횟수: **{st.session_state['switch_count']}회**")

    # ---- 오른쪽: 실시간 지표 요약 ----
    with col_right:
        st.subheader("실시간 행동 요약")

        tap_metrics = compute_tap_metrics(st.session_state["tap_times"])
        reaction_metrics = compute_reaction_metrics(st.session_state["reaction_times"])
        switches = st.session_state["switch_count"]

        if tap_metrics:
            st.metric("탭 횟수", f"{tap_metrics['tap_count']}회")
            st.metric("평균 탭 간격", f"{tap_metrics['avg_interval']:.2f}초")
            st.metric("평균 탭 속도", f"{tap_metrics['tap_rate']:.2f}회/초")
        else:
            st.write("탭 데이터가 아직 충분하지 않습니다.")

        st.markdown("---")

        if reaction_metrics:
            st.metric("반응 테스트 횟수", f"{reaction_metrics['trial_count']}회")
            st.metric("평균 반응 속도", f"{reaction_metrics['avg_reaction_time']:.2f}초")
        else:
            st.write("반응 테스트 데이터를 조금 더 모아 보세요.")

        st.markdown("---")
        st.metric("선택 변경 횟수", f"{switches}회")

        st.info("왼쪽에서 충분히 만지작거린 뒤, 상단 메뉴의 **'결과 보기'** 탭에서 감정 추정 결과를 확인할 수 있습니다.")


# ---------------------------
# 3. 페이지: 결과 보기
# ---------------------------
def page_results():
    st.title("행동 데이터 기반 감정 추정 결과")

    tap_metrics = compute_tap_metrics(st.session_state["tap_times"])
    reaction_metrics = compute_reaction_metrics(st.session_state["reaction_times"])
    switches = st.session_state["switch_count"]

    if tap_metrics is None or reaction_metrics is None:
        st.warning("아직 데이터가 충분하지 않습니다. '디지털 피젯 실험' 페이지에서 "
                   "탭과 반응 테스트를 몇 번 더 수행한 뒤 다시 확인해 주세요.")
        return

    st.subheader("1. 정량적 지표 요약")

    col1, col2, col3 = st.columns(3)
    col1.metric("탭 횟수", f"{tap_metrics['tap_count']}회")
    col1.metric("평균 탭 속도", f"{tap_metrics['tap_rate']:.2f}회/초")
    col2.metric("반응 테스트 횟수", f"{reaction_metrics['trial_count']}회")
    col2.metric("평균 반응 속도", f"{reaction_metrics['avg_reaction_time']:.2f}초")
    col3.metric("선택 변경 횟수", f"{switches}회")
    col3.metric("탭 간격 변동성", f"{tap_metrics['std_interval']:.2f}초")

    st.markdown("---")
    st.subheader("2. 시각화")

    # 간단한 바 차트로 시각화
    fig, ax = plt.subplots()
    categories = ["탭 속도(회/초)", "평균 반응속도(초)", "선택 변경 횟수"]
    values = [tap_metrics["tap_rate"], reaction_metrics["avg_reaction_time"], switches]
    ax.bar(categories, values)
    ax.set_ylabel("값")
    ax.set_title("행동 지표 요약")
    plt.xticks(rotation=10)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("3. 규칙 기반 감정 상태 추정")

    emotion_info = infer_emotion(tap_metrics, reaction_metrics, switches)
    st.markdown(f"### 🔍 추정 결과: **{emotion_info['label']}**")
    st.write(emotion_info["description"])

    st.caption(
        "※ 이 결과는 소규모 행동 데이터에 기반한 단순 규칙 모델의 출력일 뿐, "
        "의학적·심리학적 진단을 대신하지 않습니다."
    )


# ---------------------------
# 4. 페이지: 추천 보기 (크롤링 데모)
# ---------------------------
def page_recommendations():
    st.title("마음 상태에 따른 간단한 추천")

    st.write(
        "이 페이지는 **BeautifulSoup로 크롤링한 텍스트** 또는 기본 내장된 문구를 이용해, "
        "현재 감정 상태에 따라 시도해 볼 만한 간단한 행동들을 보여줍니다."
    )

    tips = load_relax_tips()
    emotion_info = infer_emotion(
        compute_tap_metrics(st.session_state["tap_times"]),
        compute_reaction_metrics(st.session_state["reaction_times"]),
        st.session_state["switch_count"],
    )

    st.markdown(f"### 현재 추정 상태: **{emotion_info['label']}**")
    st.write(emotion_info["description"])
    st.markdown("---")

    st.subheader("지금 시도해 볼 수 있는 작은 행동들")

    for i, tip in enumerate(tips, start=1):
        st.markdown(f"- {tip}")

    st.caption(
        "※ 일부 문구는 웹에서 크롤링한 뒤 필터링했으며, 네트워크 환경이나 원본 사이트 구조에 따라 "
        "항목이 달라질 수 있습니다."
    )


# ---------------------------
# 5. 페이지: 데이터 다운로드
# ---------------------------
def page_download():
    st.title("실험 데이터 다운로드")

    logs = st.session_state["experiment_logs"]
    if not logs:
        st.warning("아직 저장된 로그가 없습니다. 먼저 '디지털 피젯 실험' 페이지에서 행동을 기록해 주세요.")
        return

    df = pd.DataFrame(logs)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="CSV 파일로 다운로드",
        data=csv,
        file_name="digital_fidget_logs.csv",
        mime="text/csv",
    )

    st.caption("이 데이터를 가지고 추가적인 통계 분석, 머신러닝 모델링 등을 진행할 수 있습니다.")


# ---------------------------
# 메인 앱 엔트리포인트
# ---------------------------
def main():
    st.set_page_config(
        page_title="Digital Fidget Emotion Analyzer",
        page_icon="📱",
        layout="wide",
    )

    init_session_state()

    menu = st.sidebar.radio(
        "메뉴",
        ["디지털 피젯 실험", "결과 보기", "추천 보기(크롤링 데모)", "데이터 다운로드"],
    )

    if menu == "디지털 피젯 실험":
        page_experiment()
    elif menu == "결과 보기":
        page_results()
    elif menu == "추천 보기(크롤링 데모)":
        page_recommendations()
    elif menu == "데이터 다운로드":
        page_download()


if __name__ == "__main__":
    main()
