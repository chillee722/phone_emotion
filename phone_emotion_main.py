import streamlit as st
import time
import random
import statistics
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------
# 초기화
# ----------------------------------------------------------
def init():
    defaults = {
        "taps": [],
        "tap_start": None,
        "go_nogo_logs": [],
        "scroll_logs": [],
        "scroll_last_y": None,
        "scroll_last_time": None,
        "fatigue_window": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ----------------------------------------------------------
# 1) Finger Tapping Test
# ----------------------------------------------------------
def finger_tap_test():
    st.title("Finger Tapping Test (20초)")
    st.write("20초 동안 가능한 빠르게 탭하세요.")

    if st.button("시작하기", type="primary"):
        st.session_state["taps"] = []
        st.session_state["tap_start"] = time.time()
        st.experimental_rerun()

    if st.session_state["tap_start"]:
        elapsed = time.time() - st.session_state["tap_start"]

        if elapsed <= 20:
            if st.button("TAP!", key=str(elapsed)):
                st.session_state["taps"].append(time.time())
            st.write(f"⏱ 경과 시간: {elapsed:.1f}/20초")
            st.write(f"현재 탭 수: {len(st.session_state['taps'])}")
        else:
            st.success("테스트 종료! 결과를 보려면 '결과 보기' 탭을 확인하세요.")

# ----------------------------------------------------------
# 2) Go / No-Go Test
# ----------------------------------------------------------
def go_nogo_test():
    st.title("Go / No-Go Test (반응억제 기능 측정)")
    st.write("Go 신호(초록)는 클릭, No-Go(빨강)는 클릭하면 안 됩니다.")

    if st.button("테스트 시작"):
        st.session_state["go_nogo_logs"] = []
        run_go_nogo()

def run_go_nogo():
    st.write("테스트 중입니다…")
    placeholder = st.empty()

    for trial in range(8):
        time.sleep(random.uniform(1.0, 2.0))
        signal_type = random.choice(["go", "nogo"])
        signal_time = time.time()

        if signal_type == "go":
            button = placeholder.button("🟢 GO! 눌러!", key=f"go{trial}")
        else:
            button = placeholder.button("🔴 NO-GO! 누르지 말 것!", key=f"nogo{trial}")

        clicked = False
        click_time = None

        start = time.time()
        while time.time() - start < 1.2:
            if button:
                clicked = True
                click_time = time.time()
                break

        st.session_state["go_nogo_logs"].append({
            "signal": signal_type,
            "time": signal_time,
            "clicked": clicked,
            "click_time": click_time
        })

    placeholder.empty()
    st.success("Go/No-Go Test 종료!")

# ----------------------------------------------------------
# 3) Scroll Variability Test
# ----------------------------------------------------------
def scroll_test():
    st.title("Scroll Variability Test")
    st.write("쭉 스크롤하면서 마음 가는 대로 움직여 보세요.")

    scroll_area = st.empty()
    big_text = "\n".join([f"Line {i}" for i in range(1, 300)])
    scroll_area.text(big_text)

    y = st.slider("스크롤 위치 시뮬레이션 (본인 마음대로 움직여 보세요)", 0, 1000, 0)

    now = time.time()
    last_y = st.session_state["scroll_last_y"]
    last_t = st.session_state["scroll_last_time"]

    if last_y is not None and last_t is not None:
        dy = y - last_y
        dt = now - last_t
        if dt > 0:
            st.session_state["scroll_logs"].append({
                "time": now,
                "dy": dy,
                "dt": dt,
                "velocity": dy / dt
            })

    st.session_state["scroll_last_y"] = y
    st.session_state["scroll_last_time"] = now

    st.write("👇 스크롤 데이터를 충분히 생성한 뒤 결과 페이지로 이동하세요.")

# ----------------------------------------------------------
# 4) 분석 함수들
# ----------------------------------------------------------
def compute_finger_metrics():
    taps = st.session_state["taps"]
    if len(taps) < 3:
        return None

    intervals = [t2 - t1 for t1, t2 in zip(taps[:-1], taps[1:])]
    avg = statistics.mean(intervals)
    tap_freq = 1 / avg if avg > 0 else 0
    variability = statistics.pstdev(intervals) if len(intervals) > 1 else 0

    # Fatigue slope 분석
    n = len(intervals)
    x = np.arange(n)
    slope = np.polyfit(x, intervals, 1)[0]

    return {
        "count": len(taps),
        "avg_interval": avg,
        "tap_freq": tap_freq,
        "variability": variability,
        "fatigue_slope": slope,
    }

def compute_go_nogo_metrics():
    logs = st.session_state["go_nogo_logs"]
    if not logs:
        return None

    rt = []
    commission = 0
    omission = 0

    for l in logs:
        if l["signal"] == "go":
            if l["clicked"] and l["click_time"]:
                rt.append(l["click_time"] - l["time"])
            else:
                omission += 1
        else:
            if l["clicked"]:
                commission += 1

    avg_rt = statistics.mean(rt) if rt else None
    rt_var = statistics.pstdev(rt) if len(rt) > 1 else 0

    return {
        "avg_rt": avg_rt,
        "rt_var": rt_var,
        "commission": commission,
        "omission": omission,
    }

def compute_scroll_metrics():
    logs = st.session_state["scroll_logs"]
    if not logs:
        return None

    velocities = [abs(l["velocity"]) for l in logs]
    burst = sum(1 for v in velocities if v > np.percentile(velocities, 75))
    variability = statistics.pstdev(velocities)
    direction_switch = sum(
        1 for i in range(1, len(logs))
        if logs[i]["dy"] * logs[i - 1]["dy"] < 0
    )

    return {
        "burst": burst,
        "variability": variability,
        "direction_switch": direction_switch
    }

# ----------------------------------------------------------
# 5) 감정 지표 계산
# ----------------------------------------------------------
def normalize(value):
    return value / (1 + abs(value))

def compute_emotion():
    tap = compute_finger_metrics()
    go = compute_go_nogo_metrics()
    sc = compute_scroll_metrics()

    if not tap or not go or not sc:
        return "데이터 부족", {}

    α = normalize((1/(go["avg_rt"] or 1)) + tap["tap_freq"] + sc["burst"])
    β = normalize((go["avg_rt"] or 1) + tap["fatigue_slope"])
    γ = normalize(sc["direction_switch"] + go["commission"])

    scores = {"anxiety": α, "fatigue": β, "distraction": γ}

    emo = max(scores, key=scores.get)
    return emo, scores

# ----------------------------------------------------------
# 6) 전문가 처방 크롤링
# ----------------------------------------------------------
def crawl_treatment(emotion):
    urls = {
        "anxiety": "https://www.verywellmind.com/anxiety-4157184",
        "fatigue": "https://www.verywellmind.com/fatigue-symptoms-causes-treatment-4587047",
        "distraction": "https://www.mindful.org/meditation-for-beginners/"
    }

    url = urls.get(emotion)
    if not url:
        return ["추천 정보를 찾을 수 없습니다."]

    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        items = [li.get_text(strip=True) for li in soup.find_all("li")]
        return items[:10]
    except:
        return ["인터넷 연결 문제로 전문 처방을 불러오지 못했습니다."]

# ----------------------------------------------------------
# 결과 페이지
# ----------------------------------------------------------
def result_page():
    st.title("전문 행동 분석 결과")

    emotion, scores = compute_emotion()

    if emotion == "데이터 부족":
        st.warning("데이터가 충분하지 않습니다. 모든 테스트를 완료하세요.")
        return

    st.header(f"📌 감정 추정 결과: **{emotion.upper()}**")

    st.subheader("📊 Emotion Index")
    st.write(scores)

    # 전문 처방
    st.subheader("🧠 전문가 기반 처방")
    tips = crawl_treatment(emotion)
    for t in tips:
        st.write("- ", t)

# ----------------------------------------------------------
# 메인 메뉴
# ----------------------------------------------------------
def main():
    init()
    page = st.sidebar.radio(
        "메뉴",
        ["Finger Tapping", "Go/No-Go", "Scroll Test", "결과 보기"]
    )

    if page == "Finger Tapping":
        finger_tap_test()
    elif page == "Go/No-Go":
        go_nogo_test()
    elif page == "Scroll Test":
        scroll_test()
    else:
        result_page()

if __name__ == "__main__":
    main()
