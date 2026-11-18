"""
S&P 500 패턴 예측 웹앱
- 최근 N일(기본 14일) 주가 모양과 비슷했던 과거 구간을 찾아
  그 뒤에 주가가 올랐는지 / 내렸는지 세어서 확률로 보여줍니다.
- 실행:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view

st.set_page_config(page_title="S&P 500 패턴 예측", page_icon="📈", layout="centered")

# ─────────────────────────────────────────────
# 1. 상단: 제목 + 다크모드 버튼
# ─────────────────────────────────────────────
title_col, toggle_col = st.columns([4, 1])
with toggle_col:
    dark = st.toggle("🌙 다크", key="dark")

# 색상표 (한국식: 상승=빨강, 하락=파랑)
if dark:
    C = dict(bg="#151a21", fg="#e8eaed", muted="#8a929c", grid="#252c35",
             line="#e8eaed", up="#ff5c5c", down="#5c9dff", band="rgba(255,255,255,0.07)")
else:
    C = dict(bg="#ffffff", fg="#1b1f24", muted="#7a828c", grid="#eef0f3",
             line="#1b1f24", up="#e0262b", down="#1f5fd6", band="rgba(27,31,36,0.06)")

# 화면 전체 색을 바꾸는 CSS
st.markdown(f"""
<style>
.stApp {{ background:{C['bg']}; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stApp [data-testid="stMarkdownContainer"], .stApp summary {{ color:{C['fg']} !important; }}
[data-testid="stExpander"] details {{ border-color:{C['grid']}; background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
.price {{ font-size:2.4rem; font-weight:700; letter-spacing:-0.02em; line-height:1.1; }}
.small {{ color:{C['muted']}; font-size:0.9rem; }}
.result {{ text-align:center; padding:1.2rem 0 0.4rem; }}
.result .big {{ font-size:3.6rem; font-weight:800; letter-spacing:-0.03em; line-height:1; }}
.bar {{ display:flex; height:8px; border-radius:4px; overflow:hidden; margin:1rem auto 0.6rem; max-width:420px; }}
.counts {{ display:flex; justify-content:space-between; max-width:420px; margin:0 auto; font-weight:600; }}
</style>
""", unsafe_allow_html=True)

with title_col:
    st.markdown("### S&P 500 패턴 예측")


# ─────────────────────────────────────────────
# 2. 설정 (접혀 있음)
# ─────────────────────────────────────────────
with st.expander("⚙️ 설정"):
    window = st.slider("비교할 패턴 길이 (거래일)", 5, 30, 14)
    horizon = st.slider("며칠 뒤를 예측할까요? (거래일)", 1, 20, 5)
    threshold = st.slider("유사도 기준 (높을수록 더 닮은 패턴만 사용)", 0.60, 0.99, 0.80, 0.01)
    start_year = st.selectbox("비교에 쓸 과거 데이터 시작 연도", [1950, 1970, 1990, 2000, 2010], index=0)


# ─────────────────────────────────────────────
# 3. 데이터 불러오기 (1시간 동안 저장해 두고 재사용)
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices() -> pd.Series:
    # 1순위: 야후 파이낸스
    try:
        df = yf.Ticker("^GSPC").history(period="max", auto_adjust=False)
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        if len(s) > 1000:
            return s
    except Exception:
        pass
    # 2순위: Stooq (야후가 막혔을 때 대비)
    df = pd.read_csv("https://stooq.com/q/d/l/?s=%5Espx&i=d",
                     parse_dates=["Date"], index_col="Date")
    return df["Close"].dropna()


with st.spinner("S&P 500 데이터를 불러오는 중..."):
    try:
        prices = load_prices()
    except Exception:
        st.error("주가 데이터를 불러오지 못했습니다. 잠시 후 페이지를 새로고침해 주세요.")
        st.stop()

prices = prices[prices.index.year >= start_year]


# ─────────────────────────────────────────────
# 4. 핵심 알고리즘: 비슷한 과거 패턴 찾기
# ─────────────────────────────────────────────
def find_matches(prices: pd.Series, window: int, horizon: int, threshold: float):
    p = prices.values.astype(float)
    n = len(p)

    # 최근 window일 패턴을 '모양'만 남기도록 표준화 (평균 0, 표준편차 1)
    cur = p[-window:]
    cur_z = (cur - cur.mean()) / cur.std()

    # 비교 대상: 결과(horizon일 뒤)를 이미 알고 있고, 최근 패턴과 겹치지 않는 과거 구간
    last_start = min(n - window - horizon, n - 2 * window)
    W = sliding_window_view(p, window)[: last_start + 1]
    mu = W.mean(axis=1, keepdims=True)
    sd = W.std(axis=1, keepdims=True)
    sd[sd == 0] = np.nan
    Z = (W - mu) / sd

    # 유사도 = 상관계수 (1에 가까울수록 모양이 똑같음)
    corr = (Z @ cur_z) / window

    # 각 과거 구간의 'horizon일 뒤 수익률'
    ends = np.arange(len(W)) + window - 1
    future = p[ends + horizon] / p[ends] - 1

    # 기준을 넘는 구간 중, 서로 겹치는 건 하나만 남김 (같은 사건 중복 계산 방지)
    candidates = np.where(corr >= threshold)[0]
    order = candidates[np.argsort(-corr[candidates])]
    taken = np.zeros(n, dtype=bool)
    chosen = []
    for s in order:
        if not taken[s:s + window].any():
            chosen.append(s)
            taken[s:s + window] = True

    chosen = np.array(chosen, dtype=int)
    result = pd.DataFrame({
        "start_idx": chosen,
        "패턴 시작일": prices.index[chosen].date if len(chosen) else [],
        "패턴 종료일": prices.index[ends[chosen]].date if len(chosen) else [],
        "유사도": corr[chosen],
        "이후 수익률(%)": future[chosen] * 100,
    })
    base_up_rate = (future > 0).mean() * 100  # 평소(모든 구간)의 상승 확률
    return result, base_up_rate


matches, base_up_rate = find_matches(prices, window, horizon, threshold)


# ─────────────────────────────────────────────
# 5. 그래프 공통 스타일
# ─────────────────────────────────────────────
def style(fig, height, hover="x unified"):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(color=C["fg"]), showlegend=False, hovermode=hover,
    )
    fig.update_xaxes(showgrid=False, linecolor=C["grid"], color=C["muted"])
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False, color=C["muted"])
    return fig


# ─────────────────────────────────────────────
# 6. 현재 주가 + 최근 6개월 그래프
# ─────────────────────────────────────────────
last, prev = prices.iloc[-1], prices.iloc[-2]
chg = (last / prev - 1) * 100
chg_color = C["up"] if chg >= 0 else C["down"]
st.markdown(
    f"<div class='price'>{last:,.2f}</div>"
    f"<div class='small'>{prices.index[-1]:%Y-%m-%d} 종가 &nbsp;"
    f"<span style='color:{chg_color};font-weight:600'>{chg:+.2f}%</span></div>",
    unsafe_allow_html=True,
)

recent = prices.iloc[-130:]
fig1 = go.Figure(go.Scatter(
    x=recent.index, y=recent.values, mode="lines",
    line=dict(color=C["line"], width=2),
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
))
fig1.add_vrect(
    x0=prices.index[-window], x1=prices.index[-1], fillcolor=C["band"], line_width=0,
    annotation_text=f"최근 {window}일", annotation_position="top left",
    annotation_font_color=C["muted"],
)
st.plotly_chart(style(fig1, 340), width="stretch", theme=None,
                config={"displayModeBar": False})


# ─────────────────────────────────────────────
# 7. 예측 결과
# ─────────────────────────────────────────────
if matches.empty:
    st.info("기준을 넘는 비슷한 패턴이 없습니다. ⚙️ 설정에서 유사도 기준을 낮춰 보세요.")
    st.stop()

up = int((matches["이후 수익률(%)"] > 0).sum())
down = int((matches["이후 수익률(%)"] < 0).sum())
total = max(up + down, 1)
up_prob = up / total * 100
main_color = C["up"] if up_prob >= 50 else C["down"]
headline = f"상승 {up_prob:.0f}%" if up_prob >= 50 else f"하락 {100 - up_prob:.0f}%"
avg_ret = matches["이후 수익률(%)"].mean()
if len(matches) < 20:
    st.warning(f"비슷한 패턴이 {len(matches)}개뿐이라 결과를 믿기 어렵습니다. ⚙️ 설정에서 유사도 기준을 조금 낮춰 보세요.")

st.markdown(f"""
<div class="result">
  <div class="big" style="color:{main_color}">{headline}</div>
  <div class="small" style="margin-top:0.5rem">비슷했던 과거 {len(matches)}번 중, {horizon}거래일 뒤 결과</div>
  <div class="bar">
    <div style="width:{up_prob}%;background:{C['up']}"></div>
    <div style="width:{100 - up_prob}%;background:{C['down']}"></div>
  </div>
  <div class="counts">
    <span style="color:{C['up']}">▲ 상승 {up}회</span>
    <span style="color:{C['down']}">하락 {down}회 ▼</span>
  </div>
  <div class="small" style="margin-top:0.8rem">
    평균 수익률 {avg_ret:+.2f}% &nbsp;|&nbsp; 참고: 아무 때나 골랐을 때 상승 확률 {base_up_rate:.0f}%
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 8. 패턴 비교 그래프 (현재 패턴 vs 닮은 과거 패턴들)
# ─────────────────────────────────────────────
p = prices.values.astype(float)
x = np.arange(-(window - 1), horizon + 1)  # 오늘 = 0
fig2 = go.Figure()

top = matches.nlargest(15, "유사도")
for _, row in top.iterrows():
    s = int(row["start_idx"])
    seg = p[s: s + window + horizon]
    rel = (seg / seg[window - 1] - 1) * 100  # 패턴 마지막 날 대비 %
    color = C["up"] if rel[-1] > 0 else C["down"]
    fig2.add_trace(go.Scatter(x=x[:window], y=rel[:window], mode="lines",
                              line=dict(color=C["muted"], width=1), opacity=0.35,
                              hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=x[window - 1:], y=rel[window - 1:], mode="lines",
                              line=dict(color=color, width=1.5), opacity=0.7,
                              hovertemplate=f"{row['패턴 종료일']}<br>%{{y:+.2f}}%<extra></extra>"))

cur = p[-window:]
fig2.add_trace(go.Scatter(x=x[:window], y=(cur / cur[-1] - 1) * 100, mode="lines",
                          line=dict(color=C["line"], width=3.5),
                          hovertemplate="지금<br>%{y:+.2f}%<extra></extra>"))
fig2.add_vline(x=0, line_dash="dot", line_color=C["muted"])
fig2.update_xaxes(title_text="거래일 (0 = 오늘)")
fig2.update_yaxes(title_text="오늘 대비", ticksuffix="%")

st.markdown("<div class='small' style='margin-top:1.5rem'>굵은 선: 지금의 패턴 · 가는 선: 가장 닮았던 과거 패턴과 그 이후 움직임</div>",
            unsafe_allow_html=True)
st.plotly_chart(style(fig2, 380, hover="closest"), width="stretch", theme=None,
                config={"displayModeBar": False})


# ─────────────────────────────────────────────
# 9. 상세 목록 + 안내문
# ─────────────────────────────────────────────
with st.expander(f"비슷했던 과거 패턴 {len(matches)}개 전체 보기"):
    table = matches.drop(columns="start_idx").sort_values("유사도", ascending=False)
    st.dataframe(table.style.format({"유사도": "{:.3f}", "이후 수익률(%)": "{:+.2f}"}),
                 hide_index=True, width="stretch")

st.caption("연습용 프로젝트입니다. 과거 패턴이 미래를 보장하지 않으며, 투자 조언이 아닙니다.")
