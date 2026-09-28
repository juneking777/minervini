"""미너비니 스크리너 앱 (Streamlit)
실행: streamlit run app.py
필요 파일: minervini_screener.py (같은 폴더)
"""
import datetime as dt
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import minervini_screener as ms

st.set_page_config(page_title="Minervini Screener", page_icon="📈", layout="wide")

# Streamlit Cloud의 Secrets → 환경변수 (이메일 발송용, 선택)
for _k in ("SMTP_USER", "SMTP_PASS", "MAIL_TO"):
    try:
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
    except Exception:
        pass


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_data(universe):
    tickers = ms.load_universe(universe)
    frames = ms.download_prices(sorted(set(tickers + ["SPY", "QQQ"])))
    return tickers, frames


def screen(frames, tickers, min_rs, min_price, min_vol):
    rows = []
    for t in tickers:
        df = frames.get(t)
        if df is None:
            continue
        m = ms.trend_template(df)
        if m:
            m["ticker"] = t
            rows.append(m)
    if not rows:
        return pd.DataFrame()
    pct = pd.Series([m["rs_raw"] for m in rows]).rank(pct=True) * 100
    out = []
    for m, rs in zip(rows, pct):
        if not m["passed"] or rs < min_rs:
            continue
        if m["price"] < min_price or m["avg_vol50"] < min_vol:
            continue
        v = ms.vcp_status(frames[m["ticker"]])
        out.append(dict(
            ticker=m["ticker"], status=v["status"], price=round(m["price"], 2),
            rs=round(rs, 1), pct_from_high=round(m["pct_from_high"], 1),
            pivot=round(v["pivot"], 2), dist_to_pivot=round(v["dist_to_pivot"], 1),
            range10=round(v["range10"], 1), vol_dry=v["vol_dry"],
            vol_ratio=round(v["vol_ratio"], 2)))
    return pd.DataFrame(out)


def chart(df, ticker, pivot):
    d = df.tail(300).copy()
    full = df["Close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="종가", line=dict(width=2)))
    for n in (50, 150, 200):
        fig.add_trace(go.Scatter(x=d.index, y=full.rolling(n).mean().loc[d.index],
                                 name=f"MA{n}", line=dict(width=1)))
    fig.add_hline(y=pivot, line_dash="dot", annotation_text="피벗")
    fig.update_layout(title=ticker, height=380, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h"))
    return fig


st.title("📈 미너비니 스크리너 (미국)")

with st.sidebar:
    st.header("설정")
    universe = st.selectbox("유니버스", ["sp1500", "sp500", "sp400", "sp600"])
    min_rs = st.slider("RS 백분위 하한", 50, 95, 70)
    min_price = st.number_input("최소 주가 ($)", value=10.0, min_value=0.0)
    min_vol = st.number_input("최소 평균거래량", value=500_000, step=100_000)
    use_fund = st.checkbox("EPS/매출 성장 필터 (느림)")
    run = st.button("스캔 실행", type="primary", use_container_width=True)

if run:
    with st.spinner("가격 데이터 다운로드 중... (처음엔 몇 분 걸릴 수 있어요)"):
        tickers, frames = load_data(universe)
    res = screen(frames, tickers, min_rs, min_price, min_vol)
    if use_fund and not res.empty:
        with st.spinner("펀더멘털 조회 중..."):
            res = ms.add_fundamentals(res)
    order = {"BREAKOUT": 0, "SETUP": 1, "WATCH": 2}
    if not res.empty:
        res = res.sort_values(by=["status", "rs"],
                              key=lambda s: s.map(order) if s.name == "status" else -s)
    st.session_state["res"] = res
    st.session_state["frames"] = frames
    st.session_state["market"] = ms.market_health(frames)
    st.session_state["ts"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

if "res" not in st.session_state:
    st.info("왼쪽 설정(모바일은 좌상단 >)에서 **스캔 실행**을 눌러주세요.")
    st.stop()

res, frames = st.session_state["res"], st.session_state["frames"]
st.caption(f"스캔 시각 {st.session_state['ts']} · 시장 상태: {st.session_state['market']}")

if res.empty:
    st.warning("조건을 통과한 종목이 없습니다. (약세장에서는 정상입니다)")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("BREAKOUT", int((res.status == "BREAKOUT").sum()))
c2.metric("SETUP", int((res.status == "SETUP").sum()))
c3.metric("전체 통과", len(res))

show = st.multiselect("상태 필터", ["BREAKOUT", "SETUP", "WATCH"],
                      default=["BREAKOUT", "SETUP", "WATCH"])
view = res[res.status.isin(show)]
st.dataframe(view, hide_index=True)

if not view.empty:
    pick = st.selectbox("차트 보기", view["ticker"].tolist())
    row = view[view.ticker == pick].iloc[0]
    st.plotly_chart(chart(frames[pick], pick, row["pivot"]))

col_a, col_b = st.columns(2)
col_a.download_button("CSV 다운로드", view.to_csv(index=False).encode("utf-8-sig"),
                      file_name=f"minervini_{dt.date.today()}.csv", mime="text/csv")
if col_b.button("이메일로 보내기"):
    nb, ns = int((view.status == "BREAKOUT").sum()), int((view.status == "SETUP").sum())
    html = (f"<p><b>시장 상태</b>: {st.session_state['market']}</p>"
            + view.head(40).to_html(index=False, border=1))
    try:
        ms.send_email(f"[Minervini] BREAKOUT {nb} / SETUP {ns} / 총 {len(view)}", html)
        st.success("발송 시도 완료 (SMTP 설정이 없으면 발송되지 않습니다)")
    except Exception as e:
        st.error(f"발송 실패: {e}")

st.caption("교육/연구용 도구입니다. 투자 판단과 책임은 본인에게 있습니다.")
