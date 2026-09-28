#!/usr/bin/env python3
"""
Minervini Trend Template 일일 스크리너 (미국 주식)

사용법:
    pip install yfinance pandas numpy requests lxml
    python minervini_screener.py                 # S&P 1500 대상
    python minervini_screener.py --universe sp500
    python minervini_screener.py --tickers-file my_list.txt
    python minervini_screener.py --fundamentals  # EPS/매출 성장 필터 추가

결과: ./results/minervini_YYYY-MM-DD.csv + 콘솔 출력
※ 교육/연구용 도구입니다. 투자 판단과 책임은 본인에게 있습니다.
"""
import argparse
import datetime as dt
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from io import StringIO

import numpy as np
import pandas as pd
import requests

WIKI = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (minervini-screener)"}


# ----------------------------- 유니버스 -----------------------------
def load_universe(name, tickers_file=None):
    if tickers_file:
        with open(tickers_file) as f:
            return sorted({l.strip().upper() for l in f if l.strip()})
    keys = ["sp500", "sp400", "sp600"] if name == "sp1500" else [name]
    out = []
    for k in keys:
        html = requests.get(WIKI[k], headers=HEADERS, timeout=30).text
        tbl = pd.read_html(StringIO(html))[0]
        col = "Symbol" if "Symbol" in tbl.columns else tbl.columns[0]
        out += tbl[col].astype(str).str.replace(".", "-", regex=False).tolist()
    return sorted(set(out))


def download_prices(tickers, period="2y", batch=150):
    import yfinance as yf

    frames = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        print(f"  다운로드 {i + 1}-{i + len(chunk)} / {len(tickers)}", file=sys.stderr)
        data = yf.download(chunk, period=period, auto_adjust=True,
                           group_by="ticker", threads=True, progress=False)
        for t in chunk:
            try:
                df = data[t] if len(chunk) > 1 else data
                df = df[["High", "Low", "Close", "Volume"]].dropna()
                if len(df):
                    frames[t] = df
            except KeyError:
                continue
    return frames


# ----------------------------- 지표 계산 -----------------------------
def rs_raw(c):
    """IBD 방식 가중 수익률: 최근 3개월 40%, 6/9/12개월 각 20%"""
    def perf(n):
        return c.iloc[-1] / c.iloc[-1 - n] - 1
    return 0.4 * perf(63) + 0.2 * perf(126) + 0.2 * perf(189) + 0.2 * perf(252)


def trend_template(df):
    """8가지 트렌드 템플릿 조건 중 RS를 제외한 7개 + 지표 반환. 데이터 부족 시 None"""
    c, v = df["Close"], df["Volume"]
    if len(c) < 260:
        return None
    ma50, ma150, ma200 = (c.rolling(n).mean() for n in (50, 150, 200))
    p = c.iloc[-1]
    hi52, lo52 = c.iloc[-252:].max(), c.iloc[-252:].min()
    passed = (
        p > ma150.iloc[-1] and p > ma200.iloc[-1]
        and ma150.iloc[-1] > ma200.iloc[-1]
        and ma200.iloc[-1] > ma200.iloc[-22]          # 200일선 최소 1개월 상승
        and ma50.iloc[-1] > ma150.iloc[-1] and ma50.iloc[-1] > ma200.iloc[-1]
        and p > ma50.iloc[-1]
        and p >= 1.30 * lo52                          # 52주 저점 +30%
        and p >= 0.75 * hi52                          # 52주 고점 -25% 이내
    )
    return dict(
        passed=bool(passed), price=p, rs_raw=rs_raw(c),
        pct_from_high=(p / hi52 - 1) * 100,
        avg_vol50=v.iloc[-50:].mean(),
    )


def vcp_status(df):
    """VCP 단순 휴리스틱: 변동성 수축 + 거래량 감소 + 피벗 근접/돌파"""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    avg50 = v.iloc[-51:-1].mean()
    range10 = (h.iloc[-10:].max() - l.iloc[-10:].min()) / h.iloc[-10:].max()
    range30 = (h.iloc[-30:].max() - l.iloc[-30:].min()) / h.iloc[-30:].max()
    contracting = range10 < 0.7 * range30 and range10 <= 0.10
    vol_dry = v.iloc[-5:].mean() < 0.8 * avg50
    pivot = h.iloc[-16:-1].max()                      # 직전 15일 고점 = 피벗
    p = c.iloc[-1]
    vol_ratio = v.iloc[-1] / avg50 if avg50 else np.nan
    if p > pivot and vol_ratio >= 1.4:
        status = "BREAKOUT"
    elif contracting and p >= 0.95 * pivot:
        status = "SETUP"
    else:
        status = "WATCH"
    return dict(status=status, pivot=pivot, dist_to_pivot=(p / pivot - 1) * 100,
                range10=range10 * 100, vol_dry=vol_dry, vol_ratio=vol_ratio)


def market_health(frames):
    """SPY/QQQ가 50일선·200일선 위인지"""
    msg = []
    for t in ("SPY", "QQQ"):
        df = frames.get(t)
        if df is None or len(df) < 200:
            continue
        c = df["Close"]
        ok = c.iloc[-1] > c.rolling(50).mean().iloc[-1] and \
            c.iloc[-1] > c.rolling(200).mean().iloc[-1]
        msg.append(f"{t}: {'강세(50/200일선 위)' if ok else '약세/경계'}")
    return " | ".join(msg)


def add_fundamentals(res, min_eps=0.25, min_rev=0.15):
    import yfinance as yf

    eps, rev = [], []
    for t in res["ticker"]:
        try:
            info = yf.Ticker(t).info
            eps.append(info.get("earningsQuarterlyGrowth"))
            rev.append(info.get("revenueGrowth"))
        except Exception:
            eps.append(None)
            rev.append(None)
    res["eps_qoq"] = pd.to_numeric(pd.Series(eps, index=res.index), errors="coerce") * 100
    res["rev_growth"] = pd.to_numeric(pd.Series(rev, index=res.index), errors="coerce") * 100
    # 값이 있는데 기준 미달인 종목만 제외 (데이터 없음은 유지)
    bad = (res["eps_qoq"] < min_eps * 100) | (res["rev_growth"] < min_rev * 100)
    return res[~bad.fillna(False)]


# ----------------------------- 이메일 -----------------------------
def send_email(subject, html, attachment=None):
    """환경변수: SMTP_USER, SMTP_PASS (필수), MAIL_TO (기본: SMTP_USER),
    SMTP_HOST (기본 smtp.gmail.com), SMTP_PORT (기본 465)
    Gmail은 2단계 인증 후 '앱 비밀번호'를 발급받아 SMTP_PASS에 넣으세요."""
    user, pw = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    if not user or not pw:
        print("이메일 미발송: SMTP_USER / SMTP_PASS 환경변수를 설정하세요.", file=sys.stderr)
        return
    msg = EmailMessage()
    msg["Subject"], msg["From"] = subject, user
    msg["To"] = os.environ.get("MAIL_TO", user)
    msg.set_content("HTML 메일을 지원하는 클라이언트에서 확인하세요.")
    msg.add_alternative(html, subtype="html")
    if attachment:
        with open(attachment, "rb") as f:
            msg.add_attachment(f.read(), maintype="text", subtype="csv",
                               filename=os.path.basename(attachment))
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as s:
        s.login(user, pw)
        s.send_message(msg)
    print(f"이메일 발송 완료 → {msg['To']}", file=sys.stderr)


# ----------------------------- 메인 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp1500",
                    choices=["sp500", "sp400", "sp600", "sp1500"])
    ap.add_argument("--tickers-file")
    ap.add_argument("--min-price", type=float, default=10)
    ap.add_argument("--min-volume", type=float, default=500_000)
    ap.add_argument("--min-rs", type=float, default=70, help="RS 백분위 하한")
    ap.add_argument("--fundamentals", action="store_true")
    ap.add_argument("--email", action="store_true", help="결과를 이메일로 발송")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    print("유니버스 로딩...", file=sys.stderr)
    tickers = load_universe(a.universe, a.tickers_file)
    frames = download_prices(sorted(set(tickers + ["SPY", "QQQ"])))

    rows = []
    for t in tickers:
        df = frames.get(t)
        if df is None:
            continue
        m = trend_template(df)
        if m:
            m["ticker"] = t
            rows.append((m, df))
    if not rows:
        print("데이터를 가져오지 못했습니다.")
        return

    # RS 백분위: 전체 유니버스 기준
    raw = pd.Series([m["rs_raw"] for m, _ in rows])
    pct = raw.rank(pct=True) * 100

    out = []
    for (m, df), rs in zip(rows, pct):
        if not m["passed"] or rs < a.min_rs:
            continue
        if m["price"] < a.min_price or m["avg_vol50"] < a.min_volume:
            continue
        v = vcp_status(df)
        out.append(dict(ticker=m["ticker"], price=round(m["price"], 2),
                        rs=round(rs, 1), pct_from_high=round(m["pct_from_high"], 1),
                        status=v["status"], pivot=round(v["pivot"], 2),
                        dist_to_pivot=round(v["dist_to_pivot"], 1),
                        range10=round(v["range10"], 1), vol_dry=v["vol_dry"],
                        vol_ratio=round(v["vol_ratio"], 2)))
    res = pd.DataFrame(out)
    market = market_health(frames)
    print(f"\n[{dt.date.today()}] 시장 상태: {market}")
    if res.empty:
        print("조건을 통과한 종목이 없습니다. (약세장에서는 정상입니다)")
        if a.email:
            send_email(f"[Minervini] {dt.date.today()} 통과 종목 없음",
                       f"<p>시장 상태: {market}</p><p>조건을 통과한 종목이 없습니다.</p>")
        return

    if a.fundamentals:
        print("펀더멘털 조회 중...", file=sys.stderr)
        res = add_fundamentals(res)

    order = {"BREAKOUT": 0, "SETUP": 1, "WATCH": 2}
    res = res.sort_values(by=["status", "rs"], key=lambda s: s.map(order) if s.name == "status" else -s)
    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"minervini_{dt.date.today()}.csv")
    res.to_csv(path, index=False)

    print(f"통과 {len(res)}종목 (BREAKOUT {sum(res.status == 'BREAKOUT')}, "
          f"SETUP {sum(res.status == 'SETUP')})\n")
    print(res.head(40).to_string(index=False))
    print(f"\n저장: {path}")

    if a.email:
        nb, ns = sum(res.status == "BREAKOUT"), sum(res.status == "SETUP")
        html = (f"<p><b>시장 상태</b>: {market}</p>"
                f"<p>통과 {len(res)}종목 (BREAKOUT {nb}, SETUP {ns}) · 상위 40개 표시, 전체는 첨부 CSV</p>"
                + res.head(40).to_html(index=False, border=1))
        send_email(f"[Minervini] {dt.date.today()} BREAKOUT {nb} / SETUP {ns} / 총 {len(res)}",
                   html, attachment=path)


if __name__ == "__main__":
    main()
