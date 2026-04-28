"""
일 1회 실행되어 data.json을 갱신한다.
stock.html의 Database.stocks에서 티커 목록을 읽고, yfinance로 최신 지표를 받아와 data.json에 저장.

출력 스키마:
{
  "generatedAt": "2026-04-25T22:00:00+00:00",
  "source": "yfinance",
  "stocks": [
    {
      "ticker": "AAPL",
      "currentPrice": 170.0, "rsi14": 55, "pct52wHigh": 0.95,
      "mdd1y": 18.2, "avgVolume": 12345.6,
      "per": 28.5, "pbr": 38.2, "roe": 150.0,
      "divYield": 0.5, "targetPrice": 200
    }
  ],
  "failed": ["TICKER1", "TICKER2(ExceptionName)"]
}

주의:
- yfinance .info는 필드 누락 / None 값이 흔함. 값이 없으면 해당 키를 payload에서 생략(프론트에서는 기존 정적값 유지).
- ROE는 yfinance에서 소수(0.15 = 15%)로 옴. 100 곱해 % 단위로 저장.
- divYield도 소수(0.025 = 2.5%)로 올 수 있어 분기 처리.
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# 호출 간격 (초). Yahoo rate limit / 봇 차단 회피용.
SLEEP_BETWEEN = 0.6
# .info 재시도 횟수
INFO_RETRIES = 3

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "stock.html"
OUT = ROOT / "data.json"


def extract_tickers(html):
    """stock.html의 ticker:"XXX" 패턴을 모두 뽑아냄. 순서 보존, 중복 제거."""
    seen, out = set(), []
    for m in re.finditer(r'ticker:"([A-Z0-9.\-]+)"', html):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def compute_rsi(close, period=14):
    """Wilder 방식 단순화 (rolling mean). 데이터 부족 시 None."""
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return None if pd.isna(val) else round(float(val))


def compute_mdd_1y(close):
    """최대 낙폭(%)을 양수로 반환. 0~100 범위."""
    if close.empty:
        return None
    roll_max = close.cummax()
    dd = (close / roll_max - 1) * 100
    return round(float(-dd.min()), 1)


def _safe_num(x):
    """None / nan / 비숫자 방어. 숫자면 float, 아니면 None."""
    try:
        if x is None:
            return None
        f = float(x)
        if pd.isna(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _get_info_with_retry(tkr, ticker_name=""):
    """
    .info 호출. 빈 dict / 핵심 필드 없음 / 예외 시 백오프 재시도.
    Yahoo가 봇 의심해서 빈 응답을 줄 때 재시도하면 종종 통함.
    """
    KEY_FIELDS = ("trailingPE", "priceToBook", "returnOnEquity",
                  "dividendYield", "targetMeanPrice", "marketCap")

    last_err = None
    for attempt in range(1, INFO_RETRIES + 1):
        try:
            info = tkr.info or {}
            # 핵심 필드가 하나라도 있으면 성공으로 판정
            if info and any(k in info for k in KEY_FIELDS):
                return info
            # 빈 dict 또는 핵심 필드 전무 → 재시도
            last_err = f"empty info (keys={len(info)})"
        except Exception as e:
            last_err = f"{e.__class__.__name__}: {e}"
        if attempt < INFO_RETRIES:
            time.sleep(1.5 * attempt)  # 1.5s, 3.0s 백오프
    print(f"[warn] .info failed for {ticker_name}: {last_err}", file=sys.stderr)
    return {}


def fetch_fundamentals(tkr, ticker_name=""):
    """yfinance .info에서 펀더멘털 지표 추출. 누락 시 해당 키 생략."""
    info = _get_info_with_retry(tkr, ticker_name)
    if not info:
        return {}

    out = {}

    per = _safe_num(info.get("trailingPE"))
    if per is not None and per > 0:
        out["per"] = round(per, 2)

    pbr = _safe_num(info.get("priceToBook"))
    if pbr is not None and pbr > 0:
        out["pbr"] = round(pbr, 2)

    roe = _safe_num(info.get("returnOnEquity"))
    if roe is not None:
        out["roe"] = round(roe * 100, 2)

    div = _safe_num(info.get("dividendYield"))
    if div is not None and div >= 0:
        out["divYield"] = round(div * 100, 2) if div < 1 else round(div, 2)

    tgt = _safe_num(info.get("targetMeanPrice"))
    if tgt is not None and tgt > 0:
        out["targetPrice"] = round(tgt, 2)

    return out


def fetch_one(ticker):
    """단일 티커의 1년치 일봉 + 펀더멘털 지표를 받아옴."""
    yf_sym = ticker.replace(".", "-")
    tkr = yf.Ticker(yf_sym)
    hist = tkr.history(period="1y", auto_adjust=False)
    if hist.empty:
        return None
    close = hist["Close"].dropna()
    if close.empty:
        return None

    cur = float(close.iloc[-1])
    high52 = float(close.max())
    pct52 = round(cur / high52, 4) if high52 > 0 else None

    volume_dollar = None
    if "Volume" in hist.columns:
        avg_vol = hist["Volume"].tail(20).mean()
        avg_px = close.tail(20).mean()
        if pd.notna(avg_vol) and pd.notna(avg_px):
            volume_dollar = round(float(avg_vol) * float(avg_px) / 1_000_000, 1)

    row = {
        "ticker": ticker,
        "currentPrice": round(cur, 2),
        "rsi14": compute_rsi(close),
        "pct52wHigh": pct52,
        "mdd1y": compute_mdd_1y(close),
        "avgVolume": volume_dollar,
    }
    row.update(fetch_fundamentals(tkr, ticker))
    return row


def main():
    html = HTML.read_text(encoding="utf-8")
    tickers = extract_tickers(html)
    print(f"[info] {len(tickers)} tickers detected", file=sys.stderr)

    stocks, failed = [], []
    fund_ok = 0  # 펀더멘털(per/pbr/roe/divYield/targetPrice 중 하나 이상)을 가져온 종목 수
    for i, t in enumerate(tickers):
        try:
            row = fetch_one(t)
            if row:
                stocks.append(row)
                if any(k in row for k in ("per", "pbr", "roe", "divYield", "targetPrice")):
                    fund_ok += 1
            else:
                failed.append(t)
        except Exception as e:
            failed.append(f"{t}({e.__class__.__name__})")
        # rate limit 회피용 호출 간격
        if i < len(tickers) - 1:
            time.sleep(SLEEP_BETWEEN)
    print(f"[info] fundamentals fetched for {fund_ok}/{len(tickers)} tickers",
          file=sys.stderr)

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "yfinance",
        "stocks": stocks,
        "failed": failed,
    }
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[ok] wrote {OUT}  success={len(stocks)}  failed={len(failed)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
