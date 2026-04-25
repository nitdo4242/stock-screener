"""
일 1회 실행되어 data.json을 갱신한다.
stock.html의 Database.stocks에서 티커 목록을 읽고, yfinance로 최신 지표를 받아와 data.json에 저장.

출력 스키마:
{
  "generatedAt": "2026-04-25T22:00:00+00:00",
  "source": "yfinance",
  "stocks": [
    {"ticker": "AAPL", "currentPrice": 170.0, "rsi14": 55, "pct52wHigh": 0.95, "mdd1y": 18.2, "avgVolume": 12345.6}
  ],
  "failed": ["TICKER1", "TICKER2(ExceptionName)"]
}
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "stock.html"
OUT = ROOT / "data.json"


def extract_tickers(html: str):
    """stock.html의 ticker:"XXX" 패턴을 모두 뽑아냄. 순서 보존, 중복 제거."""
    seen, out = set(), []
    for m in re.finditer(r'ticker:"([A-Z0-9.\-]+)"', html):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def compute_rsi(close: pd.Series, period: int = 14):
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


def compute_mdd_1y(close: pd.Series):
    """최대 낙폭(%)을 양수로 반환. 0~100 범위."""
    if close.empty:
        return None
    roll_max = close.cummax()
    dd = (close / roll_max - 1) * 100
    return round(float(-dd.min()), 1)


def fetch_one(ticker: str):
    """단일 티커의 1년치 일봉을 받아 지표 계산.
    yfinance는 점(.)을 하이픈(-)으로 변환 필요 (예: BRK.B → BRK-B)."""
    yf_sym = ticker.replace(".", "-")
    hist = yf.Ticker(yf_sym).history(period="1y", auto_adjust=False)
    if hist.empty:
        return None
    close = hist["Close"].dropna()
    if close.empty:
        return None

    cur = float(close.iloc[-1])
    high52 = float(close.max())
    pct52 = round(cur / high52, 4) if high52 > 0 else None

    # 일평균 거래대금 ($M, 최근 20영업일)
    volume_dollar = None
    if "Volume" in hist.columns:
        avg_vol = hist["Volume"].tail(20).mean()
        avg_px = close.tail(20).mean()
        if pd.notna(avg_vol) and pd.notna(avg_px):
            volume_dollar = round(float(avg_vol) * float(avg_px) / 1_000_000, 1)

    return {
        "ticker": ticker,
        "currentPrice": round(cur, 2),
        "rsi14": compute_rsi(close),
        "pct52wHigh": pct52,
        "mdd1y": compute_mdd_1y(close),
        "avgVolume": volume_dollar,
    }


def main():
    html = HTML.read_text(encoding="utf-8")
    tickers = extract_tickers(html)
    print(f"[info] {len(tickers)} tickers detected", file=sys.stderr)

    stocks, failed = [], []
    for t in tickers:
        try:
            row = fetch_one(t)
            if row:
                stocks.append(row)
            else:
                failed.append(t)
        except Exception as e:
            failed.append(f"{t}({e.__class__.__name__})")

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
