
import datetime as dt
import io
import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from pykrx import stock as krx_stock
from scipy.stats import norm
from scipy.interpolate import interp1d
import plotly.graph_objects as go

load_dotenv()

st.set_page_config(page_title="워런트 블랙숄즈 가치평가", layout="wide")


# ECOS 국고채 금리 자동 조회
ECOS_STAT_CODE = "817Y002"  # 시장금리(일별)
TENOR_LABELS = {"1y": "1년", "3y": "3년", "5y": "5년", "10y": "10년", "20y": "20년", "30y": "30년"}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_ecos_bond_yields(api_key: str):
    try:
        item_list_url = (
            f"https://ecos.bok.or.kr/api/StatisticItemList/{api_key}/json/kr/1/200/{ECOS_STAT_CODE}"
        )
        resp = requests.get(item_list_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "StatisticItemList" not in data:
            err = data.get("RESULT", {}).get("MESSAGE", "알 수 없는 오류")
            return None, f"항목 목록 조회 실패: {err}"

        rows = data["StatisticItemList"]["row"]

        item_codes = {}
        for tenor_key, label in TENOR_LABELS.items():
            match = next(
                (r for r in rows if "국고채" in r["ITEM_NAME"] and label in r["ITEM_NAME"]),
                None,
            )
            if match is None:
                return None, f"'{label}' 국고채 항목을 찾지 못함"
            item_codes[tenor_key] = match["ITEM_CODE"]

        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=21)
        results = {}
        for tenor_key, item_code in item_codes.items():
            search_url = (
                f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/50/"
                f"{ECOS_STAT_CODE}/D/{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}/{item_code}"
            )
            r = requests.get(search_url, timeout=10)
            r.raise_for_status()
            d = r.json()
            if "StatisticSearch" not in d:
                err = d.get("RESULT", {}).get("MESSAGE", "알 수 없는 오류")
                return None, f"{TENOR_LABELS[tenor_key]}물 조회 실패: {err}"
            series = d["StatisticSearch"]["row"]
            latest = max(series, key=lambda x: x["TIME"])
            results[tenor_key] = float(latest["DATA_VALUE"])
            results[f"{tenor_key}_date"] = latest["TIME"]

        return results, None

    except requests.exceptions.RequestException as e:
        return None, f"네트워크 오류: {e}"
    except Exception as e:
        return None, f"조회 중 오류: {e}"


# KRX 시세로 현재가·역사적 변동성 자동계산
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_krx_price_history(ticker: str, lookback_days: int):
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=int(lookback_days * 1.7) + 15)  # 주말/공휴일 버퍼
        df = krx_stock.get_market_ohlcv_by_date(
            start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker
        )
        if df is None or df.empty or "종가" not in df.columns:
            return None, "해당 종목코드의 시세를 찾을 수 없음 (종목코드 확인)"

        df = df[df["종가"] > 0]
        if len(df) < 20:
            return None, "확보된 시세가 너무 적음 (최소 20거래일 필요)"

        closes = df["종가"].tail(lookback_days + 1)
        log_ret = np.diff(np.log(closes.values))
        realized_vol = float(np.std(log_ret, ddof=1) * np.sqrt(252))
        latest_price = float(closes.iloc[-1])
        latest_date = df.index[-1].strftime("%Y-%m-%d")

        return {
            "price": latest_price,
            "vol": realized_vol,
            "date": latest_date,
            "n_obs": len(log_ret),
        }, None

    except Exception as e:
        return None, f"KRX 조회 오류: {e}"


# DART 공시로 배당수익률(q) 자동조회
@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def fetch_dart_corp_codes(dart_key: str):
    try:
        url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={dart_key}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xml_bytes = zf.read(zf.namelist()[0])
        root = ET.fromstring(xml_bytes)
        mapping = {}
        for child in root.findall("list"):
            stock_code = (child.findtext("stock_code") or "").strip()
            corp_code = (child.findtext("corp_code") or "").strip()
            corp_name = (child.findtext("corp_name") or "").strip()
            if stock_code:
                mapping[stock_code] = {"corp_code": corp_code, "corp_name": corp_name}
        if not mapping:
            return None, "고유번호 목록이 비어있음 (인증키 확인)"
        return mapping, None
    except zipfile.BadZipFile:
        return None, "고유번호 응답 파싱 실패 (인증키가 잘못됐을 가능성)"
    except Exception as e:
        return None, f"DART 고유번호 조회 오류: {e}"


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_dart_dividend_yield(dart_key: str, ticker: str, current_price: float):
    mapping, err = fetch_dart_corp_codes(dart_key)
    if mapping is None:
        return None, err

    info = mapping.get(ticker)
    if info is None:
        return None, f"종목코드 {ticker}에 해당하는 DART 고유번호를 찾지 못함"
    corp_code = info["corp_code"]

    this_year = dt.date.today().year
    for year in [this_year - 1, this_year - 2]:
        url = (
            f"https://opendart.fss.or.kr/api/alotMatter.json?crtfc_key={dart_key}"
            f"&corp_code={corp_code}&bsns_year={year}&reprt_code=11011"
        )
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            return None, f"DART 요청 오류: {e}"

        if d.get("status") != "000":
            continue

        rows = d.get("list", [])
        dividend_row = next(
            (
                row for row in rows
                if "주당" in row.get("se", "") and "현금배당금" in row.get("se", "")
                and row.get("stock_knd") in (None, "", "보통주")
            ),
            None,
        )
        if dividend_row:
            raw = (dividend_row.get("thstrm") or "").replace(",", "").strip()
            try:
                dps = float(raw)
            except ValueError:
                continue
            q_val = (dps / current_price) if (dps > 0 and current_price > 0) else 0.0
            return {"q": q_val, "dps": dps, "year": year, "corp_name": info["corp_name"]}, None

    return None, "최근 2개년 사업보고서에서 배당 항목을 찾지 못함 (무배당 종목일 수 있음)"


# 블랙숄즈 & 그릭스
def bs_d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = bs_d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = bs_d1_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    d1, d2 = bs_d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega / 100,     # 변동성 1%p 변화당
        "Theta": theta / 365,   # 하루당
        "Rho": rho / 100,       # 금리 1%p 변화당
    }


# 국채 금리 곡선 보간
def build_curve(tenors, yields, kind="linear"):
    return interp1d(tenors, yields, kind=kind, fill_value="extrapolate")


# 국채 금리 곡선 보간
def build_curve(tenors, yields, kind="linear"):
    return interp1d(tenors, yields, kind=kind, fill_value="extrapolate")


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def build_company_options(dart_key: str):
    """selectbox에 넣을 '회사명 (종목코드)' 문자열 리스트를 이름순으로 만든다."""
    mapping, err = fetch_dart_corp_codes(dart_key)
    if mapping is None:
        return None, err
    options = sorted(f"{info['corp_name']} ({code})" for code, info in mapping.items())
    return options, None


st.sidebar.header("모델 파라미터")

dart_key = os.getenv("DART_API_KEY", "")
ticker = None

if dart_key:
    company_options, company_err = build_company_options(dart_key)
    if company_options:
        default = "삼성전자 (005930)"
        default_index = company_options.index(default) if default in company_options else 0
        picked = st.sidebar.selectbox(
            "종목 검색 (종목명 또는 코드)", company_options, index=default_index,
        )
        ticker = picked.rsplit("(", 1)[-1].rstrip(")")
    else:
        st.sidebar.error(f"종목 목록 조회 실패: {company_err}")

if ticker is None:
    ticker = st.sidebar.text_input("종목코드 (6자리)", value="005930", max_chars=6)

auto_price_vol = st.sidebar.toggle("현재가·변동성 자동으로 가져오기", value=False)
lookback_days = (
    st.sidebar.selectbox("변동성 계산 기간 (거래일)", [60, 120, 250], index=2)
    if auto_price_vol else 250
)

auto_dividend = st.sidebar.toggle("배당수익률 자동으로 가져오기", value=False)
if auto_dividend and not dart_key:
    dart_key = st.sidebar.text_input(
        "DART 인증키",
        type="password",
        help="opendart.fss.or.kr에서 무료로 발급받을 수 있음",
    )

fetched_price, fetched_vol, fetched_div = None, None, None

if auto_price_vol or auto_dividend:
    with st.spinner("KRX 시세 조회 중..."):
        krx_result, krx_err = fetch_krx_price_history(ticker, lookback_days)
    if krx_result:
        fetched_price = krx_result["price"]
        fetched_vol = krx_result["vol"]
        st.sidebar.caption(
            f"최근 종가 {fetched_price:,.0f}원 ({krx_result['date']} 기준), "
            f"최근 {krx_result['n_obs']}거래일 변동성 {fetched_vol*100:.1f}%"
        )
    else:
        st.sidebar.error(f"KRX 조회 실패: {krx_err}")

if auto_dividend:
    if not dart_key:
        st.sidebar.info("DART 인증키를 입력하면 배당수익률을 가져와 주세요.")
    else:
        basis_price = fetched_price
        if basis_price is None:
            with st.spinner("기준주가 조회 중..."):
                basis_result, _ = fetch_krx_price_history(ticker, 20)
            basis_price = basis_result["price"] if basis_result else None

        if basis_price:
            with st.spinner("DART 배당현황 조회 중..."):
                div_result, div_err = fetch_dart_dividend_yield(dart_key, ticker, basis_price)
            if div_result:
                fetched_div = div_result["q"]
                st.sidebar.caption(
                    f"{div_result['corp_name']} {div_result['year']}년 주당배당금 "
                    f"{div_result['dps']:,.0f}원, 배당수익률 {fetched_div*100:.2f}% "
                    f"(기준주가 {basis_price:,.0f}원)"
                )
            else:
                st.sidebar.error(f"DART 조회 실패: {div_err}")
        else:
            st.sidebar.error("기준주가를 확보하지 못함")


st.sidebar.header("옵션 조건")


def sync_fetched_value(widget_key: str, tracker_key: str, fetched_value):
    if fetched_value is None:
        return
    if st.session_state.get(tracker_key) != fetched_value:
        st.session_state[widget_key] = fetched_value
        st.session_state[tracker_key] = fetched_value


sync_fetched_value("S_input", "_fetched_price_seen", fetched_price)
st.session_state.setdefault("S_input", 50000.0)
S = st.sidebar.number_input("기초주식 현재가 (S)", min_value=0.01, step=100.0, key="S_input")

K = st.sidebar.number_input(
    "행사가 (K)", min_value=0.01,
    value=float(fetched_price) if fetched_price else 55000.0, step=100.0,
)
if fetched_price:
    st.sidebar.caption("행사가 입력 필요")
T = st.sidebar.number_input("잔존만기 (T, 년)", min_value=0.01, value=4.0, step=0.1)

sync_fetched_value("sigma_pct_input", "_fetched_vol_seen", float(fetched_vol * 100) if fetched_vol else None)
st.session_state.setdefault("sigma_pct_input", 45.0)
sigma_pct = st.sidebar.slider(
    "변동성 σ (연율, %)", min_value=1.0, max_value=150.0, step=0.5, key="sigma_pct_input",
)

sync_fetched_value("q_pct_input", "_fetched_div_seen", float(fetched_div * 100) if fetched_div else None)
st.session_state.setdefault("q_pct_input", 0.0)
q_pct = st.sidebar.number_input("배당수익률 q (%)", min_value=0.0, step=0.1, key="q_pct_input")
option_type = st.sidebar.radio("옵션 종류", ["콜 (매수권/워런트)", "풋"], index=0)

sigma = sigma_pct / 100
q = q_pct / 100
opt_key = "call" if option_type.startswith("콜") else "put"

# 사이드바: 국채 금리 곡선 (무위험이자율)
# 폴백값 (자동조회 실패 시 사용). 1/3/10/30은 2026-09-02 ECOS 실측값, 5/20년은 인접 만기 선형보간.
FALLBACK_YIELD_DATE = "2026-09-02 (5·20년은 보간 추정)"
FALLBACK_YIELDS = {"1y": 3.46, "3y": 3.88, "5y": 4.02, "10y": 4.37, "20y": 4.50, "30y": 4.63}

st.sidebar.header("무위험이자율 (국채 금리)")

current_yields = dict(FALLBACK_YIELDS)
yield_source_note = f"자동조회 실패로 폴백값 사용 중 (기준일 {FALLBACK_YIELD_DATE})"

ecos_key = os.getenv("ECOS_API_KEY", "")
if not ecos_key:
    ecos_key = st.sidebar.text_input(
        "ECOS 인증키",
        type="password",
        help="ecos.bok.or.kr에서 무료로 발급받을 수 있음",
    )

if ecos_key:
    with st.spinner("ECOS 국고채 금리 조회 중..."):
        fetched, error = fetch_ecos_bond_yields(ecos_key)
    if fetched:
        current_yields = {k: fetched[k] for k in TENOR_LABELS}
        fetched_date = fetched.get("1y_date", "")
        yield_source_note = f"ECOS 조회 (기준일 {fetched_date})"


st.sidebar.caption(yield_source_note)

YIELD_WIDGET_KEYS = {"1y": "y1_input", "3y": "y3_input", "5y": "y5_input",
                      "10y": "y10_input", "20y": "y20_input", "30y": "y30_input"}

for tenor_key, widget_key in YIELD_WIDGET_KEYS.items():
    st.session_state.setdefault(widget_key, current_yields[tenor_key])

if st.sidebar.button("ECOS 조회값으로 되돌리기"):
    for tenor_key, widget_key in YIELD_WIDGET_KEYS.items():
        st.session_state[widget_key] = current_yields[tenor_key]

y1 = st.sidebar.number_input("1년물 (%)", step=0.01, format="%.2f", key="y1_input")
y3 = st.sidebar.number_input("3년물 (%)", step=0.01, format="%.2f", key="y3_input")
y5 = st.sidebar.number_input("5년물 (%)", step=0.01, format="%.2f", key="y5_input")
y10 = st.sidebar.number_input("10년물 (%)", step=0.01, format="%.2f", key="y10_input")
y20 = st.sidebar.number_input("20년물 (%)", step=0.01, format="%.2f", key="y20_input")
y30 = st.sidebar.number_input("30년물 (%)", step=0.01, format="%.2f", key="y30_input")
interp_kind = st.sidebar.selectbox("보간 방식", ["linear", "quadratic", "cubic"], index=0)

tenors = np.array([1, 3, 5, 10, 20, 30])
yields = np.array([y1, y3, y5, y10, y20, y30])

curve = build_curve(tenors, yields, kind=interp_kind)
r = float(curve(T)) / 100

# 메인 화면
st.title("워런트 블랙숄즈 가치평가")

col1, col2, col3 = st.columns(3)

price = bs_call_price(S, K, T, r, sigma, q) if opt_key == "call" else bs_put_price(S, K, T, r, sigma, q)
greeks = bs_greeks(S, K, T, r, sigma, q, option_type=opt_key)

with col1:
    st.metric("옵션 이론가", f"{price:,.1f}")
with col2:
    st.metric(f"무위험이자율 r (T={T:.1f}년)", f"{r*100:.3f}%")
with col3:
    moneyness = "ITM (내가격)" if (S > K and opt_key == "call") or (S < K and opt_key == "put") else "OTM (외가격)"
    st.metric("현재 상태", moneyness)

if S / K > 3 or S / K < 0.33:
    st.caption(
        f"현재가(S={S:,.0f})와 행사가(K={K:,.0f})가 {S/K:.1f}배 차이 남. "
        "K를 자동으로 채워주지는 않으니, 실제 종목의 행사가를 넣은 게 맞는지 확인 필요."
    )

st.divider()

st.subheader("국채 금리 곡선")
curve_x = np.linspace(0.5, 32, 200)
curve_y = curve(curve_x)

NAVY = "#2C5F8A"
CORAL = "#F2994A"
EMERALD = "#27AE60"

fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(
    x=curve_x, y=curve_y, mode="lines", name="보간 곡선",
    line=dict(color=NAVY, width=3, shape="spline"),
    fill="tozeroy", fillcolor="rgba(44, 95, 138, 0.08)",
    hovertemplate="만기 %{x:.1f}년, 금리 %{y:.3f}%<extra></extra>",
))
fig_curve.add_trace(go.Scatter(
    x=tenors, y=yields, mode="markers", name="국고채 금리 (1,3,5,10,20,30년물)",
    marker=dict(size=11, color=CORAL, line=dict(color="white", width=1.5)),
    hovertemplate="%{x}년물: %{y:.3f}%<extra></extra>",
))
fig_curve.add_trace(go.Scatter(
    x=[T], y=[r * 100], mode="markers", name=f"옵션 만기 T={T:.1f}년 → r",
    marker=dict(size=16, symbol="star", color=EMERALD, line=dict(color="white", width=1.5)),
    hovertemplate=f"T={T:.1f}년: %{{y:.3f}}%<extra></extra>",
))
fig_curve.update_layout(
    template="plotly_white",
    xaxis_title="만기 (년)", yaxis_title="금리 (%)", height=380,
    font=dict(family="Arial, sans-serif", size=13, color="#333"),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    margin=dict(t=20, b=40),
    xaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    hovermode="x unified",
)
st.plotly_chart(fig_curve, width='stretch')

st.divider()


def format_greek(v):
    if v != 0 and abs(v) < 0.001:
        return f"{v:.3e}"
    return f"{v:,.5f}"


greeks_df = pd.DataFrame({
    "그릭스": list(greeks.keys()),
    "값": [format_greek(v) for v in greeks.values()],
    "의미": [
        "주가 1원 변화당 옵션가 변화",
        "델타의 주가 민감도 (곡률)",
        "변동성 1%p 변화당 옵션가 변화",
        "하루 경과당 옵션가 변화 (시간가치 감소)",
        "금리 1%p 변화당 옵션가 변화",
    ],
})
st.dataframe(greeks_df, hide_index=True, width='stretch')

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["만기별 금리 적용 비교", "금리 민감도", "변동성 민감도", "주가 민감도"])

price_fn = bs_call_price if opt_key == "call" else bs_put_price

with tab1:
    st.caption("같은 옵션 조건에 각 만기물 금리를 그대로 r로 넣으면 가격이 얼마나 달라지는지 봐요.")
    tenor_prices = [price_fn(S, K, T, yv / 100, sigma, q) for yv in yields]
    pct_diff = [(p / price - 1) * 100 for p in tenor_prices]
    tenor_df = pd.DataFrame({"국채 만기물": [f"{t}년물" for t in tenors],
                              "적용 금리(%)": yields,
                              "옵션가치": [f"{p:,.1f}" for p in tenor_prices],
                              "현재 보간가 대비": [f"{d:+.2f}%" for d in pct_diff]})
    st.dataframe(tenor_df, hide_index=True, width='stretch')

    bar_colors = ["#d62728" if d < 0 else "#1f77b4" for d in pct_diff]
    fig_tenor = go.Figure(go.Bar(
        x=[f"{t}년물" for t in tenors], y=pct_diff,
        text=[f"{d:+.2f}%" for d in pct_diff], textposition="outside",
        marker_color=bar_colors,
    ))
    fig_tenor.add_hline(y=0, line_color="gray", line_width=1)
    fig_tenor.update_layout(
        yaxis_title=f"보간가(T={T:.1f}년) 대비 차이 (%)",
        height=350,
        margin=dict(t=40),
    )
    st.plotly_chart(fig_tenor, width='stretch')
    if max(abs(d) for d in pct_diff) < 1:
        st.caption("이 조건에서는 만기물을 어떤 걸 골라도 가격이 1% 넘게 안 움직임. 지금 옵션이 깊은 내가격이라 금리 민감도가 낮기 때문.")


with tab2:
    r_range = np.linspace(max(r - 0.03, 0.001), r + 0.03, 60)
    p_range = [price_fn(S, K, T, rr, sigma, q) for rr in r_range]
    fig_r = go.Figure(go.Scatter(x=r_range * 100, y=p_range, mode="lines"))
    fig_r.add_vline(x=r * 100, line_dash="dot", line_color="green", annotation_text="현재 r")
    fig_r.update_layout(xaxis_title="무위험이자율 r (%)", yaxis_title="옵션 이론가", height=380)
    st.plotly_chart(fig_r, width='stretch')
    st.caption(f"Rho = {greeks['Rho']:.4f}, 금리가 1%p 오르면 옵션가는 약 {greeks['Rho']:,.1f} 변함.")

with tab3:
    sig_range = np.linspace(max(sigma - 0.30, 0.01), sigma + 0.30, 60)
    p_range = [price_fn(S, K, T, r, ss, q) for ss in sig_range]
    fig_sig = go.Figure(go.Scatter(x=sig_range * 100, y=p_range, mode="lines"))
    fig_sig.add_vline(x=sigma * 100, line_dash="dot", line_color="green", annotation_text="현재 σ")
    fig_sig.update_layout(xaxis_title="변동성 σ (%)", yaxis_title="옵션 이론가", height=380)
    st.plotly_chart(fig_sig, width='stretch')
    st.caption(f"Vega = {greeks['Vega']:.4f}, 변동성이 1%p 오르면 옵션가는 약 {greeks['Vega']:,.1f} 변함.")

with tab4:
    s_range = np.linspace(S * 0.5, S * 1.5, 60)
    p_range = [price_fn(ss, K, T, r, sigma, q) for ss in s_range]
    fig_s = go.Figure(go.Scatter(x=s_range, y=p_range, mode="lines"))
    fig_s.add_vline(x=S, line_dash="dot", line_color="green", annotation_text="현재 S")
    fig_s.add_vline(x=K, line_dash="dash", line_color="red", annotation_text="행사가 K")
    fig_s.update_layout(xaxis_title="기초주가 S", yaxis_title="옵션 이론가", height=380)
    st.plotly_chart(fig_s, width='stretch')
    st.caption(f"Delta = {greeks['Delta']:.4f}, 주가가 1원 오르면 옵션가는 약 {greeks['Delta']:.4f} 변함.")

st.divider()
with st.expander("계산식"):
    st.latex(r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)")
    st.latex(r"d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}")
    st.caption("r은 옵션 잔존만기 T에 맞춰 국채 금리곡선(1,3,5,10,20,30년물)을 보간한 값.")
