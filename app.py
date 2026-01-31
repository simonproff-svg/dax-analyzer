import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Seitenkonfiguration
st.set_page_config(
    page_title="DAX Pro Analyzer",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS für Dark Theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
    }
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1e293b;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #334155;
        text-align: center;
    }
    .signal-bullish { color: #4ade80; font-weight: bold; }
    .signal-bearish { color: #f87171; font-weight: bold; }
    .signal-neutral { color: #facc15; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_dax_data():
    """Holt DAX-Daten von Yahoo Finance"""
    try:
        ticker = yf.Ticker("^GDAXI")
        hist = ticker.history(period="6mo")
        
        if not hist.empty:
            data = []
            for date, row in hist.iterrows():
                data.append({
                    'date': date,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            return data, "Yahoo Finance Live", True
    except Exception as e:
        st.error(f"API Fehler: {e}")
    
    # Fallback
    return None, "Keine Daten", False

def calculate_all_indicators(data):
    """Berechnet alle 12 Indikatoren"""
    closes = np.array([d['close'] for d in data])
    highs = np.array([d['high'] for d in data])
    lows = np.array([d['low'] for d in data])
    volumes = np.array([d['volume'] for d in data])
    
    current_price = closes[-1]
    
    # 1. SMAs
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:])
    sma200 = np.mean(closes[-200:]) if len(closes) >= 200 else sma50
    
    # 2. EMAs für MACD
    ema12 = pd.Series(closes).ewm(span=12).mean().values
    ema26 = pd.Series(closes).ewm(span=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9).mean().values
    
    # 3. RSI
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    rsi = 100 - (100 / (1 + avg_gain/avg_loss)) if avg_loss > 0 else 50
    
    # 4. Bollinger Bands
    sma20_series = pd.Series(closes).rolling(window=20).mean()
    std20 = pd.Series(closes).rolling(window=20).std()
    bb_upper = (sma20_series + (std20 * 2)).values[-1]
    bb_lower = (sma20_series - (std20 * 2)).values[-1]
    
    # 5. Stochastic
    low14 = pd.Series(lows).rolling(window=14).min()
    high14 = pd.Series(highs).rolling(window=14).max()
    k_percent = 100 * (closes - low14) / (high14 - low14)
    d_percent = k_percent.rolling(window=3).mean()
    stoch_k = k_percent.values[-1] if not np.isnan(k_percent.values[-1]) else 50
    stoch_d = d_percent.values[-1] if not np.isnan(d_percent.values[-1]) else 50
    
    # 6. ADX (vereinfacht)
    plus_dm = highs[1:] - highs[:-1]
    minus_dm = lows[:-1] - lows[1:]
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                              np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-14:])
    adx = 25  # Platzhalter für echte ADX-Berechnung
    
    # 7. Williams %R
    highest_high = np.max(highs[-14:])
    lowest_low = np.min(lows[-14:])
    williams_r = -100 * (highest_high - current_price) / (highest_high - lowest_low) if (highest_high - lowest_low) != 0 else -50
    
    # 8. OBV
    obv = [volumes[0]]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    obv_sma20 = np.mean(obv[-20:])
    obv_current = obv[-1]
    
    # 9. Parabolic SAR (vereinfacht)
    sar_trend = 1 if current_price > sma50 else -1
    
    # 10. Fibonacci Levels (letzte 60 Tage)
    recent_high = np.max(highs[-60:])
    recent_low = np.min(lows[-60:])
    fib_diff = recent_high - recent_low
    fib_382 = recent_high - 0.382 * fib_diff
    fib_618 = recent_high - 0.618 * fib_diff
    
    # 11. Pivot Points (Tagesdaten)
    pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
    r1 = (2 * pivot) - lows[-1]
    r2 = pivot + (highs[-1] - lows[-1])
    s1 = (2 * pivot) - highs[-1]
    s2 = pivot - (highs[-1] - lows[-1])
    
    # 12. ATR (Volatilität)
    atr_current = atr
    
    # Punkte-Berechnung (12 Indikatoren)
    bullish_points = 0
    bearish_points = 0
    signals = []
    
    # SMA 20
    if current_price > sma20:
        bullish_points += 10
        signals.append(('bullish', 'Preis über SMA20', 10))
    else:
        bearish_points += 10
        signals.append(('bearish', 'Preis unter SMA20', 10))
    
    # SMA 50
    if current_price > sma50:
        bullish_points += 15
        signals.append(('bullish', 'Preis über SMA50', 15))
    else:
        bearish_points += 15
        signals.append(('bearish', 'Preis unter SMA50', 15))
    
    # SMA 200
    if current_price > sma200:
        bullish_points += 20
        signals.append(('bullish', 'Preis über SMA200', 20))
    else:
        bearish_points += 20
        signals.append(('bearish', 'Preis unter SMA200', 20))
    
    # Parabolic SAR Trend
    if sar_trend == 1:
        bullish_points += 15
        signals.append(('bullish', 'Parabolic SAR Bull-Trend', 15))
    else:
        bearish_points += 15
        signals.append(('bearish', 'Parabolic SAR Bear-Trend', 15))
    
    # Golden/Death Cross
    if sma50 > sma200:
        bullish_points += 10
        signals.append(('bullish', 'Golden Cross', 10))
    else:
        bearish_points += 10
        signals.append(('bearish', 'Death Cross', 10))
    
    # RSI
    if rsi < 30:
        bullish_points += 15
        signals.append(('bullish', f'RSI überverkauft ({rsi:.1f})', 15))
    elif rsi > 70:
        bearish_points += 15
        signals.append(('bearish', f'RSI überkauft ({rsi:.1f})', 15))
    elif rsi > 50:
        bullish_points += 5
        signals.append(('neutral', f'RSI bullief ({rsi:.1f})', 5))
    else:
        bearish_points += 5
        signals.append(('neutral', f'RSI bearish ({rsi:.1f})', 5))
    
    # Williams %R
    if williams_r < -80:
        bullish_points += 15
        signals.append(('bullish', f'Williams %R überverkauft ({williams_r:.1f})', 15))
    elif williams_r > -20:
        bearish_points += 15
        signals.append(('bearish', f'Williams %R überkauft ({williams_r:.1f})', 15))
    
    # MACD
    if macd_line[-1] > signal_line[-1]:
        bullish_points += 10
        signals.append(('bullish', 'MACD über Signal', 10))
    else:
        bearish_points += 10
        signals.append(('bearish', 'MACD unter Signal', 10))
    
    # Bollinger Bands
    if current_price <= bb_lower:
        bullish_points += 15
        signals.append(('bullish', f'Preis am unteren Bollinger', 15))
    elif current_price >= bb_upper:
        bearish_points += 15
        signals.append(('bearish', f'Preis am oberen Bollinger', 15))
    elif current_price > sma20:
        bullish_points += 5
        signals.append(('neutral', 'Preis über BB-Mitte', 5))
    else:
        bearish_points += 5
        signals.append(('neutral', 'Preis unter BB-Mitte', 5))
    
    # Stochastic
    if stoch_k < 20 and stoch_k > stoch_d:
        bullish_points += 15
        signals.append(('bullish', f'Stochastik Kaufsignal ({stoch_k:.1f})', 15))
    elif stoch_k > 80 and stoch_k < stoch_d:
        bearish_points += 15
        signals.append(('bearish', f'Stochastik Verkaufssignal ({stoch_k:.1f})', 15))
    elif stoch_k > stoch_d:
        bullish_points += 5
        signals.append(('neutral', f'Stochastik steigend', 5))
    else:
        bearish_points += 5
        signals.append(('neutral', f'Stochastik fallend', 5))
    
    # ADX Trendstärke
    if adx > 25:
        if current_price > sma50:
            bullish_points += 10
            signals.append(('bullish', f'Starker Trend (ADX > 25)', 10))
        else:
            bearish_points += 10
            signals.append(('bearish', f'Starker Trend (ADX > 25)', 10))
    
    # OBV
    if obv_current > obv_sma20 and current_price > sma20:
        bullish_points += 10
        signals.append(('bullish', 'OBV bestätigt Aufwärtstrend', 10))
    elif obv_current < obv_sma20 and current_price < sma20:
        bearish_points += 10
        signals.append(('bearish', 'OBV bestätigt Abwärtstrend', 10))
    
    # Fibonacci
    if current_price > fib_382:
        bullish_points += 20
        signals.append(('bullish', f'Preis über Fib 38.2%', 20))
    elif current_price < fib_618:
        bearish_points += 20
        signals.append(('bearish', f'Preis unter Fib 61.8%', 20))
    
    # Volumen
    vol_avg = np.mean(volumes[-20:])
    if volumes[-1] > vol_avg * 1.2:
        if closes[-1] > closes[-2]:
            bullish_points += 10
            signals.append(('bullish', f'Hohes Kaufvolumen', 10))
        else:
            bearish_points += 10
            signals.append(('bearish', f'Hohes Verkaufsvolumen', 10))
    
    # Gesamtbewertung
    total = bullish_points + bearish_points
    score = (bullish_points / total * 100) if total > 0 else 50
    
    # Prognose
    prob = 50
    if current_price > sma20: prob += 8
    if current_price > sma50: prob += 10
    if current_price > sma200: prob += 12
    if rsi < 30: prob += 12
    if rsi > 70: prob -= 12
    if williams_r < -80: prob += 8
    if macd_line[-1] > signal_line[-1]: prob += 8
    if current_price <= bb_lower: prob += 12
    if stoch_k < 20: prob += 8
    if obv_current > obv_sma20: prob += 6
    if current_price > fib_382: prob += 8
    
    prob = max(0, min(100, prob))
    
    # Empfehlung
    leverage_factor = 1.0 if adx > 25 else 0.5 if adx > 20 else 0
    
    if score >= 75:
        rec = "STARK BULLISH"
        lev = f"Long {'3-5x' if leverage_factor == 1.0 else '2x' if leverage_factor == 0.5 else 'Spot'}"
    elif score >= 60:
        rec = "BULLISH"
        lev = f"Long {'2-3x' if leverage_factor >= 0.5 else '1-2x'}"
    elif score >= 40:
        rec = "NEUTRAL"
        lev = "Kein Hebel empfohlen"
    elif score >= 25:
        rec = "BEARISH"
        lev = f"Short {'2-3x' if leverage_factor >= 0.5 else '1-2x'}"
    else:
        rec = "STARK BEARISH"
        lev = f"Short {'3-5x' if leverage_factor == 1.0 else '2x' if leverage_factor == 0.5 else 'Spot'}"
    
    change_5d = ((closes[-1] - closes[-6]) / closes[-6]) * 100 if len(closes) > 6 else 0
    
    return {
        'price': current_price,
        'score': score,
        'probability': prob,
        'recommendation': rec,
        'leverage': lev,
        'bullish': bullish_points,
        'bearish': bearish_points,
        'signals': sorted(signals, key=lambda x: x[2], reverse=True),
        'sma20': sma20, 'sma50': sma50, 'sma200': sma200,
        'rsi': rsi,
        'macd': macd_line[-1],
        'signal': signal_line[-1],
        'bb_upper': bb_upper, 'bb_lower': bb_lower,
        'stoch_k': stoch_k, 'stoch_d': stoch_d,
        'williams_r': williams_r,
        'adx': adx,
        'obv': obv_current,
        'sar_trend': sar_trend,
        'fib_382': fib_382, 'fib_618': fib_618,
        'pivot': pivot, 'r1': r1, 'r2': r2, 's1': s1, 's2': s2,
        'change_5d': change_5d,
        'data': data
    }

# Haupt-UI
st.title("📊 DAX Pro Analyzer")
st.markdown("Ultimate Edition - 12 Indikatoren")

# Daten laden
with st.spinner("Lade DAX-Daten..."):
    raw_data, source, success = get_dax_data()

if success and raw_data:
    a = calculate_all_indicators(raw_data)
    
    # Top Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="DAX Kurs",
            value=f"{a['price']:.0f}",
            delta=f"{a['change_5d']:.1f}% (5T)"
        )
    
    with col2:
        delta_color = "normal" if a['score'] >= 50 else "inverse"
        st.metric(
            label="Signal-Stärke",
            value=f"{a['score']:.0f}%",
            delta=a['recommendation'],
            delta_color=delta_color
        )
    
    with col3:
        st.metric(
            label="Prognose",
            value=f"{a['probability']:.0f}%",
            delta="Bullish" if a['probability'] > 60 else "Neutral" if a['probability'] > 40 else "Bearish"
        )
    
    # Hebel-Empfehlung
    color = "#4ade80" if a['score'] >= 70 else "#facc15" if a['score'] >= 40 else "#ef4444"
    st.markdown(f"""
        <div style='background-color: #1e3a8a; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; margin: 20px 0;'>
            <h3 style='margin:0; color: white;'>🎯 {a['leverage']}</h3>
            <p style='margin:5px 0 0 0; color: #93c5fd;'>{a['recommendation']} • {a['bullish']}-{a['bearish']} Punkte</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Support & Resistance
    st.subheader("🎯 Support & Resistance Levels")
    
    cols = st.columns(5)
    levels = [
        ("R2", a['r2'], "#ef4444"),
        ("R1", a['r1'], "#fca5a5"),
        ("Pivot", a['pivot'], "#ffffff"),
        ("S1", a['s1'], "#86efac"),
        ("S2", a['s2'], "#22c55e")
    ]
    
    for col, (name, value, color) in zip(cols, levels):
        col.markdown(f"""
            <div style='text-align: center; background-color: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid {color};'>
                <div style='color: {color}; font-size: 11px;'>{name}</div>
                <div style='color: white; font-weight: bold; font-size: 13px;'>{value:.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Indikatoren in Expander
    with st.expander("📊 Technische Indikatoren (12)"):
        # 4 Spalten für 12 Indikatoren
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Trend-Indikatoren**")
            st.markdown(f"SMA 20: {'🟢' if a['price'] > a['sma20'] else '🔴'} {a['sma20']:.0f}")
            st.markdown(f"SMA 50: {'🟢' if a['price'] > a['sma50'] else '🔴'} {a['sma50']:.0f}")
            st.markdown(f"SMA 200: {'🟢' if a['price'] > a['sma200'] else '🔴'} {a['sma200']:.0f}")
            st.markdown(f"Parabolic SAR: {'🟢 Bull' if a['sar_trend'] == 1 else '🔴 Bear'}")
            
            st.markdown("**Momentum**")
            rsi_color = "🟢" if a['rsi'] < 30 else "🔴" if a['rsi'] > 70 else "🟡"
            st.markdown(f"RSI: {rsi_color} {a['rsi']:.1f}")
            st.markdown(f"Williams %R: {a['williams_r']:.1f}")
            st.markdown(f"MACD: {'🟢' if a['macd'] > a['signal'] else '🔴'} {a['macd']:.2f}")
        
        with col2:
            st.markdown("**Volatilität & Volumen**")
            st.markdown(f"Bollinger: {a['bb_lower']:.0f} - {a['bb_upper']:.0f}")
            st.markdown(f"Stochastik: {a['stoch_k']:.1f}")
            st.markdown(f"ADX: {a['adx']:.1f} {'(stark)' if a['adx'] > 25 else '(schwach)'}")
            st.markdown(f"OBV: {'🟢 steigend' if a['obv'] > np.mean([a['obv']]) else '🔴 fallend'}")
            
            st.markdown("**Fibonacci**")
            st.markdown(f"38.2%: {a['fib_382']:.0f}")
            st.markdown(f"61.8%: {a['fib_618']:.0f}")
    
    # Chart
    st.subheader("📈 Chart (30 Tage)")
    df_chart = pd.DataFrame(a['data'][-30:])
    fig = go.Figure(data=[go.Candlestick(
        x=df_chart['date'],
        open=df_chart['open'],
        high=df_chart['high'],
        low=df_chart['low'],
        close=df_chart['close'],
        name="DAX"
    )])
    
    # SMAs zum Chart hinzufügen
    fig.add_trace(go.Scatter(
        x=df_chart['date'],
        y=[a['sma20']] * len(df_chart),
        name="SMA 20",
        line=dict(color='rgba(255,255,255,0.5)', width=1)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Signale
    with st.expander("🔍 Top Signale"):
        for sig_type, text, weight in a['signals'][:6]:
            emoji = "🟢" if sig_type == "bullish" else "🔴" if sig_type == "bearish" else "🟡"
            st.markdown(f"{emoji} **{text}** ({'+' if sig_type=='bullish' else '-'}{weight} pkt)")
    
    # Risikohinweis
    st.warning("⚠️ **Risikohinweis**: Gehebelte Produkte bergen ein hohes Risiko für Totalverlust. Nur Geld investieren, dessen Verlust Sie sich leisten können.")
    
    # Footer
    st.caption(f"Datenquelle: {source} | Letztes Update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto-Refresh
    st.markdown("""
        <script>
            setTimeout(function(){
                window.location.reload();
            }, 300000);  // 5 Minuten
        </script>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Keine Daten verfügbar. Yahoo Finance API temporär nicht erreichbar.")