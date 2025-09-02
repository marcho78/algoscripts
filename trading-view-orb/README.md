# Universal ORB **v2** — Pine v6 Strategy

Opening Range Breakout strategy with a compact, real-time HUD, risk-managed exits, and guardrails to avoid late entries.

---

## Table of Contents
- [What It Does](#what-it-does)
- [Quick Start](#quick-start)
- [Recommended Timeframes & Sessions](#recommended-timeframes--sessions)
- [Inputs Reference](#inputs-reference)
- [How Trades Trigger (Logic Flow)](#how-trades-trigger-logic-flow)
- [Risk & Exits (R Multiples)](#risk--exits-r-multiples)
- [HUD Guide](#hud-guide)
- [Plots & Labels](#plots--labels)
- [Backtesting Tips](#backtesting-tips)
- [Live Trading Tips](#live-trading-tips)
- [Troubleshooting](#troubleshooting)
- [Example Presets](#example-presets)
- [User Journey Example](#user-journey-example)
- [Install / Run](#install--run)
- [Changelog](#changelog)
- [Disclaimer](#disclaimer)

---

## What It Does

- Builds an **Opening Range (OR)** from the **first _N_ minutes** after session open.
- Goes **long** on breaks **above** OR high; **short** on breaks **below** OR low.
- Optional **filters**:
  - **Minimum OR width** (skip tiny, choppy ranges).
  - **Volume filter** (current volume must exceed SMA).
  - **Entry guard** (only allow entries ≤ _X_ bars after the initial break).
- **Risk-managed exits**:
  - Stop based on **ATR × multiplier**.
  - Two profit targets (**TP1**, **TP2**) at configurable **R multiples** (50/50 split).

---

## Quick Start

1. **Add the script** to a chart in TradingView (see [Install / Run](#install--run)).
2. In **Settings → Inputs**:
   - Set **Session** (e.g., `0930-1600` for US equities; a daily boundary for crypto).
   - Set **First N Minutes** (common: **5**, **15**, **30**).
   - Optionally enable **Volume Filter** and set a **Min OR Width**.
3. Pick a timeframe (1m–15m typical).  
4. After the first N minutes, the OR locks. Breakouts that pass filters/guard will **auto-enter** with TP/SL attached.
5. Review results in **Strategy Tester**. Tweak inputs to fit your market.

---

## Recommended Timeframes & Sessions

| Market      | Session Example   | First N Minutes | Chart Timeframe | Notes |
|-------------|-------------------|----------------:|----------------:|------|
| US Equities | `0930-1600`       | 5–30            | **1m–5m**       | 5–15m OR is common; 1–3m chart for precision. |
| Futures     | RTH or chosen day | 5–30            | 1m–5m           | Use exchange RTH if you trade regular hours only. |
| Crypto      | `0000-2359` (or preferred daily reset) | 15–60 | **3m–15m** | 24/7—pick a consistent daily open to define OR. |

**Heuristics**
- Shorter **N** → more trades, more noise.  
- Longer **N** → fewer trades, higher average quality.  
- Increase **Min OR Width** for very volatile symbols.

---

## Inputs Reference

### Opening Range
- **Session** (`0930-1600` by default): Trading window that defines each session/day.
- **First N Minutes**: Minutes after session open used to build the OR high/low.

### Filters
- **Min OR Width (points)**: Minimum acceptable range height; `0` disables.
- **Enable Volume Filter**: Requires `volume > SMA(volume, len)`.
- **Volume SMA Length**: Period for volume SMA.

### Risk / Targets
- **ATR Length**: ATR period for stop calculation.
- **ATR ×**: Stop distance multiplier (`stop = ATR × multiplier`).
- **TP1 = R Multiple**: First target measured in R (risk units).
- **TP2 = R Multiple**: Second target measured in R.

### Entry Guard
- **Max Bars After Break**: Reject entries that occur more than _X_ bars after the initial break.

### Visuals
- **Plot OR High/Low**: Draw the OR lines (forming vs locked).
- **Show Break Labels**: “LONG/SHORT” markers at entries.

---

## How Trades Trigger (Logic Flow)

1. **Build OR** during the first _N_ minutes of the session (track highest high & lowest low).
2. When _N_ minutes pass, **OR locks**.
3. **Detect breakouts**:
   - **Up-break** when `close` crosses **above** OR high.
   - **Down-break** when `close` crosses **below** OR low.
4. **Check filters**:
   - **Min OR Width** (if set) must pass.
   - **Volume Filter** (if enabled) must pass.
   - **Entry Guard** must pass (bars since initial break ≤ limit).
5. **Place entry** (long/short) with attached **stop** (ATR-based) and **two targets** (TP1, TP2).
6. **Exit logic**: TP1 and TP2 each close **50%** of the position; stop exits the remainder.

---

## Risk & Exits (R Multiples)

- **R** = stop distance (entry price to stop).  
- Example (long): ATR-based stop is **100** points below entry ⇒ **1R = 100**.
  - **TP1 @ 1R** = entry + 100  
  - **TP2 @ 2R** = entry + 200  
- Default split: **50%** size to TP1, **50%** to TP2.

Tune via **ATR Length**, **ATR ×**, and **TP1/TP2** R settings.

---

## HUD Guide

Compact table in the **top-right** with real-time updates on the live bar.

| Row        | Meaning |
|------------|---------|
| **Universal ORB v2 / Filters + Guard** | Header |
| **OR Mode** | “First N Min” |
| **OR Hi / OR Lo** | Current OR levels (forming, then locked) |
| **Width** | OR High − OR Low |
| **Break** | Direction of the **last** break: Up / Down / None |
| **Setup** | Playbook hint: Await Break / Await Retest Long / Await Retest Short |
| **Position** | Above Range / Below Range / In Range |
| **Min OR** | PASS / FAIL |
| **VolFilt** | ON (PASS) / ON (FAIL) / OFF |
| **Stop/TP** | Current stop and TP levels (contextual) |
| **Guard** | “≤X | now:Y” bars since initial break |

---

## Plots & Labels

- **OR High/Low**  
  - Translucent while forming; solid after locking.
- **Entry Labels**  
  - “LONG” plotted above bar on up-break entries; “SHORT” below bar on down-break entries.

---

## Backtesting Tips

- Use **1m–5m** charts for intraday ORB on equities/futures; **3m–15m** for crypto.
- Keep the **Session** aligned with the instrument (RTH vs 24h).
- Experiment with:
  - **First N Minutes**: 5 / 15 / 30
  - **ATR ×**: 1.0–2.0
  - **Min OR Width**: symbol-specific
  - **Volume Filter**: often helpful in chop
- Evaluate **Net Profit**, **Max Drawdown**, **Win Rate**, and **Trade Count**—avoid over-filtering to zero trades.

---

## Live Trading Tips

- Account for **spread & slippage**; avoid targets unrealistically close to entry.
- If you find yourself chasing, **tighten Entry Guard**.
- If you’re getting chopped, **increase N** and/or **Min OR Width**, and consider enabling **Volume Filter**.
- Many traders prefer **“first valid break only”** per session—consider that as a personal rule.

---

## Troubleshooting

- **No trades:**  
  - OR might still be forming.  
  - Filters may be blocking (Min Width, Volume, Guard).  
  - Session may not match the instrument/time.
- **Too many trades:**  
  - Increase **First N Minutes** or **Min OR Width**; enable **Volume Filter**.
- **Labels/lines missing:**  
  - Ensure **Visuals** toggles are on in Settings.

---

## Example Presets

**Equities (active, early breakout)**  
- Session: `0930-1600`  
- First N Minutes: **5–15**  
- Timeframe: **1m–3m**  
- Min OR Width: start around **0.2–0.5%** of price (symbol-dependent)  
- Volume Filter: **ON**, Length **20**  
- Entry Guard: **≤ 3 bars**

**Futures (RTH breakout)**  
- Session: Exchange RTH  
- First N Minutes: **15–30**  
- Timeframe: **1m–5m**  
- Volume Filter: ON  
- ATR ×: **1.5**  
- TP1/TP2: **1R / 2R**

**Crypto (daily OR)**  
- Session: `0000-2359` (or chosen reset)  
- First N Minutes: **30–60**  
- Timeframe: **3m–15m**  
- Min OR Width: ON (tune per pair)  
- Volume Filter: optional (exchange-dependent)

---

## User Journey Example

**Persona:** Sam, intraday trader on US equities, likes momentum out of the open.

1. **Add to Chart**  
   Sam opens the AAPL 1-minute chart, pastes the script into TradingView Pine Editor, and hits **Add to chart**.

2. **Configure**  
   - Session: `0930-1600`  
   - First N Minutes: `15`  
   - Min OR Width: `0.4` (about ~0.25–0.5% of typical AAPL price)  
   - Volume Filter: **ON**, Length `20`  
   - Entry Guard: `3` bars  
   - ATR ×: `1.5`, TP1 `1R`, TP2 `2R`

3. **Observe the Open**  
   From 9:30 to 9:45, the HUD shows **OR Hi/Lo** forming. After 9:45, the range locks and OR lines turn solid.

4. **First Breakout**  
   Price pops above **OR High** at ~9:47. The HUD shows **Break: Up**. Volume passes; Min Width passes; within 3 bars of the break. Strategy enters **LONG** with ATR stop and two targets.

5. **Trade Management**  
   The HUD line **Stop/TP** displays current SL and both TPs. Sam watches TP1 fill quickly; TP2 trails behind and eventually fills on extension. If price reverses, the ATR stop is there to cap the loss.

6. **Review & Iterate**  
   After the close, Sam checks **Strategy Tester** results. The day looks solid, but on a different symbol the range was too tight, so Sam increases **Min OR Width** to avoid similar chop tomorrow.

---

## Install / Run

1. Open **TradingView → Pine Editor**.  
2. Paste the **Universal ORB v2 (Pine v6)** code.  
3. Click **Add to chart**.  
4. Open **Settings** to configure inputs per your market/timeframe.

> The strategy is written for **Pine Script v6**.

---

## Changelog

- **v2**
  - Consolidated docs and naming.
  - Compact, real-time HUD with clear status rows.
  - ORB logic with Min Width, Volume filter, and Entry guard.
  - ATR-based stop and dual TP exits (R-based).

---

## Disclaimer

This repository is for educational purposes only and **not financial advice**. Markets involve risk. Test thoroughly in a simulator before using on live capital.
