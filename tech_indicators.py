import math

import pandas as pd
import numpy as np


####################################################
####################################################
####################################################
#  Helpers
####################################################
####################################################
####################################################

np.seterr(invalid='ignore')

def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)] # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df


def ema(series, periods):
    return series.ewm(span=periods, min_periods=0).mean()


####################################################
####################################################
####################################################
#  Momentum - 8
####################################################
####################################################
####################################################


def momentum_rsi(
    high, low, close, volume,
    n=14):
    """Relative Strength Index (RSI)
    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.
    https://www.investopedia.com/terms/r/rsi.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    diff = close.diff()
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = ema(up, n)
    emadn = ema(dn, n)

    rsi = 100 * emaup / (emaup + emadn)
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(rsi, name='momentum_rsi')


def momentum_mfi(
    high, low, close, volume,
    n=14):
    """Money Flow Index (MFI)
    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    # 0 Prepare dataframe to work
    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ['High', 'Low', 'Close', 'Volume']
    df['Up_or_Down'] = 0
    df.loc[(df['Close'] > df['Close'].shift(1)), 'Up_or_Down'] = 1
    df.loc[(df['Close'] < df['Close'].shift(1)), 'Up_or_Down'] = 2

    # 1 typical price
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0

    # 2 money flow
    mf = tp * df['Volume']

    # 3 positive and negative money flow with n periods
    df['1p_Positive_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 1, '1p_Positive_Money_Flow'] = mf
    n_positive_mf = df['1p_Positive_Money_Flow'].rolling(n).sum()

    df['1p_Negative_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 2, '1p_Negative_Money_Flow'] = mf
    n_negative_mf = df['1p_Negative_Money_Flow'].rolling(n).sum()

    # 4 money flow index
    mr = n_positive_mf / n_negative_mf
    mr = (100 - (100 / (1 + mr)))
    mr = mr.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(mr, name='momentum_mfi')


def momentum_tsi(
    high, low, close, volume,
    r=25, s=13):
    """True strength index (TSI)
    Shows both trend direction and overbought/oversold conditions.
    https://en.wikipedia.org/wiki/True_strength_index
    Args:
        close(pandas.Series): dataset 'Close' column.
        r(int): high period.
        s(int): low period.

    Returns:
        pandas.Series: New feature generated.
    """
    m = close - close.shift(1)
    m1 = m.ewm(r).mean().ewm(s).mean()
    m2 = abs(m).ewm(r).mean().ewm(s).mean()
    tsi = m1 / m2
    tsi *= 100
    tsi = tsi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(tsi, name='momentum_tsi')


def momentum_uo(
    high, low, close, volume,
    s=7, m=14, l=28, ws=4.0, wm=2.0, wl=1.0):
    """Ultimate Oscillator
    Larry Williams' (1976) signal, a momentum oscillator designed to capture momentum
    across three different timeframes.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator
    BP = Close - Minimum(Low or Prior Close).
    TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    Average7 = (7-period BP Sum) / (7-period TR Sum)
    Average14 = (14-period BP Sum) / (14-period TR Sum)
    Average28 = (28-period BP Sum) / (28-period TR Sum)
    UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        s(int): short period
        m(int): medium period
        l(int): long period
        ws(float): weight of short BP average for UO
        wm(float): weight of medium BP average for UO
        wl(float): weight of long BP average for UO
    Returns:
        pandas.Series: New feature generated.
    """
    min_l_or_pc = close.shift(1).combine(low, min)
    max_h_or_pc = close.shift(1).combine(high, max)

    bp = close - min_l_or_pc
    tr = max_h_or_pc - min_l_or_pc

    avg_s = bp.rolling(s).sum() / tr.rolling(s).sum()
    avg_m = bp.rolling(m).sum() / tr.rolling(m).sum()
    avg_l = bp.rolling(l).sum() / tr.rolling(l).sum()

    uo = 100.0 * ((ws * avg_s) + (wm * avg_m) + (wl * avg_l)) / (ws + wm + wl)
    uo = uo.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(uo, name='momentum_uo')


def momentum_stoch(
    high, low, close, volume,
    n=14):
    """Stochastic Oscillator
    Developed in the late 1950s by George Lane. The stochastic
    oscillator presents the location of the closing price of a
    stock in relation to the high and low range of the price
    of a stock over a period of time, typically a 14-day period.
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    smin = low.rolling(n).min()
    smax = high.rolling(n).max()
    stoch_k = 100 * (close - smin) / (smax - smin)

    stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(stoch_k, name='momentum_stoch')


def momentum_stoch_signal(
    high, low, close, volume,
    n=14, d_n=3):
    """Stochastic Oscillator Signal
    Shows SMA of Stochastic Oscillator. Typically a 3 day SMA.
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        d_n(int): sma period over stoch_k

    Returns:
        pandas.Series: New feature generated.
    """
    stoch_k = momentum_stoch(high, low, close, n)
    stoch_d = stoch_k.rolling(d_n).mean()

    stoch_d = stoch_d.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(stoch_d, name='momentum_stoch_signal')


def momentum_wr(
    high, low, close, volume,
    lbp=14):
    """Williams %R
    From: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r
    Developed by Larry Williams, Williams %R is a momentum indicator that is the inverse of the
    Fast Stochastic Oscillator. Also referred to as %R, Williams %R reflects the level of the close
    relative to the highest high for the look-back period. In contrast, the Stochastic Oscillator
    reflects the level of the close relative to the lowest low. %R corrects for the inversion by
    multiplying the raw value by -100. As a result, the Fast Stochastic Oscillator and Williams %R
    produce the exact same lines, only the scaling is different. Williams %R oscillates from 0 to -100.
    Readings from 0 to -20 are considered overbought. Readings from -80 to -100 are considered oversold.
    Unsurprisingly, signals derived from the Stochastic Oscillator are also applicable to Williams %R.
    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100
    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.
    From: https://www.investopedia.com/terms/w/williamsr.asp
    The Williams %R oscillates from 0 to -100. When the indicator produces readings from 0 to -20, this indicates
    overbought market conditions. When readings are -80 to -100, it indicates oversold market conditions.
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period
    Returns:
        pandas.Series: New feature generated.
    """

    hh = high.rolling(lbp).max() #highest high over lookback period lbp
    ll = low.rolling(lbp).min()  #lowest low over lookback period lbp

    wr = -100 * (hh - close) / (hh - ll)

    wr = wr.replace([np.inf, -np.inf], np.nan).fillna(-50)
    return pd.Series(wr, name='momentum_wr')


def momentum_ao(
    high, low, close, volume,
    s=5, l=34):
    """Awesome Oscillator
    From: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
    The Awesome Oscillator is an indicator used to measure market momentum. AO calculates the difference of a
    34 Period and 5 Period Simple Moving Averages. The Simple Moving Averages that are used are not calculated
    using closing price but rather each bar's midpoints. AO is generally used to affirm trends or to anticipate
    possible reversals.
    From: https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator
    Awesome Oscillator is a 34-period simple moving average, plotted through the central points of the bars (H+L)/2,
    and subtracted from the 5-period simple moving average, graphed across the central points of the bars (H+L)/2.
    MEDIAN PRICE = (HIGH+LOW)/2
    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)
    where
    SMA — Simple Moving Average.
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        s(int): short period
        l(int): long period
    Returns:
        pandas.Series: New feature generated.
    """

    mp = 0.5 * (high + low)
    ao = mp.rolling(s).mean() - mp.rolling(l).mean()

    ao = ao.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(ao, name='momentum_ao')

####################################################
####################################################
####################################################
#  Volume 8
####################################################
####################################################
####################################################


def volume_acc_dist_index(
    high, low, close, volume
    ):
    """Accumulation/Distribution Index (ADI)
    Acting as leading indicator of price movements.
    https://en.wikipedia.org/wiki/Accumulation/distribution_index
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.

    Returns:
        pandas.Series: New feature generated.
    """
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0.0) # float division by zero
    ad = clv * volume
    ad = ad + ad.shift(1)
    ad = ad.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(ad, name='volume_adi')


def volume_on_balance_volume(
    high, low, close, volume,
    ):
    """On-balance volume (OBV)
    It relates price and volume in the stock market. OBV is based on a
    cumulative total volume.
    https://en.wikipedia.org/wiki/On-balance_volume
    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    obv = df['OBV']
    obv = obv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(obv, name='volume_obv')


def volume_on_balance_volume_mean(
    high, low, close, volume,
    n=10):
    """On-balance volume mean (OBV mean)
    It's based on a cumulative total volume.
    https://en.wikipedia.org/wiki/On-balance_volume
    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    obv = df['OBV'].rolling(n).mean()
    obv = obv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(obv, name='volume_obvm')


def volume_chaikin_money_flow(
    high, low, close, volume,
    n=20):
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0) # float division by zero
    mfv *= volume
    cmf = mfv.rolling(n).sum() / volume.rolling(n).sum()
    cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(cmf, name='volume_cmf')


def volume_force_index(
    high, low, close, volume,
    n=2):
    """Force Index (FI)
    It illustrates how strong the actual buying or selling pressure is. High
    positive values mean there is a strong rising trend, and low values signify
    a strong downward trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    fi = close.diff(n) * volume.diff(n)
    fi = fi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(fi, name='volume_fi')


def volume_ease_of_movement(
    high, low, close, volume,
    n=20):
    """Ease of movement (EoM, EMV)
    It relate an asset's price change to its volume and is particularly useful
    for assessing the strength of a trend.
    https://en.wikipedia.org/wiki/Ease_of_movement
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    emv = (high.diff(1) + low.diff(1)) * (high - low) / (2 * volume)
    emv = emv.rolling(n).mean()
    emv = emv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(emv, name='volume_eom')


def volume_volume_price_trend(
    high, low, close, volume,
    ):
    """Volume-price trend (VPT)
    Is based on a running cumulative volume that adds or substracts a multiple
    of the percentage change in share price trend and current volume, depending
    upon the investment's upward or downward movements.
    https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend
    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    vpt = volume * ((close - close.shift(1)) / close.shift(1))
    vpt = vpt.shift(1) + vpt
    vpt = vpt.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(vpt, name='volume_vpt')


def volume_negative_volume_index(
    high, low, close, volume,
    ):
    """Negative Volume Index (NVI)
    From: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde
    The Negative Volume Index (NVI) is a cumulative indicator that uses the change in volume to decide when the
    smart money is active. Paul Dysart first developed this indicator in the 1930s. [...] Dysart's Negative Volume
    Index works under the assumption that the smart money is active on days when volume decreases and the not-so-smart
    money is active on days when volume increases.
    The cumulative NVI line was unchanged when volume increased from one period to the other. In other words,
    nothing was done. Norman Fosback, of Stock Market Logic, adjusted the indicator by substituting the percentage
    price change for Net Advances.
    This implementation is the Fosback version.
    If today's volume is less than yesterday's volume then:
        nvi(t) = nvi(t-1) * ( 1 + (close(t) - close(t-1)) / close(t-1) )
    Else
        nvi(t) = nvi(t-1)
    Please note: the "stockcharts.com" example calculation just adds the percentange change of price to previous
    NVI when volumes decline; other sources indicate that the same percentage of the previous NVI value should
    be added, which is what is implemented here.
    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
    Returns:
        pandas.Series: New feature generated.
    See also:
    https://en.wikipedia.org/wiki/Negative_volume_index
    """
    price_change = close.pct_change()
    vol_decrease = (volume.shift(1) > volume)

    nvi = pd.Series(data=np.nan, index=close.index, dtype='float64', name='nvi')

    nvi.iloc[0] = 1000
    for i in range(1,len(nvi)):
        if vol_decrease.iloc[i]:
            nvi.iloc[i] = nvi.iloc[i - 1] * (1.0 + price_change.iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]

    nvi = nvi.replace([np.inf, -np.inf], np.nan).fillna(1000) # IDEA: There shouldn't be any na; might be better to throw exception

    return pd.Series(nvi, name='volume_nvi')

####################################################
####################################################
####################################################
#  Volatility 15
####################################################
####################################################
####################################################


def volatility_average_true_range(
    high, low, close, volume,
    n=14):
    """Average True Range (ATR)
    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    tr = high.combine(cs, max) - low.combine(cs, min)

    atr = np.zeros(len(close))
    atr[0] = tr[1::].mean()
    for i in range(1, len(atr)):
        atr[i] = (atr[i-1] * (n-1) + tr.iloc[i]) / float(n)

    atr = pd.Series(data=atr, index=tr.index)

    atr = atr.replace([np.inf, -np.inf], np.nan).fillna(0)

    return pd.Series(atr, name='volatility_atr')


def volatility_bollinger_mavg(
    high, low, close, volume,
    n=20):
    """Bollinger Bands (BB)
    N-period simple moving average (MA).
    https://en.wikipedia.org/wiki/Bollinger_Bands
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n).mean()
    mavg = mavg.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(mavg, name='volatility_bbm').diff()


def volatility_bollinger_hband(
    high, low, close, volume,
    n=20, ndev=2):
    """Bollinger Bands (BB)
    Upper band at K times an N-period standard deviation above the moving
    average (MA + Kdeviation).
    https://en.wikipedia.org/wiki/Bollinger_Bands
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    hband = hband.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(hband, name='volatility_bbh').diff()


def volatility_bollinger_lband(
    high, low, close, volume,
    n=20, ndev=2):
    """Bollinger Bands (BB)
    Lower band at K times an N-period standard deviation below the moving
    average (MA − Kdeviation).
    https://en.wikipedia.org/wiki/Bollinger_Bands
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev * mstd
    lband = lband.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(lband, name='volatility_bbl').diff()


def volatility_bollinger_hband_indicator(
    high, low, close, volume,
    n=20, ndev=2):
    """Bollinger High Band Indicator
    Returns 1, if close is higher than bollinger high band. Else, return 0.
    https://en.wikipedia.org/wiki/Bollinger_Bands
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev * mstd
    df['hband'] = 0.0
    df.loc[close > hband, 'hband'] = 1.0
    hband = df['hband']
    hband = hband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(hband, name='volatility_bbhi')


def volatility_bollinger_lband_indicator(
    high, low, close, volume,
    n=20, ndev=2):
    """Bollinger Low Band Indicator
    Returns 1, if close is lower than bollinger low band. Else, return 0.
    https://en.wikipedia.org/wiki/Bollinger_Bands
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation
    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev * mstd
    df['lband'] = 0.0
    df.loc[close < lband, 'lband'] = 1.0
    lband = df['lband']
    lband = lband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(lband, name='volatility_bbli')


def volatility_keltner_channel_central(
    high, low, close, volume,
    n=10):
    """Keltner channel (KC)
    Showing a simple moving average line (central) of typical price.
    https://en.wikipedia.org/wiki/Keltner_channel
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    tp = (high + low + close) / 3.0
    tp = tp.rolling(n).mean()
    tp = tp.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(tp, name='volatility_kcc').diff()


def volatility_keltner_channel_hband(
    high, low, close, volume,
    n=10):
    """Keltner channel (KC)
    Showing a simple moving average line (high) of typical price.
    https://en.wikipedia.org/wiki/Keltner_channel
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    tp = ((4 * high) - (2 * low) + close) / 3.0
    tp = tp.rolling(n).mean()
    tp = tp.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(tp, name='volatility_kch').diff()


def volatility_keltner_channel_lband(
    high, low, close, volume,
    n=10):
    """Keltner channel (KC)
    Showing a simple moving average line (low) of typical price.
    https://en.wikipedia.org/wiki/Keltner_channel
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    tp = ((-2 * high) + (4 * low) + close) / 3.0
    tp = tp.rolling(n).mean()
    tp = tp.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(tp, name='volatility_kcl').diff()


def volatility_keltner_channel_hband_indicator(
    high, low, close, volume,
    n=10):
    """Keltner Channel High Band Indicator (KC)
    Returns 1, if close is higher than keltner high band channel. Else,
    return 0.
    https://en.wikipedia.org/wiki/Keltner_channel
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = ((4 * high) - (2 * low) + close) / 3.0
    df.loc[close > hband, 'hband'] = 1.0
    hband = df['hband']
    hband = hband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(hband, name='volatility_kchi')


def volatility_keltner_channel_lband_indicator(
    high, low, close, volume,
    n=10):
    """Keltner Channel Low Band Indicator (KC)
    Returns 1, if close is lower than keltner low band channel. Else, return 0.
    https://en.wikipedia.org/wiki/Keltner_channel
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = ((-2 * high) + (4 * low) + close) / 3.0
    df.loc[close < lband, 'lband'] = 1.0
    lband = df['lband']
    lband = lband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(lband, name='volatility_kcli')


def volatility_donchian_channel_hband(
    high, low, close, volume,
    n=20):
    """Donchian channel (DC)
    The upper band marks the highest price of an issue for n periods.
    https://www.investopedia.com/terms/d/donchianchannels.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    hband = close.rolling(n).max()
    hband = hband.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(hband, name='volatility_dch').diff()


def volatility_donchian_channel_lband(
    high, low, close, volume,
    n=20):
    """Donchian channel (DC)
    The lower band marks the lowest price for n periods.
    https://www.investopedia.com/terms/d/donchianchannels.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    lband = close.rolling(n).min()
    lband = lband.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(lband, name='volatility_dcl').diff()


def volatility_donchian_channel_hband_indicator(
    high, low, close, volume,
    n=20):
    """Donchian High Band Indicator
    Returns 1, if close is higher than donchian high band channel. Else,
    return 0.
    https://www.investopedia.com/terms/d/donchianchannels.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = close.rolling(n).max()
    df.loc[close >= hband, 'hband'] = 1.0
    hband = df['hband']
    hband = hband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(hband, name='volatility_dchi')


def volatility_donchian_channel_lband_indicator(
    high, low, close, volume,
    n=20):
    """Donchian Low Band Indicator
    Returns 1, if close is lower than donchian low band channel. Else, return 0.
    https://www.investopedia.com/terms/d/donchianchannels.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = close.rolling(n).min()
    df.loc[close <= lband, 'lband'] = 1.0
    lband = df['lband']
    lband = lband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(lband, name='volatility_dchl')

####################################################
####################################################
####################################################
#  Trends 25
####################################################
####################################################
####################################################


def trend_macd(
    high, low, close, volume,
    n_fast=12, n_slow=26):
    """Moving Average Convergence Divergence (MACD)
    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.
    https://en.wikipedia.org/wiki/MACD
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = ema(close, n_fast)
    emaslow = ema(close, n_slow)
    macd = emafast - emaslow
    
    macd = macd.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(macd, name='trend_macd')


def trend_macd_signal(
    high, low, close, volume,
    n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD Signal)
    Shows EMA of MACD.
    https://en.wikipedia.org/wiki/MACD
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = ema(close, n_fast)
    emaslow = ema(close, n_slow)
    macd = emafast - emaslow
    macd_signal = ema(macd, n_sign)
    
    macd_signal = macd_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(macd_signal, name='trend_macd_signal')


def trend_macd_diff(
    high, low, close, volume,
    n_fast=12, n_slow=26, n_sign=9):
    """Moving Average Convergence Divergence (MACD Diff)
    Shows the relationship between MACD and MACD Signal.
    https://en.wikipedia.org/wiki/MACD
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.
        n_slow(int): n period long-term.
        n_sign(int): n period to signal.

    Returns:
        pandas.Series: New feature generated.
    """
    emafast = ema(close, n_fast)
    emaslow = ema(close, n_slow)
    macd = emafast - emaslow
    macdsign = ema(macd, n_sign)
    macd_diff = macd - macdsign
    macd_diff = macd_diff.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(macd_diff, name='trend_macd_diff')


def trend_ema_fast(
    high, low, close, volume,
    n=12):
    """EMA
    Exponential Moving Average via Pandas
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.

    Returns:
        pandas.Series: New feature generated.
    """
    ema_ = ema(close, n)
    return pd.Series(ema_, name='trend_ema_fast').diff()


def trend_ema_slow(
    high, low, close, volume,
    n=26):
    """EMA
    Exponential Moving Average via Pandas
    Args:
        close(pandas.Series): dataset 'Close' column.
        n_fast(int): n period short-term.

    Returns:
        pandas.Series: New feature generated.
    """
    ema_ = ema(close, n)
    return pd.Series(ema_, name='trend_ema_slow').diff()


def trend_adx(
    high, low, close, volume,
    n=14):
    """Average Directional Movement Index (ADX)
    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to collectively
    as the Directional Movement Indicator (DMI).
    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.
    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    pdm = high.combine(cs, lambda x1, x2: max(x1, x2) if np.isnan(x1) == False and np.isnan(x2) == False else np.nan)
    pdn = low.combine(cs,  lambda x1, x2: min(x1, x2) if np.isnan(x1) == False and np.isnan(x2) == False else np.nan)
    tr = pdm - pdn

    trs_initial = np.zeros(n-1)
    trs = np.zeros(len(close) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1]/float(n)) + tr[n+i]

    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dip_mio = np.zeros(len(close) - (n - 1))
    dip_mio[0] = pos.dropna()[0:n].sum()

    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip_mio)-1):
        dip_mio[i] = dip_mio[i-1] - (dip_mio[i-1]/float(n)) + pos[n+i]

    din_mio = np.zeros(len(close) - (n - 1))
    din_mio[0] = neg.dropna()[0:n].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio)-1):
        din_mio[i] = din_mio[i-1] - (din_mio[i-1]/float(n)) + neg[n+i]

    dip = np.zeros(len(trs))
    for i in range(len(trs)):
        dip[i] = 100 * (dip_mio[i]/trs[i])

    din = np.zeros(len(trs))
    for i in range(len(trs)):
        din[i] = 100 * (din_mio[i]/trs[i])

    dx = 100 * np.abs((dip - din) / (dip + din))

    adx = np.zeros(len(trs))
    adx[n] = dx[0:n].mean()

    for i in range(n+1, len(adx)):
        adx[i] = ((adx[i-1] * (n - 1)) + dx[i-1]) / float(n)

    adx = np.concatenate((trs_initial, adx), axis=0)
    adx = pd.Series(data=adx, index=close.index)

    
    adx = adx.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(adx, name='trend_adx')


def trend_adx_pos(
    high, low, close, volume,
    n=14):
    """Average Directional Movement Index Positive (ADX)
    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to collectively
    as the Directional Movement Indicator (DMI).
    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.
    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    pdm = high.combine(cs, lambda x1, x2: max(x1, x2) if np.isnan(x1) == False and np.isnan(x2) == False else np.nan)
    pdn = low.combine(cs,  lambda x1, x2: min(x1, x2) if np.isnan(x1) == False and np.isnan(x2) == False else np.nan)
    tr = pdm - pdn

    trs = np.zeros(len(close) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1]/float(n)) + tr[n+i]

    up = high - high.shift(1)
    dn = low.shift(1) - low
    pos = abs(((up > dn) & (up > 0)) * up)

    dip_mio = np.zeros(len(close) - (n - 1))
    dip_mio[0] = pos.dropna()[0:n].sum()

    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip_mio)-1):
        dip_mio[i] = dip_mio[i-1] - (dip_mio[i-1]/float(n)) + pos[n+i]

    dip = np.zeros(len(close))
    for i in range(1, len(trs)-1):
        dip[i+n] = 100 * (dip_mio[i]/trs[i])

    dip = pd.Series(data=dip, index=close.index)

    
    dip = dip.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(dip, name='trend_adx_pos')


def trend_adx_neg(
    high, low, close, volume,
    n=14):
    """Average Directional Movement Index Negative (ADX)
    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to collectively
    as the Directional Movement Indicator (DMI).
    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.
    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    pdm = high.combine(cs, lambda x1, x2: max(x1, x2) if np.isnan(x1) == False and np.isnan(x2) == False else np.nan)
    pdn = low.combine(cs,  lambda x1, x2: min(x1, x2) if np.isnan(x1) == False and np.isnan(x2) == False else np.nan)
    tr = pdm - pdn

    trs = np.zeros(len(close) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1]/float(n)) + tr[n+i]

    up = high - high.shift(1)
    dn = low.shift(1) - low
    neg = abs(((dn > up) & (dn > 0)) * dn)

    din_mio = np.zeros(len(close) - (n - 1))
    din_mio[0] = neg.dropna()[0:n].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio)-1):
        din_mio[i] = din_mio[i-1] - (din_mio[i-1]/float(n)) + neg[n+i]

    din = np.zeros(len(close))
    for i in range(1, len(trs)-1):
        din[i+n] = 100 * (din_mio[i]/float(trs[i]))

    din = pd.Series(data=din, index=close.index)

    
    din = din.replace([np.inf, -np.inf], np.nan).fillna(20)
    return pd.Series(din, name='trend_adx_neg')


def trend_vortex_indicator_pos(
    high, low, close, volume,
    n=14):
    """Vortex Indicator (VI)
    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))

    vip = vmp.rolling(n).sum() / trn
    
    vip = vip.replace([np.inf, -np.inf], np.nan).fillna(1)
    return pd.Series(vip, name='trend_vortex_ind_pos')


def trend_vortex_indicator_neg(
    high, low, close, volume,
    n=14):
    """Vortex Indicator (VI)
    It consists of two oscillators that capture positive and negative trend
    movement. A bearish signal triggers when the negative trend indicator
    crosses above the positive trend indicator or a key level.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmm = np.abs(low - high.shift(1))

    vin = vmm.rolling(n).sum() / trn
    
    vin = vin.replace([np.inf, -np.inf], np.nan).fillna(1)
    return pd.Series(vin, name='trend_vortex_ind_neg')


def trend_vortex_diff(
    high, low, close, volume,
    n=14):
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))

    vin = vmm.rolling(n).sum() / trn
    vin = vin.replace([np.inf, -np.inf], np.nan).fillna(1)
    vin = pd.Series(vin, name='trend_vortex_ind_neg')
    tr = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    trn = tr.rolling(n).sum()

    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))

    vip = vmp.rolling(n).sum() / trn
    vip = vip.replace([np.inf, -np.inf], np.nan).fillna(1)
    vip = pd.Series(vip, name='trend_vortex_ind_pos')
    return vip.sub(vin).abs().rename('trend_vortex_diff')


def trend_trix(
    high, low, close, volume,
    n=15):
    """Trix (TRIX)
    Shows the percent rate of change of a triple exponentially smoothed moving
    average.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    ema1 = ema(close, n)
    ema2 = ema(ema1, n)
    ema3 = ema(ema2, n)
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix *= 100
    
    trix = trix.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(trix, name='trend_trix')


def trend_mass_index(
    high, low, close, volume,
    n=9, n2=25):
    """Mass Index (MI)
    It uses the high-low range to identify trend reversals based on range
    expansions. It identifies range bulges that can foreshadow a reversal of the
    current trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n(int): n low period.
        n2(int): n high period.

    Returns:
        pandas.Series: New feature generated.
    """
    amplitude = high - low
    ema1 = ema(amplitude, n)
    ema2 = ema(ema1, n)
    mass = ema1 / ema2
    mass = mass.rolling(n2).sum()
    
    mass = mass.replace([np.inf, -np.inf], np.nan).fillna(n2)
    return pd.Series(mass, name='trend_mass_index')


def trend_cci(
    high, low, close, volume,
    n=20, c=0.015):
    """Commodity Channel Index (CCI)
    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        c(int): constant.

    Returns:
        pandas.Series: New feature generated.
    """
    pp = (high + low + close) / 3.0
    cci = (pp - pp.rolling(n).mean()) / (c * pp.rolling(n).std())
    
    cci = cci.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(cci, name='trend_cci')


def trend_dpo(
    high, low, close, volume,
    n=20):
    """Detrended Price Oscillator (DPO)
    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    dpo = close.shift(int((0.5 * n) + 1)) - close.rolling(n).mean()
    
    dpo = dpo.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(dpo, name='trend_dpo')


def trend_kst(
    high, low, close, volume,
    r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15):
    """KST Oscillator (KST)
    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.
    https://en.wikipedia.org/wiki/KST_oscillator
    Args:
        close(pandas.Series): dataset 'Close' column.
        r1(int): r1 period.
        r2(int): r2 period.
        r3(int): r3 period.
        r4(int): r4 period.
        n1(int): n1 smoothed period.
        n2(int): n2 smoothed period.
        n3(int): n3 smoothed period.
        n4(int): n4 smoothed period.

    Returns:
        pandas.Series: New feature generated.
    """
    rocma1 = ((close - close.shift(r1)) / close.shift(r1)).rolling(n1).mean()
    rocma2 = ((close - close.shift(r2)) / close.shift(r2)).rolling(n2).mean()
    rocma3 = ((close - close.shift(r3)) / close.shift(r3)).rolling(n3).mean()
    rocma4 = ((close - close.shift(r4)) / close.shift(r4)).rolling(n4).mean()
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    
    kst = kst.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(kst, name='trend_kst')


def trend_kst_sig(
    high, low, close, volume,
    r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9):
    """KST Oscillator (KST Signal)
    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst
    Args:
        close(pandas.Series): dataset 'Close' column.
        r1(int): r1 period.
        r2(int): r2 period.
        r3(int): r3 period.
        r4(int): r4 period.
        n1(int): n1 smoothed period.
        n2(int): n2 smoothed period.
        n3(int): n3 smoothed period.
        n4(int): n4 smoothed period.
        nsig(int): n period to signal.

    Returns:
        pandas.Series: New feature generated.
    """
    rocma1 = ((close - close.shift(r1)) / close.shift(r1)).rolling(n1).mean()
    rocma2 = ((close - close.shift(r2)) / close.shift(r2)).rolling(n2).mean()
    rocma3 = ((close - close.shift(r3)) / close.shift(r3)).rolling(n3).mean()
    rocma4 = ((close - close.shift(r4)) / close.shift(r4)).rolling(n4).mean()
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    kst_sig = kst.rolling(nsig).mean()
    
    kst_sig = kst_sig.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(kst_sig, name='trend_kst_sig')


def trend_kst_diff(
    high, low, close, volume,
    r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, nsig=9):
    rocma1 = ((close - close.shift(r1)) / close.shift(r1)).rolling(n1).mean()
    rocma2 = ((close - close.shift(r2)) / close.shift(r2)).rolling(n2).mean()
    rocma3 = ((close - close.shift(r3)) / close.shift(r3)).rolling(n3).mean()
    rocma4 = ((close - close.shift(r4)) / close.shift(r4)).rolling(n4).mean()
    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    kst = kst.replace([np.inf, -np.inf], np.nan).fillna(0)
    kst_sig = kst.rolling(nsig).mean()
    kst_sig = kst_sig.replace([np.inf, -np.inf], np.nan).fillna(0)
    return kst.sub(kst_sig).rename('trend_kst_diff')


def trend_ichimoku_a(
    high, low, close, volume,
    n1=9, n2=26):
    """Ichimoku Kinkō Hyō (Ichimoku)
    It identifies the trend and look for potential signals within that trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n1(int): n1 low period.
        n2(int): n2 medium period.

    Returns:
        pandas.Series: New feature generated.
    """
    conv = 0.5 * (high.rolling(n1).max() + low.rolling(n1).min())
    base = 0.5 * (high.rolling(n2).max() + low.rolling(n2).min())

    spana = 0.5 * (conv + base)
    spana = spana.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

    return pd.Series(spana, name='trend_ichimoku_a').diff()


def trend_ichimoku_b(
    high, low, close, volume,
    n2=26, n3=52):
    """Ichimoku Kinkō Hyō (Ichimoku)
    It identifies the trend and look for potential signals within that trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n2(int): n2 medium period.
        n3(int): n3 high period.

    Returns:
        pandas.Series: New feature generated.
    """
    spanb = 0.5 * (high.rolling(n3).max() + low.rolling(n3).min())

    spanb = spanb.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

    return pd.Series(spanb, name='trend_ichimoku_b').diff()


def trend_visual_ichimoku_a(
    high, low, close, volume,
    n1=9, n2=26):
    """Ichimoku Kinkō Hyō (Ichimoku)
    It identifies the trend and look for potential signals within that trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n1(int): n1 low period.
        n2(int): n2 medium period.

    Returns:
        pandas.Series: New feature generated.
    """
    conv = 0.5 * (high.rolling(n1).max() + low.rolling(n1).min())
    base = 0.5 * (high.rolling(n2).max() + low.rolling(n2).min())

    spana = 0.5 * (conv + base)

    spana = spana.shift(n2)
    spana = spana.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

    return pd.Series(spana, name='trend_visual_ichimoku_a').diff()


def trend_visual_ichimoku_b(
    high, low, close, volume,
    n2=26, n3=52):
    """Ichimoku Kinkō Hyō (Ichimoku)
    It identifies the trend and look for potential signals within that trend.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        n2(int): n2 medium period.
        n3(int): n3 high period.

    Returns:
        pandas.Series: New feature generated.
    """
    spanb = 0.5 * (high.rolling(n3).max() + low.rolling(n3).min())

    spanb = spanb.shift(n2)
    spanb = spanb.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')

    return pd.Series(spanb, name='trend_visual_ichimoku_b').diff()


def trend_aroon_up(
    high, low, close, volume,
    n=25):
    """Aroon Indicator (AI)
    Identify when trends are likely to change direction (uptrend).
    Aroon Up - ((N - Days Since N-day High) / N) x 100
    https://www.investopedia.com/terms/a/aroon.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    aroon_up = close.rolling(n).apply(lambda x: float(np.argmax(x) + 1) / n * 100, raw=True)
    
    aroon_up = aroon_up.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(aroon_up, name='trend_aroon_up')


def trend_aroon_down(
    high, low, close, volume,
    n=25):
    """Aroon Indicator (AI)
    Identify when trends are likely to change direction (downtrend).
    Aroon Down - ((N - Days Since N-day Low) / N) x 100
    https://www.investopedia.com/terms/a/aroon.asp
    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    aroon_down = close.rolling(n).apply(lambda x: float(np.argmin(x) + 1) / n * 100, raw=True)
    
    aroon_down = aroon_down.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(aroon_down, name='trend_aroon_down')


def trend_aroon_ind(
    high, low, close, volume,
    n=25):
    aroon_up = close.rolling(n).apply(lambda x: float(np.argmax(x) + 1) / n * 100, raw=True)
    aroon_up = aroon_up.replace([np.inf, -np.inf], np.nan).fillna(0)
    aroon_up = pd.Series(aroon_up, name='trend_aroon_up')
    aroon_down = close.rolling(n).apply(lambda x: float(np.argmin(x) + 1) / n * 100, raw=True)
    aroon_down = aroon_down.replace([np.inf, -np.inf], np.nan).fillna(0)
    aroon_down = pd.Series(aroon_down, name='trend_aroon_down')
    return aroon_up.sub(aroon_down).rename('trend_aroon_ind')

####################################################
####################################################
####################################################
#  Others 2
####################################################
####################################################
####################################################


def others_daily_return(
    high, low, close, volume,
    ):
    """Daily Return (DR)
    Args:
        close(pandas.Series): dataset 'Close' column.
    Returns:
        pandas.Series: New feature generated.
    """
    dr = (close / close.shift(1)) - 1
    dr *= 100
    dr = dr.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(dr, name='others_dr')


def others_daily_log_return(
    high, low, close, volume,
    ):
    """Daily Log Return (DLR)
    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe
    Args:
        close(pandas.Series): dataset 'Close' column.
    Returns:
        pandas.Series: New feature generated.
    """
    dr = np.log(close).diff()
    dr *= 100
    dr = dr.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(dr, name='others_dlr')

####################################################
####################################################
####################################################
#  Aggregators
####################################################
####################################################
####################################################

function_lookup = {
    'momentum_rsi': momentum_rsi,
    'momentum_mfi': momentum_mfi,
    'momentum_tsi': momentum_tsi,
    'momentum_uo': momentum_uo,
    'momentum_stoch': momentum_stoch,
    'momentum_stoch_signal': momentum_stoch_signal,
    'momentum_wr': momentum_wr,
    'momentum_ao': momentum_ao,
    'volume_adi': volume_acc_dist_index,
    'volume_obv': volume_on_balance_volume,
    'volume_obvm': volume_on_balance_volume_mean,
    'volume_cmf': volume_chaikin_money_flow,
    'volume_fi': volume_force_index,
    'volume_eom': volume_ease_of_movement,
    'volume_vpt': volume_volume_price_trend,
    'volume_nvi': volume_negative_volume_index,
    'volatility_atr': volatility_average_true_range,
    'volatility_bbm': volatility_bollinger_mavg,
    'volatility_bbh': volatility_bollinger_hband,
    'volatility_bbl': volatility_bollinger_lband,
    'volatility_bbhi': volatility_bollinger_hband_indicator,
    'volatility_bbli': volatility_bollinger_lband_indicator,
    'volatility_kcc': volatility_keltner_channel_central,
    'volatility_kch': volatility_keltner_channel_hband,
    'volatility_kcl': volatility_keltner_channel_lband,
    'volatility_kchi': volatility_keltner_channel_hband_indicator,
    'volatility_kcli': volatility_keltner_channel_lband_indicator,
    'volatility_dch': volatility_donchian_channel_hband,
    'volatility_dcl': volatility_donchian_channel_lband,
    'volatility_dchi': volatility_donchian_channel_hband_indicator,
    'volatility_dchl': volatility_donchian_channel_lband_indicator,
    'trend_macd': trend_macd,
    'trend_macd_signal': trend_macd_signal,
    'trend_macd_diff': trend_macd_diff,
    'trend_ema_fast': trend_ema_fast,
    'trend_ema_slow': trend_ema_slow,
    'trend_adx': trend_adx,
    'trend_adx_pos': trend_adx_pos,
    'trend_adx_neg': trend_adx_neg,
    'trend_vortex_ind_pos': trend_vortex_indicator_pos,
    'trend_vortex_ind_neg': trend_vortex_indicator_neg,
    'trend_vortex_diff': trend_vortex_diff,
    'trend_trix': trend_trix,
    'trend_mass_index': trend_mass_index,
    'trend_cci': trend_cci,
    'trend_dpo': trend_dpo,
    'trend_kst': trend_kst,
    'trend_kst_sig': trend_kst_sig,
    'trend_kst_diff': trend_kst_diff,
    'trend_ichimoku_a': trend_ichimoku_a,
    'trend_ichimoku_b': trend_ichimoku_b,
    'trend_visual_ichimoku_a': trend_visual_ichimoku_a,
    'trend_visual_ichimoku_b': trend_visual_ichimoku_b,
    'trend_aroon_up': trend_aroon_up,
    'trend_aroon_down': trend_aroon_down,
    'trend_aroon_ind': trend_aroon_ind,
    'others_dr': others_daily_return,
    'others_dlr': others_daily_log_return
}

def get_all_ta_cols():
    return function_lookup.keys()

def get_function_lookup():
    return function_lookup

ta_cols = list(get_all_ta_cols())

def gen_ti(df, function_name_list, high_col='High', low_col='Low', close_col='Close', volume_col=None):
    if function_name_list == 'all':
        function_name_list = ta_cols
    elif function_name_list == 'fx':
        function_name_list = [col for col in ta_cols if (col[:6]!='volume')&(col not in ['momentum_mfi'])]
    elif type(function_name_list) == list:
        assert pd.Series(function_name_list).isin(ta_cols).all()
    else:
        raise ValueError("Either specify value subset, or pass 'fx' or 'all'.")
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    if not volume_col:
        job = [d_function_lookup[function](high.copy(), low.copy(), close.copy(), volume=None)
        for function in function_name_list]
    else:
        volume = df[volume_col]
        job = [d_function_lookup[function](high.copy(), low.copy(), close.copy(), volume.copy()).astype
        for function in function_name_list]
    return job
