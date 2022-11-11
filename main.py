import datetime
import logging
import random
from pathlib import Path

try:
    from numba import njit
except ImportError:
    def njit(f):
        return f
import numpy as np
import pandas as pd
import yfinance as yf
from pathvalidate import sanitize_filename
from concurrent.futures import ThreadPoolExecutor, wait, Future


# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def configure_pandas():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


@njit()
def merge_ohlcav(left, right, left_time, right_time):
    l = 0
    r = 0
    left_size = len(left)
    right_size = len(right)
    res = np.zeros((right_size + left_size, left.shape[1]), dtype=left.dtype)
    times = np.zeros(len(res), dtype=left_time.dtype)
    max_time = max(np.iinfo(left_time.dtype).max, np.iinfo(right_time.dtype).max)
    i = 0
    for i in range(res.shape[0]):
        if l >= left_size:
            lt = max_time
        else:
            lt = left_time[l]

        rt = right_time[r] if r < right_size else max_time
        if lt == rt:
            if lt == max_time:
                break
            volume_right = right[r, 5]
            volume_left = left[l, 5]
            prefer_right = volume_right >= volume_left  # select best volume provider
            if prefer_right:
                if r > 0:
                    prefer_right = right[r - 1, 5] != volume_right
            else:
                if l > 0:
                    prefer_right = left[l - 1, 5] == volume_left
            if prefer_right:
                res[i, :] = right[r, :]
            else:
                res[i, :] = left[l, :]
            if volume_right > 0 and volume_left > 0:
                res[i, 1] = max(right[r, 1], left[l, 1])  # high
                res[i, 2] = min(right[r, 2], left[l, 2])  # low

            times[i] = lt

            next_l = l + 1
            next_r = r + 1

            lt = left_time[next_l] if next_l < left_size else max_time
            rt = right_time[next_r] if next_r < right_size else max_time

            if lt < rt:
                l += 1
                if rt < max_time:
                    r = np.searchsorted(right_time[r:], lt, 'right') + r
            elif rt < lt:
                r += 1
                if lt < max_time:
                    l = np.searchsorted(left_time[l:], rt, 'right') + l
            else:
                l += 1
                r += 1

        elif lt < rt:
            res[i, :] = left[l, :]
            times[i] = lt
            l += 1
        else:
            res[i, :] = right[r, :]
            times[i] = rt
            r += 1

    return res[:i], times[:i]


def download(symbol: str, start=None, end=None):
    ohlc = yf.download(symbol,
                       start=start,
                       end=end,
                       prepost=True,
                       interval="1m",
                       ignore_tz=False,
                       threads=False,
                       timeout=30,
                       progress=False)
    ohlc.index = pd.to_datetime(ohlc.index).astype(np.int64) // 10 ** 3
    return ohlc


def combine(symbol):
    file_name = sanitize_filename(symbol.replace('/', '_') + '.csv.xz', replacement_text='_')
    prev = None
    columns = None
    if Path(file_name).exists():
        prev = pd.read_csv(file_name, dtype=np.float64, index_col=["time"])
        prev.index = prev.index.to_numpy(np.int64)
        columns = list(prev.columns)

    start = datetime.datetime.now() - datetime.timedelta(days=28)

    def merge(left, right):
        merged = merge_ohlcav(left.to_numpy(),
                              right.to_numpy(),
                              left.index.to_numpy('int64'),
                              right.index.to_numpy('int64'))
        return pd.DataFrame(*merged, columns=columns or None)

    if prev is None or datetime.datetime.now().timestamp() - prev.index.to_numpy()[-1] // 1000 > 6 * 24 * 60 * 60:
        current = download(symbol, start, start := start + datetime.timedelta(days=6))
        if prev is not None:
            prev = merge(prev, current)
        else:
            prev = current
            columns = list(current.columns)
        for i in range(1, 4):
            prev = merge(prev, download(symbol, start, start := start + datetime.timedelta(days=6)))
    current = download(symbol)

    return merge(prev, current)


def process(symbol: str):
    file_name = sanitize_filename(symbol.replace('/', '_') + '.csv.xz', replacement_text='_')
    df = combine(symbol)
    df.to_csv(file_name, index_label='time')
    return symbol


if __name__ == '__main__':
    configure_pandas()
    with open('symbols.txt') as f:
        symbols = set(l.strip() for l in f.readlines() if l and not l.startswith('#'))
    symbols = list(symbols)
    random.shuffle(symbols)
    with ThreadPoolExecutor() as pool:
        done, _ = wait([pool.submit(process, symbol) for symbol in symbols])
        fut: Future
        for fut in done:
            try:
                fut.result()
            except Exception:
                logging.warning("", exc_info=True)
