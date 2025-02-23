import logging
import random
import re
import time
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
from datetime import datetime, timedelta

EXPECTED_COLUMNS = {'Open': 0, 'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4}
COLUMN_NAMES = list(EXPECTED_COLUMNS.keys())  # Maintain explicit order


def configure_pandas():
    """Configure pandas display settings for wide terminal output"""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

def symbol_to_file_name(symbol, ext='.csv.xz', replacement_text='X'):
    """
    Generates a sanitized filename from a stock symbol.

    Args:
        symbol (str): The stock symbol.
        ext (str, optional): The file extension. Defaults to '.csv.xz'.
        replacement_text (str, optional): Text to replace invalid characters. Defaults to 'X'.

    Returns:
        str: The sanitized filename.
    """
    return sanitize_filename(
        symbol.replace('/', replacement_text).replace('^', replacement_text).replace('=', replacement_text) + ext,
        replacement_text=replacement_text)

# Helper functions
def normalize_column(raw_col) -> str:
    """Extract base column name with priority to expected columns"""
    if isinstance(raw_col, tuple):
        raw_col = raw_col[0]

    # Split into parts using non-alphanumeric characters
    parts = re.split(r'[^A-Za-z0-9]', str(raw_col))

    # Check each part for expected columns
    for part in parts:
        part_norm = part.capitalize()
        if part_norm in EXPECTED_COLUMNS:
            return part_norm

    # Fallback to first part
    return parts[0].capitalize() if parts else raw_col.capitalize()

def make_unique(base: str, seen: set) -> str:
    """Generate unique column name"""
    unique = base
    cnt = 1
    while unique in seen:
        unique = f"{base}_{cnt}"
        cnt += 1
    seen.add(unique)
    return unique


def reindex_columns(df: pd.DataFrame, column_map: dict[str, int]) -> pd.DataFrame:
    """Normalize columns with deduplication"""
    normalized = []
    seen = set()

    # First pass: collect normalized names
    for raw_col in df.columns:
        base_col = normalize_column(raw_col)  # Your existing normalization logic
        unique_col = make_unique(base_col, seen)  # Deduplication logic
        normalized.append(unique_col)

    df.columns = normalized

    # Verify required columns exist
    missing = set(column_map) - set(normalized)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df.reindex(columns=column_map.keys()).rename(
        columns=lambda x: x if x in column_map else None
    ).dropna(axis=1, how="all")


@njit
def merge_ohlcav(left: np.ndarray, right: np.ndarray,
                 left_time: np.ndarray, right_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge two OHLCV datasets using volume-based conflict resolution.

    Args:
        left: Numpy array of shape (n1, 5) containing [Open, High, Low, Close, Volume]
        right: Numpy array of shape (n2, 5) containing same columns
        left_time: Numpy array of timestamps for left data
        right_time: Numpy array of timestamps for right data

    Returns:
        tuple: (merged_data, merged_timestamps)
    """
    # Column indices from EXPECTED_COLUMNS
    OPEN_IDX = 0
    HIGH_IDX = 1
    LOW_IDX = 2
    CLOSE_IDX = 3
    VOLUME_IDX = 4

    # Initialize pointers and result arrays
    left_idx = 0
    right_idx = 0
    n_left = len(left)
    n_right = len(right)

    # Pre-allocate arrays for maximum possible size
    merged = np.empty((n_left + n_right, left.shape[1]), dtype=left.dtype)
    merged_times = np.empty(n_left + n_right, dtype=left_time.dtype)

    # Use maximum possible timestamp value for comparison
    max_ts = max(np.iinfo(left_time.dtype).max, np.iinfo(right_time.dtype).max)

    insert_idx = 0

    while left_idx < n_left or right_idx < n_right:
        # Get current timestamps
        left_ts = left_time[left_idx] if left_idx < n_left else max_ts
        right_ts = right_time[right_idx] if right_idx < n_right else max_ts

        if left_ts == right_ts:
            if left_ts == max_ts:
                break  # Both arrays exhausted

            # Conflict resolution for same timestamp
            left_vol = left[left_idx, VOLUME_IDX]
            right_vol = right[right_idx, VOLUME_IDX]

            # Prefer higher volume, with tiebreaker using previous volume
            if right_vol >= left_vol:
                # Check if right's volume changed from previous
                use_right = True
                if right_idx > 0 and right[right_idx - 1, VOLUME_IDX] == right_vol:
                    use_right = False
            else:
                # Check if left's volume stayed the same
                use_left = True
                if left_idx > 0 and left[left_idx - 1, VOLUME_IDX] != left_vol:
                    use_left = False
                use_right = not use_left

            if use_right:
                merged[insert_idx] = right[right_idx]
            else:
                merged[insert_idx] = left[left_idx]

            # Merge high/low values if both have valid volumes
            if left_vol > 0 and right_vol > 0:
                merged[insert_idx, HIGH_IDX] = max(
                    left[left_idx, HIGH_IDX],
                    right[right_idx, HIGH_IDX]
                )
                merged[insert_idx, LOW_IDX] = min(
                    left[left_idx, LOW_IDX],
                    right[right_idx, LOW_IDX]
                )

            merged_times[insert_idx] = left_ts
            insert_idx += 1

            # Advance both pointers
            left_idx += 1
            right_idx += 1

        elif left_ts < right_ts:
            # Take from left array
            merged[insert_idx] = left[left_idx]
            merged_times[insert_idx] = left_ts
            insert_idx += 1
            left_idx += 1
        else:
            # Take from right array
            merged[insert_idx] = right[right_idx]
            merged_times[insert_idx] = right_ts
            insert_idx += 1
            right_idx += 1

    # Trim unused pre-allocated space
    return merged[:insert_idx], merged_times[:insert_idx]


def download(symbol: str, start: datetime = None, end: datetime = None,
             reindex: bool = True) -> pd.DataFrame:
    """
    Download OHLCV data with robustness improvements.

    Key changes:
    - Explicitly set repair=True to handle Yahoo's price discrepancies
    - Add retry logic for transient network errors
    """
    for attempt in range(3):
        try:
            ohlc = yf.download(
                symbol, start=start, end=end, repair=True,
                prepost=True, interval="1m", ignore_tz=False,
                threads=False, timeout=30, progress=False
            )
            if ohlc.empty:
                logging.warning(f"No data for {symbol}")
                return pd.DataFrame(columns=COLUMN_NAMES)
            # Convert timestamps to UTC microseconds
            ohlc.index = pd.to_datetime(ohlc.index).astype(np.int64) // 10 ** 3
            return reindex_columns(ohlc, EXPECTED_COLUMNS) if reindex else ohlc
        except Exception as e:
            if "duplicate labels" in str(e) or "dictionary changed size during iteration" in str(e):
                yf.pdr_override = False
                yf.Ticker(symbol)._history = None  # Clear cache
            if attempt == 2:
                raise
            logging.warning(f"Retry {attempt + 1}/3 for {symbol}: {str(e)}")
            time.sleep(2 ** attempt)


def combine(symbol: str) -> pd.DataFrame:
    """
    Combine historical and recent data with improved time range handling.

    Fixes:
    - Properly handle time intervals to avoid gaps
    - Add type hints for better IDE support
    - Simplify column management using constants
    """
    # Initialize with existing data
    file_path = Path(symbol_to_file_name(symbol))
    if file_path.exists():
        hist_data = pd.read_csv(file_path, dtype=np.float64, index_col="time")
        hist_data.index = hist_data.index.astype(np.int64)
        hist_data = reindex_columns(hist_data, EXPECTED_COLUMNS)
    else:
        hist_data = None

    # Download in 7-day chunks (Yahoo's 1m data limit)
    start_date = datetime.now() - timedelta(days=28)
    for _ in range(4):  # 4 weeks = 28 days
        end_date = start_date + timedelta(days=7)
        new_data = download(symbol, start_date, end_date)
        hist_data = merge_data(hist_data, new_data)
        start_date = end_date

    # Add latest data and return
    return merge_data(hist_data, download(symbol))


def merge_data(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Wrapper for merge_ohlcav with proper column handling"""
    if existing is None or len(existing) == 0:
        return new.copy() if not new.empty else pd.DataFrame(columns=COLUMN_NAMES)

    if new.empty:
        return existing.copy()
    merged, times = merge_ohlcav(
        existing.to_numpy(), new.to_numpy(),
        existing.index.to_numpy('int64'), new.index.to_numpy('int64')
    )
    return pd.DataFrame(merged, index=times, columns=COLUMN_NAMES)

def process(symbol: str):
    """
    Downloads, combines, and saves OHLCV data for a stock symbol.

    This function orchestrates the entire process of downloading, combining, and saving
    OHLCV data for a specified stock symbol. Caller may utilizes multithreading for efficiency.

    Args:
        symbol (str): The stock symbol.

    Returns:
        str: The processed stock symbol.
    """
    file_name = symbol_to_file_name(symbol)
    df = combine(symbol)
    if not df.empty:
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
