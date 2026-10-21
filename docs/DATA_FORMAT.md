# Bar file format

wfo-gpu reads bars from a flat binary file that it memory-maps. The layout is fixed so that loading costs one `mmap` and no parsing, and so that files written on one machine load on any other.

All integers are little-endian. Floating-point values are IEEE 754 binary32. There is no compression and no padding beyond what is listed.

## Layout

```
offset 0                 64                64 + 32 * bar_count
       +-----------------+-----------------+-----------------+----
       | FileHeader (64) | Bar 0 (32)      | Bar 1 (32)      | ...
       +-----------------+-----------------+-----------------+----
```

### FileHeader (64 bytes)

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 4 | `char[4]` | `magic` | `WFBT` |
| 4 | 4 | `uint32` | `version` | `1` |
| 8 | 4 | `uint32` | `bar_count` | number of bar records that follow |
| 12 | 8 | `char[8]` | `symbol` | instrument symbol, NUL-padded |
| 20 | 4 | `uint32` | `timeframe_sec` | bar length in seconds |
| 24 | 8 | `int64` | `start_ts` | timestamp of the first bar |
| 32 | 8 | `int64` | `end_ts` | timestamp of the last bar |
| 40 | 24 | `char[24]` | `reserved` | zero |

### Bar (32 bytes)

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 8 | `int64` | `timestamp` | bar **open** time, nanoseconds since the Unix epoch, UTC |
| 8 | 4 | `float` | `open` | |
| 12 | 4 | `float` | `high` | |
| 16 | 4 | `float` | `low` | |
| 20 | 4 | `float` | `close` | |
| 24 | 4 | `uint32` | `volume` | |
| 28 | 4 | `uint32` | `reserved` | zero |

The C++ definitions are `FileHeader` and `Bar` in `src/market_data.h`. `static_assert`s pin both sizes.

## Rules the loader enforces

`MarketData::load` rejects a file when:

- the magic is not `WFBT` or the version is not `1`;
- the file is shorter than `64 + 32 * bar_count` bytes;
- `bar_count` is zero or implausibly large (above 500 million);
- `timeframe_sec` is zero;
- timestamps are not strictly increasing, or two bars are closer together than `timeframe_sec`.

The header's `timeframe_sec` decides when a higher-timeframe bar is complete and which bar is the last before the daily close, so it must not overstate the bar length.

Prices are used as given. Back-adjust continuous futures, remove bad ticks and decide how to treat missing bars before writing the file. Gaps are allowed: the session index records the gap to the previous bar for every bar, and long gaps inside core hours mark a halt from that point to the end of the session (see [METHODOLOGY.md](METHODOLOGY.md)).

## Timestamps and sessions

Timestamps are UTC bar-open times. The session model chosen at load time (`[session] model`) converts them to the market's local clock and groups bars into session days:

- `us_futures`: US Eastern time with DST. A session runs from 18:00 to 17:00 the next day, and is named by the date of its close.
- `continuous`: UTC; each UTC calendar day is a session.

The bar file itself carries no session information, so the same file can be studied under either model.

## Creating bar files

From CSV (`timestamp,open,high,low,close,volume`, header optional, columns in any order when a header is present):

```sh
wfo-gpu convert --csv bars.csv --out data/bars.bin --symbol ES
```

Timestamps may be epoch seconds, milliseconds, microseconds or nanoseconds (detected by magnitude), or ISO 8601 such as `2024-01-02T14:30:00Z` or `2024-01-02 09:30:00-05:00`. The converter infers the timeframe from the most common spacing between bars; override it with `--timeframe`. It rejects non-increasing timestamps, non-positive or non-finite prices, highs below the open or close, lows above them, and negative volume, and names the offending line.

Synthetic data with a known structure:

```sh
wfo-gpu generate --out data/synthetic.bin --days 260 --seed 7
wfo-gpu generate --out data/trending.bin  --days 260 --trend 0.03
```

From code, `write_bar_file` in `src/bar_io.h` writes the format with a crash-safe atomic replace.

## Reading the format elsewhere

The layout maps directly onto a NumPy structured dtype:

```python
import numpy as np

header = np.dtype([("magic", "S4"), ("version", "<u4"), ("bar_count", "<u4"),
                   ("symbol", "S8"), ("timeframe_sec", "<u4"),
                   ("start_ts", "<i8"), ("end_ts", "<i8"), ("reserved", "V24")])
bar = np.dtype([("timestamp", "<i8"), ("open", "<f4"), ("high", "<f4"),
                ("low", "<f4"), ("close", "<f4"), ("volume", "<u4"), ("reserved", "<u4")])

with open("data/synthetic.bin", "rb") as f:
    h = np.frombuffer(f.read(header.itemsize), header)[0]
    bars = np.fromfile(f, bar, count=h["bar_count"])
```

`tools/gen_reference_indicators.py` uses exactly this to build the cross-validation reference data.
