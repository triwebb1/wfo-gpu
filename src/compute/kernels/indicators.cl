// =============================================================================
// OpenCL Indicator Compute Kernels -- Batched Dispatch
//
// Each kernel is launched with global_size = N (number of indicators of that
// type).  Each work-item computes one indicator series using parameters from
// a packed params buffer and writes to a strided region of the output buffer.
//
// Work-item i:
//   - Reads params from params[i]
//   - Reads shared bar data from close[0..num_bars-1] (or high/low/close)
//   - Writes output to output[i * num_bars .. (i+1) * num_bars - 1]
//
// Data layout (SoA):
//   close[0..N-1]  - contiguous float array of close prices
//   high[0..N-1]   - contiguous float array of high prices
//   low[0..N-1]    - contiguous float array of low prices
//   output[i * num_bars + 0 .. i * num_bars + N-1] - output for work-item i
//
// This file is kept in sync with the embedded source in cl_indicators.cpp.
// =============================================================================

#ifndef NO_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// ---------------------------------------------------------------------------
// Parameter structs for batched dispatch
// ---------------------------------------------------------------------------

typedef struct {
    int period;
    int warmup_bars;
} EMAParams;

typedef struct {
    int period;
    int warmup_bars;
} SMAParams;

typedef struct {
    int period;
    int warmup_bars;
} RSIParams;

typedef struct {
    int period;
    int warmup_bars;
} ATRParams;

typedef struct {
    int fast_period;
    int slow_period;
    int signal_period;
    int series_idx;
    int warmup_bars;
    int _pad0;  // align to 8-byte boundary
} MACDParams;

typedef struct {
    int period;
    float multiplier;
    int series_idx;
    int warmup_bars;
} BollingerParams;

typedef struct {
    int ma_type_int;
    int ma_period;
    int atr_period;
    float atr_multiplier;
    int series_idx;
    int warmup_bars;
    int _pad0;
    int _pad1;
} KeltnerParams;

typedef struct {
    int period;
    int warmup_bars;
} CCIParams;

typedef struct {
    int period;
    int warmup_bars;
} ROCParams;

typedef struct {
    int smoothing_period;
    int warmup_bars;
} OBVParams;

typedef struct {
    int period;
    int warmup_bars;
} HMAParams;

// ---------------------------------------------------------------------------
// EMA kernel: Exponential Moving Average (batched)
// ---------------------------------------------------------------------------
__kernel void compute_ema(__global const float* close,
                          __global float* output,
                          __global const EMAParams* params,
                          int num_bars) {
    int id = get_global_id(0);
    EMAParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

    float alpha = 2.0f / ((float)p.period + 1.0f);
    float one_minus_alpha = 1.0f - alpha;
    float ema = 0.0f;
    for (int i = 0; i < num_bars; ++i) {
        if (i == 0) { ema = close[i]; } else { ema = alpha * close[i] + one_minus_alpha * ema; }
        output[offset + i] = (i < p.warmup_bars) ? NAN : ema;
    }
}

// ---------------------------------------------------------------------------
// SMA kernel: Simple Moving Average (batched)
// ---------------------------------------------------------------------------
__kernel void compute_sma(__global const float* close,
                          __global float* output,
                          __global const SMAParams* params,
                          int num_bars) {
    int id = get_global_id(0);
    SMAParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

#ifndef NO_FP64
    // ---- Double-precision path ----
    double running_sum = 0.0;
    double period_d = (double)p.period;
    for (int i = 0; i < num_bars; ++i) {
        running_sum += (double)close[i];
        if (i >= p.period) { running_sum -= (double)close[i - p.period]; }
        if (i >= p.period - 1 && i >= p.warmup_bars) {
            double sma_d = running_sum / period_d;
            output[offset + i] = (float)sma_d;
        } else {
            output[offset + i] = NAN;
        }
    }
#else
    // ---- Single-precision fallback (preserves prior behavior) ----
    float running_sum = 0.0f;
    for (int i = 0; i < num_bars; ++i) {
        running_sum += close[i];
        if (i >= p.period) { running_sum -= close[i - p.period]; }
        if (i >= p.period - 1 && i >= p.warmup_bars) { output[offset + i] = running_sum / (float)p.period; }
        else { output[offset + i] = NAN; }
    }
#endif
}

// ---------------------------------------------------------------------------
// RSI kernel: Relative Strength Index (batched)
// ---------------------------------------------------------------------------
__kernel void compute_rsi(__global const float* close,
                          __global float* output,
                          __global const RSIParams* params,
                          int num_bars) {
    int id = get_global_id(0);
    RSIParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

    float period_f = (float)p.period;
    float period_minus_1 = period_f - 1.0f;
    float avg_gain = 0.0f, avg_loss = 0.0f;
    int initialized = 0, changes_counted = 0;
    for (int i = 0; i < num_bars; ++i) {
        if (i == 0) { output[offset + i] = NAN; continue; }
        float diff = close[i] - close[i-1];
        float gain = (diff > 0.0f) ? diff : 0.0f;
        float loss = (diff < 0.0f) ? -diff : 0.0f;
        if (initialized == 0) {
            avg_gain += gain; avg_loss += loss; changes_counted++;
            if (changes_counted == p.period) {
                avg_gain /= period_f; avg_loss /= period_f; initialized = 1;
                if (i < p.warmup_bars) { output[offset + i] = NAN; }
                else if (avg_loss == 0.0f) { output[offset + i] = (avg_gain == 0.0f) ? 50.0f : 100.0f; }
                else { float rs = avg_gain / avg_loss; output[offset + i] = 100.0f - 100.0f / (1.0f + rs); }
            } else { output[offset + i] = NAN; }
        } else {
            avg_gain = (avg_gain * period_minus_1 + gain) / period_f;
            avg_loss = (avg_loss * period_minus_1 + loss) / period_f;
            if (i < p.warmup_bars) { output[offset + i] = NAN; }
            else if (avg_loss == 0.0f) { output[offset + i] = (avg_gain == 0.0f) ? 50.0f : 100.0f; }
            else { float rs = avg_gain / avg_loss; output[offset + i] = 100.0f - 100.0f / (1.0f + rs); }
        }
    }
}

// ---------------------------------------------------------------------------
// ATR kernel: Average True Range (batched)
// ---------------------------------------------------------------------------
__kernel void compute_atr(__global const float* high, __global const float* low,
                          __global const float* close,
                          __global float* output,
                          __global const ATRParams* params,
                          int num_bars) {
    int id = get_global_id(0);
    ATRParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

    float period_f = (float)p.period;
    float period_minus_1 = period_f - 1.0f;
    float atr_val = 0.0f, tr_sum = 0.0f;
    int initialized = 0;
    for (int i = 0; i < num_bars; ++i) {
        float tr;
        if (i == 0) { tr = high[i] - low[i]; }
        else {
            float pc = close[i-1], hl = high[i]-low[i];
            float hpc = fabs(high[i]-pc), lpc = fabs(low[i]-pc);
            tr = fmax(hl, fmax(hpc, lpc));
        }
        if (initialized == 0) {
            tr_sum += tr;
            if (i == p.period - 1) { atr_val = tr_sum / period_f; initialized = 1;
                output[offset + i] = (i < p.warmup_bars) ? NAN : atr_val; }
            else { output[offset + i] = NAN; }
        } else {
            atr_val = (atr_val * period_minus_1 + tr) / period_f;
            output[offset + i] = (i < p.warmup_bars) ? NAN : atr_val;
        }
    }
}

// ---------------------------------------------------------------------------
// MACD: fast EMA - slow EMA, signal = EMA of MACD, histogram = MACD - signal
// (batched)
// ---------------------------------------------------------------------------
__kernel void compute_macd(__global const float* close,
                           __global float* output,
                           __global const MACDParams* params,
                           int num_bars) {
    int id = get_global_id(0);
    MACDParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

    float fast_alpha = 2.0f / ((float)p.fast_period + 1.0f);
    float slow_alpha = 2.0f / ((float)p.slow_period + 1.0f);
    float sig_alpha  = 2.0f / ((float)p.signal_period + 1.0f);

    float fast_ema = 0.0f;
    float slow_ema = 0.0f;
    float signal_ema = 0.0f;
    int signal_initialized = 0;

    for (int i = 0; i < num_bars; ++i) {
        float c = close[i];

        if (i == 0) {
            fast_ema = c;
            slow_ema = c;
        } else {
            fast_ema = fast_alpha * c + (1.0f - fast_alpha) * fast_ema;
            slow_ema = slow_alpha * c + (1.0f - slow_alpha) * slow_ema;
        }

        float macd_val = fast_ema - slow_ema;

        if (signal_initialized == 0) {
            signal_ema = macd_val;
            signal_initialized = 1;
        } else {
            signal_ema = sig_alpha * macd_val + (1.0f - sig_alpha) * signal_ema;
        }

        float hist_val = macd_val - signal_ema;

        float result;
        if (p.series_idx == 0)      result = macd_val;
        else if (p.series_idx == 1) result = signal_ema;
        else                        result = hist_val;

        output[offset + i] = (i < p.warmup_bars) ? NAN : result;
    }
}

// ---------------------------------------------------------------------------
// Bollinger Bands: SMA + stddev bands (batched)
// ---------------------------------------------------------------------------
__kernel void compute_bollinger(__global const float* close,
                                __global float* output,
                                __global const BollingerParams* params,
                                int num_bars) {
    int id = get_global_id(0);
    BollingerParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

#ifndef NO_FP64
    // ---- Double-precision path (accurate running sums) ----
    double running_sum = 0.0;
    double running_sum_sq = 0.0;
    double period_d = (double)p.period;

    for (int i = 0; i < num_bars; ++i) {
        double c = (double)close[i];
        running_sum += c;
        running_sum_sq += c * c;

        if (i >= p.period) {
            double old_c = (double)close[i - p.period];
            running_sum -= old_c;
            running_sum_sq -= old_c * old_c;
        }

        if (i >= p.period - 1) {
            double sma_d = running_sum / period_d;
            float sma = (float)sma_d;

            double variance = running_sum_sq / period_d - sma_d * sma_d;
            if (variance < 0.0) variance = 0.0;
            float stddev = (float)sqrt(variance);

            if (i < p.warmup_bars) {
                output[offset + i] = NAN;
            } else {
                if (p.series_idx == 0)      output[offset + i] = sma;
                else if (p.series_idx == 1) output[offset + i] = sma + p.multiplier * stddev;
                else                        output[offset + i] = sma - p.multiplier * stddev;
            }
        } else {
            output[offset + i] = NAN;
        }
    }
#else
    // ---- Single-precision path (Welford's online algorithm) ----
    float period_f = (float)p.period;
    float running_sum = 0.0f;
    float w_mean = 0.0f;
    float w_m2   = 0.0f;
    int   w_n    = 0;

    for (int i = 0; i < num_bars; ++i) {
        float c = close[i];
        running_sum += c;

        w_n++;
        float delta = c - w_mean;
        w_mean += delta / (float)w_n;
        float delta2 = c - w_mean;
        w_m2 += delta * delta2;

        if (i >= p.period) {
            float old_c = close[i - p.period];
            running_sum -= old_c;

            float old_delta = old_c - w_mean;
            w_n--;
            if (w_n > 0) {
                float new_mean = w_mean - old_delta / (float)w_n;
                float old_delta2 = old_c - new_mean;
                w_m2 -= old_delta * old_delta2;
                w_mean = new_mean;
            } else {
                w_mean = 0.0f;
                w_m2 = 0.0f;
            }
            if (w_m2 < 0.0f) w_m2 = 0.0f;
        }

        if (i >= p.period - 1) {
            float sma = running_sum / period_f;

            float variance = w_m2 / period_f;
            if (variance < 0.0f) variance = 0.0f;
            float stddev = sqrt(variance);

            if (i < p.warmup_bars) {
                output[offset + i] = NAN;
            } else {
                if (p.series_idx == 0)      output[offset + i] = sma;
                else if (p.series_idx == 1) output[offset + i] = sma + p.multiplier * stddev;
                else                        output[offset + i] = sma - p.multiplier * stddev;
            }
        } else {
            output[offset + i] = NAN;
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Keltner Channels: center (EMA or SMA) +/- multiplier * ATR (batched)
// ---------------------------------------------------------------------------
__kernel void compute_keltner(__global const float* high, __global const float* low,
                              __global const float* close,
                              __global float* output,
                              __global const KeltnerParams* params,
                              int num_bars) {
    int id = get_global_id(0);
    KeltnerParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

    float ma_period_f = (float)p.ma_period;
    float ma_alpha = 2.0f / (ma_period_f + 1.0f);
    float ema_val = 0.0f;
#ifndef NO_FP64
    double sma_sum = 0.0;
    double ma_period_d = (double)p.ma_period;
#else
    float sma_sum = 0.0f;
#endif

    float atr_period_f = (float)p.atr_period;
    float atr_period_minus_1 = atr_period_f - 1.0f;
    float atr_val = 0.0f;
    float tr_sum = 0.0f;
    int atr_initialized = 0;

    for (int i = 0; i < num_bars; ++i) {
        float c = close[i];

        // Center line
        float center = NAN;
        if (p.ma_type_int == 0) {
            if (i == 0) { ema_val = c; }
            else { ema_val = ma_alpha * c + (1.0f - ma_alpha) * ema_val; }
            center = ema_val;
        } else {
#ifndef NO_FP64
            sma_sum += (double)c;
            if (i >= p.ma_period) { sma_sum -= (double)close[i - p.ma_period]; }
            if (i >= p.ma_period - 1) { center = (float)(sma_sum / ma_period_d); }
#else
            sma_sum += c;
            if (i >= p.ma_period) { sma_sum -= close[i - p.ma_period]; }
            if (i >= p.ma_period - 1) { center = sma_sum / ma_period_f; }
#endif
        }

        // ATR
        float tr;
        if (i == 0) {
            tr = high[i] - low[i];
        } else {
            float pc = close[i - 1];
            float hl = high[i] - low[i];
            float hpc = fabs(high[i] - pc);
            float lpc = fabs(low[i] - pc);
            tr = fmax(hl, fmax(hpc, lpc));
        }

        float atr = NAN;
        if (atr_initialized == 0) {
            tr_sum += tr;
            if (i == p.atr_period - 1) {
                atr_val = tr_sum / atr_period_f;
                atr_initialized = 1;
                atr = atr_val;
            }
        } else {
            atr_val = (atr_val * atr_period_minus_1 + tr) / atr_period_f;
            atr = atr_val;
        }

        // Combine
        if (i < p.warmup_bars || isnan(center) || isnan(atr)) {
            output[offset + i] = NAN;
        } else {
            if (p.series_idx == 0)      output[offset + i] = center;
            else if (p.series_idx == 1) output[offset + i] = center + p.atr_multiplier * atr;
            else                        output[offset + i] = center - p.atr_multiplier * atr;
        }
    }
}

// ---------------------------------------------------------------------------
// CCI: (TP - SMA(TP)) / (0.015 * MAD) (batched)
// ---------------------------------------------------------------------------
__kernel void compute_cci(__global const float* high, __global const float* low,
                          __global const float* close,
                          __global float* output,
                          __global const CCIParams* params,
                          int num_bars) {
    int id = get_global_id(0);
    CCIParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

    float period_f = (float)p.period;
    float cci_scale = 0.015f;
#ifndef NO_FP64
    double tp_sum = 0.0;
    double period_d = (double)p.period;
#else
    float tp_sum = 0.0f;
#endif

    for (int i = 0; i < num_bars; ++i) {
        float tp = (high[i] + low[i] + close[i]) / 3.0f;
#ifndef NO_FP64
        tp_sum += (double)tp;
#else
        tp_sum += tp;
#endif

        if (i >= p.period) {
            float old_tp = (high[i - p.period] + low[i - p.period] + close[i - p.period]) / 3.0f;
#ifndef NO_FP64
            tp_sum -= (double)old_tp;
#else
            tp_sum -= old_tp;
#endif
        }

        if (i >= p.period - 1) {
#ifndef NO_FP64
            float sma_tp = (float)(tp_sum / period_d);
#else
            float sma_tp = tp_sum / period_f;
#endif

            float mad_sum = 0.0f;
            for (int j = i - p.period + 1; j <= i; ++j) {
                float tp_j = (high[j] + low[j] + close[j]) / 3.0f;
                mad_sum += fabs(tp_j - sma_tp);
            }
            float mean_dev = mad_sum / period_f;

            if (i < p.warmup_bars) {
                output[offset + i] = NAN;
            } else if (mean_dev == 0.0f) {
                output[offset + i] = 0.0f;
            } else {
                output[offset + i] = (tp - sma_tp) / (cci_scale * mean_dev);
            }
        } else {
            output[offset + i] = NAN;
        }
    }
}

// ---------------------------------------------------------------------------
// ROC: Rate of Change (batched)
// ---------------------------------------------------------------------------
__kernel void compute_roc(__global const float* close,
                          __global float* output,
                          __global const ROCParams* params,
                          int num_bars) {
    int id = get_global_id(0);
    ROCParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

    for (int i = 0; i < num_bars; ++i) {
        if (i >= p.period) {
            float prev_close = close[i - p.period];
            if (prev_close == 0.0f) {
                output[offset + i] = (i < p.warmup_bars) ? NAN : 0.0f;
            } else {
                float roc_val = ((close[i] - prev_close) / prev_close) * 100.0f;
                output[offset + i] = (i < p.warmup_bars) ? NAN : roc_val;
            }
        } else {
            output[offset + i] = NAN;
        }
    }
}

// ---------------------------------------------------------------------------
// OBV: On-Balance Volume with optional SMA smoothing (batched)
// ---------------------------------------------------------------------------
__kernel void compute_obv(__global const float* close, __global const uint* volume,
                          __global float* output,
                          __global const OBVParams* params,
                          int num_bars) {
    int id = get_global_id(0);
    OBVParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

#ifndef NO_FP64
    double obv = 0.0;
#else
    float obv = 0.0f;
#endif

    if (p.smoothing_period <= 0) {
        for (int i = 0; i < num_bars; ++i) {
            if (i == 0) {
#ifndef NO_FP64
                obv = 0.0;
#else
                obv = 0.0f;
#endif
            } else {
                if (close[i] > close[i - 1]) {
#ifndef NO_FP64
                    obv += (double)volume[i];
#else
                    obv += (float)volume[i];
#endif
                } else if (close[i] < close[i - 1]) {
#ifndef NO_FP64
                    obv -= (double)volume[i];
#else
                    obv -= (float)volume[i];
#endif
                }
            }
            output[offset + i] = (float)obv;
        }
    } else {
        // Pass 1: raw OBV
        for (int i = 0; i < num_bars; ++i) {
            if (i == 0) {
#ifndef NO_FP64
                obv = 0.0;
#else
                obv = 0.0f;
#endif
            } else {
                if (close[i] > close[i - 1]) {
#ifndef NO_FP64
                    obv += (double)volume[i];
#else
                    obv += (float)volume[i];
#endif
                } else if (close[i] < close[i - 1]) {
#ifndef NO_FP64
                    obv -= (double)volume[i];
#else
                    obv -= (float)volume[i];
#endif
                }
            }
            output[offset + i] = (float)obv;
        }

        // Pass 2: SMA of raw OBV (right-to-left)
        float sp_f = (float)p.smoothing_period;
        for (int i = num_bars - 1; i >= 0; --i) {
            if (i < p.smoothing_period - 1 || i < p.warmup_bars) {
                output[offset + i] = NAN;
            } else {
                float sum = 0.0f;
                for (int j = i - p.smoothing_period + 1; j <= i; ++j) {
                    sum += output[offset + j];
                }
                output[offset + i] = sum / sp_f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HMA: Hull Moving Average (batched)
// temp buffers are sized count * num_bars; work-item i uses temp[i*num_bars..]
// ---------------------------------------------------------------------------
__kernel void compute_hma(__global const float* close,
                          __global float* output,
                          __global float* temp1, __global float* temp2,
                          __global const HMAParams* params,
                          int num_bars) {
    int id = get_global_id(0);
    HMAParams p = params[id];
    size_t offset = (size_t)id * (size_t)num_bars;

    int half_period = p.period / 2;
    int sqrt_period = (int)floor(sqrt((float)p.period));
    if (sqrt_period < 1) sqrt_period = 1;

    // Step 1: WMA(half_period, close) -> temp1
    {
        float denom = (float)half_period * (float)(half_period + 1) / 2.0f;
        for (int i = 0; i < num_bars; ++i) {
            if (i < half_period - 1) {
                temp1[offset + i] = NAN;
            } else {
                float sum = 0.0f;
                int has_nan = 0;
                for (int j = 0; j < half_period; ++j) {
                    float val = close[i - half_period + 1 + j];
                    if (isnan(val)) { has_nan = 1; break; }
                    sum += (float)(j + 1) * val;
                }
                temp1[offset + i] = has_nan ? NAN : (sum / denom);
            }
        }
    }

    // Step 2: WMA(period, close) -> temp2
    {
        float denom = (float)p.period * (float)(p.period + 1) / 2.0f;
        for (int i = 0; i < num_bars; ++i) {
            if (i < p.period - 1) {
                temp2[offset + i] = NAN;
            } else {
                float sum = 0.0f;
                int has_nan = 0;
                for (int j = 0; j < p.period; ++j) {
                    float val = close[i - p.period + 1 + j];
                    if (isnan(val)) { has_nan = 1; break; }
                    sum += (float)(j + 1) * val;
                }
                temp2[offset + i] = has_nan ? NAN : (sum / denom);
            }
        }
    }

    // Step 3: diff = 2 * temp1 - temp2 (store in temp1)
    for (int i = 0; i < num_bars; ++i) {
        if (isnan(temp1[offset + i]) || isnan(temp2[offset + i])) {
            temp1[offset + i] = NAN;
        } else {
            temp1[offset + i] = 2.0f * temp1[offset + i] - temp2[offset + i];
        }
    }

    // Step 4: WMA(sqrt_period, diff) -> output
    {
        float denom = (float)sqrt_period * (float)(sqrt_period + 1) / 2.0f;
        for (int i = 0; i < num_bars; ++i) {
            if (i < sqrt_period - 1) {
                output[offset + i] = NAN;
            } else {
                float sum = 0.0f;
                int has_nan = 0;
                for (int j = 0; j < sqrt_period; ++j) {
                    float val = temp1[offset + i - sqrt_period + 1 + j];
                    if (isnan(val)) { has_nan = 1; break; }
                    sum += (float)(j + 1) * val;
                }
                if (has_nan) {
                    output[offset + i] = NAN;
                } else {
                    output[offset + i] = (i < p.warmup_bars) ? NAN : (sum / denom);
                }
            }
        }
    }
}
