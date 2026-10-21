#pragma once
#include "indicator.h"

// Pivot Points
//
// Computation uses PRIOR session's High, Low, Close (no lookahead).
// Session boundaries detected via timestamp gaps (>55 min = new session,
// matching CME maintenance window heuristic used by VWAP).
//
// Classic:
//   PP = (H + L + C) / 3
//   S1 = 2*PP - H,   R1 = 2*PP - L
//   S2 = PP - (H-L), R2 = PP + (H-L)
//   S3 = L - 2*(H - PP), R3 = H + 2*(PP - L)
//
// Fibonacci:
//   PP = (H + L + C) / 3
//   S1 = PP - 0.382*(H-L), R1 = PP + 0.382*(H-L)
//   S2 = PP - 0.618*(H-L), R2 = PP + 0.618*(H-L)
//   S3 = PP - 1.000*(H-L), R3 = PP + 1.000*(H-L)
//
// Camarilla:
//   PP = (H + L + C) / 3
//   S1 = C - 1.1*(H-L)/12, R1 = C + 1.1*(H-L)/12
//   S2 = C - 1.1*(H-L)/6,  R2 = C + 1.1*(H-L)/6
//   S3 = C - 1.1*(H-L)/4,  R3 = C + 1.1*(H-L)/4
//
// Output: 7 series -- PP, S1, S2, S3, R1, R2, R3
//
// Warmup: 1 session (~2640 bars at 30s) -- need at least one completed prior
// session to compute levels.  We use 2640 as the warmup constant.
//
// Parameters: type (0=Classic, 1=Fibonacci, 2=Camarilla)
enum class PivotType : int { CLASSIC = 0, FIBONACCI = 1, CAMARILLA = 2 };

class PivotPoints : public Indicator {
public:
    explicit PivotPoints(PivotType type = PivotType::CLASSIC);

    void        compute(const Bar* bars, int start, int end, float* output) const override;
    int         num_series() const override;
    int         warmup_bars() const override;
    const char* name() const override;

    PivotType pivot_type() const { return m_type; }

private:
    PivotType m_type;
};
