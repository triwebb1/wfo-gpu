#include "wfo_grid.h"

WFOGrid::WFOGrid() {
    constexpr WFOFrequency frequencies[] = {WFOFrequency::WEEK_1, WFOFrequency::WEEK_2,
                                            WFOFrequency::MONTH_1, WFOFrequency::QUARTER_1};

    constexpr WFOLookback lookbacks[] = {
        WFOLookback::MONTH_1, WFOLookback::MONTH_2, WFOLookback::MONTH_3, WFOLookback::MONTH_6,
        WFOLookback::YEAR_1,  WFOLookback::YEAR_2,  WFOLookback::YEAR_3,  WFOLookback::YEAR_5};

    // ---- Fixed-interval configs: 4 freq x 8 lookback, pruned ----
    for (WFOFrequency freq : frequencies) {
        for (WFOLookback lb : lookbacks) {
            // Prune: lookback must be >= frequency
            int lb_days   = lookback_approx_days(lb);
            int freq_days = frequency_approx_days(freq);
            if (lb_days < freq_days) continue;

            WFOScheduleConfig cfg;
            cfg.frequency = freq;
            cfg.lookback  = lb;
            cfg.trigger   = WFOTrigger::FIXED_INTERVAL;
            m_entries.push_back(cfg);
        }
    }

    // ---- Adaptive trigger configs: 4 triggers x 8 lookbacks x 1 freq (1 week) ----
    // Adaptive triggers always step at 1-week intervals (the smallest frequency)
    // but only re-optimize when the trigger fires. Between triggers, reuse
    // the last winner.
    constexpr WFOTrigger adaptive_triggers[] = {WFOTrigger::DD_TRIGGER_5PCT,
                                                WFOTrigger::DD_TRIGGER_10PCT, WFOTrigger::LOSERS_5,
                                                WFOTrigger::LOSERS_10};

    for (WFOTrigger trig : adaptive_triggers) {
        for (WFOLookback lb : lookbacks) {
            WFOScheduleConfig cfg;
            cfg.frequency = WFOFrequency::WEEK_1;  // always 1-week step
            cfg.lookback  = lb;
            cfg.trigger   = trig;
            m_entries.push_back(cfg);
        }
    }
}

int WFOGrid::find(const WFOScheduleConfig& c) const {
    for (int i = 0; i < static_cast<int>(m_entries.size()); ++i) {
        if (m_entries[i] == c) return i;
    }
    return -1;
}
