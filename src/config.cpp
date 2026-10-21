#include "config.h"

#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

#define TOML_EXCEPTIONS 1
#include <toml++/toml.hpp>

namespace {

struct ConfigError {
    std::string message;
};

[[noreturn]] void fail(const std::string& message) {
    throw ConfigError{message};
}

// One [section]: typed getters plus a check that every key was recognised.
class Section {
public:
    Section(const toml::table* table, std::string name) : m_table(table), m_name(std::move(name)) {}

    bool present() const { return m_table != nullptr; }

    std::optional<std::string> str(const char* key) {
        const toml::node* n = node(key);
        if (!n) return std::nullopt;
        if (auto v = n->value<std::string>()) return *v;
        fail(where(key) + " must be a string");
    }

    std::optional<double> number(const char* key) {
        const toml::node* n = node(key);
        if (!n) return std::nullopt;
        if (auto v = n->value<double>(); v && std::isfinite(*v)) return *v;
        fail(where(key) + " must be a number");
    }

    std::optional<int64_t> integer(const char* key) {
        const toml::node* n = node(key);
        if (!n) return std::nullopt;
        if (auto v = n->value<int64_t>()) return *v;
        fail(where(key) + " must be an integer");
    }

    std::optional<bool> boolean(const char* key) {
        const toml::node* n = node(key);
        if (!n) return std::nullopt;
        if (auto v = n->value<bool>()) return *v;
        fail(where(key) + " must be true or false");
    }

    // A bar index (integer) or a date/time string.
    BarBound bound(const char* key) {
        BarBound          b;
        const toml::node* n = node(key);
        if (!n) return b;
        if (auto i = n->value<int64_t>()) {
            if (*i < 0) fail(where(key) + " must not be negative");
            b.index = *i;
        } else if (auto s = n->value<std::string>()) {
            b.timestamp = *s;
        } else {
            fail(where(key) + " must be a bar index or a date string");
        }
        return b;
    }

    const toml::array* array(const char* key) {
        const toml::node* n = node(key);
        if (!n) return nullptr;
        if (const auto* a = n->as_array()) return a;
        fail(where(key) + " must be an array");
    }

    std::string where(const char* key) const { return "[" + m_name + "] " + key; }

    void reject_unknown() const {
        if (!m_table) return;
        for (const auto& [key, value] : *m_table) {
            (void)value;
            const std::string k(key.str());
            if (!m_seen.count(k)) fail("unknown key " + where(k.c_str()));
        }
    }

private:
    const toml::node* node(const char* key) {
        m_seen.insert(key);
        return m_table ? m_table->get(key) : nullptr;
    }

    const toml::table*    m_table;
    std::string           m_name;
    std::set<std::string> m_seen;
};

double positive(Section& s, const char* key, double value) {
    if (!(value > 0.0)) fail(s.where(key) + " must be positive");
    return value;
}

int32_t clock_time(Section& s, const char* key, const std::string& text) {
    int32_t sec = 0;
    if (!parse_clock_time(text, sec)) fail(s.where(key) + " must be \"HH:MM\"");
    return sec;
}

RunConfig build(const toml::table& root) {
    static const std::set<std::string> known = {"data",   "instrument", "session",  "screen",
                                                "floors", "compute",    "resources"};
    for (const auto& [key, value] : root) {
        const std::string k(key.str());
        if (!known.count(k)) fail("unknown section [" + k + "]");
        if (!value.is_table()) fail("[" + k + "] must be a table");
    }
    auto section = [&](const char* name) { return Section(root[name].as_table(), name); };

    RunConfig c;

    Section data = section("data");
    if (auto v = data.str("bars")) c.bars = *v;
    if (auto v = data.str("output")) c.output = *v;
    data.reject_unknown();

    Section inst            = section("instrument");
    c.instrument_configured = inst.present();
    if (inst.present()) {
        if (auto v = inst.str("symbol")) c.symbol = *v;
        const auto tick_size  = inst.number("tick_size");
        const auto tick_value = inst.number("tick_value");
        if (!tick_size || !tick_value) fail("[instrument] needs tick_size and tick_value");
        c.cost.tick_size     = static_cast<float>(positive(inst, "tick_size", *tick_size));
        c.cost.tick_value    = static_cast<float>(positive(inst, "tick_value", *tick_value));
        c.cost.point_value   = static_cast<float>(positive(
            inst, "point_value", inst.number("point_value").value_or(*tick_value / *tick_size)));
        const int64_t spread = inst.integer("spread_ticks").value_or(c.cost.round_trip_ticks);
        if (spread < 0) fail(inst.where("spread_ticks") + " must not be negative");
        c.cost.round_trip_ticks = static_cast<int>(spread);
        const double slip       = inst.number("slippage_ticks").value_or(c.cost.slippage_ticks);
        if (slip < 0.0) fail(inst.where("slippage_ticks") + " must not be negative");
        c.cost.slippage_ticks   = static_cast<float>(slip);
        c.cost.capital_required = static_cast<float>(
            positive(inst, "capital", inst.number("capital").value_or(c.cost.capital_required)));
        if (std::abs(c.cost.point_value * c.cost.tick_size - c.cost.tick_value) >
            1e-3f * c.cost.tick_value)
            fail("[instrument] tick_value must equal tick_size * point_value");
        inst.reject_unknown();
    }

    Section sess = section("session");
    if (sess.present()) {
        if (auto v = sess.str("model")) {
            if (!SessionModel::from_name(*v, c.session))
                fail(sess.where("model") + " must be \"us_futures\" or \"continuous\"");
        }
        if (auto v = sess.str("session_open"))
            c.session.session_open_sec = clock_time(sess, "session_open", *v);
        if (auto v = sess.str("session_close")) {
            c.session.session_close_sec =
                (*v == "none") ? -1 : clock_time(sess, "session_close", *v);
            if (c.session.session_close_sec < 0) c.session.flat_at_close = false;
        }
        if (auto v = sess.boolean("flat_at_close")) c.session.flat_at_close = *v;
        if (const auto* hours = sess.array("core_hours")) {
            if (hours->size() != 3) fail(sess.where("core_hours") + " needs [open, split, close]");
            int32_t t[3];
            for (size_t i = 0; i < 3; ++i) {
                auto s = (*hours)[i].value<std::string>();
                if (!s) fail(sess.where("core_hours") + " entries must be \"HH:MM\"");
                t[i] = clock_time(sess, "core_hours", *s);
            }
            c.session.core_open_sec  = t[0];
            c.session.core_split_sec = t[1];
            c.session.core_close_sec = t[2] == 0 ? 86400 : t[2];
            // The session-signal module (opening range, lunch compression,
            // RTH levels, ...) is built on the 09:30-16:00 ET equity session.
            // Other core hours would feed it the wrong windows, so it is off.
            if (c.session.core_open_sec != 9 * 3600 + 30 * 60 ||
                c.session.core_close_sec != 16 * 3600)
                c.session.session_signals = false;
        }
        if (auto v = sess.number("halt_gap_minutes")) {
            c.session.detect_halts = *v > 0.0;
            c.session.halt_gap_sec = static_cast<int32_t>(*v * 60.0);
        }
        if (auto v = sess.str("early_close_calendar")) c.session.early_close_calendar = *v;
        sess.reject_unknown();
    }
    std::string model_error;
    if (!c.session.validate(&model_error)) fail("[session] " + model_error);

    Section screen = section("screen");
    c.screen_start = screen.bound("start");
    c.screen_end   = screen.bound("end");
    if (auto v = screen.integer("max_strategies")) {
        if (*v <= 0) fail(screen.where("max_strategies") + " must be positive");
        c.max_strategies = *v;
    }
    if (auto v = screen.number("robustness_floor")) {
        if (*v < 0.0 || *v > 1.0) fail(screen.where("robustness_floor") + " must be in [0, 1]");
        c.robustness_floor = static_cast<float>(*v);
    }
    if (auto v = screen.boolean("early_termination")) c.early_termination = *v;
    if (auto v = screen.integer("entry_lag")) {
        if (*v < 1 || *v > 2) fail(screen.where("entry_lag") + " must be 1 or 2");
        c.entry_lag = static_cast<int>(*v);
    }
    screen.reject_unknown();

    Section floors = section("floors");
    if (auto v = floors.integer("min_trades")) c.floors.min_total_trades = static_cast<int>(*v);
    if (auto v = floors.number("min_profit_factor"))
        c.floors.min_profit_factor = static_cast<float>(*v);
    if (auto v = floors.number("min_annual_roc")) c.floors.min_annual_roc = static_cast<float>(*v);
    if (auto v = floors.number("min_total_pnl")) c.floors.min_total_pnl = static_cast<float>(*v);
    if (auto v = floors.integer("max_negative_years"))
        c.floors.max_negative_years = static_cast<int>(*v);
    if (c.floors.min_total_trades < 0) fail("[floors] min_trades must not be negative");
    if (c.floors.max_negative_years < 0) fail("[floors] max_negative_years must not be negative");
    if (c.floors.min_profit_factor < 0.0f) fail("[floors] min_profit_factor must not be negative");
    floors.reject_unknown();

    Section compute = section("compute");
    if (auto v = compute.str("device")) {
        if (!parse_device(*v, c.device))
            fail(compute.where("device") + " must be auto, gpu, opencl-cpu or native");
    }
    if (auto v = compute.boolean("indicators_on_device")) c.indicators_on_device = *v;
    compute.reject_unknown();

    Section res = section("resources");
    if (auto v = res.number("max_cpu_pct")) {
        if (*v <= 0.0 || *v > 100.0) fail(res.where("max_cpu_pct") + " must be in (0, 100]");
        c.max_cpu_pct = static_cast<float>(*v);
    }
    if (auto v = res.number("max_ram_gb"))
        c.max_ram_gb = static_cast<float>(positive(res, "max_ram_gb", *v));
    res.reject_unknown();

    return c;
}

}  // namespace

bool parse_clock_time(const std::string& text, int32_t& seconds) {
    if (text.size() != 5 || text[2] != ':') return false;
    for (int i : {0, 1, 3, 4})
        if (text[i] < '0' || text[i] > '9') return false;
    const int h = (text[0] - '0') * 10 + (text[1] - '0');
    const int m = (text[3] - '0') * 10 + (text[4] - '0');
    if (h > 23 || m > 59) return false;
    seconds = h * 3600 + m * 60;
    return true;
}

bool parse_device(const std::string& text, DeviceType& out) {
    if (text == "auto")
        out = DeviceType::AUTO;
    else if (text == "gpu")
        out = DeviceType::GPU;
    else if (text == "opencl-cpu")
        out = DeviceType::CPU;
    else if (text == "native")
        out = DeviceType::NATIVE;
    else
        return false;
    return true;
}

ConfigResult parse_run_config(const std::string& toml_text, const std::string& source_name) {
    ConfigResult r;
    try {
        const toml::table root = toml::parse(toml_text, source_name);
        r.config               = build(root);
    } catch (const toml::parse_error& e) {
        std::ostringstream os;
        os << source_name << ":" << e.source().begin.line << ":" << e.source().begin.column << ": "
           << e.description();
        r.error = os.str();
    } catch (const ConfigError& e) {
        r.error = source_name + ": " + e.message;
    }
    return r;
}

ConfigResult load_run_config(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        ConfigResult r;
        r.error = "cannot open config file " + path;
        return r;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_run_config(ss.str(), path);
}
