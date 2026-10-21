#pragma once
// Minimal "--name value" / "--name=value" / "--flag" argument parser for the
// wfo-gpu subcommands. Every option is consumed explicitly; anything left
// over is reported as an error.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cli {

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Args {
public:
    Args(int argc, char** argv, int first) {
        for (int i = first; i < argc; ++i) m_items.emplace_back(argv[i]);
        m_used.assign(m_items.size(), false);
    }

    bool flag(const std::string& name) {
        for (size_t i = 0; i < m_items.size(); ++i) {
            if (!m_used[i] && m_items[i] == name) {
                m_used[i] = true;
                return true;
            }
        }
        return false;
    }

    std::optional<std::string> value(const std::string& name) {
        const std::string prefix = name + "=";
        for (size_t i = 0; i < m_items.size(); ++i) {
            if (m_used[i]) continue;
            if (m_items[i] == name) {
                if (i + 1 >= m_items.size()) throw UsageError(name + " needs a value");
                m_used[i] = m_used[i + 1] = true;
                return m_items[i + 1];
            }
            if (m_items[i].rfind(prefix, 0) == 0) {
                m_used[i] = true;
                return m_items[i].substr(prefix.size());
            }
        }
        return std::nullopt;
    }

    std::string required(const std::string& name) {
        auto v = value(name);
        if (!v) throw UsageError(name + " is required");
        return *v;
    }

    std::optional<int64_t> integer(const std::string& name) {
        auto v = value(name);
        if (!v) return std::nullopt;
        try {
            size_t          pos = 0;
            const long long n   = std::stoll(*v, &pos);
            if (pos != v->size()) throw std::invalid_argument("trailing");
            return n;
        } catch (const std::exception&) {
            throw UsageError(name + " expects an integer, got '" + *v + "'");
        }
    }

    std::optional<double> number(const std::string& name) {
        auto v = value(name);
        if (!v) return std::nullopt;
        try {
            size_t       pos = 0;
            const double d   = std::stod(*v, &pos);
            if (pos != v->size()) throw std::invalid_argument("trailing");
            return d;
        } catch (const std::exception&) {
            throw UsageError(name + " expects a number, got '" + *v + "'");
        }
    }

    // Throw if any argument was not consumed.
    void finish() const {
        for (size_t i = 0; i < m_items.size(); ++i) {
            if (!m_used[i]) throw UsageError("unexpected argument '" + m_items[i] + "'");
        }
    }

private:
    std::vector<std::string> m_items;
    std::vector<bool>        m_used;
};

}  // namespace cli
