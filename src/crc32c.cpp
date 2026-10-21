#include "crc32c.h"

// ---------------------------------------------------------------------------
// CRC32C (Castagnoli) table-driven implementation.
//
// Reflected polynomial: 0x82F63B78. The table is built once on first use.
// ---------------------------------------------------------------------------

namespace {

struct Crc32cTable {
    uint32_t t[256];
    Crc32cTable() {
        constexpr uint32_t POLY = 0x82F63B78u;  // reflected 0x1EDC6F41
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (POLY ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
    }
};

const Crc32cTable& table() {
    static const Crc32cTable tbl;
    return tbl;
}

}  // namespace

uint32_t crc32c_update(uint32_t state, const void* data, std::size_t len) {
    const auto&    tbl = table();
    const uint8_t* p   = static_cast<const uint8_t*>(data);
    uint32_t       c   = state;
    for (std::size_t i = 0; i < len; ++i) {
        c = tbl.t[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    }
    return c;
}
