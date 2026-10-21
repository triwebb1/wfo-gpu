#pragma once
// ---------------------------------------------------------------------------
// CRC32C (Castagnoli) -- table-driven checksum
//
// Streaming CRC32C for durable file writes. Used to detect
// bit-rot and partial-write corruption in checkpoint and all_results.bin.
//
// Polynomial: 0x1EDC6F41 (reflected form 0x82F63B78). Initial state
// 0xFFFFFFFFu; final value XOR'd with 0xFFFFFFFFu.
//
// Usage (streaming):
//   uint32_t s = crc32c_init();
//   s = crc32c_update(s, data1, len1);
//   s = crc32c_update(s, data2, len2);
//   uint32_t final_crc = crc32c_finalize(s);
// ---------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>

constexpr uint32_t CRC32C_INITIAL_STATE = 0xFFFFFFFFu;
constexpr uint32_t CRC32C_FINALIZE_XOR  = 0xFFFFFFFFu;

inline uint32_t crc32c_init() {
    return CRC32C_INITIAL_STATE;
}

uint32_t crc32c_update(uint32_t state, const void* data, std::size_t len);

inline uint32_t crc32c_finalize(uint32_t state) {
    return state ^ CRC32C_FINALIZE_XOR;
}

// One-shot helper: compute the full CRC32C of a buffer.
inline uint32_t crc32c(const void* data, std::size_t len) {
    return crc32c_finalize(crc32c_update(crc32c_init(), data, len));
}
