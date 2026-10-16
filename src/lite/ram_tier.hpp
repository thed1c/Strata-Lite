// Strata-Lite: the RAM tier. The experts a 12 GB card does not hold live in system RAM: locked (page-locked, the
// arena the copy stream reads, so swaps run in the background) up to a budget, and the rest in the model file's
// mapping, which the OS keeps in its file cache while it has room and reads from the disk (an NVMe drive) when not.
// Whole expert tensors (a layer's gate/up or down) are the unit, in the file's layer order. The bytes are the same
// either way, so the arithmetic does not change; only where a miss or a swap reads from does.
//
// Budget: --ram-budget GiB; 0 = low-RAM mode (nothing locked, swaps in the step); default (-1): what the OS can give
// now less a headroom for the OS and other programs (Linux max(4 GiB, 15 % of RAM), Windows max(8 GiB, 25 %)).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include "ggml.h"
#include "platform.hpp"

namespace lite {

inline size_t ram_lock_budget(double budget_gib, size_t want) {
    if (budget_gib >= 0) return std::min(want, (size_t) (budget_gib * 1073741824.0));
    const double avail = available_ram_bytes();
    if (avail <= 0) return want;   // unknown: lock everything, as before the tier (a failed lock still falls back)
    double total = total_ram_gb() * 1e9;   // the PC's RAM, or the cgroup limit when the process runs under a smaller one
    if (const double cap = cgroup_limit_bytes(); cap > 0 && (total == 0 || cap < total)) total = cap;
#ifdef _WIN32
    const double headroom = std::max(8.0 * 1073741824.0, 0.25 * total);
#else
    const double headroom = std::max(4.0 * 1073741824.0, 0.15 * total);
#endif
    return avail > headroom ? std::min(want, (size_t) (avail - headroom)) : 0;
}

struct RamTier {
    void * arena = nullptr;
    size_t arena_size = 0, want = 0;
    std::unordered_set<const ggml_tensor *> locked;
    bool in_arena(const ggml_tensor * t) const { return locked.count(t) != 0; }
};

// Picks and locks: the first expert tensors (`_exps.` in the name) that fit the budget; a lock the OS refuses is
// tried again at half the size, down to 1 GiB, then nothing is locked (low-RAM mode). on_cpu: (tensor, file offset).
inline RamTier ram_tier_lock(const std::vector<std::pair<ggml_tensor *, size_t>> & on_cpu, double budget_gib, size_t align) {
    RamTier r;
    auto pad = [&](size_t n) { return (n + align - 1) / align * align; };
    auto is_exps = [](const ggml_tensor * t) { return strstr(ggml_get_name(t), "_exps.") != nullptr; };
    for (auto & [t, off] : on_cpu) if (is_exps(t)) r.want += pad(ggml_nbytes(t));
    size_t budget = ram_lock_budget(budget_gib, r.want);
    while (budget > 0) {
        r.locked.clear(); r.arena_size = 0;
        for (auto & [t, off] : on_cpu)
            if (is_exps(t) && r.arena_size + pad(ggml_nbytes(t)) <= budget) { r.locked.insert(t); r.arena_size += pad(ggml_nbytes(t)); }
        if (r.arena_size == 0) break;
        if (cudaHostAlloc(&r.arena, r.arena_size, cudaHostAllocPortable) == cudaSuccess) break;
        cudaGetLastError();
        r.arena = nullptr;
        fprintf(stderr, "strata-lite: the OS would not lock %.1f GB of RAM; trying half\n", r.arena_size / 1e9);
        budget = r.arena_size / 2;
        if (budget < ((size_t) 1 << 30)) budget = 0;
    }
    if (!r.arena) { r.locked.clear(); r.arena_size = 0; }
    if (r.arena_size == 0)
        fprintf(stderr, "strata-lite: low-RAM mode: the experts (%.1f GB) stay in the model file and are read in as needed (slower)\n",
                r.want / 1e9);
    else if (r.arena_size < r.want)
        fprintf(stderr, "strata-lite: RAM tier: %.1f of the experts' %.1f GB locked in RAM; the rest is read from the model file "
                        "(the OS file cache, else the disk)\n", r.arena_size / 1e9, r.want / 1e9);
    return r;
}

}  // namespace lite
