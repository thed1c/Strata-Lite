// Strata-Lite: the few operating-system calls the engines and servers make, for Linux and Windows. Everything else
// they use is portable C++ or ggml / CUDA.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <vector>
#else
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <fstream>
#include <set>
#endif

namespace lite {

#ifdef _WIN32
inline std::wstring widen(const std::string & s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}
#endif

// A read-only mapping of a whole file (the GGUF): its RAM-resident weights are used in place, as llama.cpp does with
// mmap. nullptr on failure.
inline void * map_file(const std::string & path, size_t * size) {
    *size = 0;
#ifdef _WIN32
    HANDLE f = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return nullptr;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz)) { CloseHandle(f); return nullptr; }
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(f);
    if (!m) return nullptr;
    void * p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);   // the view keeps the mapping alive
    if (!p) return nullptr;
    *size = (size_t) sz.QuadPart;
    return p;
#else
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat sb;
    if (fstat(fd, &sb) != 0) { close(fd); return nullptr; }
    void * p = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return nullptr;
    *size = (size_t) sb.st_size;
    return p;
#endif
}

inline void unmap_file(void * p, size_t size) {
    if (!p) return;
#ifdef _WIN32
    (void) size;
    UnmapViewOfFile(p);
#else
    munmap(p, size);
#endif
}

// Physical cores among the CPUs this process may run on (Linux: the affinity mask, so numactl is respected; one per
// set of hyper-thread siblings). 0 when unknown.
inline int physical_cores() {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return 0;
    std::vector<uint8_t> buf(len);
    auto * info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *) buf.data();
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return 0;
    int cores = 0;
    for (DWORD off = 0; off < len;) {
        auto * e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *) (buf.data() + off);
        if (e->Relationship == RelationProcessorCore) cores++;
        off += e->Size;
    }
    return cores;
#else
    cpu_set_t set; CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return 0;
    std::set<std::string> seen;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        std::string sib; std::getline(f, sib);
        seen.insert(sib.empty() ? std::to_string(c) : sib);
    }
    return (int) seen.size();
#endif
}

inline int physical_cores_or_half() {
    const int c = physical_cores();
    if (c > 0) return c;
    const unsigned h = std::thread::hardware_concurrency() / 2;
    return h > 0 ? (int) h : 1;
}

// Installed RAM in GB (1e9 bytes); 0 when unknown.
inline double total_ram_gb() {
#ifdef _WIN32
    MEMORYSTATUSEX ms; ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (double) ms.ullTotalPhys / 1e9 : 0.0;
#else
    struct sysinfo si;
    return sysinfo(&si) == 0 ? (double) si.totalram * si.mem_unit / 1e9 : 0.0;
#endif
}

// The process's cgroup v2 memory limit in bytes (Linux; 0 when none or elsewhere): a container's or a service's cap.
inline double cgroup_limit_bytes() {
#ifdef _WIN32
    return 0;
#else
    std::ifstream cg("/proc/self/cgroup");
    std::string line;
    while (std::getline(cg, line)) {
        if (line.rfind("0::", 0) != 0) continue;
        std::ifstream mx("/sys/fs/cgroup" + line.substr(3) + "/memory.max");
        std::string m;
        if (mx >> m && m != "max") return std::stod(m);
    }
    return 0;
#endif
}

// RAM the OS can give this process now, in bytes (Linux: MemAvailable, within the process's cgroup v2 limit when it
// has one; Windows: available physical memory). Both count the file cache the OS would drop. 0 when unknown.
inline double available_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX ms; ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (double) ms.ullAvailPhys : 0.0;
#else
    double avail = 0;
    {
        std::ifstream f("/proc/meminfo");
        std::string key, unit; double kb = 0;
        while (f >> key >> kb >> unit) if (key == "MemAvailable:") { avail = kb * 1024; break; }
    }
    std::ifstream cg("/proc/self/cgroup");
    std::string line;
    while (std::getline(cg, line)) {
        if (line.rfind("0::", 0) != 0) continue;
        const std::string dir = "/sys/fs/cgroup" + line.substr(3);
        std::ifstream mx(dir + "/memory.max"), cur(dir + "/memory.current");
        std::string m; double c = 0;
        if (mx >> m && m != "max" && cur >> c) {
            const double room = std::max(0.0, std::stod(m) - c);
            if (avail == 0 || room < avail) avail = room;
        }
    }
    return avail;
#endif
}

// The folder of the running executable, with its trailing separator ("" when unknown).
inline std::string self_dir() {
#ifdef _WIN32
    wchar_t w[32768];
    const DWORD n = GetModuleFileNameW(nullptr, w, 32768);
    if (n == 0 || n >= 32768) return "";
    const int k = WideCharToMultiByte(CP_UTF8, 0, w, (int) n, nullptr, 0, nullptr, nullptr);
    std::string s(k, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, (int) n, s.data(), k, nullptr, nullptr);
    return s.substr(0, s.find_last_of("\\/") + 1);
#else
    char self[4096];
    const ssize_t k = readlink("/proc/self/exe", self, sizeof self - 1);
    if (k <= 0) return "";
    self[k] = 0;
    const std::string s(self);
    return s.substr(0, s.rfind('/') + 1);
#endif
}

}  // namespace lite
