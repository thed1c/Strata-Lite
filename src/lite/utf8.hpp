// Strata-Lite: generated text made valid UTF-8 before llama.cpp's chat parser sees it. Byte-level tokens can leave a
// broken sequence anywhere in an answer (and an answer cut off by n_predict can end inside a character); the parser
// rejects any invalid byte ("does not match the expected Content-only format"), which came back as a 500. An
// incomplete tail is dropped, every other invalid sequence becomes U+FFFD, as a display would show it.
#pragma once
#include <string>

namespace lite {

inline std::string utf8_clean(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    const auto * p = (const unsigned char *) s.data();
    const size_t n = s.size();
    for (size_t i = 0; i < n;) {
        const unsigned char c = p[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        bool ok = len > 0 && !(len == 2 && c < 0xC2) && !(len == 4 && c > 0xF4);
        if (ok && i + len > n) break;   // an incomplete character at the end: dropped
        for (size_t k = 1; ok && k < len; ++k) ok = (p[i + k] & 0xC0) == 0x80;
        if (ok && len == 3) {   // no overlong forms, no surrogates
            const unsigned char c1 = p[i + 1];
            ok = !(c == 0xE0 && c1 < 0xA0) && !(c == 0xED && c1 >= 0xA0);
        }
        if (ok && len == 4) {   // no overlong forms, nothing past U+10FFFF
            const unsigned char c1 = p[i + 1];
            ok = !(c == 0xF0 && c1 < 0x90) && !(c == 0xF4 && c1 >= 0x90);
        }
        if (ok) { out.append(s, i, len); i += len; }
        else    { out += "\xEF\xBF\xBD"; ++i; }
    }
    return out;
}

}  // namespace lite
