// logit-cmp A.logits B.logits N_VOCAB [STEPS]: per step, top-1 agreement and KL(softmax(A) || softmax(B)) in double.
// The Gate 1 bar for stages e/f (the maintainer, 2026-10-09): the port must stay within llama.cpp's own
// placement-to-placement top-1 agreement and KL on the same five prompts.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: logit-cmp A B n_vocab [steps [per-step-out]]\n"); return 2; }
    FILE * fa = fopen(argv[1], "rb"); FILE * fb = fopen(argv[2], "rb");
    if (!fa || !fb) { fprintf(stderr, "cannot open\n"); return 2; }
    const int nv = atoi(argv[3]); const int max_steps = argc > 4 ? atoi(argv[4]) : 1 << 30;
    FILE * fs = argc > 5 ? fopen(argv[5], "w") : nullptr;   // optional: per-step "kl top1_same" for bootstraps
    std::vector<float> a(nv), b(nv);
    int steps = 0, top1 = 0; double kl_sum = 0, kl_max = 0;
    while (steps < max_steps && fread(a.data(), 4, nv, fa) == (size_t) nv && fread(b.data(), 4, nv, fb) == (size_t) nv) {
        const int ia = std::max_element(a.begin(), a.end()) - a.begin(), ib = std::max_element(b.begin(), b.end()) - b.begin();
        top1 += ia == ib;
        const double ma = a[ia], mb = b[ib];
        double za = 0, zb = 0;
        for (int i = 0; i < nv; ++i) { za += exp(a[i] - ma); zb += exp(b[i] - mb); }
        const double la = log(za), lb = log(zb);
        double kl = 0;
        for (int i = 0; i < nv; ++i) {
            const double lpa = a[i] - ma - la, lpb = b[i] - mb - lb;
            kl += exp(lpa) * (lpa - lpb);
        }
        kl = std::max(kl, 0.0);
        kl_sum += kl; kl_max = std::max(kl_max, kl); steps++;
        if (fs) fprintf(fs, "%.9g %d\n", kl, (int) (ia == ib));
    }
    printf("{\"steps\":%d,\"top1_agree\":%d,\"top1_rate\":%.4f,\"kl_mean\":%.6g,\"kl_max\":%.6g}\n",
           steps, top1, steps ? (double) top1 / steps : 0.0, steps ? kl_sum / steps : 0.0, kl_max);
    return 0;
}
