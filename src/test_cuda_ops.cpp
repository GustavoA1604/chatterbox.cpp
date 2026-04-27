// Standalone validation for the CUDA kernel(s) we patch in ggml.
//
// Currently covers:
//   - GGML_OP_CONV_TRANSPOSE_1D: warp-cooperative kernel rewrite shipped
//     in patches/ggml-cuda-chatterbox-ops.patch.  The patched kernel
//     parallelises IC across a 32-thread warp + narrows the input range
//     analytically, so its FP-reduction order differs from the stock
//     scalar kernel (and from CPU) — we tolerate up to 1e-3 absolute /
//     1e-3 relative.
//
// Each test runs the same op twice (once on CPU, once on CUDA) with
// identical inputs and compares element-by-element.  Exits non-zero on
// any mismatch.  Mirrors src/test_metal_ops.cpp for the Metal path.
//
// Companion shell smoke test for end-to-end audio-output equivalence
// (chatterbox binary, FORCE_GRAPHS opt-in) lives in
// scripts/test-chatterbox-cuda.sh.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int test_conv_transpose_1d(ggml_backend_t cpu, ggml_backend_t gpu,
                                  int IL, int IC, int OC, int K, int s0,
                                  const char * label) {
    fprintf(stderr, "[conv_transp_1d %-9s] ", label);

    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> kdata(K * OC * IC);
    std::vector<float> xdata(IL * IC);
    for (auto & v : kdata) v = dist(rng);
    for (auto & v : xdata) v = dist(rng);

    auto run_one = [&](ggml_backend_t backend) {
        // 256 MiB headroom — biggest test case below outputs ~5 MB,
        // tensor overhead is small, but ggml_gallocr_reserve wants
        // workspace too.
        static size_t buf_size = 256 * 1024 * 1024;
        std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, OC, IC);
        ggml_set_name(k, "k"); ggml_set_input(k);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IL, IC);
        ggml_set_name(x, "x"); ggml_set_input(x);
        ggml_tensor * y = ggml_conv_transpose_1d(ctx, k, x, s0, 0, 1);
        ggml_set_name(y, "y"); ggml_set_output(y);
        ggml_build_forward_expand(gf, y);
        auto * allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(allocr, gf);
        ggml_gallocr_alloc_graph(allocr, gf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "k"),
                                kdata.data(), 0, kdata.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"),
                                xdata.data(), 0, xdata.size() * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        ggml_tensor * out = ggml_graph_get_tensor(gf, "y");
        std::vector<float> res(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
        return res;
    };

    auto ref = run_one(cpu);
    auto got = run_one(gpu);

    if (ref.size() != got.size()) {
        fprintf(stderr, "FAIL: size mismatch cpu=%zu cuda=%zu\n", ref.size(), got.size());
        return 1;
    }

    int   bad     = 0;
    float max_abs = 0.f, max_rel = 0.f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        const float r = d / std::max(std::fabs(ref[i]), 1e-6f);
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
        // 1e-3 abs / 1e-3 rel: comfortably above warp-reduce FP-order
        // noise on the IC=512 cases (measured ~1e-5 typical, ~5e-5 worst)
        // and well below anything that would shift HiFT audio output.
        if (d > 1e-3f && r > 1e-3f) {
            if (bad < 5) {
                fprintf(stderr, "\n  mismatch @ %zu: cpu=%.6g cuda=%.6g abs=%.3e rel=%.3e",
                        i, ref[i], got[i], d, r);
            }
            ++bad;
        }
    }
    if (bad == 0) {
        fprintf(stderr,
                "OK (IL=%-5d IC=%-3d OC=%-3d K=%-2d s0=%d, max_abs=%.1e max_rel=%.1e, n=%zu)\n",
                IL, IC, OC, K, s0, max_abs, max_rel, ref.size());
        return 0;
    }
    fprintf(stderr, "\n[conv_transp_1d] FAIL: %d / %zu mismatched (max_abs=%.3e)\n",
            bad, ref.size(), max_abs);
    return 1;
}

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "CPU backend init failed\n"); return 1; }

    ggml_backend_t gpu = nullptr;
#ifdef GGML_USE_CUDA
    gpu = ggml_backend_cuda_init(0);
    fprintf(stderr, "Using CUDA backend\n");
#endif
    if (!gpu) {
        fprintf(stderr, "No CUDA backend compiled in; nothing to validate.\n");
        return 0;
    }

    int rc = 0;

    // HiFT-realistic upsample shapes (Turbo / MTL share the same vocoder).
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/130,  /*IC=*/512, /*OC=*/256, /*K=*/16, /*s0=*/8, "ups[0]");
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/1040, /*IC=*/256, /*OC=*/128, /*K=*/15, /*s0=*/5, "ups[1]");
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/5200, /*IC=*/128, /*OC=*/64,  /*K=*/11, /*s0=*/3, "ups[2]");

    // Warp-reduction edge cases: IC < warp width (some warp lanes get
    // zero work and contribute zero to the __shfl_xor sum).
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/64,   /*IC=*/8,   /*OC=*/16,  /*K=*/8,  /*s0=*/4, "ic_lt_warp");
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/64,   /*IC=*/16,  /*OC=*/16,  /*K=*/8,  /*s0=*/4, "ic_half_warp");
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/64,   /*IC=*/32,  /*OC=*/16,  /*K=*/8,  /*s0=*/4, "ic_eq_warp");
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/64,   /*IC=*/33,  /*OC=*/16,  /*K=*/8,  /*s0=*/4, "ic_warp+1");

    // Stride / kernel-size edge cases.
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/16,   /*IC=*/64,  /*OC=*/64,  /*K=*/4,  /*s0=*/4, "k_eq_s0");    // no overlap
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/16,   /*IC=*/64,  /*OC=*/64,  /*K=*/8,  /*s0=*/1, "s0_1");       // full overlap
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/16,   /*IC=*/64,  /*OC=*/64,  /*K=*/16, /*s0=*/1, "k_gt_il");    // K > IL ratio extreme

    // Tiny: catches out-of-bounds errors in i_start/i_end clamping.
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/1,    /*IC=*/1,   /*OC=*/1,   /*K=*/1,  /*s0=*/1, "1x1");
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/2,    /*IC=*/2,   /*OC=*/2,   /*K=*/3,  /*s0=*/1, "2x2");
    rc |= test_conv_transpose_1d(cpu, gpu, /*IL=*/10,   /*IC=*/3,   /*OC=*/4,   /*K=*/5,  /*s0=*/2, "tiny");

    fprintf(stderr, "\n%s\n", rc == 0 ? "All CUDA op tests PASSED" : "Some CUDA op tests FAILED");

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return rc;
}
