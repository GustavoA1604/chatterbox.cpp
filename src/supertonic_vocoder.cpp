#include "supertonic_internal.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace tts_cpp::supertonic::detail {
namespace {

struct f32_tensor {
    std::vector<float> data;
    int64_t ne[4] = {1, 1, 1, 1}; // ggml order; ONNX row-major is reversed
};

f32_tensor read_f32(const supertonic_model & m, const std::string & source_name) {
    ggml_tensor * t = require_source_tensor(m, source_name);
    f32_tensor out;
    for (int i = 0; i < 4; ++i) out.ne[i] = t->ne[i];
    out.data.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data.data(), 0, ggml_nbytes(t));
    return out;
}

float scalar_f32(const supertonic_model & m, const std::string & source_name) {
    f32_tensor t = read_f32(m, source_name);
    if (t.data.empty()) throw std::runtime_error("empty scalar tensor: " + source_name);
    return t.data[0];
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f));
}

void linear1x1(const std::vector<float> & x, int L, int IC,
               const f32_tensor & w, const f32_tensor * b,
               int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    // ONNX Conv weight is row-major [OC, IC, 1]; raw index ((oc*IC + ic)*1).
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[oc] : 0.0f;
            const size_t woff = (size_t) oc * IC;
            for (int ic = 0; ic < IC; ++ic) {
                sum += w.data[woff + ic] * x[(size_t) t * IC + ic];
            }
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

void conv1d_causal(const std::vector<float> & x, int L, int IC,
                   const f32_tensor & w, const f32_tensor * b,
                   int K, int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    const int pad_left = K - 1;
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[oc] : 0.0f;
            for (int ic = 0; ic < IC; ++ic) {
                const size_t wbase = ((size_t) oc * IC + ic) * K;
                for (int k = 0; k < K; ++k) {
                    int src_t = t + k - pad_left;
                    if (src_t < 0) src_t = 0; // replicate pad
                    sum += w.data[wbase + k] * x[(size_t) src_t * IC + ic];
                }
            }
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

void depthwise_conv1d_causal(const std::vector<float> & x, int L, int C,
                             const f32_tensor & w, const f32_tensor & b,
                             int K, int dilation, std::vector<float> & y) {
    y.assign((size_t) L * C, 0.0f);
    const int pad_left = (K - 1) * dilation;
    // ONNX depthwise Conv weight is [C, 1, K].
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float sum = b.data[c];
            const size_t wbase = (size_t) c * K;
            for (int k = 0; k < K; ++k) {
                int src_t = t + k * dilation - pad_left;
                if (src_t < 0) src_t = 0;
                sum += w.data[wbase + k] * x[(size_t) src_t * C + c];
            }
            y[(size_t) t * C + c] = sum;
        }
    }
}

void layer_norm_channel(std::vector<float> & x, int L, int C,
                        const f32_tensor & gamma, const f32_tensor & beta,
                        float eps = 1e-6f) {
    for (int t = 0; t < L; ++t) {
        float mean = 0.0f;
        for (int c = 0; c < C; ++c) mean += x[(size_t) t * C + c];
        mean /= (float) C;
        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            float d = x[(size_t) t * C + c] - mean;
            var += d * d;
        }
        float inv = 1.0f / std::sqrt(var / (float) C + eps);
        for (int c = 0; c < C; ++c) {
            float v = (x[(size_t) t * C + c] - mean) * inv;
            x[(size_t) t * C + c] = v * gamma.data[c] + beta.data[c];
        }
    }
}

void batch_norm_channel(std::vector<float> & x, int L, int C,
                        const f32_tensor & gamma, const f32_tensor & beta,
                        const f32_tensor & running_mean, const f32_tensor & running_var,
                        float eps = 1e-5f) {
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float v = (x[(size_t) t * C + c] - running_mean.data[c]) /
                      std::sqrt(running_var.data[c] + eps);
            x[(size_t) t * C + c] = v * gamma.data[c] + beta.data[c];
        }
    }
}

void convnext_block(const supertonic_model & m, int idx,
                    std::vector<float> & x, int L, int C) {
    const std::string p = "vocoder:tts.ae.decoder.convnext." + std::to_string(idx);
    f32_tensor dw_w = read_f32(m, p + ".dwconv.net.weight");
    f32_tensor dw_b = read_f32(m, p + ".dwconv.net.bias");
    f32_tensor ln_g = read_f32(m, p + ".norm.norm.weight");
    f32_tensor ln_b = read_f32(m, p + ".norm.norm.bias");
    f32_tensor pw1_w = read_f32(m, p + ".pwconv1.weight");
    f32_tensor pw1_b = read_f32(m, p + ".pwconv1.bias");
    f32_tensor pw2_w = read_f32(m, p + ".pwconv2.weight");
    f32_tensor pw2_b = read_f32(m, p + ".pwconv2.bias");
    f32_tensor gamma = read_f32(m, p + ".gamma");

    std::vector<float> residual = x;
    std::vector<float> y;
    const int K = (int) dw_w.ne[0];
    static const int dilations[10] = {1, 2, 4, 1, 2, 4, 1, 1, 1, 1};
    depthwise_conv1d_causal(x, L, C, dw_w, dw_b, K, dilations[idx], y);
    layer_norm_channel(y, L, C, ln_g, ln_b);

    std::vector<float> z;
    const int hidden = (int) pw1_w.ne[2];
    linear1x1(y, L, C, pw1_w, &pw1_b, hidden, z);
    for (float & v : z) v = gelu(v);
    linear1x1(z, L, hidden, pw2_w, &pw2_b, C, y);

    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            x[(size_t) t * C + c] = residual[(size_t) t * C + c] +
                                    gamma.data[c] * y[(size_t) t * C + c];
        }
    }
}

} // namespace

bool supertonic_vocoder_forward_cpu(const supertonic_model & model,
                                    const float * latent,
                                    int latent_len,
                                    std::vector<float> & wav_out,
                                    std::string * error) {
    try {
        const int C_latent = model.hparams.latent_dim;       // 24
        const int factor = model.hparams.ttl_chunk_compress_factor; // 6
        const int latent_channels = model.hparams.latent_channels;  // 144
        if (latent_len <= 0) throw std::runtime_error("latent_len must be positive");

        // Input latent is NumPy/PyTorch row-major [1, 144, L].  Vocoder unpacks
        // it as [1, 24, 6, L] -> [1, 24, L, 6] -> [1, 24, L*6].
        const int T0 = latent_len * factor;
        std::vector<float> x((size_t) T0 * C_latent);
        for (int c = 0; c < C_latent; ++c) {
            for (int t = 0; t < latent_len; ++t) {
                for (int r = 0; r < factor; ++r) {
                    int src_c = c * factor + r;
                    x[(size_t) (t * factor + r) * C_latent + c] =
                        latent[(size_t) src_c * latent_len + t];
                }
            }
        }

        float normalizer_scale = scalar_f32(model, "vocoder:tts.ttl.normalizer.scale");
        f32_tensor mean = read_f32(model, "vocoder:tts.ae.latent_mean");
        f32_tensor std = read_f32(model, "vocoder:tts.ae.latent_std");
        for (int t = 0; t < T0; ++t) {
            for (int c = 0; c < C_latent; ++c) {
                float v = x[(size_t) t * C_latent + c] / normalizer_scale;
                x[(size_t) t * C_latent + c] = v * std.data[c] + mean.data[c];
            }
        }

        f32_tensor embed_w = read_f32(model, "vocoder:onnx::Conv_1440");
        f32_tensor embed_b = read_f32(model, "vocoder:onnx::Conv_1441");
        std::vector<float> y;
        conv1d_causal(x, T0, C_latent, embed_w, &embed_b,
                      (int) embed_w.ne[0], (int) embed_w.ne[2], y);
        x.swap(y);
        const int C = (int) embed_w.ne[2]; // 512

        for (int i = 0; i < 10; ++i) {
            convnext_block(model, i, x, T0, C);
        }

        batch_norm_channel(
            x, T0, C,
            read_f32(model, "vocoder:tts.ae.decoder.final_norm.norm.weight"),
            read_f32(model, "vocoder:tts.ae.decoder.final_norm.norm.bias"),
            read_f32(model, "vocoder:tts.ae.decoder.final_norm.norm.running_mean"),
            read_f32(model, "vocoder:tts.ae.decoder.final_norm.norm.running_var"));

        f32_tensor h1_w = read_f32(model, "vocoder:tts.ae.decoder.head.layer1.net.weight");
        f32_tensor h1_b = read_f32(model, "vocoder:tts.ae.decoder.head.layer1.net.bias");
        conv1d_causal(x, T0, C, h1_w, &h1_b, (int) h1_w.ne[0], (int) h1_w.ne[2], y);
        float prelu = scalar_f32(model, "vocoder:onnx::PRelu_1505");
        for (float & v : y) {
            if (v < 0.0f) v *= prelu;
        }

        f32_tensor h2_w = read_f32(model, "vocoder:tts.ae.decoder.head.layer2.weight");
        std::vector<float> z;
        linear1x1(y, T0, (int) h1_w.ne[2], h2_w, nullptr, (int) h2_w.ne[2], z);

        wav_out = std::move(z);
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace tts_cpp::supertonic::detail
