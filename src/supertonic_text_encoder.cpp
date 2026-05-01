#include "supertonic_internal.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tts_cpp::supertonic::detail {
namespace {

struct f32_tensor {
    std::vector<float> data;
    int64_t ne[4] = {1, 1, 1, 1};
};

f32_tensor read_f32(const supertonic_model & m, const std::string & source_name) {
    ggml_tensor * t = require_source_tensor(m, source_name);
    f32_tensor out;
    for (int i = 0; i < 4; ++i) out.ne[i] = t->ne[i];
    out.data.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data.data(), 0, ggml_nbytes(t));
    return out;
}

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f)); }
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

void linear1x1(const std::vector<float> & x, int L, int IC,
               const f32_tensor & w, const f32_tensor * b,
               int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[oc] : 0.0f;
            const size_t woff = (size_t) oc * IC;
            for (int ic = 0; ic < IC; ++ic) sum += w.data[woff + ic] * x[(size_t) t * IC + ic];
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

void dense_time(const std::vector<float> & x, int L, int IC,
                const f32_tensor & w, const f32_tensor & b,
                int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b.data[oc];
            for (int ic = 0; ic < IC; ++ic) sum += w.data[(size_t) oc * IC + ic] * x[(size_t) t * IC + ic];
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

void dense_time_matmul(const std::vector<float> & x, int L, int IC,
                       const f32_tensor & w, const f32_tensor & b,
                       int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    // ONNX MatMul constants are row-major [IC, OC]; PyTorch Linear weights
    // transposed these at load time, but here we consume the raw ONNX tensor.
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b.data[oc];
            for (int ic = 0; ic < IC; ++ic) sum += x[(size_t) t * IC + ic] * w.data[(size_t) ic * OC + oc];
            y[(size_t) t * OC + oc] = sum;
        }
    }
}

void depthwise_conv1d_same(const std::vector<float> & x, int L, int C,
                           const f32_tensor & w, const f32_tensor & b,
                           int K, int dilation, std::vector<float> & y) {
    y.assign((size_t) L * C, 0.0f);
    const int total_pad = (K - 1) * dilation;
    const int pad_left = total_pad / 2;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float sum = b.data[c];
            const size_t wbase = (size_t) c * K;
            for (int k = 0; k < K; ++k) {
                int src_t = t + k * dilation - pad_left;
                src_t = std::max(0, std::min(L - 1, src_t));
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

void convnext_block(const supertonic_model & m, const std::string & p,
                    std::vector<float> & x, int L, int C) {
    f32_tensor dw_w = read_f32(m, p + ".dwconv.weight");
    f32_tensor dw_b = read_f32(m, p + ".dwconv.bias");
    f32_tensor ln_g = read_f32(m, p + ".norm.norm.weight");
    f32_tensor ln_b = read_f32(m, p + ".norm.norm.bias");
    f32_tensor pw1_w = read_f32(m, p + ".pwconv1.weight");
    f32_tensor pw1_b = read_f32(m, p + ".pwconv1.bias");
    f32_tensor pw2_w = read_f32(m, p + ".pwconv2.weight");
    f32_tensor pw2_b = read_f32(m, p + ".pwconv2.bias");
    f32_tensor gamma = read_f32(m, p + ".gamma");
    std::vector<float> residual = x;
    std::vector<float> y, z;
    depthwise_conv1d_same(x, L, C, dw_w, dw_b, (int) dw_w.ne[0], 1, y);
    layer_norm_channel(y, L, C, ln_g, ln_b);
    linear1x1(y, L, C, pw1_w, &pw1_b, (int) pw1_w.ne[2], z);
    for (float & v : z) v = gelu(v);
    linear1x1(z, L, (int) pw1_w.ne[2], pw2_w, &pw2_b, C, y);
    for (size_t i = 0; i < x.size(); ++i) {
        int c = (int) (i % C);
        x[i] = residual[i] + gamma.data[c] * y[i];
    }
}

void relpos_attention(const supertonic_model & m, int idx, std::vector<float> & x, int L, int C) {
    const int H = 4;
    const int D = C / H;
    const float scale = 1.0f / std::sqrt((float) D);
    const std::string p = "text_encoder:tts.ttl.text_encoder.attn_encoder.attn_layers." + std::to_string(idx);
    f32_tensor q_w = read_f32(m, p + ".conv_q.weight");
    f32_tensor q_b = read_f32(m, p + ".conv_q.bias");
    f32_tensor k_w = read_f32(m, p + ".conv_k.weight");
    f32_tensor k_b = read_f32(m, p + ".conv_k.bias");
    f32_tensor v_w = read_f32(m, p + ".conv_v.weight");
    f32_tensor v_b = read_f32(m, p + ".conv_v.bias");
    f32_tensor o_w = read_f32(m, p + ".conv_o.weight");
    f32_tensor o_b = read_f32(m, p + ".conv_o.bias");
    f32_tensor rel_k = read_f32(m, p + ".emb_rel_k");
    f32_tensor rel_v = read_f32(m, p + ".emb_rel_v");
    std::vector<float> q, k, v;
    linear1x1(x, L, C, q_w, &q_b, C, q);
    linear1x1(x, L, C, k_w, &k_b, C, k);
    linear1x1(x, L, C, v_w, &v_b, C, v);
    std::vector<float> out((size_t) L * C, 0.0f);
    std::vector<float> scores(L), probs(L);
    const int half_window = 4;
    for (int h = 0; h < H; ++h) {
        for (int qi = 0; qi < L; ++qi) {
            float max_score = -INFINITY;
            for (int kj = 0; kj < L; ++kj) {
                float s = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s += q[(size_t) qi * C + h * D + d] * scale * k[(size_t) kj * C + h * D + d];
                }
                int rel_pos = kj - qi;
                if (rel_pos >= -half_window && rel_pos <= half_window) {
                    int ridx = rel_pos + half_window;
                    for (int d = 0; d < D; ++d) {
                        s += q[(size_t) qi * C + h * D + d] * scale * rel_k.data[(size_t) ridx * D + d];
                    }
                }
                scores[kj] = s;
                max_score = std::max(max_score, s);
            }
            float denom = 0.0f;
            for (int kj = 0; kj < L; ++kj) {
                probs[kj] = std::exp(scores[kj] - max_score);
                denom += probs[kj];
            }
            for (int kj = 0; kj < L; ++kj) probs[kj] /= denom;
            for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int kj = 0; kj < L; ++kj) {
                    sum += probs[kj] * v[(size_t) kj * C + h * D + d];
                    int rel_pos = kj - qi;
                    if (rel_pos >= -half_window && rel_pos <= half_window) {
                        int ridx = rel_pos + half_window;
                        sum += probs[kj] * rel_v.data[(size_t) ridx * D + d];
                    }
                }
                out[(size_t) qi * C + h * D + d] = sum;
            }
        }
    }
    std::vector<float> proj;
    linear1x1(out, L, C, o_w, &o_b, C, proj);
    x.swap(proj);
}

void ffn_block(const supertonic_model & m, int idx, std::vector<float> & x, int L, int C) {
    const std::string p = "text_encoder:tts.ttl.text_encoder.attn_encoder.ffn_layers." + std::to_string(idx);
    f32_tensor w1 = read_f32(m, p + ".conv_1.weight");
    f32_tensor b1 = read_f32(m, p + ".conv_1.bias");
    f32_tensor w2 = read_f32(m, p + ".conv_2.weight");
    f32_tensor b2 = read_f32(m, p + ".conv_2.bias");
    std::vector<float> y;
    linear1x1(x, L, C, w1, &b1, (int) w1.ne[2], y);
    for (float & v : y) v = relu(v);
    linear1x1(y, L, (int) w1.ne[2], w2, &b2, C, x);
}

void speech_prompted_attention(const supertonic_model & m, int idx,
                               std::vector<float> & x_lc, int L,
                               const float * style_ttl,
                               std::vector<float> & out_lc) {
    const int C = 256;
    const int half = 128;
    const int Lctx = 50;
    const float scale = 1.0f / 16.0f;
    const int attn_num = idx + 1;
    const std::string p = "text_encoder:tts.ttl.speech_prompted_text_encoder.attention" + std::to_string(attn_num);
    f32_tensor q_w = read_f32(m, "text_encoder:" + std::string(idx == 0 ? "onnx::MatMul_3678" : "onnx::MatMul_3682"));
    f32_tensor q_b = read_f32(m, p + ".W_query.linear.bias");
    f32_tensor kv_w = read_f32(m, "text_encoder:" + std::string(idx == 0 ? "onnx::MatMul_3680" : "onnx::MatMul_3684"));
    f32_tensor kv_b = read_f32(m, p + ".W_value.linear.bias");
    f32_tensor out_w = read_f32(m, "text_encoder:" + std::string(idx == 0 ? "onnx::MatMul_3681" : "onnx::MatMul_3685"));
    f32_tensor out_b = read_f32(m, p + ".out_fc.linear.bias");
    f32_tensor tanh_k = read_f32(m, "text_encoder:/speech_prompted_text_encoder/attention" + std::to_string(attn_num) + "/tanh/Tanh_output_0");

    std::vector<float> q;
    dense_time_matmul(x_lc, L, C, q_w, q_b, C, q);

    std::vector<float> style((size_t) Lctx * C);
    // style_ttl is NumPy row-major [1, 50, 256].
    for (int t = 0; t < Lctx; ++t) {
        for (int c = 0; c < C; ++c) style[(size_t) t * C + c] = style_ttl[(size_t) t * C + c];
    }
    std::vector<float> kv;
    dense_time_matmul(style, Lctx, C, kv_w, kv_b, C, kv);

    std::vector<float> attn((size_t) 2 * L * Lctx);
    std::vector<float> scores(Lctx);
    for (int part = 0; part < 2; ++part) {
        for (int t = 0; t < L; ++t) {
            float max_score = -INFINITY;
            for (int j = 0; j < Lctx; ++j) {
                float s = 0.0f;
                for (int d = 0; d < half; ++d) {
                    // tanh_k shape [2, 1, 128, 50] row-major.
                    float k = tanh_k.data[((size_t) part * half + d) * Lctx + j];
                    s += q[(size_t) t * C + part * half + d] * k * scale;
                }
                scores[j] = s;
                max_score = std::max(max_score, s);
            }
            float denom = 0.0f;
            for (int j = 0; j < Lctx; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                denom += scores[j];
            }
            for (int j = 0; j < Lctx; ++j) attn[((size_t) part * L + t) * Lctx + j] = scores[j] / denom;
        }
    }

    std::vector<float> merged((size_t) L * C, 0.0f);
    for (int part = 0; part < 2; ++part) {
        for (int t = 0; t < L; ++t) {
            for (int d = 0; d < half; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < Lctx; ++j) {
                    // kv_stacked = [k, v]; part 0 uses k split, part 1 uses v split.
                    sum += attn[((size_t) part * L + t) * Lctx + j] * kv[(size_t) j * C + part * half + d];
                }
                merged[(size_t) t * C + part * half + d] = sum;
            }
        }
    }
    dense_time_matmul(merged, L, C, out_w, out_b, C, out_lc);
}

} // namespace

bool supertonic_text_encoder_forward_cpu(const supertonic_model & model,
                                         const int64_t * text_ids,
                                         int text_len,
                                         const float * style_ttl,
                                         std::vector<float> & text_emb_out,
                                         std::string * error) {
    try {
        const int C = 256;
        const int L = text_len;
        f32_tensor emb = read_f32(model, "text_encoder:tts.ttl.text_encoder.text_embedder.char_embedder.weight");
        std::vector<float> x((size_t) L * C);
        for (int t = 0; t < L; ++t) {
            int64_t id = text_ids[t];
            if (id < 0 || id >= emb.ne[1]) throw std::runtime_error("text id out of range");
            for (int c = 0; c < C; ++c) x[(size_t) t * C + c] = emb.data[(size_t) id * C + c];
        }

        for (int i = 0; i < 6; ++i) {
            convnext_block(model, "text_encoder:tts.ttl.text_encoder.convnext.convnext." + std::to_string(i), x, L, C);
        }
        std::vector<float> convnext_out = x;

        for (int i = 0; i < 4; ++i) {
            std::vector<float> residual = x;
            relpos_attention(model, i, x, L, C);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.bias"));
            residual = x;
            ffn_block(model, i, x, L, C);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "text_encoder:tts.ttl.text_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.bias"));
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += convnext_out[i];

        std::vector<float> shared_residual = x; // [L, C]
        std::vector<float> attn_out;
        speech_prompted_attention(model, 0, x, L, style_ttl, attn_out);
        for (size_t i = 0; i < x.size(); ++i) x[i] = shared_residual[i] + attn_out[i];
        speech_prompted_attention(model, 1, x, L, style_ttl, attn_out);
        for (size_t i = 0; i < x.size(); ++i) x[i] = shared_residual[i] + attn_out[i];

        layer_norm_channel(
            x, L, C,
            read_f32(model, "text_encoder:tts.ttl.speech_prompted_text_encoder.norm.norm.weight"),
            read_f32(model, "text_encoder:tts.ttl.speech_prompted_text_encoder.norm.norm.bias"));

        // Return in ONNX/PyTorch shape [1, 256, L] row-major.
        text_emb_out.assign((size_t) C * L, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < L; ++t) text_emb_out[(size_t) c * L + t] = x[(size_t) t * C + c];
        }
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace tts_cpp::supertonic::detail
