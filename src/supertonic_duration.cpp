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

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
inline float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f)); }

void linear1x1(const std::vector<float> & x, int L, int IC,
               const f32_tensor & w, const f32_tensor * b,
               int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
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
                src_t = std::max(0, std::min(L - 1, src_t)); // replicate pad
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
                    std::vector<float> & x, int L, int C, int dilation = 1) {
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
    std::vector<float> y;
    depthwise_conv1d_same(x, L, C, dw_w, dw_b, (int) dw_w.ne[0], dilation, y);
    layer_norm_channel(y, L, C, ln_g, ln_b);

    std::vector<float> z;
    int hidden = (int) pw1_w.ne[2];
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

void self_attention(const supertonic_model & m, int idx, std::vector<float> & x, int L, int C) {
    const int H = 2;
    const int D = C / H;
    const float scale = 1.0f / std::sqrt((float) D);
    const std::string p = "duration:tts.dp.sentence_encoder.attn_encoder.attn_layers." + std::to_string(idx);

    f32_tensor q_w = read_f32(m, p + ".conv_q.weight");
    f32_tensor q_b = read_f32(m, p + ".conv_q.bias");
    f32_tensor k_w = read_f32(m, p + ".conv_k.weight");
    f32_tensor k_b = read_f32(m, p + ".conv_k.bias");
    f32_tensor v_w = read_f32(m, p + ".conv_v.weight");
    f32_tensor v_b = read_f32(m, p + ".conv_v.bias");
    f32_tensor o_w = read_f32(m, p + ".conv_o.weight");
    f32_tensor o_b = read_f32(m, p + ".conv_o.bias");
    f32_tensor rel_k = read_f32(m, p + ".emb_rel_k"); // [1, 9, D]
    f32_tensor rel_v = read_f32(m, p + ".emb_rel_v");

    std::vector<float> q, k, v;
    linear1x1(x, L, C, q_w, &q_b, C, q);
    linear1x1(x, L, C, k_w, &k_b, C, k);
    linear1x1(x, L, C, v_w, &v_b, C, v);

    std::vector<float> out((size_t) L * C, 0.0f);
    std::vector<float> scores(L);
    std::vector<float> probs(L);
    const int half_window = 4;

    for (int h = 0; h < H; ++h) {
        for (int qi = 0; qi < L; ++qi) {
            float max_score = -INFINITY;
            for (int kj = 0; kj < L; ++kj) {
                float s = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s += q[(size_t) qi * C + h * D + d] * scale *
                         k[(size_t) kj * C + h * D + d];
                }
                int rel_pos = kj - qi;
                if (rel_pos >= -half_window && rel_pos <= half_window) {
                    int ridx = rel_pos + half_window;
                    for (int d = 0; d < D; ++d) {
                        s += q[(size_t) qi * C + h * D + d] * scale *
                             rel_k.data[(size_t) ridx * D + d];
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
    const std::string p = "duration:tts.dp.sentence_encoder.attn_encoder.ffn_layers." + std::to_string(idx);
    f32_tensor w1 = read_f32(m, p + ".conv_1.weight");
    f32_tensor b1 = read_f32(m, p + ".conv_1.bias");
    f32_tensor w2 = read_f32(m, p + ".conv_2.weight");
    f32_tensor b2 = read_f32(m, p + ".conv_2.bias");
    std::vector<float> y;
    linear1x1(x, L, C, w1, &b1, (int) w1.ne[2], y);
    for (float & v : y) v = relu(v);
    linear1x1(y, L, (int) w1.ne[2], w2, &b2, C, x);
}

void dense(const std::vector<float> & x, const f32_tensor & w, const f32_tensor & b,
           int IC, int OC, std::vector<float> & y) {
    y.assign(OC, 0.0f);
    for (int oc = 0; oc < OC; ++oc) {
        float sum = b.data[oc];
        for (int ic = 0; ic < IC; ++ic) sum += w.data[(size_t) oc * IC + ic] * x[ic];
        y[oc] = sum;
    }
}

} // namespace

bool supertonic_duration_forward_cpu(const supertonic_model & model,
                                     const int64_t * text_ids,
                                     int text_len,
                                     const float * style_dp,
                                     float & duration_out,
                                     std::string * error) {
    try {
        const int C = 64;
        const int L = text_len + 1;
        f32_tensor emb = read_f32(model, "duration:tts.dp.sentence_encoder.text_embedder.char_embedder.weight");
        f32_tensor sentence = read_f32(model, "duration:tts.dp.sentence_encoder.sentence_token");

        std::vector<float> x((size_t) L * C, 0.0f);
        for (int c = 0; c < C; ++c) x[c] = sentence.data[c];
        for (int t = 0; t < text_len; ++t) {
            int64_t id = text_ids[t];
            if (id < 0 || id >= emb.ne[1]) throw std::runtime_error("text id out of range");
            for (int c = 0; c < C; ++c) {
                x[(size_t) (t + 1) * C + c] = emb.data[(size_t) id * C + c];
            }
        }

        for (int i = 0; i < 6; ++i) {
            const std::string p = "duration:tts.dp.sentence_encoder.convnext.convnext." + std::to_string(i);
            convnext_block(model, p, x, L, C);
        }
        std::vector<float> convnext_out = x;

        for (int i = 0; i < 2; ++i) {
            std::vector<float> residual = x;
            self_attention(model, i, x, L, C);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_1." + std::to_string(i) + ".norm.bias"));

            residual = x;
            ffn_block(model, i, x, L, C);
            for (size_t j = 0; j < x.size(); ++j) x[j] += residual[j];
            layer_norm_channel(
                x, L, C,
                read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.weight"),
                read_f32(model, "duration:tts.dp.sentence_encoder.attn_encoder.norm_layers_2." + std::to_string(i) + ".norm.bias"));
        }

        for (size_t i = 0; i < x.size(); ++i) x[i] += convnext_out[i];

        std::vector<float> sentence_repr(C);
        for (int c = 0; c < C; ++c) sentence_repr[c] = x[c];
        std::vector<float> projected;
        f32_tensor proj_w = read_f32(model, "duration:tts.dp.sentence_encoder.proj_out.net.weight");
        linear1x1(sentence_repr, 1, C, proj_w, nullptr, C, projected);

        std::vector<float> combined(192);
        for (int c = 0; c < C; ++c) combined[c] = projected[c];
        for (int i = 0; i < 128; ++i) combined[C + i] = style_dp[i];

        std::vector<float> h;
        dense(combined,
              read_f32(model, "duration:tts.dp.predictor.layers.0.weight"),
              read_f32(model, "duration:tts.dp.predictor.layers.0.bias"),
              192, 128, h);
        float prelu = read_f32(model, "duration:tts.dp.predictor.activation.weight").data[0];
        for (float & v : h) if (v < 0.0f) v *= prelu;
        std::vector<float> out;
        dense(h,
              read_f32(model, "duration:tts.dp.predictor.layers.1.weight"),
              read_f32(model, "duration:tts.dp.predictor.layers.1.bias"),
              128, 1, out);
        duration_out = std::exp(out[0]);
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace tts_cpp::supertonic::detail
