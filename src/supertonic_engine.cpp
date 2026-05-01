#include "tts-cpp/supertonic/engine.h"

#include "supertonic_internal.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace tts_cpp::supertonic {

using namespace detail;

namespace {

std::vector<float> read_tensor_f32(ggml_tensor * t) {
    std::vector<float> out((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

} // namespace

SynthesisResult synthesize(const EngineOptions & opts, const std::string & text) {
    if (opts.model_gguf_path.empty()) throw std::runtime_error("Supertonic model_gguf_path is required");
    if (text.empty()) throw std::runtime_error("Supertonic text is empty");
    if (opts.steps <= 0) throw std::runtime_error("Supertonic steps must be positive");
    if (opts.speed <= 0.0f) throw std::runtime_error("Supertonic speed must be positive");

    supertonic_model model;
    if (!load_supertonic_gguf(opts.model_gguf_path, model)) {
        throw std::runtime_error("failed to load Supertonic GGUF: " + opts.model_gguf_path);
    }

    try {
        auto vit = model.voices.find(opts.voice);
        if (vit == model.voices.end()) throw std::runtime_error("unknown Supertonic voice: " + opts.voice);
        std::vector<float> style_ttl = read_tensor_f32(vit->second.ttl);
        std::vector<float> style_dp  = read_tensor_f32(vit->second.dp);

        std::vector<int32_t> text_ids_i32;
        std::string normalized;
        std::string error;
        if (!supertonic_text_to_ids(model, text, opts.language, text_ids_i32, &normalized, &error)) {
            throw std::runtime_error("text preprocessing failed: " + error);
        }
        std::vector<int64_t> text_ids(text_ids_i32.begin(), text_ids_i32.end());

        float duration_raw = 0.0f;
        if (!supertonic_duration_forward_cpu(model, text_ids.data(), (int) text_ids.size(),
                                             style_dp.data(), duration_raw, &error)) {
            throw std::runtime_error("duration failed: " + error);
        }
        const float duration_s = duration_raw / opts.speed;
        const int sample_rate = model.hparams.sample_rate;
        const int chunk = model.hparams.base_chunk_size * model.hparams.ttl_chunk_compress_factor;
        const int wav_len = (int) (duration_s * sample_rate);
        const int latent_len = std::max(1, (wav_len + chunk - 1) / chunk);

        std::vector<float> text_emb;
        if (!supertonic_text_encoder_forward_cpu(model, text_ids.data(), (int) text_ids.size(),
                                                 style_ttl.data(), text_emb, &error)) {
            throw std::runtime_error("text encoder failed: " + error);
        }

        std::mt19937 rng(opts.seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> latent((size_t) model.hparams.latent_channels * latent_len);
        for (float & v : latent) v = normal(rng);
        std::vector<float> latent_mask((size_t) latent_len, 1.0f);

        std::vector<float> next;
        for (int step = 0; step < opts.steps; ++step) {
            if (!supertonic_vector_step_cpu(model, latent.data(), latent_len,
                                            text_emb.data(), (int) text_ids.size(),
                                            style_ttl.data(), latent_mask.data(),
                                            step, opts.steps, next, &error)) {
                throw std::runtime_error("vector estimator failed: " + error);
            }
            latent.swap(next);
        }

        std::vector<float> wav_full;
        if (!supertonic_vocoder_forward_cpu(model, latent.data(), latent_len, wav_full, &error)) {
            throw std::runtime_error("vocoder failed: " + error);
        }

        SynthesisResult result;
        result.sample_rate = sample_rate;
        result.duration_s = duration_s;
        result.pcm.assign(wav_full.begin(), wav_full.begin() + std::min((size_t) wav_len, wav_full.size()));
        free_supertonic_model(model);
        return result;
    } catch (...) {
        free_supertonic_model(model);
        throw;
    }
}

} // namespace tts_cpp::supertonic
