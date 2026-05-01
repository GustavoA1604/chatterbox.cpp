#pragma once

#include <string>
#include <vector>

namespace tts_cpp::supertonic {

struct EngineOptions {
    std::string model_gguf_path;
    std::string voice = "M1";
    std::string language = "en";
    int steps = 5;
    float speed = 1.05f;
    int seed = 42;
};

struct SynthesisResult {
    std::vector<float> pcm;
    int sample_rate = 44100;
    float duration_s = 0.0f;
};

SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

} // namespace tts_cpp::supertonic
