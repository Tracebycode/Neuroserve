#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <span>
#include <optional>

namespace neuroserve {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;

enum class DataType {
    Float32,
    Int64,
    Int32,
    Unknown
};

struct TensorData {
    std::string name;
    DataType dtype{DataType::Float32};
    std::vector<int64_t> shape;
    std::vector<float> float_data;
    std::vector<int64_t> int64_data;

    size_t element_count() const {
        if (shape.empty()) return 0;
        size_t count = 1;
        for (auto dim : shape) {
            if (dim > 0) count *= static_cast<size_t>(dim);
        }
        return count;
    }
};

struct InferenceRequest {
    uint64_t request_id{0};
    std::string model_name;
    std::vector<TensorData> inputs;
    TimePoint arrival_time{Clock::now()};
    TimePoint dequeue_time{};
    TimePoint start_inference_time{};
    TimePoint end_inference_time{};
};

struct InferenceResponse {
    uint64_t request_id{0};
    bool success{false};
    int status_code{200};
    std::string error_message;
    std::vector<TensorData> outputs;
    Microseconds queue_latency{0};
    Microseconds inference_latency{0};
    Microseconds total_latency{0};
};

} // namespace neuroserve

