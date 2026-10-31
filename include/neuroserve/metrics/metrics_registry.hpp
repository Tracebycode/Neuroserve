#pragma once

#include <atomic>
#include <vector>
#include <string>
#include <mutex>
#include <cstdint>

namespace neuroserve {

struct Percentiles {
    double p50{0.0};
    double p95{0.0};
    double p99{0.0};
};

class MetricsRegistry {
public:
    static MetricsRegistry& instance();

    void record_request() noexcept { requests_total_.fetch_add(1, std::memory_order_relaxed); }
    void record_success() noexcept { requests_success_total_.fetch_add(1, std::memory_order_relaxed); }
    void record_failure() noexcept { requests_failed_total_.fetch_add(1, std::memory_order_relaxed); }
    void record_rejection() noexcept { requests_rejected_total_.fetch_add(1, std::memory_order_relaxed); }
    void record_model_inference(size_t batch_size = 1) noexcept {
        model_inference_count_.fetch_add(1, std::memory_order_relaxed);
        total_batched_items_.fetch_add(batch_size, std::memory_order_relaxed);
    }

    void record_inference_latency_us(uint64_t us);
    void record_queue_latency_us(uint64_t us);
    void record_total_latency_us(uint64_t us);

    Percentiles get_inference_percentiles();
    Percentiles get_queue_percentiles();
    Percentiles get_total_percentiles();

    std::string to_prometheus(size_t current_queue_depth, size_t active_workers) ;

    uint64_t total_requests() const noexcept { return requests_total_.load(std::memory_order_relaxed); }
    uint64_t rejected_requests() const noexcept { return requests_rejected_total_.load(std::memory_order_relaxed); }

    void reset();

private:
    MetricsRegistry() = default;

    Percentiles compute_percentiles(std::vector<uint64_t>& samples);

    std::atomic<uint64_t> requests_total_{0};
    std::atomic<uint64_t> requests_success_total_{0};
    std::atomic<uint64_t> requests_failed_total_{0};
    std::atomic<uint64_t> requests_rejected_total_{0};
    std::atomic<uint64_t> model_inference_count_{0};
    std::atomic<uint64_t> total_batched_items_{0};

    mutable std::mutex latency_mutex_;
    std::vector<uint64_t> inference_samples_us_;
    std::vector<uint64_t> queue_samples_us_;
    std::vector<uint64_t> total_samples_us_;
    static constexpr size_t MAX_SAMPLES = 20000;
};

} // namespace neuroserve

