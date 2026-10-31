#include "neuroserve/metrics/metrics_registry.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace neuroserve {

MetricsRegistry& MetricsRegistry::instance() {
    static MetricsRegistry reg;
    return reg;
}

void MetricsRegistry::record_inference_latency_us(uint64_t us) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    if (inference_samples_us_.size() >= MAX_SAMPLES) {
        inference_samples_us_.erase(inference_samples_us_.begin());
    }
    inference_samples_us_.push_back(us);
}

void MetricsRegistry::record_queue_latency_us(uint64_t us) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    if (queue_samples_us_.size() >= MAX_SAMPLES) {
        queue_samples_us_.erase(queue_samples_us_.begin());
    }
    queue_samples_us_.push_back(us);
}

void MetricsRegistry::record_total_latency_us(uint64_t us) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    if (total_samples_us_.size() >= MAX_SAMPLES) {
        total_samples_us_.erase(total_samples_us_.begin());
    }
    total_samples_us_.push_back(us);
}

Percentiles MetricsRegistry::compute_percentiles(std::vector<uint64_t>& samples) {
    if (samples.empty()) {
        return {0.0, 0.0, 0.0};
    }
    std::sort(samples.begin(), samples.end());
    auto get_val = [&](double p) -> double {
        size_t idx = static_cast<size_t>(p * (samples.size() - 1));
        return static_cast<double>(samples[idx]);
    };
    return {get_val(0.50), get_val(0.95), get_val(0.99)};
}

Percentiles MetricsRegistry::get_inference_percentiles() {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    return compute_percentiles(inference_samples_us_);
}

Percentiles MetricsRegistry::get_queue_percentiles() {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    return compute_percentiles(queue_samples_us_);
}

Percentiles MetricsRegistry::get_total_percentiles() {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    return compute_percentiles(total_samples_us_);
}

void MetricsRegistry::reset() {
    requests_total_.store(0, std::memory_order_relaxed);
    requests_success_total_.store(0, std::memory_order_relaxed);
    requests_failed_total_.store(0, std::memory_order_relaxed);
    requests_rejected_total_.store(0, std::memory_order_relaxed);
    model_inference_count_.store(0, std::memory_order_relaxed);
    total_batched_items_.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(latency_mutex_);
    inference_samples_us_.clear();
    queue_samples_us_.clear();
    total_samples_us_.clear();
}

std::string MetricsRegistry::to_prometheus(size_t current_queue_depth, size_t active_workers) {
    Percentiles inf_pct;
    Percentiles q_pct;
    Percentiles tot_pct;
    {
        std::lock_guard<std::mutex> lock(latency_mutex_);
        inf_pct = compute_percentiles(inference_samples_us_);
        q_pct = compute_percentiles(queue_samples_us_);
        tot_pct = compute_percentiles(total_samples_us_);
    }

    uint64_t calls = model_inference_count_.load(std::memory_order_relaxed);
    uint64_t batched = total_batched_items_.load(std::memory_order_relaxed);
    double avg_batch = calls > 0 ? (static_cast<double>(batched) / static_cast<double>(calls)) : 1.0;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);

    ss << "# HELP neuroserve_requests_total Total number of inference requests received\n";
    ss << "# TYPE neuroserve_requests_total counter\n";
    ss << "neuroserve_requests_total " << requests_total_.load() << "\n\n";

    ss << "# HELP neuroserve_requests_success_total Total successful requests\n";
    ss << "# TYPE neuroserve_requests_success_total counter\n";
    ss << "neuroserve_requests_success_total " << requests_success_total_.load() << "\n\n";

    ss << "# HELP neuroserve_requests_failed_total Total failed requests\n";
    ss << "# TYPE neuroserve_requests_failed_total counter\n";
    ss << "neuroserve_requests_failed_total " << requests_failed_total_.load() << "\n\n";

    ss << "# HELP neuroserve_requests_rejected_total Total rejected requests due to load shedding\n";
    ss << "# TYPE neuroserve_requests_rejected_total counter\n";
    ss << "neuroserve_requests_rejected_total " << requests_rejected_total_.load() << "\n\n";

    ss << "# HELP neuroserve_queue_depth Current number of pending requests in queue\n";
    ss << "# TYPE neuroserve_queue_depth gauge\n";
    ss << "neuroserve_queue_depth " << current_queue_depth << "\n\n";

    ss << "# HELP neuroserve_active_workers Current number of workers actively running tasks\n";
    ss << "# TYPE neuroserve_active_workers gauge\n";
    ss << "neuroserve_active_workers " << active_workers << "\n\n";

    ss << "# HELP neuroserve_model_inference_count Total ONNX session execution runs\n";
    ss << "# TYPE neuroserve_model_inference_count counter\n";
    ss << "neuroserve_model_inference_count " << calls << "\n\n";

    ss << "# HELP neuroserve_batch_size_average Average batch size of model execution\n";
    ss << "# TYPE neuroserve_batch_size_average gauge\n";
    ss << "neuroserve_batch_size_average " << avg_batch << "\n\n";

    ss << "# HELP neuroserve_inference_latency_ms Model execution latency in milliseconds\n";
    ss << "# TYPE neuroserve_inference_latency_ms summary\n";
    ss << "neuroserve_inference_latency_ms{quantile=\"0.5\"} " << (inf_pct.p50 / 1000.0) << "\n";
    ss << "neuroserve_inference_latency_ms{quantile=\"0.95\"} " << (inf_pct.p95 / 1000.0) << "\n";
    ss << "neuroserve_inference_latency_ms{quantile=\"0.99\"} " << (inf_pct.p99 / 1000.0) << "\n\n";

    ss << "# HELP neuroserve_queue_latency_ms Queue waiting latency in milliseconds\n";
    ss << "# TYPE neuroserve_queue_latency_ms summary\n";
    ss << "neuroserve_queue_latency_ms{quantile=\"0.5\"} " << (q_pct.p50 / 1000.0) << "\n";
    ss << "neuroserve_queue_latency_ms{quantile=\"0.95\"} " << (q_pct.p95 / 1000.0) << "\n";
    ss << "neuroserve_queue_latency_ms{quantile=\"0.99\"} " << (q_pct.p99 / 1000.0) << "\n\n";

    ss << "# HELP neuroserve_total_latency_ms End-to-end server latency in milliseconds\n";
    ss << "# TYPE neuroserve_total_latency_ms summary\n";
    ss << "neuroserve_total_latency_ms{quantile=\"0.5\"} " << (tot_pct.p50 / 1000.0) << "\n";
    ss << "neuroserve_total_latency_ms{quantile=\"0.95\"} " << (tot_pct.p95 / 1000.0) << "\n";
    ss << "neuroserve_total_latency_ms{quantile=\"0.99\"} " << (tot_pct.p99 / 1000.0) << "\n";

    return ss.str();
}

} // namespace neuroserve

