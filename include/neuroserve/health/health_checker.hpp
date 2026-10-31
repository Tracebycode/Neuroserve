#pragma once

#include <atomic>
#include <string>

namespace neuroserve {

class HealthChecker {
public:
    static HealthChecker& instance() {
        static HealthChecker hc;
        return hc;
    }

    void set_alive(bool alive) noexcept { alive_.store(alive, std::memory_order_relaxed); }
    void set_ready(bool ready) noexcept { ready_.store(ready, std::memory_order_relaxed); }

    bool is_alive() const noexcept { return alive_.load(std::memory_order_relaxed); }
    bool is_ready() const noexcept { return ready_.load(std::memory_order_relaxed); }

    std::string liveness_json() const {
        return alive_.load(std::memory_order_relaxed) ? "{\"status\":\"UP\"}" : "{\"status\":\"DOWN\"}";
    }

    std::string readiness_json() const {
        return ready_.load(std::memory_order_relaxed) ? "{\"status\":\"READY\"}" : "{\"status\":\"NOT_READY\"}";
    }

private:
    HealthChecker() = default;
    std::atomic<bool> alive_{true};
    std::atomic<bool> ready_{false};
};

} // namespace neuroserve

