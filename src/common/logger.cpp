#include "neuroserve/common/logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>

namespace neuroserve {

static std::shared_ptr<spdlog::logger> g_logger;

void Logger::init(const std::string& log_level) {
    if (!g_logger) {
        g_logger = spdlog::stdout_color_mt("neuroserve");
        g_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");
    }

    if (log_level == "trace") {
        g_logger->set_level(spdlog::level::trace);
    } else if (log_level == "debug") {
        g_logger->set_level(spdlog::level::debug);
    } else if (log_level == "info") {
        g_logger->set_level(spdlog::level::info);
    } else if (log_level == "warn") {
        g_logger->set_level(spdlog::level::warn);
    } else if (log_level == "error") {
        g_logger->set_level(spdlog::level::err);
    } else if (log_level == "critical") {
        g_logger->set_level(spdlog::level::critical);
    } else {
        g_logger->set_level(spdlog::level::info);
    }
}

std::shared_ptr<spdlog::logger>& Logger::get() {
    if (!g_logger) {
        init("info");
    }
    return g_logger;
}

} // namespace neuroserve

