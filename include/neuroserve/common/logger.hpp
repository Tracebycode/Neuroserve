#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <string>

namespace neuroserve {

class Logger {
public:
    static void init(const std::string& log_level = "info");
    static std::shared_ptr<spdlog::logger>& get();
};

#define NS_LOG_TRACE(...) ::neuroserve::Logger::get()->trace(__VA_ARGS__)
#define NS_LOG_DEBUG(...) ::neuroserve::Logger::get()->debug(__VA_ARGS__)
#define NS_LOG_INFO(...)  ::neuroserve::Logger::get()->info(__VA_ARGS__)
#define NS_LOG_WARN(...)  ::neuroserve::Logger::get()->warn(__VA_ARGS__)
#define NS_LOG_ERROR(...) ::neuroserve::Logger::get()->error(__VA_ARGS__)
#define NS_LOG_CRITICAL(...) ::neuroserve::Logger::get()->critical(__VA_ARGS__)

} // namespace neuroserve

