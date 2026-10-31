#include "neuroserve/config/config.hpp"
#include "neuroserve/common/logger.hpp"
#include "neuroserve/model/model_manager.hpp"
#include "neuroserve/inference/inference_service.hpp"
#include "neuroserve/server/http_server.hpp"

#include <csignal>
#include <iostream>
#include <memory>
#include <condition_variable>

namespace {
std::atomic<bool> g_shutdown_requested{false};
std::mutex g_signal_mutex;
std::condition_variable g_signal_cv;

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_shutdown_requested.store(true);
        g_signal_cv.notify_all();
    }
}
} // namespace

int main(int argc, char* argv[]) {
    std::string config_path = "config/config.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }

    try {
        // 1. Load strongly-typed configuration
        auto config = neuroserve::Config::load_from_file(config_path);

        // 2. Initialize structured logging
        neuroserve::Logger::init(config.logging.level);
        NS_LOG_INFO("==================================================");
        NS_LOG_INFO("  Starting NeuroServe C++20 High-Performance Server");
        NS_LOG_INFO("==================================================");
        NS_LOG_INFO("Configuration loaded from: {}", config_path);

        // 3. Register signals for graceful shutdown
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // 4. Initialize Model Manager & Load configured model
        auto model_manager = std::make_shared<neuroserve::ModelManager>();
        model_manager->load_model(config.model.name, config.model.path, config.model.instances);

        // 5. Initialize Inference Service (Queue, Batcher, Worker Pool)
        auto inference_service = std::make_shared<neuroserve::InferenceService>(config, model_manager);
        inference_service->start();

        // 6. Initialize HTTP Network Server
        auto http_server = std::make_unique<neuroserve::HttpServer>(config, inference_service);
        http_server->start();

        NS_LOG_INFO("NeuroServe is running at http://{}:{}", config.server.host, config.server.port);
        NS_LOG_INFO("Endpoints:");
        NS_LOG_INFO("  Inference: POST http://{}:{}/v1/infer", config.server.host, config.server.port);
        NS_LOG_INFO("  Health:    GET  http://{}:{}/health", config.server.host, config.server.port);
        NS_LOG_INFO("  Readiness: GET  http://{}:{}/ready", config.server.host, config.server.port);
        NS_LOG_INFO("  Metrics:   GET  http://{}:{}/metrics", config.server.host, config.server.port);

        // 7. Wait for shutdown signal
        {
            std::unique_lock<std::mutex> lock(g_signal_mutex);
            g_signal_cv.wait(lock, [] {
                return g_shutdown_requested.load();
            });
        }

        NS_LOG_INFO("Shutdown signal received. Initiating graceful shutdown...");

        // 8. Graceful shutdown sequence
        // Step A: Stop HTTP server (stop accepting new network connections)
        http_server->stop();

        // Step B: Drain and stop inference pipeline & worker pool
        inference_service->stop();

        NS_LOG_INFO("NeuroServe exited cleanly. Resources released.");
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "[CRITICAL ERROR] " << ex.what() << std::endl;
        return 1;
    }
}

