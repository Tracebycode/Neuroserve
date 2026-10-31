#pragma once

#include "neuroserve/inference/inference_service.hpp"
#include "neuroserve/config/config.hpp"
#include <httplib.h>
#include <memory>
#include <string>
#include <atomic>
#include <thread>

namespace neuroserve {

class HttpServer {
public:
    HttpServer(const Config& config, std::shared_ptr<InferenceService> inference_service);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void start();
    void stop();

private:
    void setup_routes();

    Config config_;
    std::shared_ptr<InferenceService> inference_service_;
    std::unique_ptr<httplib::Server> server_;
    std::atomic<bool> running_{false};
    std::jthread server_thread_;
    std::atomic<uint64_t> request_counter_{0};
};

} // namespace neuroserve

