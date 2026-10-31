#pragma once

#include <string>
#include <cstdint>
#include <cstddef>

namespace neuroserve {

struct ServerConfig {
    std::string host{"0.0.0.0"};
    uint16_t port{8080};
};

struct WorkersConfig {
    size_t count{4};
};

struct QueueConfig {
    size_t capacity{1024};
};

struct BatchingConfig {
    bool enabled{true};
    size_t max_batch_size{8};
    uint32_t max_wait_ms{5};
};

struct ModelConfig {
    std::string name{"default"};
    std::string path{"models/model.onnx"};
    size_t instances{2};
};

struct LoadSheddingConfig {
    bool enabled{true};
    uint32_t max_queue_wait_ms{100};
};

struct LoggingConfig {
    std::string level{"info"};
};

struct Config {
    ServerConfig server;
    WorkersConfig workers;
    QueueConfig queue;
    BatchingConfig batching;
    ModelConfig model;
    LoadSheddingConfig load_shedding;
    LoggingConfig logging;

    static Config load_from_file(const std::string& filepath);
    void validate() const;
};

} // namespace neuroserve

