#include "neuroserve/config/config.hpp"
#include "neuroserve/common/error.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace neuroserve {

Config Config::load_from_file(const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) {
        throw ConfigurationException("Config file does not exist: " + filepath);
    }

    YAML::Node root;
    try {
        root = YAML::LoadFile(filepath);
    } catch (const std::exception& ex) {
        throw ConfigurationException("Failed to parse YAML config: " + std::string(ex.what()));
    }

    Config cfg;

    if (root["server"]) {
        auto node = root["server"];
        if (node["host"]) cfg.server.host = node["host"].as<std::string>();
        if (node["port"]) cfg.server.port = node["port"].as<uint16_t>();
    }

    if (root["workers"]) {
        auto node = root["workers"];
        if (node["count"]) cfg.workers.count = node["count"].as<size_t>();
    }

    if (root["queue"]) {
        auto node = root["queue"];
        if (node["capacity"]) cfg.queue.capacity = node["capacity"].as<size_t>();
    }

    if (root["batching"]) {
        auto node = root["batching"];
        if (node["enabled"]) cfg.batching.enabled = node["enabled"].as<bool>();
        if (node["max_batch_size"]) cfg.batching.max_batch_size = node["max_batch_size"].as<size_t>();
        if (node["max_wait_ms"]) cfg.batching.max_wait_ms = node["max_wait_ms"].as<uint32_t>();
    }

    if (root["model"]) {
        auto node = root["model"];
        if (node["name"]) cfg.model.name = node["name"].as<std::string>();
        if (node["path"]) cfg.model.path = node["path"].as<std::string>();
        if (node["instances"]) cfg.model.instances = node["instances"].as<size_t>();
    }

    if (root["load_shedding"]) {
        auto node = root["load_shedding"];
        if (node["enabled"]) cfg.load_shedding.enabled = node["enabled"].as<bool>();
        if (node["max_queue_wait_ms"]) cfg.load_shedding.max_queue_wait_ms = node["max_queue_wait_ms"].as<uint32_t>();
    }

    if (root["logging"]) {
        auto node = root["logging"];
        if (node["level"]) cfg.logging.level = node["level"].as<std::string>();
    }

    cfg.validate();
    return cfg;
}

void Config::validate() const {
    if (server.port == 0) {
        throw ConfigurationException("Server port cannot be 0");
    }
    if (server.host.empty()) {
        throw ConfigurationException("Server host cannot be empty");
    }
    if (workers.count == 0) {
        throw ConfigurationException("Worker count must be > 0");
    }
    if (queue.capacity == 0) {
        throw ConfigurationException("Queue capacity must be > 0");
    }
    if (batching.enabled && batching.max_batch_size == 0) {
        throw ConfigurationException("Batching max_batch_size must be > 0 when batching is enabled");
    }
    if (model.path.empty()) {
        throw ConfigurationException("Model path cannot be empty");
    }
    if (model.instances == 0) {
        throw ConfigurationException("Model instance count must be > 0");
    }
}

} // namespace neuroserve

