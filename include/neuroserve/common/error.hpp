#pragma once

#include <stdexcept>
#include <string>

namespace neuroserve {

class NeuroServeException : public std::runtime_error {
public:
    explicit NeuroServeException(const std::string& msg)
        : std::runtime_error(msg) {}
};

class ConfigurationException : public NeuroServeException {
public:
    explicit ConfigurationException(const std::string& msg)
        : NeuroServeException("Configuration Error: " + msg) {}
};

class ModelLoadException : public NeuroServeException {
public:
    explicit ModelLoadException(const std::string& msg)
        : NeuroServeException("Model Load Error: " + msg) {}
};

class InferenceException : public NeuroServeException {
public:
    explicit InferenceException(const std::string& msg)
        : NeuroServeException("Inference Error: " + msg) {}
};

class QueueFullException : public NeuroServeException {
public:
    explicit QueueFullException(const std::string& msg = "Queue capacity reached")
        : NeuroServeException("Queue Full: " + msg) {}
};

} // namespace neuroserve

