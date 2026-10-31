#include "neuroserve/model/model_manager.hpp"
#include "neuroserve/common/error.hpp"
#include "neuroserve/common/logger.hpp"

namespace neuroserve {

ModelInstancePool::ModelInstancePool(Ort::Env& env, const std::string& model_path, size_t pool_size,
                                     const Ort::SessionOptions& options)
    : total_size_(pool_size) {
    if (pool_size == 0) {
        throw ConfigurationException("ModelInstancePool size must be greater than 0");
    }

    pool_.reserve(pool_size);
    for (size_t i = 0; i < pool_size; ++i) {
        NS_LOG_INFO("Initializing model instance {}/{} for {}", i + 1, pool_size, model_path);
        pool_.push_back(std::make_unique<ModelInstance>(env, model_path, options));
    }
}

ModelInstancePool::~ModelInstancePool() {
    shutdown();
}

void ModelInstancePool::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutting_down_ = true;
    cv_.notify_all();
}

ModelInstancePool::ScopedLease ModelInstancePool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
        return shutting_down_ || !pool_.empty();
    });

    if (shutting_down_ && pool_.empty()) {
        throw InferenceException("ModelInstancePool is shutting down");
    }

    auto instance = std::move(pool_.back());
    pool_.pop_back();
    return ScopedLease(*this, std::move(instance));
}

std::optional<ModelInstancePool::ScopedLease> ModelInstancePool::try_acquire_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cv_.wait_for(lock, timeout, [this] {
        return shutting_down_ || !pool_.empty();
    })) {
        return std::nullopt;
    }

    if (shutting_down_ && pool_.empty()) {
        return std::nullopt;
    }

    auto instance = std::move(pool_.back());
    pool_.pop_back();
    return ScopedLease(*this, std::move(instance));
}

void ModelInstancePool::release(std::unique_ptr<ModelInstance> instance) {
    if (!instance) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(std::move(instance));
    }
    cv_.notify_one();
}

size_t ModelInstancePool::available_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.size();
}

ModelManager::ModelManager()
    : env_(ORT_LOGGING_LEVEL_WARNING, "NeuroServeEnv") {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

void ModelManager::load_model(const std::string& model_name, const std::string& model_path, size_t instances) {
    std::lock_guard<std::mutex> lock(mutex_);
    NS_LOG_INFO("Loading model '{}' from '{}' with {} instances", model_name, model_path, instances);

    auto pool = std::make_shared<ModelInstancePool>(env_, model_path, instances, session_options_);
    {
        auto lease = pool->acquire();
        metadata_cache_[model_name] = lease->metadata();
    }
    pools_[model_name] = pool;
    NS_LOG_INFO("Successfully loaded model '{}'", model_name);
}

std::shared_ptr<ModelInstancePool> ModelManager::get_pool(const std::string& model_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pools_.find(model_name);
    if (it == pools_.end()) {
        throw ModelLoadException("Model not found in registry: " + model_name);
    }
    return it->second;
}

bool ModelManager::has_model(const std::string& model_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pools_.find(model_name) != pools_.end();
}

const ModelMetadata& ModelManager::get_metadata(const std::string& model_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = metadata_cache_.find(model_name);
    if (it == metadata_cache_.end()) {
        throw ModelLoadException("Metadata not found for model: " + model_name);
    }
    return it->second;
}

} // namespace neuroserve

