#pragma once

#include "neuroserve/model/model_instance.hpp"
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>
#include <optional>
#include <chrono>

namespace neuroserve {

class ModelInstancePool {
public:
    ModelInstancePool(Ort::Env& env, const std::string& model_path, size_t pool_size,
                      const Ort::SessionOptions& options);
    ~ModelInstancePool();

    class ScopedLease {
    public:
        ScopedLease(ModelInstancePool& pool, std::unique_ptr<ModelInstance> instance)
            : pool_(pool), instance_(std::move(instance)) {}

        ~ScopedLease() {
            if (instance_) {
                pool_.release(std::move(instance_));
            }
        }

        ScopedLease(const ScopedLease&) = delete;
        ScopedLease& operator=(const ScopedLease&) = delete;

        ScopedLease(ScopedLease&& other) noexcept
            : pool_(other.pool_), instance_(std::move(other.instance_)) {}

        ScopedLease& operator=(ScopedLease&& other) noexcept {
            if (this != &other) {
                if (instance_) {
                    pool_.release(std::move(instance_));
                }
                instance_ = std::move(other.instance_);
            }
            return *this;
        }

        ModelInstance& get() noexcept { return *instance_; }
        ModelInstance* operator->() noexcept { return instance_.get(); }
        const ModelInstance& get() const noexcept { return *instance_; }
        const ModelInstance* operator->() const noexcept { return instance_.get(); }

    private:
        ModelInstancePool& pool_;
        std::unique_ptr<ModelInstance> instance_;
    };

    ScopedLease acquire();
    std::optional<ScopedLease> try_acquire_for(std::chrono::milliseconds timeout);

    size_t available_count() const;
    size_t total_count() const noexcept { return total_size_; }
    void shutdown();

private:
    friend class ScopedLease;
    void release(std::unique_ptr<ModelInstance> instance);

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<ModelInstance>> pool_;
    const size_t total_size_;
    bool shutting_down_{false};
};

class ModelManager {
public:
    ModelManager();
    ~ModelManager() = default;

    void load_model(const std::string& model_name, const std::string& model_path, size_t instances);
    std::shared_ptr<ModelInstancePool> get_pool(const std::string& model_name) const;
    bool has_model(const std::string& model_name) const;

    const ModelMetadata& get_metadata(const std::string& model_name) const;

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<ModelInstancePool>> pools_;
    std::unordered_map<std::string, ModelMetadata> metadata_cache_;
};

} // namespace neuroserve

