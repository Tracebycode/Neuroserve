#pragma once

#include "neuroserve/common/types.hpp"
#include "neuroserve/model/model_metadata.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <span>

namespace neuroserve {

class ModelInstance {
public:
    ModelInstance(Ort::Env& env, const std::string& model_path, const Ort::SessionOptions& options);
    ~ModelInstance() = default;

    ModelInstance(const ModelInstance&) = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
    ModelInstance(ModelInstance&&) noexcept = default;
    ModelInstance& operator=(ModelInstance&&) noexcept = default;

    // Single request inference
    InferenceResponse run(const InferenceRequest& req);

    // Dynamic batched inference combining multiple requests along dim 0
    std::vector<InferenceResponse> run_batch(std::span<const InferenceRequest> batch_requests);

    const ModelMetadata& metadata() const noexcept { return metadata_; }

private:
    void inspect_metadata(const std::string& model_path);

    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    ModelMetadata metadata_;
};

} // namespace neuroserve

