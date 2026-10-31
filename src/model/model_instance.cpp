#include "neuroserve/model/model_instance.hpp"
#include "neuroserve/common/error.hpp"
#include "neuroserve/common/logger.hpp"
#include <numeric>

namespace neuroserve {

ModelInstance::ModelInstance(Ort::Env& env, const std::string& model_path, const Ort::SessionOptions& options)
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    try {
        session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), options);
        inspect_metadata(model_path);
    } catch (const std::exception& ex) {
        throw ModelLoadException("Failed to load model from " + model_path + ": " + ex.what());
    }
}

void ModelInstance::inspect_metadata(const std::string& model_path) {
    metadata_.model_path = model_path;
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_inputs = session_->GetInputCount();
    metadata_.inputs.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto allocated_name = session_->GetInputNameAllocated(i, allocator);
        std::string name(allocated_name.get());
        auto type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();
        metadata_.inputs.push_back(TensorInfo{std::move(name), DataType::Float32, std::move(dims)});
    }

    size_t num_outputs = session_->GetOutputCount();
    metadata_.outputs.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto allocated_name = session_->GetOutputNameAllocated(i, allocator);
        std::string name(allocated_name.get());
        auto type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();
        metadata_.outputs.push_back(TensorInfo{std::move(name), DataType::Float32, std::move(dims)});
    }
}

InferenceResponse ModelInstance::run(const InferenceRequest& req) {
    InferenceResponse resp;
    resp.request_id = req.request_id;

    if (metadata_.inputs.empty()) {
        resp.success = false;
        resp.error_message = "Model has no inputs defined";
        resp.status_code = 500;
        return resp;
    }

    try {
        std::vector<const char*> input_names;
        std::vector<Ort::Value> input_tensors;
        input_names.reserve(req.inputs.size());
        input_tensors.reserve(req.inputs.size());

        for (const auto& in_data : req.inputs) {
            input_names.push_back(in_data.name.c_str());
            auto shape = in_data.shape;
            if (shape.empty()) {
                shape = {1, static_cast<int64_t>(in_data.float_data.size())};
            }

            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info_,
                const_cast<float*>(in_data.float_data.data()),
                in_data.float_data.size(),
                shape.data(),
                shape.size()
            );
            input_tensors.push_back(std::move(tensor));
        }

        std::vector<const char*> output_names;
        output_names.reserve(metadata_.outputs.size());
        for (const auto& out_meta : metadata_.outputs) {
            output_names.push_back(out_meta.name.c_str());
        }

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& ort_tensor = output_tensors[i];
            auto tensor_info = ort_tensor.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            size_t total_elements = tensor_info.GetElementCount();

            const float* float_arr = ort_tensor.GetTensorData<float>();
            TensorData out_data;
            out_data.name = metadata_.outputs[i].name;
            out_data.shape = shape;
            out_data.float_data.assign(float_arr, float_arr + total_elements);

            resp.outputs.push_back(std::move(out_data));
        }

        resp.success = true;
        resp.status_code = 200;
    } catch (const std::exception& ex) {
        resp.success = false;
        resp.status_code = 500;
        resp.error_message = ex.what();
    }

    return resp;
}

std::vector<InferenceResponse> ModelInstance::run_batch(std::span<const InferenceRequest> batch_requests) {
    if (batch_requests.empty()) {
        return {};
    }

    if (batch_requests.size() == 1) {
        return {run(batch_requests[0])};
    }

    const size_t batch_size = batch_requests.size();
    std::vector<InferenceResponse> responses(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        responses[i].request_id = batch_requests[i].request_id;
    }

    try {
        // Collect single item element count from first request
        const auto& first_req = batch_requests[0];
        if (first_req.inputs.empty()) {
            throw InferenceException("Batch request missing inputs");
        }

        const size_t num_inputs = first_req.inputs.size();
        std::vector<std::vector<float>> batched_buffers(num_inputs);
        std::vector<std::vector<int64_t>> batched_shapes(num_inputs);

        for (size_t in_idx = 0; in_idx < num_inputs; ++in_idx) {
            const size_t per_item_elements = first_req.inputs[in_idx].float_data.size();
            batched_buffers[in_idx].reserve(batch_size * per_item_elements);

            for (size_t b = 0; b < batch_size; ++b) {
                const auto& item_data = batch_requests[b].inputs[in_idx].float_data;
                if (item_data.size() != per_item_elements) {
                    throw InferenceException("Inconsistent input tensor size in dynamic batch");
                }
                batched_buffers[in_idx].insert(batched_buffers[in_idx].end(), item_data.begin(), item_data.end());
            }

            std::vector<int64_t> shape = first_req.inputs[in_idx].shape;
            if (shape.empty()) {
                shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(per_item_elements)};
            } else {
                shape[0] = static_cast<int64_t>(batch_size);
            }
            batched_shapes[in_idx] = std::move(shape);
        }

        std::vector<const char*> input_names;
        std::vector<Ort::Value> input_tensors;
        input_names.reserve(num_inputs);
        input_tensors.reserve(num_inputs);

        for (size_t in_idx = 0; in_idx < num_inputs; ++in_idx) {
            input_names.push_back(first_req.inputs[in_idx].name.c_str());
            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info_,
                batched_buffers[in_idx].data(),
                batched_buffers[in_idx].size(),
                batched_shapes[in_idx].data(),
                batched_shapes[in_idx].size()
            );
            input_tensors.push_back(std::move(tensor));
        }

        std::vector<const char*> output_names;
        output_names.reserve(metadata_.outputs.size());
        for (const auto& out_meta : metadata_.outputs) {
            output_names.push_back(out_meta.name.c_str());
        }

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size()
        );

        // De-batch: Slice output tensors across the batch items
        for (size_t out_idx = 0; out_idx < output_tensors.size(); ++out_idx) {
            auto& ort_tensor = output_tensors[out_idx];
            auto tensor_info = ort_tensor.GetTensorTypeAndShapeInfo();
            auto out_shape = tensor_info.GetShape();
            size_t total_elements = tensor_info.GetElementCount();
            size_t elements_per_item = total_elements / batch_size;

            const float* float_arr = ort_tensor.GetTensorData<float>();
            std::vector<int64_t> single_shape = out_shape;
            if (!single_shape.empty()) {
                single_shape[0] = 1;
            }

            for (size_t b = 0; b < batch_size; ++b) {
                TensorData item_out;
                item_out.name = metadata_.outputs[out_idx].name;
                item_out.shape = single_shape;
                const float* item_start = float_arr + (b * elements_per_item);
                item_out.float_data.assign(item_start, item_start + elements_per_item);
                responses[b].outputs.push_back(std::move(item_out));
                responses[b].success = true;
                responses[b].status_code = 200;
            }
        }
    } catch (const std::exception& ex) {
        for (auto& resp : responses) {
            resp.success = false;
            resp.status_code = 500;
            resp.error_message = ex.what();
        }
    }

    return responses;
}

} // namespace neuroserve

