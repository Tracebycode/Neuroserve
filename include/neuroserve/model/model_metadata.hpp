#pragma once

#include "neuroserve/common/types.hpp"
#include <string>
#include <vector>

namespace neuroserve {

struct TensorInfo {
    std::string name;
    DataType dtype{DataType::Float32};
    std::vector<int64_t> dims; // may contain -1 for dynamic dimensions
};

struct ModelMetadata {
    std::string model_name;
    std::string model_path;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;

    bool validate_input(const TensorData& tensor) const {
        for (const auto& in : inputs) {
            if (in.name == tensor.name) {
                // If shape dimensions are defined (excluding dynamic batch dimension 0)
                if (in.dims.size() == tensor.shape.size()) {
                    for (size_t i = 1; i < in.dims.size(); ++i) {
                        if (in.dims[i] > 0 && tensor.shape[i] > 0 && in.dims[i] != tensor.shape[i]) {
                            return false;
                        }
                    }
                }
                return true;
            }
        }
        return false;
    }
};

} // namespace neuroserve

