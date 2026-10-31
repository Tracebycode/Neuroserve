#include <gtest/gtest.h>
#include "neuroserve/model/model_manager.hpp"
#include "neuroserve/common/error.hpp"

using namespace neuroserve;

TEST(ModelManagerTest, LoadValidModelAndInspectMetadata) {
    ModelManager manager;
    ASSERT_NO_THROW(manager.load_model("test_mlp", "models/model.onnx", 2));
    EXPECT_TRUE(manager.has_model("test_mlp"));

    const auto& meta = manager.get_metadata("test_mlp");
    EXPECT_EQ(meta.inputs.size(), 1);
    EXPECT_EQ(meta.inputs[0].name, "input");
    EXPECT_EQ(meta.outputs.size(), 1);
    EXPECT_EQ(meta.outputs[0].name, "output");
}

TEST(ModelManagerTest, InvalidModelThrows) {
    ModelManager manager;
    EXPECT_THROW(manager.load_model("nonexistent", "models/missing_file.onnx", 1), ModelLoadException);
    EXPECT_FALSE(manager.has_model("nonexistent"));
}

TEST(ModelManagerTest, AcquireAndReleasePoolInstances) {
    ModelManager manager;
    manager.load_model("test_mlp", "models/model.onnx", 2);

    auto pool = manager.get_pool("test_mlp");
    EXPECT_EQ(pool->total_count(), 2);
    EXPECT_EQ(pool->available_count(), 2);

    {
        auto lease1 = pool->acquire();
        EXPECT_EQ(pool->available_count(), 1);

        auto lease2 = pool->acquire();
        EXPECT_EQ(pool->available_count(), 0);

        // Try acquire with 0 available should timeout
        auto lease3 = pool->try_acquire_for(std::chrono::milliseconds(50));
        EXPECT_FALSE(lease3.has_value());
    } // lease1 and lease2 go out of scope and release instances

    EXPECT_EQ(pool->available_count(), 2);
}

TEST(ModelManagerTest, SingleInferenceExecution) {
    ModelManager manager;
    manager.load_model("test_mlp", "models/model.onnx", 1);
    auto pool = manager.get_pool("test_mlp");
    auto lease = pool->acquire();

    InferenceRequest req;
    req.request_id = 1;
    req.model_name = "test_mlp";
    TensorData in;
    in.name = "input";
    in.shape = {1, 8};
    in.float_data = {0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, 0.8f};
    req.inputs.push_back(in);

    auto resp = lease->run(req);
    EXPECT_TRUE(resp.success);
    EXPECT_EQ(resp.status_code, 200);
    ASSERT_EQ(resp.outputs.size(), 1);
    EXPECT_EQ(resp.outputs[0].name, "output");
    EXPECT_EQ(resp.outputs[0].float_data.size(), 2);
}

