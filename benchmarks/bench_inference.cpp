#include <benchmark/benchmark.h>
#include "neuroserve/model/model_manager.hpp"

using namespace neuroserve;

static void BM_Inference_Batch_Scaling(benchmark::State& state) {
    const size_t batch_size = state.range(0);

    ModelManager manager;
    manager.load_model("bench_mlp", "models/model.onnx", 1);
    auto pool = manager.get_pool("bench_mlp");
    auto lease = pool->acquire();

    std::vector<InferenceRequest> batch;
    batch.reserve(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        InferenceRequest req;
        req.request_id = b;
        TensorData in;
        in.name = "input";
        in.shape = {1, 8};
        in.float_data = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f};
        req.inputs.push_back(in);
        batch.push_back(req);
    }

    for (auto _ : state) {
        auto responses = lease->run_batch(batch);
        benchmark::DoNotOptimize(responses);
    }

    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_Inference_Batch_Scaling)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

static void BM_ModelInstance_Acquisition(benchmark::State& state) {
    ModelManager manager;
    manager.load_model("bench_mlp", "models/model.onnx", 4);
    auto pool = manager.get_pool("bench_mlp");

    for (auto _ : state) {
        auto lease = pool->acquire();
        benchmark::DoNotOptimize(lease);
    }
}
BENCHMARK(BM_ModelInstance_Acquisition);

