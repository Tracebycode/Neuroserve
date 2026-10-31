#include <benchmark/benchmark.h>
#include "neuroserve/queue/bounded_queue.hpp"
#include <thread>
#include <vector>

using namespace neuroserve;

static void BM_Queue_SingleThread_Throughput(benchmark::State& state) {
    BoundedQueue<int> q(1024);
    int val = 42;
    for (auto _ : state) {
        q.try_enqueue(val);
        auto out = q.dequeue_for(std::chrono::microseconds(1));
        benchmark::DoNotOptimize(out);
    }
}
BENCHMARK(BM_Queue_SingleThread_Throughput);

static void BM_Queue_Concurrent_Contention(benchmark::State& state) {
    const size_t num_threads = state.range(0);
    BoundedQueue<int> q(2048);
    std::atomic<bool> stop{false};

    std::vector<std::thread> consumers;
    for (size_t i = 0; i < num_threads; ++i) {
        consumers.emplace_back([&q, &stop] {
            while (!stop.load(std::memory_order_relaxed)) {
                auto out = q.dequeue_for(std::chrono::microseconds(10));
                benchmark::DoNotOptimize(out);
            }
        });
    }

    for (auto _ : state) {
        q.try_enqueue(100);
    }

    stop.store(true);
    q.shutdown();
    for (auto& c : consumers) c.join();
}
BENCHMARK(BM_Queue_Concurrent_Contention)->Arg(1)->Arg(2)->Arg(4)->Arg(8);

