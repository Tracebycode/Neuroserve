#include <gtest/gtest.h>
#include "neuroserve/batching/dynamic_batcher.hpp"
#include <atomic>
#include <chrono>

using namespace neuroserve;

TEST(DynamicBatcherTest, FullBatchDispatchedImmediately) {
    constexpr size_t max_batch = 4;
    std::atomic<size_t> dispatched_batches{0};
    std::atomic<size_t> total_items_processed{0};

    DynamicBatcher batcher(
        max_batch,
        std::chrono::milliseconds(500), // Long wait
        64,
        [&](std::vector<QueueItem> items) {
            dispatched_batches.fetch_add(1);
            total_items_processed.fetch_add(items.size());
            for (auto& it : items) {
                InferenceResponse resp;
                resp.success = true;
                resp.status_code = 200;
                it.promise.set_value(resp);
            }
        }
    );
    batcher.start();

    std::vector<std::future<InferenceResponse>> futures;
    for (size_t i = 0; i < max_batch; ++i) {
        InferenceRequest req;
        req.request_id = i;
        bool shed = false;
        futures.push_back(batcher.submit(std::move(req), shed));
        EXPECT_FALSE(shed);
    }

    // Since max_batch was reached, all should be fulfilled quickly without waiting 500ms
    for (auto& fut : futures) {
        ASSERT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::ready);
        auto resp = fut.get();
        EXPECT_TRUE(resp.success);
    }

    EXPECT_EQ(dispatched_batches.load(), 1);
    EXPECT_EQ(total_items_processed.load(), max_batch);
    batcher.stop();
}

TEST(DynamicBatcherTest, PartialBatchDispatchedOnTimeout) {
    std::atomic<size_t> dispatched_batches{0};
    std::atomic<size_t> total_items_processed{0};

    DynamicBatcher batcher(
        8, // Expects 8
        std::chrono::milliseconds(30), // Short timeout
        64,
        [&](std::vector<QueueItem> items) {
            dispatched_batches.fetch_add(1);
            total_items_processed.fetch_add(items.size());
            for (auto& it : items) {
                InferenceResponse resp;
                resp.success = true;
                it.promise.set_value(resp);
            }
        }
    );
    batcher.start();

    // Submit only 2 items (< 8)
    bool shed = false;
    InferenceRequest r1; r1.request_id = 1;
    InferenceRequest r2; r2.request_id = 2;
    auto f1 = batcher.submit(std::move(r1), shed);
    auto f2 = batcher.submit(std::move(r2), shed);

    ASSERT_EQ(f1.wait_for(std::chrono::milliseconds(200)), std::future_status::ready);
    ASSERT_EQ(f2.wait_for(std::chrono::milliseconds(200)), std::future_status::ready);

    EXPECT_TRUE(f1.get().success);
    EXPECT_TRUE(f2.get().success);
    EXPECT_EQ(dispatched_batches.load(), 1);
    EXPECT_EQ(total_items_processed.load(), 2);

    batcher.stop();
}

