#include <gtest/gtest.h>
#include "neuroserve/queue/bounded_queue.hpp"
#include <thread>
#include <vector>
#include <atomic>

using namespace neuroserve;

TEST(BoundedQueueTest, BasicEnqueueDequeue) {
    BoundedQueue<int> q(5);
    EXPECT_EQ(q.size(), 0);
    EXPECT_TRUE(q.empty());

    auto res = q.try_enqueue(42);
    EXPECT_EQ(res, BoundedQueue<int>::EnqueueResult::Success);
    EXPECT_EQ(q.size(), 1);
    EXPECT_FALSE(q.empty());

    auto item = q.dequeue_for(std::chrono::milliseconds(50));
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(*item, 42);
    EXPECT_TRUE(q.empty());
}

TEST(BoundedQueueTest, FullQueueLoadShedding) {
    BoundedQueue<int> q(2);
    EXPECT_EQ(q.try_enqueue(1), BoundedQueue<int>::EnqueueResult::Success);
    EXPECT_EQ(q.try_enqueue(2), BoundedQueue<int>::EnqueueResult::Success);
    EXPECT_EQ(q.size(), 2);

    // Queue is full, must return Full (no unbounded growth)
    EXPECT_EQ(q.try_enqueue(3), BoundedQueue<int>::EnqueueResult::Full);
    EXPECT_EQ(q.size(), 2);
}

TEST(BoundedQueueTest, ShutdownWakesWaitingThreads) {
    BoundedQueue<int> q(2);
    std::atomic<bool> dequeued_none{false};

    std::thread consumer([&] {
        auto item = q.dequeue_for(std::chrono::seconds(2));
        if (!item.has_value()) {
            dequeued_none = true;
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    q.shutdown();
    consumer.join();

    EXPECT_TRUE(dequeued_none);
    EXPECT_TRUE(q.is_shutdown());
}

TEST(BoundedQueueTest, ConcurrentProducersAndConsumers) {
    constexpr size_t num_producers = 4;
    constexpr size_t num_consumers = 4;
    constexpr size_t items_per_producer = 500;
    constexpr size_t total_items = num_producers * items_per_producer;

    BoundedQueue<int> q(64);
    std::atomic<size_t> consumed_count{0};
    std::atomic<uint64_t> sum_consumed{0};

    std::vector<std::thread> producers;
    for (size_t p = 0; p < num_producers; ++p) {
        producers.emplace_back([&q, p] {
            for (size_t i = 1; i <= items_per_producer; ++i) {
                while (q.enqueue_for(static_cast<int>(i), std::chrono::milliseconds(10)) !=
                       BoundedQueue<int>::EnqueueResult::Success) {
                    if (q.is_shutdown()) return;
                }
            }
        });
    }

    std::vector<std::thread> consumers;
    for (size_t c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&q, &consumed_count, &sum_consumed] {
            while (consumed_count.load() < total_items) {
                auto val = q.dequeue_for(std::chrono::milliseconds(10));
                if (val.has_value()) {
                    sum_consumed.fetch_add(*val);
                    consumed_count.fetch_add(1);
                }
            }
        });
    }

    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    EXPECT_EQ(consumed_count.load(), total_items);
}

