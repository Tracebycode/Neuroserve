#include <gtest/gtest.h>
#include "neuroserve/worker/worker_pool.hpp"
#include <atomic>
#include <chrono>

using namespace neuroserve;

TEST(WorkerPoolTest, StartupAndShutdown) {
    WorkerPool pool(4);
    EXPECT_EQ(pool.worker_count(), 4);
    EXPECT_FALSE(pool.is_shutdown());

    pool.shutdown();
    EXPECT_TRUE(pool.is_shutdown());
    // Submitting after shutdown fails
    EXPECT_FALSE(pool.submit([]{}));
}

TEST(WorkerPoolTest, ConcurrentTaskExecution) {
    WorkerPool pool(4);
    constexpr size_t task_count = 100;
    std::atomic<size_t> counter{0};

    for (size_t i = 0; i < task_count; ++i) {
        EXPECT_TRUE(pool.submit([&counter] {
            counter.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    // Wait until all tasks are executed
    auto start = std::chrono::steady_clock::now();
    while (counter.load() < task_count &&
           std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() < 3) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    EXPECT_EQ(counter.load(), task_count);
    pool.shutdown();
}

TEST(WorkerPoolTest, ExceptionHandlingDoesNotKillWorkers) {
    WorkerPool pool(2);
    std::atomic<bool> subsequent_task_ran{false};

    // Submit task that throws
    EXPECT_TRUE(pool.submit([] {
        throw std::runtime_error("Simulated task exception");
    }));

    // Submit another task to verify workers are still alive and functioning
    EXPECT_TRUE(pool.submit([&subsequent_task_ran] {
        subsequent_task_ran.store(true);
    }));

    auto start = std::chrono::steady_clock::now();
    while (!subsequent_task_ran.load() &&
           std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    EXPECT_TRUE(subsequent_task_ran.load());
    pool.shutdown();
}

