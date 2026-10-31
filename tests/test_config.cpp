#include <gtest/gtest.h>
#include "neuroserve/config/config.hpp"
#include "neuroserve/common/error.hpp"

TEST(ConfigTest, LoadValidYaml) {
    auto cfg = neuroserve::Config::load_from_file("config/config.yaml");
    EXPECT_EQ(cfg.server.port, 8080);
    EXPECT_EQ(cfg.server.host, "0.0.0.0");
    EXPECT_EQ(cfg.workers.count, 4);
    EXPECT_EQ(cfg.queue.capacity, 1024);
    EXPECT_TRUE(cfg.batching.enabled);
    EXPECT_EQ(cfg.batching.max_batch_size, 8);
    EXPECT_EQ(cfg.model.name, "demo_mlp");
    EXPECT_EQ(cfg.model.instances, 2);
}

TEST(ConfigTest, ValidationFailures) {
    neuroserve::Config invalid_cfg;
    invalid_cfg.server.port = 0;
    EXPECT_THROW(invalid_cfg.validate(), neuroserve::ConfigurationException);

    invalid_cfg.server.port = 8080;
    invalid_cfg.workers.count = 0;
    EXPECT_THROW(invalid_cfg.validate(), neuroserve::ConfigurationException);

    invalid_cfg.workers.count = 2;
    invalid_cfg.queue.capacity = 0;
    EXPECT_THROW(invalid_cfg.validate(), neuroserve::ConfigurationException);

    invalid_cfg.queue.capacity = 100;
    invalid_cfg.model.path = "";
    EXPECT_THROW(invalid_cfg.validate(), neuroserve::ConfigurationException);
}

