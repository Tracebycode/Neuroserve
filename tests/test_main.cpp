#include <gtest/gtest.h>
#include "neuroserve/common/logger.hpp"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    neuroserve::Logger::init("warn");
    return RUN_ALL_TESTS();
}

