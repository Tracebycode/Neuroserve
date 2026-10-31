#include <benchmark/benchmark.h>
#include "neuroserve/common/logger.hpp"

int main(int argc, char** argv) {
    neuroserve::Logger::init("error");
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}

