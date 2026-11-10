// Benchmarks for RegularGridInterpolator lookups.
//
// Arguments: {number of dimensions, interpolation method (0 = linear, 1 = cubic), number of data sets}

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include <btwxt/btwxt.h>

namespace {

class SilentCourier : public Btwxt::BtwxtDefaultCourier {
  protected:
    void write_message(const std::string&, const std::string&) override {}
};

constexpr std::size_t points_per_axis = 8;
constexpr std::size_t number_of_targets = 1024;

Btwxt::RegularGridInterpolator make_interpolator(std::size_t dimensions,
                                                 Btwxt::InterpolationMethod method,
                                                 std::size_t data_sets)
{
    std::vector<double> axis(points_per_axis);
    for (std::size_t i = 0; i < points_per_axis; ++i) {
        axis[i] = static_cast<double>(i);
    }
    std::vector<std::vector<double>> grid(dimensions, axis);

    std::size_t number_of_points = 1;
    for (std::size_t d = 0; d < dimensions; ++d) {
        number_of_points *= points_per_axis;
    }

    std::vector<std::vector<double>> data(data_sets, std::vector<double>(number_of_points));
    for (std::size_t set = 0; set < data_sets; ++set) {
        for (std::size_t point = 0; point < number_of_points; ++point) {
            data[set][point] = std::sin(0.1 * static_cast<double>(point) + static_cast<double>(set));
        }
    }

    Btwxt::RegularGridInterpolator interpolator(
        grid, data, "benchmark", std::make_shared<SilentCourier>());
    for (std::size_t d = 0; d < dimensions; ++d) {
        interpolator.set_axis_interpolation_method(d, method);
    }
    return interpolator;
}

std::vector<std::vector<double>> make_targets(std::size_t dimensions)
{
    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(0.0,
                                                        static_cast<double>(points_per_axis - 1));
    std::vector<std::vector<double>> targets(number_of_targets, std::vector<double>(dimensions));
    for (auto& target : targets) {
        for (auto& value : target) {
            value = distribution(generator);
        }
    }
    return targets;
}

Btwxt::InterpolationMethod method_from(const benchmark::State& state)
{
    return state.range(1) == 0 ? Btwxt::InterpolationMethod::linear
                               : Btwxt::InterpolationMethod::cubic;
}

// Random targets across the grid, with the hypercube cache already warm: every lookup is a cache hit.
void BM_WarmCacheRandomTargets(benchmark::State& state)
{
    const auto dimensions = static_cast<std::size_t>(state.range(0));
    const auto data_sets = static_cast<std::size_t>(state.range(2));
    auto interpolator = make_interpolator(dimensions, method_from(state), data_sets);
    const auto targets = make_targets(dimensions);

    for (const auto& target : targets) { // warm the cache
        benchmark::DoNotOptimize(interpolator(target));
    }

    std::size_t i = 0;
    for (auto _ : state) {
        auto result = interpolator(targets[i]);
        benchmark::DoNotOptimize(result.data());
        i = (i + 1) % targets.size();
    }
    state.SetItemsProcessed(state.iterations());
}

// Targets that all fall in the same grid cell.
void BM_SameCell(benchmark::State& state)
{
    const auto dimensions = static_cast<std::size_t>(state.range(0));
    const auto data_sets = static_cast<std::size_t>(state.range(2));
    auto interpolator = make_interpolator(dimensions, method_from(state), data_sets);

    std::vector<std::vector<double>> targets(2, std::vector<double>(dimensions, 3.25));
    std::fill(targets[1].begin(), targets[1].end(), 3.75);

    std::size_t i = 0;
    for (auto _ : state) {
        auto result = interpolator(targets[i]);
        benchmark::DoNotOptimize(result.data());
        i ^= 1U;
    }
    state.SetItemsProcessed(state.iterations());
}

// Cold cache: each lookup builds the hypercube data (cache miss path).
void BM_ColdCache(benchmark::State& state)
{
    const auto dimensions = static_cast<std::size_t>(state.range(0));
    const auto data_sets = static_cast<std::size_t>(state.range(2));
    const auto method = method_from(state);
    const auto targets = make_targets(dimensions);

    std::size_t i = 0;
    for (auto _ : state) {
        state.PauseTiming();
        auto interpolator = make_interpolator(dimensions, method, data_sets);
        state.ResumeTiming();
        auto result = interpolator(targets[i]);
        benchmark::DoNotOptimize(result.data());
        i = (i + 1) % targets.size();
    }
    state.SetItemsProcessed(state.iterations());
}

// Mimics EnergyPlus UnitaryHybridAC_DedicatedOutsideAir: 6-D cubic grids with 3 points per axis
// (Toa, Woa, Tra, Wra, Msa, OAF), one data set per Table:Lookup on the same independent variable list.
// Each timestep the model sweeps Msa x OAF settings and evaluates every output at each setting.
void BM_UnitaryHybridAC(benchmark::State& state)
{
    constexpr std::size_t dimensions = 6;
    const auto data_sets = static_cast<std::size_t>(state.range(0));
    const std::vector<double> axis {-100.0, 0.0, 100.0};
    std::vector<std::vector<double>> grid(dimensions, axis);

    std::size_t number_of_points = 1;
    for (std::size_t d = 0; d < dimensions; ++d) {
        number_of_points *= axis.size();
    }
    std::vector<std::vector<double>> data(data_sets, std::vector<double>(number_of_points));
    for (std::size_t set = 0; set < data_sets; ++set) {
        for (std::size_t point = 0; point < number_of_points; ++point) {
            data[set][point] = std::sin(0.1 * static_cast<double>(point) + static_cast<double>(set));
        }
    }
    Btwxt::RegularGridInterpolator interpolator(
        grid, data, "UnitaryHybridAC", std::make_shared<SilentCourier>());
    for (std::size_t d = 0; d < dimensions; ++d) {
        interpolator.set_axis_interpolation_method(d, Btwxt::InterpolationMethod::cubic);
    }

    // Toa, Woa, Tra, Wra fixed for the timestep; Msa and OAF swept from 1.0 down to 0.0 in 0.2 steps
    std::vector<std::vector<double>> targets;
    for (int msa = 5; msa >= 0; --msa) {
        for (int oaf = 5; oaf >= 0; --oaf) {
            targets.push_back({30.0, 0.012, 24.0, 0.009, 0.2 * msa, 0.2 * oaf});
        }
    }

    std::size_t i = 0;
    for (auto _ : state) {
        for (std::size_t output = 0; output < data_sets; ++output) { // one CurveValue call per output
            benchmark::DoNotOptimize(interpolator(targets[i])[output]);
        }
        i = (i + 1) % targets.size();
    }
    state.SetItemsProcessed(state.iterations()); // items = settings evaluated
}

void arguments(benchmark::Benchmark* b)
{
    b->ArgNames({"dims", "cubic", "sets"});
    for (int dimensions : {1, 2, 3, 4}) {
        for (int cubic : {0, 1}) {
            for (int data_sets : {1, 8}) {
                b->Args({dimensions, cubic, data_sets});
            }
        }
    }
}

} // namespace

BENCHMARK(BM_WarmCacheRandomTargets)->Apply(arguments);
BENCHMARK(BM_SameCell)->Apply(arguments);
BENCHMARK(BM_ColdCache)->Apply(arguments)->Iterations(2000);
BENCHMARK(BM_UnitaryHybridAC)->ArgName("sets")->Arg(1)->Arg(6);

BENCHMARK_MAIN();
