#include <hnsw/HnswIndex.hpp>
#include <hnsw/HnswBruteIndex.hpp>
#include <hnsw/detail/EuclideanDistance.hpp>
#include <hnsw/detail/CpuFeatures.hpp>
#include "Recall.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>

using Clock = std::chrono::steady_clock;

std::vector<float> random_vector(size_t dim) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> v(dim);

    for (float& x : v) {
        x = dist(rng);
    }

    return v;
}

int main()
{
    constexpr std::size_t NUM_VECTORS = 25'000;
    constexpr std::size_t NUM_QUERIES = 1'000;
    constexpr std::size_t VECTOR_DIM = 16;
    constexpr std::size_t K = 10;
    constexpr std::size_t DIM = 2;

    constexpr std::size_t M = 16;
    constexpr std::size_t EF_CONSTRUCTION = 200;

    std::vector<std::vector<float>> vectors;
    vectors.reserve(NUM_VECTORS);

    for (std::size_t i = 0; i < NUM_VECTORS; ++i) {
        std::vector<float> vec = random_vector(VECTOR_DIM);
        vectors.push_back(std::move(vec));
    }

    std::vector<std::vector<float>> queries;
    queries.reserve(NUM_QUERIES);

    for (std::size_t i = 0; i < NUM_QUERIES; ++i) {
        std::vector<float> query = random_vector(VECTOR_DIM);
        queries.push_back(std::move(query));
    }


    hnsw::HnswBruteIndex brute_index{};

    for (const auto& vec : vectors) {
        brute_index.add(vec);
    }

    std::vector<std::vector<std::size_t>> exact_results;
    exact_results.reserve(NUM_QUERIES);

    for (const auto& query : queries) {
        exact_results.push_back(
            brute_index.search(query, K)
        );
    }

    std::cout << "Vendor: " << cpu_features::InstructionSet::Vendor() << "\n";
    std::cout << "Brand: " << cpu_features::InstructionSet::Brand() << "\n";
    std::cout << "AVX supported:  " << cpu_features::InstructionSet::AVX2() << "\n";
    std::cout << "FMA supported:  " << cpu_features::InstructionSet::FMA() << "\n";

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\n";
    std::cout << "HNSW benchmark\n";
    std::cout << "--------------\n";
    std::cout << "Vectors:        " << NUM_VECTORS << '\n';
    std::cout << "Dimensions:     " << VECTOR_DIM << '\n';
    std::cout << "Queries:        " << NUM_QUERIES << '\n';
    std::cout << "K:              " << K << '\n';
    std::cout << "M:              " << M << '\n';
    std::cout << "efConstruction: " << EF_CONSTRUCTION << '\n';
    std::cout << "\n";

    std::cout
        << std::left
        << std::setw(12) << "efSearch"
        << std::setw(15) << "Recall@K"
        << std::setw(20) << "Brute (ms/query)"
        << std::setw(20) << "HNSW (ms/query)"
        << '\n';

    std::cout << std::string(67, '-') << '\n';

    for (const std::size_t ef_search : {10UL, 20UL, 50UL, 100UL, 200UL}) {

        if (ef_search < K) {
            continue;
        }

        hnsw::HnswConfig config{
            .M = M,
            .ef_construction = EF_CONSTRUCTION,
            .ef_search = ef_search,
        };

        hnsw::HnswIndex<hnsw::EuclideanDistance> index(config);

        // Build HNSW using exactly the same dataset.
        for (const auto& vec : vectors) {
            index.add(vec);
        }

        auto brute_start = Clock::now();

        for (const auto& query : queries) {
            [[maybe_unused]]
            auto result = brute_index.search(query, K);
        }

        auto brute_end = Clock::now();

        const double brute_total_ms =
            std::chrono::duration<double, std::milli>(
                brute_end - brute_start
            ).count();

        const double brute_avg_ms =
            brute_total_ms / static_cast<double>(NUM_QUERIES);


        double total_recall = 0.0;

        auto hnsw_start = Clock::now();

        for (std::size_t i = 0; i < NUM_QUERIES; ++i) {
            const auto approximate_neighbors =
                index.search(queries[i], K);

            total_recall += recallAtK(
                exact_results[i],
                approximate_neighbors,
                K
            );
        }

        auto hnsw_end = Clock::now();

        const double hnsw_total_ms =
            std::chrono::duration<double, std::milli>(
                hnsw_end - hnsw_start
            ).count();

        const double hnsw_avg_ms =
            hnsw_total_ms / static_cast<double>(NUM_QUERIES);

        const double avg_recall =
            total_recall / static_cast<double>(NUM_QUERIES);


        std::cout
            << std::left
            << std::setw(12) << ef_search
            << std::setw(15) << avg_recall
            << std::setw(20) << brute_avg_ms
            << std::setw(20) << hnsw_avg_ms
            << '\n';
    }

    std::cout << '\n';

    return 0;
}