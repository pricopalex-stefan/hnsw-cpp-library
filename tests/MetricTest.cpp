#include "../src/CosineDistance.hpp"
#include "../src/DotProductDistance.hpp"
#include "../src/EuclideanDistance.hpp"
#include  <hnsw/Metric.hpp>
#include  <hnsw/Vector.hpp>

class BadMetric {
    public:
        float operator()(
            const hnsw::Vector& u,
            const hnsw::Vector& v
        ) const 
        {
            return 0.0f;
        }
};

static_assert(hnsw::Metric<hnsw::EuclideanDistance>);
static_assert(hnsw::Metric<hnsw::CosineDistance>);
static_assert(hnsw::Metric<hnsw::DotProductDistance>);
static_assert(!hnsw::Metric<BadMetric>);

int main() {
    return 0;
}