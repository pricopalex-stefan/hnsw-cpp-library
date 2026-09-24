#pragma once

#include <hnsw/detail/Vector.hpp>
#include <hnsw/detail/CpuFeatures.hpp>
#include <cmath>
#include <cstddef>

namespace hnsw {
    class EuclideanDistance final {
        public:
            [[nodiscard]]
            float operator()(
                const Vector& u,
                const Vector& v
            ) const noexcept
            {
                if(cpu_features::InstructionSet::AVX512F()) {
                    return l2_avx512f(u, v);
                }
                if(cpu_features::InstructionSet::AVX2()) {
                    return l2_avx(u, v);
                }
                return l2_scalar(u, v);
            }

        private:
            [[nodiscard]]
            static float l2_scalar(
                const Vector& u,
                const Vector& v
            ) noexcept
            {
                float sum = 0.0f;
                for(std::size_t i = 0; i < u.size(); i++) {
                    const float diff = u[i] - v[i];
                    sum += diff * diff;
                }
                return sqrt(sum);
            }

            [[nodiscard]]
            #if defined(__clang__) || defined(__GNUC__)
                    __attribute__((target("avx2,fma")))
            #endif
            static float l2_avx(
                const Vector& u,
                const Vector& v
            ) noexcept
            {
                #ifdef HAS_CPUID_INTRIN
                __m256 sum = _mm256_setzero_ps();

                std::size_t i = 0;

                for (; i + 8 <= u.size(); i += 8) { 
                    const __m256 u_values = _mm256_loadu_ps(u.data() + i); 

                    const __m256 v_values = _mm256_loadu_ps(v.data() + i); 

                    const __m256 diff = _mm256_sub_ps(u_values, v_values); 

                    sum = _mm256_fmadd_ps(diff, diff, sum); 
                }

                const __m128 lo =  _mm256_extractf128_ps(sum, 0);
                const __m128 hi = _mm256_extractf128_ps(sum, 1);

                __m128 result = _mm_add_ps(lo, hi);

                result = _mm_hadd_ps(result, result); 
                result = _mm_hadd_ps(result, result);

                float distance_squared = _mm_cvtss_f32(result);

                // Remaining elements
                for (; i < u.size(); ++i) { 
                    const float diff = u[i] - v[i]; 
                    distance_squared += diff * diff; 
                } 

                return std::sqrt(distance_squared);
                #endif
                return 0.0f;
            }

            [[nodiscard]]
            #if defined(__clang__) || defined(__GNUC__)
                    __attribute__((target("avx512f")))
            #endif
            static float l2_avx512f(
                const Vector& u,
                const Vector& v
            ) noexcept
            {
                #ifdef HAS_CPUID_INTRIN
                __m512 sum = _mm512_setzero_ps();

                std::size_t i = 0;

                for (; i + 16 <= u.size(); i += 16) { 
                    const __m512 u_values = _mm512_loadu_ps(u.data() + i); 

                    const __m512 v_values = _mm512_loadu_ps(v.data() + i); 

                    const __m512 diff = _mm512_sub_ps(u_values, v_values); 

                    sum = _mm512_fmadd_ps(diff, diff, sum); 
                }

                float distance_squared = _mm512_reduce_add_ps(sum);

                // Remaining elements
                for (; i < u.size(); ++i) { 
                    const float diff = u[i] - v[i]; 
                    distance_squared += diff * diff; 
                } 

                return std::sqrt(distance_squared);
                #endif
                return 0.0f;
            }
    };
}