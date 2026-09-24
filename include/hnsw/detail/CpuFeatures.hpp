#pragma once 

#include <bitset>
#include <string>
#include <vector>
#include <array>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__) || \
    defined(_M_X64) || defined(_M_IX86)

    #ifdef _MSC_VER

        #include <intrin.h>
        #define HAS_CPUID_INTRIN 1

    #elif defined(__GNUC__) || defined(__clang__)

        #include <cpuid.h>
        #define HAS_CPUID_INTRIN 1

    #endif

#endif

namespace cpu_features {

class InstructionSet {

    private:

        class InstructionSet_Internal {

#if HAS_CPUID_INTRIN

            private:
                static void cpuid(
                    std::array<int, 4>& out,
                    int leaf 
                )
                {
#ifdef _MSC_VER
                    std::array<int, 4> reg;

                    __cpuidex(reg.data(), leaf, 0);

                    out = reg;

#else

                    // eax, ebx, ecx, edx
                    std::array<int, 4> reg;

                    __cpuid_count(
                        leaf,
                        0,
                        reg[0],
                        reg[1],
                        reg[2],
                        reg[3]
                    );

                    out = reg;
#endif
                }
        
            public:
                InstructionSet_Internal() 
                    : f_1_ECX_{ 0 },
                    f_1_EDX_{ 0 },
                    f_7_EBX_{ 0 },
                    f_7_ECX_{ 0 }
                {
                    // int cpuInfo[4] =  {-1}
                    std::array<int, 4> cpui;

                    // Calling __cpui with 0x0 as the function_id argument
                    // gets the number of the highest valid function id
                    cpuid(cpui, 0x0);
                    nIds_ = cpui[0];

                    // For each CPUID function_id execute and save the registers into data
                    for (int i = 0; i <= nIds_; ++i) {
                        cpuid(cpui, i);
                        data_.push_back(cpui);
                    }

                    // Capture vendor string
                    std::array<char, 32> vendor{};
                    std::memcpy(vendor.data(), &data_[0][1], sizeof(int));
                    std::memcpy(vendor.data() + 4, &data_[0][3], sizeof(int));
                    std::memcpy(vendor.data() + 8, &data_[0][2], sizeof(int));
                    vendor_ = vendor.data();

                    if (vendor_ == "GenuineIntel") {
                        isIntel_ = true;
                    } else if(vendor_ == "AuthenticAMD") {
                        isAMD_ = true;
                    }

                    // load bitset with flags for function 0x00000001
                    if (nIds_ >= 1) {
                        f_1_ECX_ = data_[1][2];
                        f_1_EDX_ = data_[1][3];
                    }

                    // load bitset with flags for function 0x00000007
                    if (nIds_ >= 7)
                    {
                        f_7_EBX_ = data_[7][1];
                        f_7_ECX_ = data_[7][2];
                    }

                    // Calling __cpuid with 0x80000000 as the function_id argument
                    // gets the number of the highest valid extended ID.
                    cpuid(cpui, 0x80000000);
                    nExIds_ = cpui[0];

                    std::array<char, 64> brand{};
                    
                    for (int i = 0x80000000; i <= nExIds_; ++i) {
                        cpuid(cpui, i);
                        extdata_.push_back(cpui);
                    }
                    

                    // Interpret CPU brand string if reported
                    if (nExIds_ >= 0x80000004) {
                        std::memcpy(brand.data(), extdata_[2].data(), sizeof(cpui));
                        std::memcpy(brand.data() + 16, extdata_[3].data(), sizeof(cpui));
                        std::memcpy(brand.data() + 32, extdata_[4].data(), sizeof(cpui));
                        brand_ = brand.data();
                    }
                }

                int nIds_ { 0 };
                int nExIds_ { 0 };
                std::string vendor_;
                std::string brand_;
                bool isIntel_ { false };
                bool isAMD_ { false };
                std::bitset<32> f_1_ECX_;
                std::bitset<32> f_1_EDX_;
                std::bitset<32> f_7_EBX_;
                std::bitset<32> f_7_ECX_;
                std::vector<std::array<int, 4>> data_;
                std::vector<std::array<int, 4>> extdata_;
            
            #endif

        };

        inline static const InstructionSet_Internal CPU_Rep{};

        public:
            static std::string Vendor() { 
                #ifdef HAS_CPUID_INTRIN
                    return CPU_Rep.vendor_; 
                #endif
                return "";
            }
            static std::string Brand() { 
                #ifdef HAS_CPUID_INTRIN
                    return CPU_Rep.brand_; 
                #endif
                return "";
            }

            static bool FMA() {
                #ifdef HAS_CPUID_INTRIN
                    return CPU_Rep.f_1_ECX_[12]; 
                #endif
                return false;
            }
            static bool AVX2() { 
                #ifdef HAS_CPUID_INTRIN
                    return CPU_Rep.f_7_EBX_[5]; 
                #endif
                return false;
            }
             static bool AVX512F() { 
                #ifdef HAS_CPUID_INTRIN
                    return CPU_Rep.f_7_EBX_[16]; 
                #endif
                return false;
            }

    };
}
