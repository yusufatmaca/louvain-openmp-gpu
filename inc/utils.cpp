#include <iostream>
#include <omp.h>
#include "utils.h"

int get_thread_group_size(int degree) {
    if (degree <= 4) return 4;          // 2^2 threads
    else if (degree <= 16) return 8;    // 2^3 threads
    else if (degree <= 64) return 16;   // 2^4 threads
    else return 32;                     // 2^5 threads
}

int checkDevice() {
    int num_devices = omp_get_num_devices();
    int result = 0;
    if (num_devices > 0) {
        #pragma omp target
        {
            if (!omp_is_initial_device()) {
                printf("OpenMP can handle offloaded to GPU successfully...\n");
                printf("---------------------------------------------------\n");
                result = 1;
            } else {
                printf("OpenMP can _NOT_ handle offloaded to GPU. There may be a missing module(s).\n");
                printf("---------------------------------------------------\n");
                result = -1;
            }
        }
    } else {
        printf("No GPU devices available\n");
        printf("---------------------------------------------------\n");
        result = -1;
    }

    return result;
}

int checkArgs(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mtx_file> [output_csr_file]" << std::endl;
        return -1;
    }
    return 1;
}