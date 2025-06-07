#include <iomanip>
#include <iostream>
#include <string>

struct Times {
    double buildGrid = 0.f;
    double sphUpdate = 0.f;
    double memcpy = 0.f;
    int iters = 0;
};

inline void displayTimes(Times *times) {
    double avgBuildGrid = times->iters ? times->buildGrid / times->iters : 0.0;
    double avgSphUpdate = times->iters ? times->sphUpdate / times->iters : 0.0;
    double avgMemcpy = times->iters ? times->memcpy / times->iters : 0.0;

    std::cout << std::fixed << std::setprecision(5);

    std::cout << std::left << std::setw(12) << "Operation" << std::right
              << std::setw(18) << "Per frame" << std::setw(12) << "Total"
              << std::endl;

    std::cout << std::string(45, '-') << std::endl;

    std::cout << std::left << std::setw(11) << "Grid construction" << std::right
              << std::setw(11) << avgBuildGrid << std::setw(15)
              << times->buildGrid << std::endl;

    std::cout << std::left << std::setw(12) << "SPH update" << std::right
              << std::setw(16) << avgSphUpdate << std::setw(15)
              << times->sphUpdate << std::endl;

    std::cout << std::left << std::setw(12) << "Data transfer" << std::right
              << std::setw(15) << avgMemcpy << std::setw(15) << times->memcpy
              << std::endl;
}