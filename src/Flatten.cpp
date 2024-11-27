#include "Flatten.h"

std::vector<double> Flatten::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    std::vector<double> output;
    for (const auto& channel : input) {
        for (const auto& row : channel) {
            for (const auto& value : row) {
                output.push_back(value);
            }
        }
    }
    return output;
}
