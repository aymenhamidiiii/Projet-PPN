#ifndef FLATTEN_H
#define FLATTEN_H

#include <vector>

class Flatten {
public:
    std::vector<double> forward(const std::vector<std::vector<std::vector<double>>>& input);
};

#endif
