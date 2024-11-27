// include/MNISTReader.h
#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <vector>
#include <string>

void readMNISTImages(const std::string &filename, std::vector<std::vector<double>> &images, int &numImages, int &imageSize);
void readMNISTLabels(const std::string &filename, std::vector<int> &labels, int &numLabels);

#endif // MNISTREADER_H