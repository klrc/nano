#ifndef XNNC_CUSTOM_SLICE_HPP
#define XNNC_CUSTOM_SLICE_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "xnnc/cpp_custom_layer.hpp"
using namespace xnnc;
using namespace std;

void slice_copy(const vector<Tensor<float>*>& inputs, const vector<Tensor<float>*>& outputs);

#endif