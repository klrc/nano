#ifndef XNNC_YOLOXPP_HPP
#define XNNC_YOLOXPP_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "xnnc/cpp_custom_layer.hpp"
using namespace xnnc;
using namespace std;

void post_process(const vector<Tensor<float>*>& inputs, const vector<Tensor<float>*>& outputs, const int num_strides);

#endif