#ifndef XNNC_YOLOXPP_HPP_
#define XNNC_YOLOXPP_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include "xnnc/cpp_custom_layer.hpp"
using namespace xnnc;
using namespace std;

void post_process(const vector<Tensor<float>*>& inputs, const vector<Tensor<float>*>& outputs);

#endif