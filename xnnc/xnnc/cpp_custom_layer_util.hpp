/*
 * Copyright (c) 2018 by Cadence Design Systems, Inc.  ALL RIGHTS RESERVED.
 * These coded instructions, statements, and computer programs are the
 * copyrighted works and confidential proprietary information of
 * Cadence Design Systems Inc.  They may be adapted and modified by bona fide
 * purchasers for internal use, but neither the original nor any adapted
 * or modified version may be disclosed or distributed to third parties
 * in any manner, medium, or form, in whole or in part, without the prior
 * written consent of Cadence Design Systems Inc.  This software and its
 * derivatives are to be executed solely on products incorporating a Cadence
 * Design Systems processor.
 */

#ifndef CPP_CUSTOM_LAYER_UTILS_HPP_
#define CPP_CUSTOM_LAYER_UTILS_HPP_

#include "xnnc/cpp_custom_layer.hpp"



namespace xnnc {

#define XNNC_MAX_IN_KEY    "xnnc_max_in"
#define XNNC_MAX_OUT_KEY   "xnnc_max_out"
#define XNNC_MIN_IN_KEY    "xnnc_min_in"
#define XNNC_MIN_OUT_KEY   "xnnc_min_out"

//=======================================================
// Simple tensor implementation
//=======================================================

template <typename T>
class SimpleTensor : public Tensor<T> {
public:
  SimpleTensor() {
    num_items_ = 0;
    num_items_allocated_ = 0;
    data_ = NULL;
  }

  SimpleTensor(TensorShape& shape) {
    // Handle the shape
    shape_.resize(shape.size());
    for(int dim = 0; dim<shape.size(); dim++) {
      if (shape[dim]<=0) {
        // TODO: fail
      }
      shape_[dim] = shape[dim];
    }

    // Allocate the memory
    num_items_ = 1;
    for(int i=0;i<shape_.size();i++) { num_items_*= shape_[i]; }
    data_ = new T[num_items_];
    num_items_allocated_ = num_items_;
    name_ = "";
  }

  ~SimpleTensor() {
    delete data_;
  }

  // Data
  virtual const T *getData() { return data_; }
  virtual T *getMutableData() { return data_; }
  virtual size_t getNumItems() const { return num_items_; }

  // Shape
  virtual const TensorShape& getShape() const { return shape_; }
  virtual void reshape(const TensorShape &shape) {
    // Handle the shape
    shape_.resize(shape.size());
    for(int i = 0; i<shape.size(); i++) {
      if (shape[i]<=0) {
        // TODO: fail
      }
      shape_[i] = shape[i];
    }
    // Handle the memory
    num_items_ = 1;
    for(int i=0;i<shape_.size();i++) {
      num_items_ *= shape_[i];
    }
    if (num_items_ > num_items_allocated_) {
      if (data_ != NULL) {
        delete[] data_;
      }
      data_ = new T[num_items_];
      num_items_allocated_ = num_items_;
    }
  }
  virtual void reshape(int n, int d, int h, int w) {
    int array[4] = {n, d, h, w};
    TensorShape shape(array, array+4);
    reshape(shape);
  }

  //TODO: DHW for now (Caffe layout). Should be customizable by the user.
  //TODO: These function follow the Caffe approach, but this is not the most
  //      natural. For instance, a 2D tensor will be considered as a
  //      Batch x Depth object.
  int getDim(int i) const
  {
    if ((i<0) || (i>=shape_.size())) { return 1; }
    return shape_[i];
  }
  virtual int getBatch() const
  {
    return getDim(0);
  }
  virtual int getDepth() const
  {
    return getDim(1);
  }
  virtual int getHeight() const
  {
    return getDim(2);
  }
  virtual int getWidth() const
  {
    return getDim(3);
  }

  virtual size_t getOffset(int n = 0, int c = 0, int h = 0, int w = 0) const
  {
    return ((n * getDepth() + c) * getHeight() + h) * getWidth() + w;
  }

  virtual const std::string &getName() const { return name_; }

private:
  T *data_;
  size_t num_items_;
  size_t num_items_allocated_;
  TensorShape shape_;
  std::string name_;
};


template class SimpleTensor<float>; 
template class SimpleTensor<double>;
template class SimpleTensor<unsigned short>;
template class SimpleTensor<int>;



#define READ_INT_PARAMS(PARAM_MAP, KEY,DEFAULT_VAL,DEST_VAR)     int KEY = DEFAULT_VAL; \
    if (PARAM_MAP.count(#KEY) != 0) { \
      KEY = strtol(PARAM_MAP[#KEY].c_str(), NULL, 10); \
    } \
    DEST_VAR = KEY; 

#define READ_FLOAT_PARAMS(PARAM_MAP,KEY,DEFAULT_VAL,DEST_VAR)     float KEY = DEFAULT_VAL; \
    if (PARAM_MAP.count(#KEY) != 0) { \
      KEY = strtof(PARAM_MAP[#KEY].c_str(), NULL); \
    } \
    DEST_VAR = KEY; 

#define CHECK_CONDITION(PASS_CONDITION,ERROR_MESSAGE_STRING)      if (!(PASS_CONDITION)) \
{ \
  stringstream message; \
  message << ERROR_MESSAGE_STRING << endl; \
  throw XnncError(message.str()); \
} \

using namespace std;
vector<std::string> getParamVec(map<string, string> param_map, std::string key);


}  // namespace xnnc

#endif
