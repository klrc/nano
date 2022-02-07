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

#ifndef XNNC_TENSOR_HPP_
#define XNNC_TENSOR_HPP_

#include <cstddef>
#include <vector>
#include <string>

//=======================================================
// Tensor interface
//=======================================================

namespace xnnc {

typedef std::vector<int> TensorShape;

template <typename T>
class Tensor {
public:
  // General info
  virtual const std::string &getName() const = 0;

  // Data
  virtual const T *getData() = 0;
  virtual T *getMutableData() = 0;

  // Shape
  virtual const TensorShape& getShape() const = 0;
  virtual void reshape(const TensorShape &new_shape) = 0;
  virtual size_t getNumItems() const = 0;
  inline int getNumDims() const { return getShape().size(); }

  // TODO: very specific function for images
  virtual void reshape(int b, int d, int h, int w) = 0;
  virtual int getHeight() const = 0;
  virtual int getWidth() const = 0;
  virtual int getDepth() const = 0;
  virtual int getBatch() const = 0;
  virtual size_t getOffset(int b = 0, int d = 0, int h = 0, int w = 0) const = 0;
};

}  // namespace xnnc

#endif

