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

#ifndef CPP_CUSTOM_LAYER_HPP_
#define CPP_CUSTOM_LAYER_HPP_

#include <map>
#include <string>
#include <vector>
#include <cstddef>

#include "xnnc/tensor.hpp"
#include "xnnc/xnnc_error.hpp"

#ifdef _WIN32
#define XNNC_EXPORT __declspec( dllexport )
#else // _WIN32
#define XNNC_EXPORT
#endif // _WIN32

namespace xnnc {

//=======================================================
// Low level C function information
//=======================================================

// Important note:
// the XI CNN library uses notation convention for the tensor layout
// that differs from Caffe and Tensorflow
// - Caffe and TensorTlow specifies the fastest moving dimension
//   last (C/C++ way)
// - The XI CNN lib specifies the fastest moving first
//   (Fortran way)
// This means that:
// - The default Caffe DHW (also noted CHW) layout corresponds to
//   the XI_WHD layout
// - The default TensorFlow HWD (also noted HWC) layout corresponds to
//   the XI_DWH layout
typedef enum {
  MEMORY_LAYOUT_ANY = 0,
  MEMORY_LAYOUT_XI_DWHN = 1,
  MEMORY_LAYOUT_XI_WHDN = 2
} memory_layout_e;

typedef enum
{
  eXICNN_TYPE_F32 = 1,
  eXICNN_TYPE_F16 = 2,
  eXICNN_TYPE_S32 = 3,
  eXICNN_TYPE_S16 = 4,
  eXICNN_TYPE_S8 = 5,
  eXICNN_TYPE_U32 = 8,
  eXICNN_TYPE_U16 = 9,
  eXICNN_TYPE_U8 = 10
} xicnn_type_e;

class CFunctionDescr {
public:
  virtual void setName(std::string) = 0;
  virtual void addInputLayout(memory_layout_e) = 0;
  virtual void addOutputLayout(memory_layout_e) = 0;
  virtual void addExtraParams(const void* p, size_t size) = 0;
  virtual void addConstTensors(const std::vector<Tensor<float>*> &tables, const std::vector<xicnn_type_e> &table_types)=0;
};


//=======================================================
// Base custom layer class
//=======================================================

// The user custom layer interface
class CppCustomLayer {
public:
  // XNNC inference functions
  virtual void reshape(const std::vector<Tensor<float>*>& inputs,
                       const std::vector<Tensor<float>*>& outputs) = 0;
  virtual void forward(const std::vector<Tensor<float>*>& inputs,
                       const std::vector<Tensor<float>*>& outputs) = 0;
  virtual std::string getTypeName() const = 0;

  // Low-level bridge function
  virtual void getCfunctionDescr(CFunctionDescr &descr) = 0;
};


//=======================================================
// Callback function
//=======================================================

// The custom layer needs to define the 'createCppCustomLayer'
// function with the cpp_custom_layer_factory_f prototype
typedef CppCustomLayer*
    (*cpp_custom_layer_factory_f)(std::map<std::string, std::string>& param_map);


}  // namespace xnnc

#endif
