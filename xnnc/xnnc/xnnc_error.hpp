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

#ifndef XNNC_ERROR_HPP_
#define XNNC_ERROR_HPP_

#include <exception>
#include <string>

namespace xnnc {

class XnncError : public std::exception {
public:
  explicit XnncError(const std::string& message) :
    message_(message)
  { }

  XnncError(const std::exception &e, const std::string& message)
  {
    message_ = message + std::string(e.what());
  }

  virtual ~XnncError() throw() {}

  virtual const char* what() const throw() {
    return message_.c_str();
  }

private:
  std::string message_;
};

}  // namespace xnnc

#endif
