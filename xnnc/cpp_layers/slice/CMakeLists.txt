# Copyright (c) 2018 by Cadence Design Systems, Inc. ALL RIGHTS RESERVED.
#
# These coded instructions, statements, and computer programs are the
# copyrighted works and confidential proprietary information of Cadence Design
# Systems Inc. They may be adapted and modified by bona fide purchasers for
# internal use, but neither the original nor any adapted or modified version
# may be disclosed or distributed to third parties in any manner, medium, or
# form, in whole or in part, without the prior written consent of Cadence
# Design Systems Inc or in accordance with the terms of your Design Technology
# License Agreement with Cadence Design Systems. This software and its
# derivatives are to be executed solely on products incorporating a Cadence
# Design Systems processor.
#

cmake_minimum_required(VERSION 3.0)
set(cpp_layer_name "slice")

file(GLOB XNNC_LAYER_SOURCES *.cpp)
file(GLOB XNNC_LAYER_HEADERS *.hpp)

include_directories(../../../../Analyzer/include)

add_library(${cpp_layer_name} SHARED
  ${XNNC_LAYER_SOURCES}
   # For IDEs only
  ${XNNC_LAYER_HEADERS}
  )

set_property(TARGET ${cpp_layer_name} PROPERTY POSITION_INDEPENDENT_CODE ON)

install (TARGETS ${cpp_layer_name} DESTINATION .)