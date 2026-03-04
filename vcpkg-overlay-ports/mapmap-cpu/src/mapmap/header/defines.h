/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#ifndef __MAPMAP_DEFINES_H_
#define __MAPMAP_DEFINES_H_

#include <chrono>
#include <string>
#include <map>

/*
 * *****************************************************************************
 * ************************* Parallelization control ***************************
 * *****************************************************************************
 */

#define BFS_ROOTS 32u
#define DIV_UP(A, V) (A / V + (A % V == 0 ? 0 : 1))

/*
 * *****************************************************************************
 * ************************** OS-dependent brainfuck ***************************
 * *****************************************************************************
 */
#ifdef _MSC_VER
    #define FORCEINLINE __forceinline
#else
    #define FORCEINLINE __attribute__((always_inline)) inline
#endif

/*
 * *****************************************************************************
 * *************************** Namespace definitions ***************************
 * *****************************************************************************
 */

#define NS_MAPMAP mapmap

#define NS_MAPMAP_BEGIN namespace NS_MAPMAP {
#define NS_MAPMAP_END }

#if defined(_MSC_VER) && !defined(_MSVC_LANG)
#define _MSVC_LANG 0L
#endif

#if defined(_MSVC_LANG)
#define MAPMAP_LANG_LEVEL _MSVC_LANG
#else
#define MAPMAP_LANG_LEVEL __cplusplus
#endif

#if MAPMAP_LANG_LEVEL > 201103L
#define MAPMAP_USE_STD_SHUFFLE 1
#else
#define MAPMAP_USE_STD_SHUFFLE 0
#endif

#endif /* __MAPMAP_DEFINES_H_ */