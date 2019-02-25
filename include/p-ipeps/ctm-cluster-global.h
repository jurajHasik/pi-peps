#ifndef __CTM_CLS_GLOBAL_
#define __CTM_CLS_GLOBAL_

#include "p-ipeps/config.h"
#include <string>
DISABLE_WARNINGS
#include "itensor/all.h"
ENABLE_WARNINGS

/*
 * ENVironment of nxm cluster 
 *  [type==t, name==n, prime==p]
 *
 *    (t: Ulink, n: I_U, p: 0)  (t: Ulink, n: I_U, p: m)
 *          V                                V
 * |C_LU|--I()--|T_U0|--I()--...--|T_Um-1|--I()--|C_RU|
 *   |            |                 |              |  
 *  I(t: Llink, n: I_L, p: 0)                     I(t: Rlink, n: I_R, p: 0)
 *   |                                             |
 * |T_L0  |--                                  --|T_R0  |     
 *   |                                             |
 *  ...                                           ...
 *   |                                             |
 * |T_Ln-1|--                                  --|T_Rn-1|
 *   |                                             |
 *  I(t: Llink, n: I_L, p: m)                     I(t: Rlink, n: I_R, p: m)
 *   |            |                 |              |   
 * |C_LD|--I()--|T_D0|--I()--...--|T_Dm-1|--I()--|C_RD|
 *          ^                                ^
 *     (t: Dlink, n: I_D, p: 0)  (t: Dlink, n: I_D, p: m)
 */

const char* const TAG_IT_ULINK  = "Ulink";
const char* const TAG_IT_RLINK  = "Rlink";
const char* const TAG_IT_DLINK  = "Dlink";
const char* const TAG_IT_LLINK  = "Llink";
const char* const TAG_IT_HSLINK = "HSlink";
const char* const TAG_IT_VSLINK = "VSlink";

// types for auxiliary indices of environment 
const auto ULINK = itensor::IndexType(TAG_IT_ULINK);
const auto RLINK = itensor::IndexType(TAG_IT_RLINK);
const auto DLINK = itensor::IndexType(TAG_IT_DLINK);
const auto LLINK = itensor::IndexType(TAG_IT_LLINK);

// types for auxiliary indices of on-Site tensors (H-horizontal, V-vertical)
// contracted through physical index
const auto HSLINK = itensor::IndexType(TAG_IT_HSLINK);
const auto VSLINK = itensor::IndexType(TAG_IT_VSLINK);

// types for auxiliary indices used in Isometry
const auto SVD_LINK = itensor::IndexType("SVDlink");

// Index names - ENV auxiliary indices
const std::string TAG_I_U = "I_U";
const std::string TAG_I_R = "I_R";
const std::string TAG_I_D = "I_D";
const std::string TAG_I_L = "I_L";
// SITE auxiliary indices (Contracted by physical index)
const std::string TAG_I_XV = "I_XV";
const std::string TAG_I_XH = "I_XH";

// ISOMETRY auxiliary indices
// TODO unused, delete 
const std::string TAG_I_FXD = "I_FXD";
const std::string TAG_I_TX  = "I_TX";

// ENVironment tensor names
const std::string TAG_C_LU = "C_LU";
const std::string TAG_C_RU = "C_RU";
const std::string TAG_C_RD = "C_RD";
const std::string TAG_C_LD = "C_LD";

const std::string TAG_T_U = "T_U";
const std::string TAG_T_R = "T_R";
const std::string TAG_T_D = "T_D";
const std::string TAG_T_L = "T_L";

#endif
