#ifndef __CLS_TRANSFER_OP_
#define __CLS_TRANSFER_OP_

#include "pi-peps/config.h"

#ifdef PEPS_WITH_ARPACK

#  include "pi-peps/cluster-ev-builder.h"
#  include "pi-peps/linalg/arpack-rcdn.h"

namespace itensor {

  struct TransferOpVecProd {
    CtmEnv::DIRECTION dir;
    Vertex v_ref;
    EVBuilder const& ev;

    TransferOpVecProd(EVBuilder const& ev,
                      Vertex const& v,
                      CtmEnv::DIRECTION dir);

    void operator()(double const* const x, double* const y, bool DBG = false);
  };

  // ARPACK required
  void analyzeTransferMatrix(EVBuilder const& ev,
                             Vertex const& v,
                             CtmEnv::DIRECTION dir = CtmEnv::DIRECTION::RIGHT,
                             int num_eigs = 2,
                             std::string alg_type = "ARPACK");

}  // namespace itensor

#endif

#endif