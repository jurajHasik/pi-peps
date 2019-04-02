#include "pi-peps/config.h"
#include "pi-peps/svdsolver-factory.h"
#include "pi-peps/linalg/arpack-rcdn.h"
#include "pi-peps/linalg/lapacksvd-solver.h"
#include "pi-peps/linalg/rsvd-solver.h"

SvdSolverFactory::SvdSolverFactory() {
  registerSolver("default", &itensor::SvdSolver::create);
  registerSolver("itensor", &itensor::SvdSolver::create);
  registerSolver("gesdd", &itensor::GESDDSolver::create);
#ifdef PEPS_WITH_RSVD
  registerSolver("rsvd", &itensor::RsvdSolver::create);
#endif
#ifdef PEPS_WITH_ARPACK
  registerSolver("arpack", &itensor::ArpackSvdSolver::create);
#endif
}

bool SvdSolverFactory::registerSolver(std::string const& name,
                                      TCreateMethod funcCreate) {
  auto it = s_methods.find(name);
  if (it == s_methods.end()) {
    s_methods[name] = funcCreate;
    return true;
  }
  return false;
}

std::unique_ptr<itensor::SvdSolver> SvdSolverFactory::create(
  std::string const& name) {
  auto it = s_methods.find(name);
  if (it != s_methods.end())
    return it->second();  // call the "create" function

  std::string message = "[SvdSolverFactory] Invalid svd solver: " + name;
  throw std::runtime_error(message);

  return nullptr;
}
