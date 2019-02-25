#ifndef __SVDSOLVER_FACTORY_
#define __SVDSOLVER_FACTORY_

#include "pi-peps/config.h"
#include "json.hpp"
#include "pi-peps/linalg/itensor-svd-solvers.h"

class SvdSolverFactory {
	public:
		using TCreateMethod = std::function<std::unique_ptr<itensor::SvdSolver>()>;

		SvdSolverFactory();
		virtual ~SvdSolverFactory() = default;

		bool registerSolver(std::string const& name, TCreateMethod funcCreate);

		std::unique_ptr<itensor::SvdSolver> create(std::string const& name);

	private:
		std::map<std::string, TCreateMethod> s_methods;
};

#endif
