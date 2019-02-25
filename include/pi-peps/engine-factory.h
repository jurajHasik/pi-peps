#ifndef __ENGINE_FACTORY_
#define __ENGINE_FACTORY_

#include "pi-peps/config.h"
#include "json.hpp"
#include "pi-peps/engine.h"

class EngineFactory {
	public:
		using TCreateMethod = std::function<
			std::unique_ptr<Engine>(nlohmann::json & json_model)>;

		EngineFactory();
		virtual ~EngineFactory() = default;

		bool registerEngine(std::string const& name, TCreateMethod funcCreate);

		std::unique_ptr<Engine> build(nlohmann::json & json_model, 
			itensor::LinSysSolver * pSolver = nullptr);

	private:
		std::map<std::string, TCreateMethod> s_methods;
};

#endif
