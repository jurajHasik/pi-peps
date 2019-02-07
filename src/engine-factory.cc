#include "engine-factory.h"
#include "ising-2x2-ABCD.h"

EngineFactory::EngineFactory() {
	registerEngine("ISING_2X2_ABCD", &itensor::IsingModel_2x2_ABCD::buildEngine);
}

bool EngineFactory::registerEngine(std::string const& name, 
    TCreateMethod funcCreate) {
    if (auto it = s_methods.find(name); it == s_methods.end()) { 
        s_methods[name] = funcCreate;
        return true;
    }
    return false;
}

std::unique_ptr<Engine> EngineFactory::build(nlohmann::json & json_model, 
    itensor::LinSysSolver * pSolver) {
    
    std::string model_type = json_model.value("type","NOT_FOUND");

    if (auto it = s_methods.find(model_type); it != s_methods.end()) {
        auto engine = it->second(json_model); // call the "build" function
        engine->pSolver = pSolver;
        return engine;
    }

    std::string message = "[EngineFactory] Invalid model: "+ model_type;
    throw std::runtime_error(message);

    return nullptr;
}