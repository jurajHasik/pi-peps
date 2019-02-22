#include "p-ipeps/engine-factory.h"
#include "id-2x2-ABCD.h"
#include "ising-2x2-ABCD.h"
#include "hb-2x2-ABCD.h"
#include "aklt-2x2-ABCD.h"
#include "j1j2-2x2-ABCD.h"
#include "ladders-2x2-ABCD.h"

EngineFactory::EngineFactory() {
    registerEngine("ID_2X2_ABCD", &itensor::IdentityModel_2x2_ABCD::buildEngine);
    registerEngine("ID_2X2_AB", &itensor::IdentityModel_2x2_AB::buildEngine);
	registerEngine("ISING_2X2_ABCD", &itensor::IsingModel_2x2_ABCD::buildEngine);
    registerEngine("ISING_2X2_AB", &itensor::IsingModel_2x2_AB::buildEngine);
    registerEngine("HB_2X2_ABCD", &itensor::HeisenbergModel_2x2_ABCD::buildEngine);
    registerEngine("HB_2X2_AB", &itensor::HeisenbergModel_2x2_AB::buildEngine);
    registerEngine("AKLT_2X2_ABCD", &itensor::AKLTModel_2x2_ABCD::buildEngine);
    registerEngine("AKLT_2X2_AB", &itensor::AKLTModel_2x2_AB::buildEngine);
    registerEngine("J1J2_2X2_ABCD", &itensor::J1J2Model_2x2_ABCD::buildEngine);
    registerEngine("LADDERS_2X2_ABCD", &itensor::LaddersModel_2x2_ABCD::buildEngine);
    registerEngine("LADDERS_4X2_ABCD", &itensor::LaddersModel_4x2_ABCD::buildEngine);
}

bool EngineFactory::registerEngine(std::string const& name, 
    TCreateMethod funcCreate) {

    auto it = s_methods.find(name);
    if (it == s_methods.end()) { 
        s_methods[name] = funcCreate;
        return true;
    }
    return false;
}

std::unique_ptr<Engine> EngineFactory::build(nlohmann::json & json_model, 
    itensor::LinSysSolver * pSolver) {
    
    std::string model_type = json_model.value("type","NOT_FOUND");

    auto it = s_methods.find(model_type);
    if (it != s_methods.end()) {
        auto engine = it->second(json_model); // call the "build" function
        engine->pSolver = pSolver;
        return engine;
    }

    std::string message = "[EngineFactory] Invalid model: "+ model_type;
    throw std::runtime_error(message);

    return nullptr;
}