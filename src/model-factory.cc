#include "model-factory.h"
#include "ising-2x2-ABCD.h"

ModelFactory::ModelFactory() {
	registerModel("ISING_2X2_ABCD", &itensor::IsingModel_2x2_ABCD::create);
}

bool ModelFactory::registerModel(std::string const& name, 
    TCreateMethod funcCreate) {
    if (auto it = s_methods.find(name); it == s_methods.end()) { 
        s_methods[name] = funcCreate;
        return true;
    }
    return false;
}

std::unique_ptr<Model> ModelFactory::create(nlohmann::json & json_model) {
    
    std::string model_type = json_model.value("type","NOT_FOUND");

    return create(model_type,json_model);

    return nullptr;
}

std::unique_ptr<Model> ModelFactory::create(std::string const& name, 
	nlohmann::json & json_model) {
    if (auto it = s_methods.find(name); it != s_methods.end())
        return it->second(json_model); // call the "create" function

    std::string message = "[ModelFactory] Invalid model: "+ name;
    throw std::runtime_error(message);

    return nullptr;
}