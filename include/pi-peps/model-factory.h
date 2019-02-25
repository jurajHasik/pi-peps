#ifndef __MODEL_FACTORY_
#define __MODEL_FACTORY_

#include "pi-peps/config.h"
#include "json.hpp"
#include "pi-peps/models.h"

class ModelFactory {
 public:
  using TCreateMethod =
    std::function<std::unique_ptr<Model>(nlohmann::json& json_model)>;

  ModelFactory();
  virtual ~ModelFactory() = default;

  bool registerModel(std::string const& name, TCreateMethod funcCreate);

  std::unique_ptr<Model> create(nlohmann::json& json_model);

  std::unique_ptr<Model> create(std::string const& name,
                                nlohmann::json& json_model);

 private:
  std::map<std::string, TCreateMethod> s_methods;
};

#endif
