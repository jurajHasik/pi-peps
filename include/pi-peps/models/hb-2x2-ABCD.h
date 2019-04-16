#ifndef __HB_2X2_ABCD_
#define __HB_2X2_ABCD_

#include "pi-peps/config.h"
#include "pi-peps/engine.h"
#include "pi-peps/models.h"

namespace itensor {

  // ----- Trotter gates (2site, ...) MPOs ------------------------------
  MPO_2site getMPO2s_HB(double tau, double J, double h, double del);

  MPO_3site getMPO3s_HB(double tau, double J, double h, double del);

  OpNS getOP4s_HB(double tau, double J, double h, double del);
  // ----- END Trotter gates (3site, ...) MPOs --------------------------

  // ----- Definition of model base class and its particular instances --
  class HeisenbergModel_2x2_ABCD : public Model {
   public:
    double J, h, del;

    HeisenbergModel_2x2_ABCD(double arg_J, double arg_h, double arg_del);

    void setObservablesHeader(std::ofstream& output) override;

    void computeAndWriteObservables(EVBuilder const& ev,
                                    std::ofstream& output,
                                    Args& metaInf) override;

    static std::unique_ptr<Model> create(nlohmann::json& json_model);

    static std::unique_ptr<Engine> buildEngine(nlohmann::json& json_model);
  };
  // ----- END Definition of model class --------------------------------

  // ----- Definition of model base class and its particular instances --
  class HeisenbergModel_2x2_AB : public Model {
   public:
    double J, h, del;

    HeisenbergModel_2x2_AB(double arg_J, double arg_h, double arg_del);

    void setObservablesHeader(std::ofstream& output) override;

    void computeAndWriteObservables(EVBuilder const& ev,
                                    std::ofstream& output,
                                    Args& metaInf) override;

    static std::unique_ptr<Model> create(nlohmann::json& json_model);

    static std::unique_ptr<Engine> buildEngine(nlohmann::json& json_model);
  };
  // ----- END Definition of model class --------------------------------

  // ----- Definition of model base class and its particular instances --
  class HeisenbergModel_1x1_A : public Model {
   public:
    double J, h, del;

    HeisenbergModel_1x1_A(double arg_J, double arg_h, double arg_del);

    void setObservablesHeader(std::ofstream& output) override;

    void computeAndWriteObservables(EVBuilder const& ev,
                                    std::ofstream& output,
                                    Args& metaInf) override;

    static std::unique_ptr<Model> create(nlohmann::json& json_model);

    static std::unique_ptr<Engine> buildEngine(nlohmann::json& json_model);
  };
  // ----- END Definition of model class --------------------------------
}  // namespace itensor

#endif
