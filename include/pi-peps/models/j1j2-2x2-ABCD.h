#ifndef __J1J2_2X2_ABCD_
#define __J1J2_2X2_ABCD_

#include "pi-peps/config.h"
#include "pi-peps/engine.h"
#include "pi-peps/models.h"

namespace itensor {

  // ----- Trotter gates (2site, ...) MPOs ------------------------------
  MPO_3site getMPO3s_J1J2(double tau, double J1, double J2, double del);

  OpNS getOP4s_J1J2(double tau, double J1, double J2, double del);
  // ----- END Trotter gates (3site, ...) MPOs --------------------------

  // ----- Definition of model base class and its particular instances --
  class J1J2Model_2x2_ABCD : public Model {
   public:
    double J1, J2, del;

    J1J2Model_2x2_ABCD(double arg_J1, double arg_J2, double arg_del);

    void setObservablesHeader(std::ofstream& output) override;

    void computeAndWriteObservables(EVBuilder const& ev,
                                    std::ofstream& output,
                                    Args& metaInf) override;

    static std::unique_ptr<Model> create(nlohmann::json& json_model);

    static std::unique_ptr<Engine> buildEngine(nlohmann::json& json_model);
  };
  // ----- END Definition of model class --------------------------------

  // ----- Definition of model base class and its particular instances --
  class J1J2Model_1x1_A : public Model {
   public:
    double J1, J2, del;

    J1J2Model_1x1_A(double arg_J1, double arg_J2, double arg_del);

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
