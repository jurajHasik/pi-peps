#ifndef __ID_2X2_ABCD_
#define __ID_2X2_ABCD_

#include "pi-peps/config.h"
#include "pi-peps/models.h"
#include "pi-peps/engine.h"

namespace itensor {

// ----- Trotter gates (2site, ...) MPOs ------------------------------
MPO_2site getMPO2s_ID(int pd);

MPO_3site getMPO3s_ID(int pd);

OpNS getOP4s_ID(int pd);
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
class IdentityModel_2x2_ABCD : public Model {
    public:

        IdentityModel_2x2_ABCD(int pd);

        void setObservablesHeader(std::ofstream & output) override;

        void computeAndWriteObservables(EVBuilder const& ev, std::ofstream & output, 
            Args & metaInf) override;

        static std::unique_ptr<Model> create(nlohmann::json & json_model);

        static std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);
};
// ----- END Definition of model class --------------------------------

// ----- Definition of model base class and its particular instances --
class IdentityModel_2x2_AB : public Model {
    public:

        IdentityModel_2x2_AB(int pd);

        void setObservablesHeader(std::ofstream & output) override;

        void computeAndWriteObservables(EVBuilder const& ev, std::ofstream & output, 
            Args & metaInf) override;

        static std::unique_ptr<Model> create(nlohmann::json & json_model);

        static std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);
};
// ----- END Definition of model class --------------------------------

} //namespace itensor

#endif
