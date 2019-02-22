#ifndef __ISING_2X2_ABCD_
#define __ISING_2X2_ABCD_

#include "models.h"
#include "engine.h"

namespace itensor {

// ----- Trotter gates (2site, ...) MPOs ------------------------------
MPO_2site getMPO2s_Ising_2site(double tau, double J, double h);

MPO_3site getMPO3s_Ising_3site(double tau, double J, double h);
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
class IsingModel_2x2_ABCD : public Model {
    public:
        double J1, h;

        IsingModel_2x2_ABCD(double arg_J1, double arg_h);

        void setObservablesHeader(std::ofstream & output) override;

        void computeAndWriteObservables(EVBuilder const& ev, std::ofstream & output, 
            itensor::Args & metaInf) override;

        static std::unique_ptr<Model> create(nlohmann::json & json_model);

        static std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);
};

class IsingModel_2x2_AB : public Model {
    public:
        double J1, h;

        IsingModel_2x2_AB(double arg_J1, double arg_h);

        void setObservablesHeader(std::ofstream & output) override;

        void computeAndWriteObservables(EVBuilder const& ev, std::ofstream & output, 
            itensor::Args & metaInf) override;

        static std::unique_ptr<Model> create(nlohmann::json & json_model);

        static std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);
};
// ----- END Definition of model class --------------------------------

} //namespace itensor

#endif