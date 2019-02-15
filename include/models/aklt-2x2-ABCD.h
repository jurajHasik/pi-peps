#ifndef __AKLT_2X2_ABCD_
#define __AKLT_2X2_ABCD_

#include "models.h"
#include "engine.h"

namespace itensor {

// ----- Trotter gates (2site, ...) MPOs ------------------------------
ITensor Projector_S2tpS2_S1(Index & s1, Index & s2);

ITensor Projector_S2tpS2_S1_v2(Index & s1, Index & s2);

MPO_2site getMPO2s_AKLT(double tau, double lambda);

MPO_3site getMPO3s_AKLT(double tau);
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
class AKLTModel_2x2_ABCD : public Model {
    public:

        AKLTModel_2x2_ABCD();

        void setObservablesHeader(std::ofstream & output) override;

        void computeAndWriteObservables(EVBuilder const& ev, std::ofstream & output, 
            Args & metaInf) override;

        static std::unique_ptr<Model> create(nlohmann::json & json_model);

        static std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);
};
// ----- END Definition of model class --------------------------------

// ----- Definition of model base class and its particular instances --
class AKLTModel_2x2_AB : public Model {
    public:

        AKLTModel_2x2_AB();

        void setObservablesHeader(std::ofstream & output) override;

        void computeAndWriteObservables(EVBuilder const& ev, std::ofstream & output, 
            Args & metaInf) override;

        static std::unique_ptr<Model> create(nlohmann::json & json_model);

        static std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);
};
// ----- END Definition of model class --------------------------------

} //namespace itensor

#endif