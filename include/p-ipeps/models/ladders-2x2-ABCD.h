#ifndef __LADDERS_2X2_ABCD_
#define __LADDERS_2X2_ABCD_

#include "p-ipeps/models.h"
#include "p-ipeps/engine.h"

namespace itensor {

// ----- Trotter gates (2site, ...) MPOs ------------------------------
MPO_2site getMPO2s_Ladders(double tau, double J);

MPO_3site getMPO3s_Ladders(double tau, double J, double alpha);

OpNS getOP4s_Ladders(double tau, double J, double alpha);
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
class LaddersModel_2x2_ABCD : public Model {
	public:
		double J, alpha;

		LaddersModel_2x2_ABCD(double arg_J, double arg_alpha);

		void setObservablesHeader(std::ofstream & output) override;

		void computeAndWriteObservables(EVBuilder const& ev, 
		    std::ofstream & output, Args & metaInf) override;

		static std::unique_ptr<Model> create(nlohmann::json & json_model);

		static std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);
};
// ----- END Definition of model class --------------------------------

// ----- Definition of model base class and its particular instances --
class LaddersModel_4x2_ABCD : public Model {
	public:
		double J, alpha;

		LaddersModel_4x2_ABCD(double arg_J, double arg_alpha);

		void setObservablesHeader(std::ofstream & output) override;

		void computeAndWriteObservables(EVBuilder const& ev, 
		    std::ofstream & output, Args & metaInf) override;

		static std::unique_ptr<Model> create(nlohmann::json & json_model);

		static std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);
};
// ----- END Definition of model class --------------------------------

} //namespace itensor

#endif