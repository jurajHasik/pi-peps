#ifndef __MODELS_
#define __MODELS_

#include "p-ipeps/config.h"
#include <string>
#include <iostream>
#include <chrono>
#include "json.hpp"
#include "p-ipeps/su2.h"
#include "p-ipeps/cluster-ev-builder.h"
#include "p-ipeps/ctm-cluster.h"
#include "p-ipeps/ctm-cluster-global.h"
#include "p-ipeps/mpo.h"
#include "itensor/all.h"

// ----- Trotter gates (3site, ...) MPOs ------------------------------
MPO_3site getMPO3s_Ising3Body(double tau, double J1, double J2, double h);
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Trotter gates (4site, ...) MPOs ------------------------------
OpNS getOP4s_J1Q(double tau, double J1, double Q);
// ----- END Trotter gates (4site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
class Model {
    public:
        int physDim;

        virtual void setObservablesHeader(std::ofstream & output) = 0;

        virtual void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args & metaInf) = 0;

        virtual ~Model() = default;
};

class J1QModel : public Model {
    public:
        double J1, Q;

        J1QModel(double arg_J1, double arg_Q);

        void setObservablesHeader(std::ofstream & output);
        
        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args & metaInf);
};

// class Ising3BodyModel : public Model {
//     public:
//         double J1, J2, h;

//         Ising3BodyModel(double arg_J1, double arg_J2, double arg_h);

//         void setObservablesHeader(std::ofstream & output);

//         void computeAndWriteObservables(EVBuilder const& ev, 
//             std::ofstream & output, itensor::Args & metaInf);
// };
// ----- END Definition of model class --------------------------------


// ----- Model Definitions --------------------------------------------
std::unique_ptr<Model> getModel_J1Q(nlohmann::json & json_model);

// std::unique_ptr<Model> getModel_Ising3Body(nlohmann::json & json_model);

std::unique_ptr<Model> getModel(nlohmann::json & json_model);
// ----- END Model Definitions ----------------------------------------

#endif
