#ifndef __MODELS_
#define __MODELS_

#include <string>
#include <iostream>
#include <chrono>
#include "json.hpp"
#include "su2.h"
#include "cluster-ev-builder.h"
#include "ctm-cluster.h"
#include "ctm-cluster-global.h"
#include "mpo.h"
#include "itensor/all.h"

// ----- Trotter gates (2site, ...) MPOs ------------------------------
MPO_2site getMPO2s_Id(int physDim);

MPO_2site getMPO2s_NNH_2site(double tau, double J, double h);

MPO_2site getMPO2s_Ising_2site(double tau, double J, double h);
// ----- END Trotter gates (2site, ...) MPOs --------------------------


// ----- Trotter gates (3site, ...) MPOs ------------------------------
MPO_3site getMPO3s_Id(int physDim);

MPO_3site getMPO3s_Id_v2(int physDim, bool dbg = false);

MPO_3site getMPO3s_Uj1j2(double tau, double J1, double J2);

MPO_3site getMPO3s_Uj1j2_v2(double tau, double J1, double J2, double lambda);

MPO_3site getMPO3s_ANISJ1J2(double tau, double J1, double J2, double del);

MPO_3site getMPO3s_Uladder(double tau, double J, double Jp);

MPO_3site getMPO3s_Uladder_v2(double tau, double J, double Jp);

MPO_3site getMPO3s_NNHLadder_2site(double tau, double J, double alpha);

MPO_3site getMPO3s_Ising_v2(double tau, double J, double h);

MPO_3site getMPO3s_Ising_2site(double tau, double J, double h);

MPO_3site getMPO3s_Ising3Body(double tau, double J1, double J2, double h);
// ----- END Trotter gates (3site, ...) MPOs --------------------------


// ----- Trotter gates (4site, ...) MPOs ------------------------------
OpNS getOP4s_J1J2(double tau, double J1, double J2);
// ----- END Trotter gates (4site, ...) MPOs --------------------------


// ----- Definition of model base class and its particular instances --
class Model {
    public:
        virtual void setObservablesHeader(std::ofstream & output) = 0;

        virtual void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args & metaInf) = 0;
};

class J1J2Model : public Model {
    public:
        double J1, J2;

        J1J2Model(double arg_J1, double arg_J2);

        void setObservablesHeader(std::ofstream & output);
        
        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args & metaInf);
};

class NNHModel_2x2Cell_AB : public Model {
    public:
        double J1, h, del;

        NNHModel_2x2Cell_AB(double arg_J1, double arg_h, double arg_del);

        void setObservablesHeader(std::ofstream & output);
        
        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args & metaInf);
};

class NNHModel_2x2Cell_ABCD : public Model {
    public:
        double J1, h, del;

        NNHModel_2x2Cell_ABCD(double arg_J1, double arg_h, double arg_del);

        void setObservablesHeader(std::ofstream & output);
        
        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args & metaInf);
};

class NNHLadderModel : public Model {
    public:
        double J1, alpha;

        NNHLadderModel(double arg_J1, double arg_alpha);

        void setObservablesHeader(std::ofstream & output);

        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args & metaInf);
};

class IsingModel : public Model {
    public:
        double J1, h;

        IsingModel(double arg_J1, double arg_h);

        void setObservablesHeader(std::ofstream & output);

        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args & metaInf);
};

class Ising3BodyModel : public Model {
    public:
        double J1, J2, h;

        Ising3BodyModel(double arg_J1, double arg_J2, double arg_h);

        void setObservablesHeader(std::ofstream & output);

        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args & metaInf);
};
// ----- END Definition of model class --------------------------------


// ----- Model Definitions --------------------------------------------
std::unique_ptr<Model> getModel_J1J2(nlohmann::json & json_model);

// void getModel_NNH_2x2Cell_AB(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_2site > & gateMPO,
//     std::vector< MPO_2site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds);

std::unique_ptr<Model> getModel_NNH_2x2Cell_ABCD(nlohmann::json & json_model);

std::unique_ptr<Model> getModel_NNH_2x2Cell_Ladder(nlohmann::json & json_model);

// void getModel_Ising(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_3site > & gateMPO,
//     std::vector< MPO_3site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds);

// void getModel_Ising_2x2Cell_ABCD(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_2site > & gateMPO,
//     std::vector< MPO_2site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds);

std::unique_ptr<Model> getModel_Ising3Body(nlohmann::json & json_model);

// void getModel_3site(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_3site > & gateMPO,
//     std::vector< MPO_3site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds);

// void getModel_2site(nlohmann::json & json_model,
//     std::unique_ptr<Model> & ptr_model,
//     std::vector< MPO_2site > & gateMPO,
//     std::vector< MPO_2site *> & ptr_gateMPO,
//     std::vector< std::vector<std::string> > & gates,
//     std::vector< std::vector<int> > & gate_auxInds);

std::unique_ptr<Model> getModel(nlohmann::json & json_model);
// ----- END Model Definitions ----------------------------------------

#endif