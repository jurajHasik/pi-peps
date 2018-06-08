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

// ----- Trotter gates (2Site, 3site, ...) MPOs -----------------------
MPO_3site getMPO3s_Id(int physDim);

MPO_3site getMPO3s_Id_v2(int physDim, bool dbg = false);

MPO_3site getMPO3s_Uj1j2(double tau, double J1, double J2);

MPO_3site getMPO3s_Uj1j2_v2(double tau, double J1, double J2, double lambda);

MPO_3site getMPO3s_Uladder(double tau, double J, double Jp);

MPO_3site getMPO3s_Uladder_v2(double tau, double J, double Jp);

MPO_3site getMPO3s_NNHLadder_2site(double tau, double J, double alpha);

MPO_3site getMPO3s_Ising_v2(double tau, double J, double h);

MPO_3site getMPO3s_Ising_2site(double tau, double J, double h);

MPO_3site getMPO3s_Ising3Body(double tau, double J1, double J2, double h);
// ----- END Trotter gates (2Site, 3site, ...) MPOs -------------------

// ----- Definition of model base class and its particular instances --
class Model {
    public:
        virtual void setObservablesHeader(std::ofstream & output) = 0;

        virtual void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args const& metaInf) = 0;
};

class J1J2Model : public Model {
    public:
        double J1, J2;

        J1J2Model(double arg_J1, double arg_J2);

        void setObservablesHeader(std::ofstream & output);
        
        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args const& metaInf);
};

class NNHLadderModel : public Model {
    public:
        double J1, alpha;

        NNHLadderModel(double arg_J1, double arg_alpha);

        void setObservablesHeader(std::ofstream & output);

        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args const& metaInf);
};

class IsingModel : public Model {
    public:
        double J1, h;

        IsingModel(double arg_J1, double arg_h);

        void setObservablesHeader(std::ofstream & output);

        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args const& metaInf);
};

class Ising3BodyModel : public Model {
    public:
        double J1, J2, h;

        Ising3BodyModel(double arg_J1, double arg_J2, double arg_h);

        void setObservablesHeader(std::ofstream & output);

        void computeAndWriteObservables(EVBuilder const& ev, 
            std::ofstream & output, itensor::Args const& metaInf);
};
// ----- END Definition of model class --------------------------------


// ----- Model Definitions --------------------------------------------
void getModel_J1J2(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds);

void getModel_NNHLadder(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
	std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
	std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds);

void getModel_Ising(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds);

void getModel_Ising3Body(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds);

void getModel(nlohmann::json & json_model,
    std::unique_ptr<Model> & ptr_model,
    std::vector< MPO_3site > & gateMPO,
    std::vector< MPO_3site *> & ptr_gateMPO,
    std::vector< std::vector<std::string> > & gates,
    std::vector< std::vector<int> > & gate_auxInds);
// ----- END Model Definitions ----------------------------------------

#endif