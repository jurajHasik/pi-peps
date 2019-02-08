#ifndef __ENGINE_
#define __ENGINE_

#include <string>
#include "json.hpp"
#include "itensor/all.h"
#include "cluster-ev-builder.h"
#include "ctm-cluster.h"
#include "ctm-cluster-global.h"
#include "mpo.h"
#include "models.h"
#include "simple-update_v2.h"
#include "full-update.h"
#include "itensor-linsys-solvers.h"

template<class T>
class TrotterDecomposition {
    public:
		std::vector< T >	gateMPO;
		std::vector< T * >	ptr_gateMPO;
		std::vector< std::vector<int> >         gate_auxInds;
		std::vector< std::vector<std::string> > gates;

		bool symmetrized    = false;
		int currentPosition = -1;

		void symmetrize() {
		    // For symmetric Trotter decomposition
		    if ( !symmetrized ) {
		        int init_gate_size = gates.size();
		        for (int i=0; i<init_gate_size; i++) {
		            ptr_gateMPO.push_back(ptr_gateMPO[init_gate_size-1-i]);
		            gates.push_back(gates[init_gate_size-1-i]);
		            gate_auxInds.push_back(gate_auxInds[init_gate_size-1-i]);
		        }
		    }
		    // can't symmetrize twice
		    symmetrized = true;

		    std::cout<<"TrotterDecomposition symmetrized"<<std::endl;
		}

		int nextCyclicIndex() {
		    currentPosition = (currentPosition + 1) % gates.size();
		    return currentPosition;
		}
};

class Engine {
	 public:
	 	itensor::LinSysSolver * pSolver;

        virtual itensor::Args performSimpleUpdate(
        	Cluster & cls, itensor::Args const& args) = 0;

        virtual itensor::Args performFullUpdate(
        	Cluster & cls, CtmEnv const& ctmEnv, itensor::Args const& args) = 0;
};

template <class T>
class TrotterEngine : public Engine {
	public:
		TrotterDecomposition<T> td;

		itensor::Args performSimpleUpdate(Cluster & cls, itensor::Args const& args);

		itensor::Args performFullUpdate(Cluster & cls, CtmEnv const& ctmEnv, itensor::Args const& args);
};

// std::unique_ptr<Engine> buildEngine_ISING3BODY(nlohmann::json & json_model);

std::unique_ptr<Engine> buildEngine_IDENTITY(nlohmann::json & json_model);

std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model, 
	itensor::LinSysSolver * solver);

std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);

template<> itensor::Args TrotterEngine<MPO_2site>::performSimpleUpdate(
	Cluster & cls, itensor::Args const& args);
template<> itensor::Args TrotterEngine<MPO_3site>::performSimpleUpdate(
	Cluster & cls, itensor::Args const& args);


template<> itensor::Args TrotterEngine<MPO_2site>::performFullUpdate(
	Cluster & cls, CtmEnv const& ctmEnv, itensor::Args const& args);
// template<> itensor::Args TrotterEngine<MPO_3site>::performFullUpdate(
// 	Cluster & cls, CtmEnv const& ctmEnv, itensor::Args const& args);
// template<> itensor::Args TrotterEngine<OpNS>::performFullUpdate(
// 	Cluster & cls, CtmEnv const& ctmEnv, itensor::Args const& args);


#endif