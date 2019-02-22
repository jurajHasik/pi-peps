#ifndef __ENGINE_
#define __ENGINE_

#include <string>
#include "json.hpp"
#include "itensor/all.h"
#include "p-ipeps/cluster-ev-builder.h"
#include "p-ipeps/ctm-cluster.h"
#include "p-ipeps/ctm-cluster-global.h"
#include "p-ipeps/mpo.h"
#include "p-ipeps/models.h"
#include "p-ipeps/simple-update_v2.h"
#include "p-ipeps/full-update.h"
#include "p-ipeps/linalg/itensor-linsys-solvers.h"

template<class T>
class TrotterGate {
	public:
		Vertex init_vertex;
		std::vector<Shift> disp;
		T * ptr_gate;

	TrotterGate(Vertex const& init_v, std::vector<Shift> const& ddisp, 
		T * pptr_gate) : init_vertex(init_v), disp(ddisp), ptr_gate(pptr_gate) {}
};

template<class T>
class TrotterDecomposition {
    public:
    	std::vector< T >	          gateMPO;
		std::vector< TrotterGate<T> > tgates;

		bool symmetrized    = false;
		int currentPosition = -1;

		void symmetrize() {
		    // For symmetric Trotter decomposition
		    if ( !symmetrized ) {
		        int init_gate_size = tgates.size();
		        for (int i=0; i<init_gate_size; i++) {
		        	tgates.push_back(tgates[init_gate_size-1-i]);
		        }
		    }
		    // can't symmetrize twice
		    symmetrized = true;

		    std::cout<<"TrotterDecomposition symmetrized"<<std::endl;
		}

		int nextCyclicIndex() {
		    currentPosition = (currentPosition + 1) % tgates.size();
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

		itensor::Args performSimpleUpdate(Cluster & cls, 
			itensor::Args const& args) override;

		itensor::Args performFullUpdate(Cluster & cls, CtmEnv const& ctmEnv, 
			itensor::Args const& args) override;
};

// std::unique_ptr<Engine> buildEngine_ISING3BODY(nlohmann::json & json_model);

// std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model, 
// 	itensor::LinSysSolver * solver);

// std::unique_ptr<Engine> buildEngine(nlohmann::json & json_model);

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