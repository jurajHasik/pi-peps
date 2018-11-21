#include "ctm-cluster-env_v2.h"

using namespace itensor;

// C1 T1 C2
// T4 X  T2
// C4 T3 C3

// Four possible directions:
// Left, Up, Down, Right

// Left
// C1*T1 -> C1(X + (1,0))  
// T4*X  -> T4(X + (1,0))
// C4*T3 -> C4(X + (1,0))

// Up
// C1*T4 T1*X C2*T2 -> C1(X+(0,-1)) T1(X+(0,-1)) C2(X+(0,-1))

// Right
// T1*C2 -> C2(X + (-1.0))
// X*T2  -> T2(X + (-1,0))
// T3*C3 -> C3(x + (-1,0))

// Down 
// T4*C4 X*T3 T2*C3

// Define map for direction to tensors needed for the move

void CtmEnv::move_singleDirection(unsigned int direction, Cluster const& c) {

	Shift shift;
 	switch (direction) {
 		case 1: { shift = Shift( 1, 0); break; } // left
 		case 2: { shift = Shift( 0,-1); break; } // up
 		case 3:	{ shift = Shift(-1, 0); break; } // right
 		case 4: { shift = Shift( 0, 1); break; } // down
 		default: throw std::runtime_error("[move_singleDirection] Invalid direction");
 	}

 	// map C T C according to selected direction	
 	// to Cu T Cv where u,v 1,2 or 2,3 or 3,4 or 4,1

 	ITensor P, Pt;
	// computeIsometries(direction) vector of P, Pt

	// copy current C,T,C for each X O(x^2) + O(x^2 D^2)
	// depending on the direction

	// clockwise
	std::vector<ITensor> C;
	std::vector<ITensor> T;
	std::vector<ITensor> Ct;
	std::vector<ITensor> * ptr_Taux;
	std::vector<ITensor> * ptr_Tauxt;
	switch (direction) {
		case 1:	{ C = C_LU; ptr_Taux = &T_U; T = T_L; Ct = C_LD; ptr_Tauxt = &T_D; break; }
		case 2: { C = C_RU; ptr_Taux = &T_R; T = T_U; Ct = C_LU; ptr_Tauxt = &T_L; break; }  
		case 3: { C = C_RD; ptr_Taux = &T_D; T = T_R; Ct = C_RU; ptr_Tauxt = &T_U; break; }  
		case 4: { C = C_LD; ptr_Taux = &T_L; T = T_D; Ct = C_RD; ptr_Tauxt = &T_R; break; } 
	}
	std::vector<ITensor> const& Taux 	= *ptr_Taux; 
	std::vector<ITensor> const& Tauxt 	= *ptr_Tauxt; 

	// iterate over pairs (Vertex, Id) within elementary cell of cluster
	// Id identifies tensor belonging to Vertex
	for (auto const& v_id : c.vToId) {
		Vertex const& v = v_id.first;
		std::string id  = v_id.second;

		Vertex shifted = v + shift;

		// TODO santize this
		// glue that maps from Vertex to position within C,T,Ct,Taux,Tauxt
		// auto vTo[const& c] (Vertex const& v) { c.get } 
		auto vToPos = [&c] (Vertex const& v)->int { return c.SI.at(c.vertexToId(v)); };
		
		int v_pos = vToPos(v);
		int shifted_pos = vToPos(shifted);

		C[shifted_pos] 	= (C[v_pos] * Taux[v_pos]) * P;

		T[shifted_pos] 	= (((P * T[v_pos]) * c.getSiteRefc(v)) * dag(c.getSite(v)) ) * Pt;
	
		Ct[shifted_pos] = (Ct[v_pos] * Tauxt[v_pos]) * Pt;
	}
}