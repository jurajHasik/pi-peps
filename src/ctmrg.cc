#include "ctm-cluster-env_v2.h"

using namespace itensor;

void CtmEnv::move_unidirectional(DIRECTION direction, 
	ISOMETRY iso_type, std::vector<double> & accT) {
	
	int length;
	switch (direction) {
 		case DIRECTION::LEFT:  { length = p_cluster->lX; break; }
 		case DIRECTION::UP:    { length = p_cluster->lY; break; }
 		case DIRECTION::RIGHT: { length = p_cluster->lX; break; }
 		case DIRECTION::DOWN:  { length = p_cluster->lY; break; }
 		default: throw std::runtime_error("[move_singleDirection] Invalid direction");
 	}

	for (unsigned int i=0; i<length; i++) {
		move_singleDirection(direction, iso_type, accT);
	}
}

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

void CtmEnv::move_singleDirection(DIRECTION direction, ISOMETRY iso_type, 
	std::vector<double> & accT) 
{
	using time_point = std::chrono::high_resolution_clock::time_point;
	auto BRAKET_OFFSET = p_cluster->BRAKET_OFFSET;
	auto sites = p_cluster->sites;

	auto get_mS = [](time_point ti, time_point tf) { return std::chrono::duration_cast
            <std::chrono::microseconds>(tf - ti).count()/1000.0; };

    auto vToId  = [this] (Vertex const& v) { return p_cluster->vertexToId(v); };

    auto readyToContract = [this](ITensor & t, CtmEnv::DIRECTION direction, 
    	Vertex const& v0, int dir0, Vertex const& v1, int dir1) {
    	// relabel env auxiliary indices
    	t *= delta(tauxByVertex(direction,v0,dir0), tauxByVertex(direction,v1,dir1));
    	// relabel site auxiliary indices
    	t *= p_cluster->DContract(v0,dir0,v1,dir1);
    	t *= prime(p_cluster->DContract(v0,dir0,v1,dir1), p_cluster->BRAKET_OFFSET);
    };

    // Compute isometries
    time_point t_iso_begin, t_iso_end;
 	t_iso_begin = std::chrono::high_resolution_clock::now();
 	std::map<std::string, Index> ip, ipt;
 	std::map<std::string, ITensor> P, Pt;
 	switch(iso_type) {
        // case ISOMETRY_T3: {
        //     compute_IsometriesT3(direction, c, ip, ipt, ia, P, Pt, accT);
        //     break;
        // }
        case ISOMETRY_T4: {
            compute_IsometriesT4(direction, ip, ipt, P, Pt, accT);
            break;
        }
    }
	t_iso_end = std::chrono::high_resolution_clock::now();
    accT[0] += get_mS(t_iso_begin, t_iso_end);

	Shift shift, p_shift;
	int dir0, dir1;
	DIRECTION opposite_direction;
 	switch (direction) {
 		case DIRECTION::LEFT: { 
 			// C(v)  * Taux(v)  * Pt(v+(0,1))        -> nC(v+(1,0)) 
 			// T(v)  * site(v)  * P(v+(0,1)) * Pt(v) -> nT(v+(1,0))
 			// Ct(v) * Tauxt(v) * P(v)               -> nCt(v+(1,0))
 			//
 			// C(v)--Taux(v)
 			// 1     1
 			// 1     1
 			// Pt(v+(0,1))
 			// P(v+(0,1))
 			// 3     3
 			// 1     1
 			// T(v)--site(v)
 			// 3     3
 			// 1     1
 			// Pt(v)
 			// P(v)
 			// 3      3
 			// 3      3
 			// Ct(v)--Tauxt(v)
 			//
 			shift= Shift( 1, 0);
 			p_shift= Shift( 0, 1);
 			dir0= 1;
 			dir1= 3;
 			opposite_direction = DIRECTION::RIGHT;
 			break; 
 		}
 		case DIRECTION::UP: {
 			// C(v)  * Taux(v)  * Pt(v+(-1,0))        -> nC(v+(0,-1)) 
 			// T(v)  * site(v)  * P(v+(-1,0)) * Pt(v) -> nT(v+(0,-1))
 			// Ct(v) * Tauxt(v) * P(v)               -> nCt(v+(0,-1))
 			//
 			// Ct(v)-----0 0--P(v)Pt(v)--2 0--T(v)-----2 0--P(v+(-1,0))Pt(v+(-1,0))--2 2--C(v) 
 			// |                              |
 			// Tauxt(v)--0 0--         --2 0--site(v)--2 0--                       --2 2--Taux(v)
 			//
 			shift= Shift( 0,-1); 
 			p_shift= Shift(-1, 0);
 			dir0= 2;
 			dir1= 0;
 			opposite_direction = DIRECTION::DOWN;
 			break; 
 		}
 		case DIRECTION::RIGHT: { 
 			shift = Shift(-1, 0);
 			p_shift = Shift( 0,-1);
 			dir0= 3;
 			dir1= 1;
 			opposite_direction = DIRECTION::LEFT;
 			break; 
 		}
 		case DIRECTION::DOWN: { 
 			shift = Shift( 0, 1);
 			p_shift = Shift( 1, 0);
 			dir0= 0;
 			dir1= 2;
 			opposite_direction = DIRECTION::UP;
 			break; 
 		}
 		default: throw std::runtime_error("[move_singleDirection] Invalid direction");
 	}
 	
 	


    // map C T C according to selected direction	
 	// to Cu T Cv where u,v 1,2 or 2,3 or 3,4 or 4,1

	// copy current C,T,C for each X O(x^2) + O(x^2 auxBondDim^2)
	// depending on the direction

	// clockwise
	std::map< std::string, ITensor >  nC, nT, nCt;
	std::map< std::string, ITensor > * ptr_oldC;
	std::map< std::string, ITensor > * ptr_oldT;
	std::map< std::string, ITensor > * ptr_oldCt;
	std::map< std::string, ITensor > * ptr_Taux;
	std::map< std::string, ITensor > * ptr_Tauxt;
	switch (direction) {
		case DIRECTION::LEFT: { 
			ptr_oldC = &C_LU;
			ptr_Taux = &T_U;
			ptr_oldT = &T_L;
			ptr_oldCt = &C_LD;
			ptr_Tauxt = &T_D; 
			break; 
		}
		case DIRECTION::UP: { 
			ptr_oldC = &C_RU;
			ptr_Taux = &T_R;
			ptr_oldT = &T_U;
			ptr_oldCt = &C_LU;
			ptr_Tauxt = &T_L; 
			break; 
		}
		case DIRECTION::RIGHT: { 
			ptr_oldC = &C_RD;
			ptr_Taux = &T_D;
			ptr_oldT = &T_R;
			ptr_oldCt = &C_RU;
			ptr_Tauxt = &T_U;
			break; }
		case DIRECTION::DOWN: { 
			ptr_oldC = &C_LD;
			ptr_Taux = &T_L;
			ptr_oldT = &T_D;
			ptr_oldCt = &C_RD;
			ptr_Tauxt = &T_R;
			break; }
	}
	std::map< std::string, ITensor > & C  = *ptr_oldC;
	std::map< std::string, ITensor > & T  = *ptr_oldT;
	std::map< std::string, ITensor > & Ct = *ptr_oldCt;
	std::map< std::string, ITensor > const& Taux  = *ptr_Taux;
	std::map< std::string, ITensor > const& Tauxt = *ptr_Tauxt;


	// iterate over pairs (Vertex, Id) within elementary cell of cluster
	// Id identifies tensor belonging to Vertex
	t_iso_begin = std::chrono::high_resolution_clock::now();
	for (auto const& v_id : p_cluster->vToId) {
		Vertex const& v    = v_id.first;
		Vertex v_shifted   = v + shift;   // Shift of site
		Vertex v_p_shifted = v + p_shift; // Shift of projector
		std::string id           = v_id.second;
		std::string id_shift     = vToId(v_shifted);
		std::string id_p_shifted = vToId(v_p_shifted);

		// (dbg) std::cout<<"======================================================================"<<std::endl;
		// Print(P[v_pos]); Print(Pt[v_pos]);
		// Print(P[p_shifted_pos]); Print(Pt[p_shifted_pos]);
		// std::cout<< v-p_shift <<"="<< c.vertexToId(v-p_shift) << std::endl;
		// std::cout<< v <<"="<< id <<" --> "<< shifted <<"="<<c.vertexToId(shifted)<<std::endl;
		// std::cout<< p_shifted <<"="<< c.vertexToId(p_shifted) << std::endl;
		// Print(c.aux[v_pos]);

		nC[id_shift] = Taux.at(id) * C[id];
		readyToContract(nC[id_shift],direction,v,dir0,v_p_shifted,dir0);
		nC[id_shift] *= Pt[id];

		nT[id_shift] = T[id];
		readyToContract(nT[id_shift],direction,v,dir0,v,dir1);
		nT[id_shift] *= P[id];
		nT[id_shift] *= (sites.at(id) * dag(prime(sites.at(id),AUXLINK,BRAKET_OFFSET)));
		readyToContract(nT[id_shift],direction,v,dir1,v,dir0);
		nT[id_shift] *= Pt[id_p_shifted];
	
		nCt[id_shift] = Tauxt.at(id) * Ct[id];
		readyToContract(nCt[id_shift],direction,v,dir1,v_p_shifted,dir1);
		nCt[id_shift] *= P[id_p_shifted];

		// (dbg) Print(nC[shifted_pos]); Print(nT[shifted_pos]); Print(nCt[shifted_pos]);
	}
	t_iso_end = std::chrono::high_resolution_clock::now();
	accT[1] += get_mS(t_iso_begin, t_iso_end);

	// Post-process the indices of new environment tensors
	t_iso_begin = std::chrono::high_resolution_clock::now();
	
	

	for (auto const& v_id : p_cluster->vToId) {
		Vertex const& v    = v_id.first;
		Vertex v_shifted   = v + shift;   // Shift of site
		Vertex v_p_shifted = v + p_shift; // Shift of projector
		std::string id           = v_id.second;
		std::string id_shift     = vToId(v_shifted);
		std::string id_p_shifted = vToId(v_p_shifted);

		nC[id_shift] *= delta(
			tauxByVertex(dir0,v,opposite_direction),
			tauxByVertex(dir0,v_shifted,direction));
		nC[id_shift] *= delta(ipt.at(id), tauxByVertex(direction,v_shifted,dir0));
		

		nT[id_shift] *= p_cluster->DContract(id,opposite_direction,id_shift,direction);
		nT[id_shift] *= prime(p_cluster->DContract(id,opposite_direction,id_shift,direction),
			p_cluster->BRAKET_OFFSET);
		nT[id_shift] *= delta(ip.at(id), tauxByVertex(direction,v_shifted,dir0));
		nT[id_shift] *= delta(ipt.at(id_p_shifted), tauxByVertex(direction,v_shifted,dir1));


		nCt[id_shift] *= delta(
			tauxByVertex(dir1,v,opposite_direction),
			tauxByVertex(dir1,v_shifted,direction));
		nCt[id_shift] *= delta(ip.at(id_p_shifted), tauxByVertex(direction,v_shifted,dir1));
	}

	// 	std::cout<<"=DONE====================================================================="<<std::endl;
	// 	std::cout<< v-p_shift <<"="<< c.vertexToId(v-p_shift) << std::endl;
	// 	std::cout<< v <<"="<< id <<" --> "<< shifted <<"="<<c.vertexToId(shifted)
	// 		<<" "<< c.aux[vToPos(shifted)] << std::endl;
	// 	std::cout<< v+p_shift <<"="<< c.vertexToId(v+p_shift) << std::endl;
	
	// 	Print(nC[shifted_pos]); Print(nT[shifted_pos]); Print(nCt[shifted_pos]);
	// }

	// Normalize new corner tensors
	auto normalizeBLE_T = [](ITensor& t) {
		double m = 0.;
        auto max_m = [&m](double d)
        {
            if(std::abs(d) > m) m = std::abs(d);
        };

        t.visit(max_m);
        t *= 1.0/m;
	};
	for (auto const& id : p_cluster->siteIds ) { 
		normalizeBLE_T(nC[id]); 
		normalizeBLE_T(nT[id]);
		normalizeBLE_T(nCt[id]);
	}

	// Update environment tensors
	C = nC;
	T = nT;
	Ct= nCt;

	t_iso_end = std::chrono::high_resolution_clock::now();
	accT[3] += get_mS(t_iso_begin, t_iso_end);
}


// C---I_U,a2,a6
// |
// I_L, a3,a7
// I_L1,a1,a5    
// |
// Ct--I_D,a2,a6
//
// delta(I_L,I_L1) == delta(I[0],prime(I[0]))
// delta(a3,a1)    == delta(prime(ap,pl_pi),prime(apt,pl_pti))
// delta(a3+4,a1+4)== prime(delta(a3,a1))
//
// SVD
// I_U,a2,a6--U--S--V^+--I_D,a2,a6
//
// P
// I_L,a3,a7---C---I_U,a2,a6---U^+-----S^-1/2--Ip
//
// Pt
// I_L1,a1,a5--Ct--I_D,a2,a6--(V^+)^+--S^-1/2--Ipt
//
// P
// I_L,a1,a5<===I_L,a3,a7---P---Ip
//
// Pt
// I_L1,a3,a7<==I_L1,a1,a5--Pt--Ipt
//
// void CtmEnv::compute_IsometriesT3(unsigned int direction, Cluster const& c,
//         Index & ip, Index & ipt, Index & ia,
//         std::vector<ITensor> & P, std::vector<ITensor> & Pt,
//         std::vector<double> & accT) const
// {
// 	double const machine_eps = std::numeric_limits<double>::epsilon();
// 	int const BRAKET_OFFSET  = 4;
// 	using time_point = std::chrono::steady_clock::time_point;
// 	time_point t_iso_begin, t_iso_end;
// 	auto get_mS = [](time_point ti, time_point tf) { return std::chrono::duration_cast
//             <std::chrono::microseconds>(tf - ti).count()/1000.0; };

// 	auto argsSVDRRt = Args(
//         "Cutoff",-1.0,
//         "Maxm",x,
//         "SVDThreshold",1E-2,
//         "SVD_METHOD",SVD_METHOD,
//         "rsvd_power",rsvd_power,
//         "rsvd_reortho",rsvd_reortho,
//         "rsvd_oversampling",rsvd_oversampling
//     );

//     // Take the square-root of SV's
//     double loc_psdInvCutoff = isoPseudoInvCutoff;
    

// 	// TODO santize this
// 	// glue that maps from Vertex to position within C,T,Ct,Taux,Tauxt
// 	auto vToPos = [&c] (Vertex const& v)->int { return c.SI.at(c.vertexToId(v)); };
// 	auto vToId  = [&c] (Vertex const& v)->std::string { return c.vertexToId(v); };

// 	// Corners to be used in construction of projectors
// 	Shift shift;
// 	int corner_i, corner_it;
//     switch (direction) {								// pl_oi--T--pl_pi--C--pl_pl_pti--T--
// 		case 0: { corner_i = 1; corner_it = 4; shift = Shift( 0, 1); break; } // 2--U--3--L--1--D
// 		case 1: { corner_i = 2; corner_it = 1; shift = Shift(-1, 0); break; } // R--U--L  
// 		case 2: { corner_i = 3; corner_it = 2; shift = Shift( 0,-1); break; } // D--R--U 
// 		case 3: { corner_i = 4; corner_it = 3; shift = Shift( 1, 0); break; } // L--D--R
// 	}

// 	int pl_pi, pl_pti, pl_oi;
// 	std::vector<Index> I(4);
// 	switch (direction) {
// 		// I_U,2,6--R---I_L,3,7
// 		// I_L,1,5--Rt--I_D,2,6
// 		// I_U,2,6--R--I_L,3,7--dai(pl_pi,pl_pti)&&(I[0],prime(I[0]))--I_L,1,5--Rt--I_D,2,6
// 		case 0:	{ I[0] = I_L; I[1] = I_XH; I[2] = I_U; I[3] = I_D; 
// 			pl_oi = 2; pl_pi = 3; pl_pti = 1; break; }
// 		case 1: { I[0] = I_U; I[1] = I_XV; I[2] = I_R; I[3] = I_L;
// 			pl_oi = 3; pl_pi = 0; pl_pti = 2; break; }
// 		case 2: { I[0] = I_R; I[1] = prime(I_XH); I[2] = prime(I_D); I[3] = prime(I_U); 
// 			pl_oi = 0; pl_pi = 1; pl_pti = 3; break; }
// 		case 3: { I[0] = I_D; I[1] = prime(I_XV); I[2] = prime(I_L); I[3] = prime(I_R);
// 			pl_oi = 1; pl_pi = 2; pl_pti = 0; break; } 
// 	}

// 	// Prepare vectors P, Pt to hold projectors
// 	P  = std::vector<ITensor>( c.sites.size() );
// 	Pt = std::vector<ITensor>( c.sites.size() );
// 	ia  = Index("AUX",c.auxBondDim,AUXLINK);
// 	ip  = Index("AUXP", x);
// 	ipt = Index("AUXPt",x);

//     for(auto const& v_id : c.vToId) {
//     	Vertex const& v = v_id.first;
//     	auto ap  = c.aux[vToPos(v)];
//     	auto apt = c.aux[vToPos(v+shift)];
//     	std::string id  = v_id.second; // == vToId(v)
//     	std::string idt = vToId(v+shift);
//     	auto da = delta(prime(ap,pl_pi), prime(apt,pl_pti));

//     	// Compute enlarged corners
//     	t_iso_begin = std::chrono::steady_clock::now();
//     	ITensor U, S, V;
//     	auto R  = build_corner_V2(corner_i,  c, v);
//     	auto Rt = build_corner_V2(corner_it, c, v+shift);
//   		R *= da;
//   		R *= prime(da,BRAKET_OFFSET);
//     	R *= delta(I[0],prime(I[0]));
//     	t_iso_end = std::chrono::steady_clock::now();
//     	accT[4] += get_mS(t_iso_begin,t_iso_end);

//     	// truncated SVD
//     	t_iso_begin = std::chrono::steady_clock::now();
//     	U = ITensor(I[2], prime(ap,pl_oi), prime(ap,pl_oi+BRAKET_OFFSET));
// 	    svd( R * Rt, U, S, V, solver, argsSVDRRt);
// 	    if( S.real(S.inds().front()(1),S.inds().back()(1)) > isoMaxElemWarning ||
// 	        S.real(S.inds().front()(1),S.inds().back()(1)) < isoMinElemWarning ) {
// 	        std::cout << "WARNING: CTM-Iso3 " << direction << " [col:row]= "<< v 
// 	    		<<" Max Sing. val.: "<< S.real(S.inds().front()(1),S.inds().back()(1))
// 	            << std::endl;
// 	    }
// 	    t_iso_end = std::chrono::steady_clock::now();
//     	accT[6] += get_mS(t_iso_begin,t_iso_end);
// 	    // (dbg) Print(U); Print(V);


// 	    // Create pseudo-inverse matrix and compute projectors
//     	t_iso_begin = std::chrono::steady_clock::now();
//     	auto sIU = commonIndex(U,S);
//     	auto sIV = commonIndex(S,V);
//     	int rank = std::max(sIU.m(),sIV.m());
//     	double max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
//     	double est_tol = std::sqrt(max_sv * rank * machine_eps);
//     	double arg_tol = std::sqrt(max_sv) * loc_psdInvCutoff;
//     	// if ( (not default_pinv_cutoff) && (est_tol > arg_tol) )  std::cout<<
//     	// 	"[compute_IsometriesT4] WARNING: est_tol > loc_psdInvCutoff*max_sv"<< std::endl;
//     	double const tol = (default_pinv_cutoff) ? est_tol : arg_tol;
//     	auto oneOverSqrtT = [&tol](Real r) 
//         	{ return (r > tol) ? 1.0/sqrt(r) : 0.0; };
//     	S.apply(oneOverSqrtT);

//     	// Inner indices are back to original state
//     	R *= da;
//   		R *= prime(da,BRAKET_OFFSET);
//   		R *= delta(I[0],prime(I[0]));

// 	    P[ vToPos(v)] = (R* U.dag())*S*delta(sIV, ip );
// 	   	Pt[vToPos(v)] = (Rt*V.dag())*S*delta(sIU, ipt);
	
// 	 	// Post-process indices of projectors
// 	 	auto dR  = delta(prime(ap,pl_pi  ),prime(ia,pl_pi ));
// 	 	auto dRt = delta(prime(apt,pl_pti),prime(ia,pl_pti));
// 	   	P[ vToPos(v)] *= dR;  P [vToPos(v)] *= prime(dR,BRAKET_OFFSET);
// 	   	Pt[vToPos(v)] *= dRt; Pt[vToPos(v)] *= prime(dRt,BRAKET_OFFSET);

// 	   	t_iso_end = std::chrono::steady_clock::now();
//     	accT[7] += get_mS(t_iso_begin,t_iso_end);
// 	   	// (dbg) Print(P[vToPos(v)]); Print(Pt[vToPos(v)]);
// 	}
// }

void CtmEnv::compute_IsometriesT4(DIRECTION direction, 
        std::map<std::string, Index> & ip,  std::map<std::string, Index> & ipt,
        std::map<std::string, ITensor> & P, std::map<std::string, ITensor> & Pt,
        std::vector<double> & accT) const
{
	using time_point = std::chrono::steady_clock::time_point;

	double const machine_eps = std::numeric_limits<double>::epsilon();

	auto get_mS = [](time_point ti, time_point tf) { return std::chrono::duration_cast
            <std::chrono::microseconds>(tf - ti).count()/1000.0; };

    auto vToId  = [this] (Vertex const& v) { return p_cluster->vertexToId(v); };

    auto readyToContract = [this](ITensor & t, CtmEnv::DIRECTION direction, 
    	Vertex const& v0, int dir0, Vertex const& v1, int dir1) {
    	// relabel env auxiliary indices
    	t *= delta(tauxByVertex(direction,v0,dir0), tauxByVertex(direction,v1,dir1));
    	// relabel site auxiliary indices
    	t *= p_cluster->DContract(v0,dir0,v1,dir1);
    	t *= prime(p_cluster->DContract(v0,dir0,v1,dir1), p_cluster->BRAKET_OFFSET);
    };

    auto edgeIndices = [this](CtmEnv::DIRECTION direction, Vertex const& v, int dir) {
    	std::vector<Index> tmp = p_cluster->AIBraKetPair(v,dir);
    	tmp.emplace_back(tauxByVertex(direction,v,dir));
    	return tmp;
    };

	auto argsSVDRRt = Args(
        "Cutoff",-1.0,
        "Maxm",x,
        "SVDThreshold",1E-2,
        "SVD_METHOD",SVD_METHOD,
        "rsvd_power",rsvd_power,
        "rsvd_reortho",rsvd_reortho,
        "rsvd_oversampling",rsvd_oversampling,
        "dbg",false
    );

    // Take the square-root of SV's
    double loc_psdInvCutoff = isoPseudoInvCutoff;
    
	// Corners to be used in construction of projectors
	DIRECTION opposite_direction;
	Shift shift, shift_oi;
    int dir0, dir1;
    switch (direction) {
		case DIRECTION::LEFT: { 
			// P   v--------v+(1,0) 
			//     3        3 <-- indices of U
			//     1        1 
			// Pt  v+(0,1)--v+(1,1)
			opposite_direction = DIRECTION::RIGHT;
			shift    = Shift( 0, 1);
			shift_oi = Shift( 1, 0);
			dir0 = 3;
			dir1 = 1;
			break; 
		}
		case DIRECTION::UP: { 
			// Pt              P
			// v+(-1,0)--2 0---v
			// |               |
            // v+(-1,1)--2 0---v+(0,1)
			//             ^--indices of U
			opposite_direction = DIRECTION::DOWN;
			shift    = Shift(-1, 0);
			shift_oi = Shift( 0, 1); 
			dir0 = 0;
			dir1 = 2;
			break; 
		}  
		case DIRECTION::RIGHT: {
			//                  v+(-1,-1)--v+(0,-1) Pt
			//                  3          3
			// indices of U --> 1          1
			//                  v+(-1,0)---v        P
			opposite_direction = DIRECTION::LEFT;
			shift    = Shift( 0,-1);
			shift_oi = Shift(-1, 0);
			dir0 = 1;
			dir1 = 3;
			break; 
		} 
		case DIRECTION::DOWN: {
		    // indices of U --V
			//      v+(0,-1)--2 0--v+(1,-1)
			//      |              |
            //      v---------2 0--v+(1,0)
            //      P              Pt
            opposite_direction = DIRECTION::UP;
			shift = Shift( 1, 0); 
			shift_oi = Shift( 0,-1);
			dir0 = 2;
			dir1 = 0; 
			break; 
		}
	}

	time_point t_iso_begin, t_iso_end;
    for(auto const& v_id : p_cluster->vToId) {
    	std::string id = v_id.second;   
    	Vertex const& v = v_id.first;
    	Vertex v_shift  = v + shift;
    	Vertex v_shift_oi = v + shift_oi;

    	// Compute two halfs of 2x2 density matrix
    	t_iso_begin = std::chrono::steady_clock::now();
    	ITensor U, S, V, R, Rt;
    	build_halves_V2(direction, v, R, Rt);
    	t_iso_end = std::chrono::steady_clock::now();
    	accT[4] += get_mS(t_iso_begin,t_iso_end);


    	// truncated SVD
    	t_iso_begin = std::chrono::steady_clock::now();
    	U = ITensor(edgeIndices(opposite_direction,v_shift_oi,dir0));
    	readyToContract(R,direction,v,dir0,v_shift,dir1);
	    svd( R * Rt, U, S, V, solver, argsSVDRRt);
	    if( S.real(S.inds().front()(1),S.inds().back()(1)) > isoMaxElemWarning ||
	        S.real(S.inds().front()(1),S.inds().back()(1)) < isoMinElemWarning ) {
	        std::cout << "WARNING: CTM-Iso4 " << direction << " [col:row]= "<< v 
	    		<<" Max Sing. val.: "<< S.real(S.inds().front()(1),S.inds().back()(1))
	            << std::endl;
	    }
	    t_iso_end = std::chrono::steady_clock::now();
    	accT[6] += get_mS(t_iso_begin,t_iso_end);


	    // Create pseudo-inverse matrix and compute projectors
    	t_iso_begin = std::chrono::steady_clock::now();
    	auto sIU = commonIndex(U,S);
    	auto sIV = commonIndex(S,V);
    	int rank = std::max(sIU.m(),sIV.m());
    	double max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
    	double est_tol = std::sqrt(max_sv * rank * machine_eps);
    	double arg_tol = std::sqrt(max_sv) * loc_psdInvCutoff;
    	// TODO expose debug setting here
    	// if ( dbg && (not default_pinv_cutoff) && (est_tol > arg_tol) )  std::cout<<
    		// "[compute_IsometriesT4] WARNING: est_tol > loc_psdInvCutoff*max_sv"<< std::endl;
    	double const tol = (default_pinv_cutoff) ? est_tol : arg_tol;
    	auto oneOverSqrtT = [&tol](Real r) 
        	{ return (r > tol) ? 1.0/sqrt(r) : 0.0; };
    	S.apply(oneOverSqrtT);

    	// set innner indices back to original state
    	readyToContract(R,direction,v,dir0,v_shift,dir1);

    	ip[ id] = Index("P_"+id,tauxByVertex(direction,v,dir0).m());
    	ipt[id] = Index("Pt_"+id,tauxByVertex(direction,v,dir1).m());
	    P[ id] = (R* U.dag())*S*delta(sIV, ip[id] );
	   	Pt[id] = (Rt*V.dag())*S*delta(sIU, ipt[id]);

	   	t_iso_end = std::chrono::steady_clock::now();
    	accT[7] += get_mS(t_iso_begin,t_iso_end);
	}
}

ITensor CtmEnv::build_corner_V2(CORNER cornerType, Vertex const& v) const 
{
	auto BRAKET_OFFSET = p_cluster->BRAKET_OFFSET;
	auto sites = p_cluster->sites;

	std::string siteId = p_cluster->vertexToId(v);

	ITensor ct;	
    switch(cornerType) {
        case CORNER::LU: {
            // build left upper corner
            ct  = C_LU.at( siteId ) * T_L.at( siteId );
            ct *= T_U.at( siteId );
			ct *= (sites.at(siteId) * dag(prime(sites.at(siteId),AUXLINK,BRAKET_OFFSET)));
            break;
        }
        case CORNER::RU: {
            // build right upper corner
            ct  = C_RU.at( siteId ) * T_U.at( siteId );
            ct *= T_R.at( siteId );
            ct *= (sites.at(siteId) * dag(prime(sites.at(siteId),AUXLINK,BRAKET_OFFSET)));
            break;
        }
        case CORNER::RD: {
            // build right lower corner
            ct  = C_RD.at( siteId ) * T_R.at( siteId );
            ct *= T_D.at( siteId );
            ct *= (sites.at(siteId) * dag(prime(sites.at(siteId),AUXLINK,BRAKET_OFFSET)));
            break;
        }
        case CORNER::LD: {
            // build left lower corner
            ct  = C_LD.at( siteId ) *  T_D.at( siteId );
            ct *= T_L.at( siteId );
            ct *= sites.at(siteId) * dag(prime(sites.at(siteId),AUXLINK,BRAKET_OFFSET));
            break;
        }
    }
    return ct;
}

void CtmEnv::build_halves_V2(DIRECTION direction, Vertex const& v, 
	ITensor & H, ITensor & Ht) const 
{
    auto readyToContract = [this](ITensor & t, DIRECTION direction, 
    	Vertex const& v0, int dir0, Vertex const& v1, int dir1) {
    	// relabel env auxiliary indices
    	t *= delta(tauxByVertex(direction,v0,dir0), tauxByVertex(direction,v1,dir1));
    	// relabel site auxiliary indices
    	t *= p_cluster->DContract(v0,dir0,v1,dir1);
    	t *= prime(p_cluster->DContract(v0,dir0,v1,dir1), p_cluster->BRAKET_OFFSET);
    };

    switch (direction) {
        case DIRECTION::LEFT: { // left
        	std::vector<Vertex> vs({v, v+Shift(1,0), v+Shift(0,1), v+Shift(1,1) });
            
            // build upper half
            H =  build_corner_V2(CORNER::LU, vs[0]); // LU
            readyToContract(H,DIRECTION::UP,vs[0],2,vs[1],0);
            H *= build_corner_V2(CORNER::RU, vs[1]); // RU
            // build lower half
            Ht  = build_corner_V2(CORNER::LD, vs[2]); // LD
            readyToContract(Ht,DIRECTION::DOWN,vs[2],2,vs[3],0);
            Ht *= build_corner_V2(CORNER::RD, vs[3]); // RD
            break;
        }
        case DIRECTION::UP: { // up
        	std::vector<Vertex> vs({v, v+Shift(0,1), v+Shift(-1,0), v+Shift(-1,1) });

            // build right half
            H  = build_corner_V2(CORNER::RU, vs[0]); // RU
            readyToContract(H,DIRECTION::RIGHT,vs[0],3,vs[1],1);
            H *= build_corner_V2(CORNER::RD, vs[1]); // RD
            // build left half (upper_h)
            Ht  = build_corner_V2(CORNER::LU, vs[2]); // LU
			readyToContract(Ht,DIRECTION::LEFT,vs[2],3,vs[3],1);
            Ht *= build_corner_V2(CORNER::LD, vs[3]); // LD
            break;
        }
        case DIRECTION::RIGHT: { // right
        	std::vector<Vertex> vs({v, v+Shift(-1,0), v+Shift(0,-1), v+Shift(-1,-1) });

            // build lower half
            H  = build_corner_V2(CORNER::RD, vs[0]); // RD
            readyToContract(H,DIRECTION::DOWN,vs[0],0,vs[1],2);
            H *= build_corner_V2(CORNER::LD, vs[1]); // LD
            // build upper half
            Ht  = build_corner_V2(CORNER::RU, vs[2]); // RU
            readyToContract(Ht,DIRECTION::UP,vs[2],0,vs[3],2);
            Ht *= build_corner_V2(CORNER::LU, vs[3]); // LU
            break;
        }
        case DIRECTION::DOWN: { // down
        	std::vector<Vertex> vs({v, v+Shift(0,-1), v+Shift(1,0), v+Shift(1,-1) });

            // build left half
            H  = build_corner_V2(CORNER::LD, vs[0]); // LD
            readyToContract(H,DIRECTION::LEFT,vs[0],1,vs[1],3);
            H *= build_corner_V2(CORNER::LU, vs[1]); // LU
            // build right half
            Ht  = build_corner_V2(CORNER::RD, vs[2]); // RD
            readyToContract(Ht,DIRECTION::RIGHT,vs[2],1,vs[3],3);
            Ht *= build_corner_V2(CORNER::RU, vs[3]); // RU
            break;
        }
    }
}
