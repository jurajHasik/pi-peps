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

void CtmEnv::move_singleDirection(unsigned int direction, Cluster const& c,
	std::vector<double> & accT) 
{
	int const BRAKET_OFFSET = 4;
	using time_point = std::chrono::high_resolution_clock::time_point;
	time_point t_iso_begin, t_iso_end;
	auto get_mS = [](time_point ti, time_point tf) { return std::chrono::duration_cast
            <std::chrono::microseconds>(tf - ti).count()/1000.0; };

	Shift shift, p_shift;
 	switch (direction) {
 		case 0: { shift = Shift( 1, 0); p_shift = Shift( 0, 1); break; } // left
 		case 1: { shift = Shift( 0,-1); p_shift = Shift(-1, 0); break; } // up
 		case 2:	{ shift = Shift(-1, 0); p_shift = Shift( 0,-1); break; } // right
 		case 3: { shift = Shift( 0, 1); p_shift = Shift( 1, 0); break; } // down
 		default: throw std::runtime_error("[move_singleDirection] Invalid direction");
 	}

 	// map C T C according to selected direction	
 	// to Cu T Cv where u,v 1,2 or 2,3 or 3,4 or 4,1

 	t_iso_begin = std::chrono::high_resolution_clock::now();
 	Index ip, ipt, ia;
 	std::vector<ITensor> P, Pt;
	computeIsometries(direction, c, ip, ipt, ia, P, Pt, accT);
	t_iso_end = std::chrono::high_resolution_clock::now();
    accT[0] += get_mS(t_iso_begin, t_iso_end);


	// copy current C,T,C for each X O(x^2) + O(x^2 auxBondDim^2)
	// depending on the direction

	// clockwise
	std::vector<Index> I(10);
	std::vector<ITensor>  nC( c.vToId.size() );
	std::vector<ITensor>  nT( c.vToId.size() );
	std::vector<ITensor> nCt( c.vToId.size() );
	std::vector<ITensor> * ptr_oldC;
	std::vector<ITensor> * ptr_oldT;
	std::vector<ITensor> * ptr_oldCt;
	std::vector<ITensor> * ptr_Taux;
	std::vector<ITensor> * ptr_Tauxt;
	int p_T, p_Tt;
	switch (direction) {
		case 0:	{ 
			ptr_oldC = &C_LU; ptr_Taux = &T_U; ptr_oldT = &T_L; ptr_oldCt = &C_LD; ptr_Tauxt = &T_D; 
			I[0] = I_L; I[1] = I_XH; I[2] = I_XV; I[3] = prime(I_XV);
			I[4] = prime(I_U); I[5] = I_U; I[6] = prime(I_D); I[7] = I_D; I[8] = I_L; I[9] = prime(I_L);
			p_T = 3; p_Tt = 1; break; }
		case 1: { 
			ptr_oldC = &C_RU; ptr_Taux = &T_R; ptr_oldT = &T_U; ptr_oldCt = &C_LU; ptr_Tauxt = &T_L; 
			I[0] = I_U; I[1] = I_XV; I[2] = prime(I_XH); I[3] = I_XH;
			I[4] = prime(I_R); I[5] = I_R; I[6] = prime(I_L); I[7] = I_L; I[8] = prime(I_U); I[9] = I_U;
			p_T = 0; p_Tt = 2; break; }
		case 2: { 
			ptr_oldC = &C_RD; ptr_Taux = &T_D; ptr_oldT = &T_R; ptr_oldCt = &C_RU; ptr_Tauxt = &T_U; 
			I[0] = I_R; I[1] = prime(I_XH); I[2] = prime(I_XV); I[3] = I_XV; 
			I[4] = I_D; I[5] = prime(I_D); I[6] = I_U; I[7] = prime(I_U); I[8] = prime(I_R); I[9] = I_R;
			p_T = 1; p_Tt = 3; break; }
		case 3: { 
			ptr_oldC = &C_LD; ptr_Taux = &T_L; ptr_oldT = &T_D; ptr_oldCt = &C_RD; ptr_Tauxt = &T_R; 
			I[0] = I_D; I[1] = prime(I_XV); I[2] = I_XH; I[3] = prime(I_XH);
			I[4] = I_L; I[5] = prime(I_L); I[6] = I_R; I[7] = prime(I_R); I[8] = I_D; I[9] = prime(I_D);
			p_T = 2; p_Tt = 0; break; }
	}
	std::vector<ITensor> & C 	= *ptr_oldC;
	std::vector<ITensor> & T 	= *ptr_oldT;
	std::vector<ITensor> & Ct 	= *ptr_oldCt;
	std::vector<ITensor> const& Taux 	= *ptr_Taux;
	std::vector<ITensor> const& Tauxt 	= *ptr_Tauxt;

	// TODO santize this
	// glue that maps from Vertex to position within C,T,Ct,Taux,Tauxt
	auto vToPos = [&c] (Vertex const& v)->int { return c.SI.at(c.vertexToId(v)); };
	// glue that maps from Vertex to aux index of sites
	auto vToAux = [&c] (Vertex const& v)->Index { return c.aux[c.SI.at(c.vertexToId(v))]; };


	// iterate over pairs (Vertex, Id) within elementary cell of cluster
	// Id identifies tensor belonging to Vertex
	t_iso_begin = std::chrono::high_resolution_clock::now();
	for (auto const& v_id : c.vToId) {
		Vertex const& v = v_id.first;
		std::string id  = v_id.second;
		Vertex shifted   = v + shift;   // Shift of site
		Vertex p_shifted = v + p_shift; // Shift of projector
		auto av  = c.aux[vToPos(v)];
    	auto apt = c.aux[vToPos(v+p_shift)];
		int v_pos         = vToPos(v);
		int p_shifted_pos = vToPos(p_shifted);
		int shifted_pos   = vToPos(shifted);

		// (dbg) std::cout<<"======================================================================"<<std::endl;
		// Print(P[v_pos]); Print(Pt[v_pos]);
		// Print(P[p_shifted_pos]); Print(Pt[p_shifted_pos]);
		// std::cout<< v-p_shift <<"="<< c.vertexToId(v-p_shift) << std::endl;
		// std::cout<< v <<"="<< id <<" --> "<< shifted <<"="<<c.vertexToId(shifted)<<std::endl;
		// std::cout<< p_shifted <<"="<< c.vertexToId(p_shifted) << std::endl;
		// Print(c.aux[v_pos]);

		// glue to unfuse auxiliary indices on T, Taux, Tauxt
		auto avia = delta(av,ia);
		auto link = delta(I[8],I[9]);

		nC[shifted_pos]  = Taux[v_pos];
		nC[shifted_pos] *= delta(I[2],combinedIndex(CMB[id][p_Tt]));
		nC[shifted_pos] *= CMB.at(id)[p_Tt];
		nC[shifted_pos] *= prime(avia,p_Tt);
		nC[shifted_pos] *= prime(avia,p_Tt+BRAKET_OFFSET);
		nC[shifted_pos] *= C[v_pos];
		nC[shifted_pos] *= link; // delta(I[0],prime(I[0]));
		nC[shifted_pos] *= Pt[p_shifted_pos];

		nT[shifted_pos]  = T[v_pos];
		nT[shifted_pos] *= delta(I[1],combinedIndex(CMB[id][direction]));
		nT[shifted_pos] *= CMB.at(id)[direction];
		nT[shifted_pos] *= P[p_shifted_pos];
		nT[shifted_pos] *= delta(prime(av,p_Tt),prime(ia,p_T));
		nT[shifted_pos] *= prime(delta(prime(av,p_Tt),prime(ia,p_T)),BRAKET_OFFSET);
		nT[shifted_pos] *= c.getSiteRefc(v);
		nT[shifted_pos] *= dag(prime(c.getSite(v),AUXLINK,BRAKET_OFFSET));
		nT[shifted_pos] *= delta(prime(av,p_T),prime(ia,p_Tt));
		nT[shifted_pos] *= prime(delta(prime(av,p_T),prime(ia,p_Tt)),BRAKET_OFFSET);
		nT[shifted_pos] *= Pt[v_pos];
	
		nCt[shifted_pos]  = Tauxt[v_pos];
		
		nCt[shifted_pos] *= delta(I[3],combinedIndex(CMB[id][p_T]));
		nCt[shifted_pos] *= CMB.at(id)[p_T];
		nCt[shifted_pos] *= prime(avia,p_T);
		nCt[shifted_pos] *= prime(avia,p_T+BRAKET_OFFSET);
		nCt[shifted_pos] *= Ct[v_pos];
		nCt[shifted_pos] *= link; // delta(I[0],prime(I[0]));
		nCt[shifted_pos] *= P[v_pos];

		// (dbg) Print(nC[shifted_pos]); Print(nT[shifted_pos]); Print(nCt[shifted_pos]);
	}
	t_iso_end = std::chrono::high_resolution_clock::now();
	accT[1] += get_mS(t_iso_begin, t_iso_end);

	// Post-process the indices of new environment tensors
	t_iso_begin = std::chrono::high_resolution_clock::now();
	
	for (auto & t : nC ) { t *= delta(I[4], I[5]); t *= delta(ipt, I[8]); }
	for (auto & t : nCt) { t *= delta(I[6], I[7]); t *= delta(ip,  I[9]); }
	for (auto & t : nT ) { t *= delta(ip, I[8]); t *= delta(ipt, I[9]); }
	
	// prepare combiner from aux indices to I_XH or I_XV with appropriate 
	// primelevel
	int pl;
	switch (direction) {
		case 0: { pl = 2; break; }
		case 1: { pl = 3; break; }  
		case 2: { pl = 0; break; }  
		case 3: { pl = 1; break; } 
	}
	for (auto const& v_id : c.vToId) { 
		Vertex const& v = v_id.first;
		std::string id  = v_id.second;
		Vertex shifted = v + shift;
		int v_pos = vToPos(v);
		int shifted_pos = vToPos(shifted);

		nT[shifted_pos] *= CMB.at(id)[pl];
		nT[shifted_pos] *= delta(combinedIndex(CMB.at(id)[pl]), I[1]);
	}

	// (dbg) for (auto const& v_id : c.vToId) { 
	// 	Vertex const& v = v_id.first;
	// 	std::string id  = v_id.second;
	// 	Vertex shifted   = v + shift;   // Shift of site
	// 	Vertex p_shifted = v + p_shift; // Shift of projector
	// 	int shifted_pos   = vToPos(shifted);

	// 	std::cout<<"=DONE====================================================================="<<std::endl;
	// 	std::cout<< v-p_shift <<"="<< c.vertexToId(v-p_shift) << std::endl;
	// 	std::cout<< v <<"="<< id <<" --> "<< shifted <<"="<<c.vertexToId(shifted)
	// 		<<" "<< c.aux[vToPos(shifted)] << std::endl;
	// 	std::cout<< v+p_shift <<"="<< c.vertexToId(v+p_shift) << std::endl;
	
	// 	Print(nC[shifted_pos]); Print(nT[shifted_pos]); Print(nCt[shifted_pos]);
	// }

	auto normalizeBLE_T = [](ITensor& t) {
		double m = 0.;
        auto max_m = [&m](double d)
        {
            if(std::abs(d) > m) m = std::abs(d);
        };

        t.visit(max_m);
        t *= 1.0/m;
	};
	for (auto & t : nC ) { normalizeBLE_T(t); }
	for (auto & t : nCt) { normalizeBLE_T(t); }
	for (auto & t : nT ) { normalizeBLE_T(t); }


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
void CtmEnv::computeIsometries(unsigned int direction, Cluster const& c,
        Index & ip, Index & ipt, Index & ia,
        std::vector<ITensor> & P, std::vector<ITensor> & Pt,
        std::vector<double> & accT) const
{
	int const BRAKET_OFFSET  = 4;
	using time_point = std::chrono::steady_clock::time_point;
	time_point t_iso_begin, t_iso_end;
	auto get_mS = [](time_point ti, time_point tf) { return std::chrono::duration_cast
            <std::chrono::microseconds>(tf - ti).count()/1000.0; };

	auto argsSVDRRt = Args(
        "Cutoff",-1.0,
        "Maxm",x,
        "SVDThreshold",1E-2,
        "SVD_METHOD",SVD_METHOD,
        "rsvd_power",rsvd_power,
        "rsvd_reortho",rsvd_reortho,
        "rsvd_oversampling",rsvd_oversampling
    );

    // Take the square-root of SV's
    double loc_psdInvCutoff = isoPseudoInvCutoff;
    

	// TODO santize this
	// glue that maps from Vertex to position within C,T,Ct,Taux,Tauxt
	auto vToPos = [&c] (Vertex const& v)->int { return c.SI.at(c.vertexToId(v)); };
	auto vToId  = [&c] (Vertex const& v)->std::string { return c.vertexToId(v); };

	// Corners to be used in construction of projectors
	Shift shift;
	int corner_i, corner_it;
    switch (direction) {								// pl_oi--T--pl_pi--C--pl_pl_pti--T--
		case 0: { corner_i = 1; corner_it = 4; shift = Shift( 0, 1); break; } // 2--U--3--L--1--D
		case 1: { corner_i = 2; corner_it = 1; shift = Shift(-1, 0); break; } // R--U--L  
		case 2: { corner_i = 3; corner_it = 2; shift = Shift( 0,-1); break; } // D--R--U 
		case 3: { corner_i = 4; corner_it = 3; shift = Shift( 1, 0); break; } // L--D--R
	}

	int pl_pi, pl_pti, pl_oi;
	std::vector<Index> I(4);
	switch (direction) {
		// I_U,2,6--R---I_L,3,7
		// I_L,1,5--Rt--I_D,2,6
		// I_U,2,6--R--I_L,3,7--dai(pl_pi,pl_pti)&&(I[0],prime(I[0]))--I_L,1,5--Rt--I_D,2,6
		case 0:	{ I[0] = I_L; I[1] = I_XH; I[2] = I_U; I[3] = I_D; 
			pl_oi = 2; pl_pi = 3; pl_pti = 1; break; }
		case 1: { I[0] = I_U; I[1] = I_XV; I[2] = I_R; I[3] = I_L;
			pl_oi = 3; pl_pi = 0; pl_pti = 2; break; }
		case 2: { I[0] = I_R; I[1] = prime(I_XH); I[2] = prime(I_D); I[3] = prime(I_U); 
			pl_oi = 0; pl_pi = 1; pl_pti = 3; break; }
		case 3: { I[0] = I_D; I[1] = prime(I_XV); I[2] = prime(I_L); I[3] = prime(I_R);
			pl_oi = 1; pl_pi = 2; pl_pti = 0; break; } 
	}

	// Prepare vectors P, Pt to hold projectors
	P  = std::vector<ITensor>( c.sites.size() );
	Pt = std::vector<ITensor>( c.sites.size() );
	ia  = Index("AUX",c.auxBondDim,AUXLINK);
	ip  = Index("AUXP", x);
	ipt = Index("AUXPt",x);

    for(auto const& v_id : c.vToId) {
    	Vertex const& v = v_id.first;
    	auto ap  = c.aux[vToPos(v)];
    	auto apt = c.aux[vToPos(v+shift)];
    	std::string id  = v_id.second; // == vToId(v)
    	std::string idt = vToId(v+shift);
    	auto da = delta(prime(ap,pl_pi), prime(apt,pl_pti));

    	// Compute enlarged corners
    	t_iso_begin = std::chrono::steady_clock::now();
    	ITensor U, S, V;
    	auto R  = build_corner_V2(corner_i,  c, v);
    	auto Rt = build_corner_V2(corner_it, c, v+shift);
  		R *= da;
  		R *= prime(da,BRAKET_OFFSET);
    	R *= delta(I[0],prime(I[0]));
    	t_iso_end = std::chrono::steady_clock::now();
    	accT[4] += get_mS(t_iso_begin,t_iso_end);

    	// truncated SVD
    	t_iso_begin = std::chrono::steady_clock::now();
    	U = ITensor(I[2], prime(ap,pl_oi), prime(ap,pl_oi+BRAKET_OFFSET));
	    svd( R * Rt, U, S, V, solver, argsSVDRRt);
	    if( S.real(S.inds().front()(1),S.inds().back()(1)) > isoMaxElemWarning ||
	        S.real(S.inds().front()(1),S.inds().back()(1)) < isoMinElemWarning ) {
	        std::cout << "WARNING: CTM-Iso3 " << direction << " [col:row]= "<< v 
	    		<<" Max Sing. val.: "<< S.real(S.inds().front()(1),S.inds().back()(1))
	            << std::endl;
	    }
	    t_iso_end = std::chrono::steady_clock::now();
    	accT[6] += get_mS(t_iso_begin,t_iso_end);
	    // (dbg) Print(U); Print(V);


	    // Create pseudo-inverse matrix and compute projectors
    	t_iso_begin = std::chrono::steady_clock::now();
    	auto sIU = commonIndex(U,S);
    	auto sIV = commonIndex(S,V);
    	double max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
    	auto oneOverSqrtT = [&max_sv, &loc_psdInvCutoff](Real r) 
        	{ return (r/max_sv > loc_psdInvCutoff) ? 1.0/sqrt(r) : 0.0; };
    	S.apply(oneOverSqrtT);

    	// Inner indices are back to original state
    	R *= da;
  		R *= prime(da,BRAKET_OFFSET);
  		R *= delta(I[0],prime(I[0]));

	    P[ vToPos(v)] = (R* U.dag())*S*delta(sIV, ip );
	   	Pt[vToPos(v)] = (Rt*V.dag())*S*delta(sIU, ipt);
	
	 	// Post-process indices of projectors
	 	auto dR  = delta(prime(ap,pl_pi  ),prime(ia,pl_pi ));
	 	auto dRt = delta(prime(apt,pl_pti),prime(ia,pl_pti));
	   	P[ vToPos(v)] *= dR;  P [vToPos(v)] *= prime(dR,BRAKET_OFFSET);
	   	Pt[vToPos(v)] *= dRt; Pt[vToPos(v)] *= prime(dRt,BRAKET_OFFSET);

	   	t_iso_end = std::chrono::steady_clock::now();
    	accT[7] += get_mS(t_iso_begin,t_iso_end);
	   	// (dbg) Print(P[vToPos(v)]); Print(Pt[vToPos(v)]);
	}
}

ITensor CtmEnv::build_corner_V2(unsigned int direction, Cluster const& c, 
	Vertex const& v) const 
{
	int const BRAKET_OFFSET = 4;
    ITensor ct;

    // TODO santize this
	// glue that maps from Vertex to position within C,T,Ct,Taux,Tauxt
	auto vToPos = [&c] (Vertex const& v)->int { return c.SI.at(c.vertexToId(v)); };
	auto vToId  = [&c] (Vertex const& v)->std::string { return c.vertexToId(v); };
	int siteIndex      = vToPos(v);
	std::string siteId = vToId(v);
	
	auto unfuseI = [this, &v, &siteId](int lat_dir, ITensor & t) { 
		t *= delta(fusedSiteI[lat_dir], combinedIndex(CMB.at(siteId)[lat_dir]));
		t *= CMB.at(siteId)[lat_dir]; };

    
    switch(direction) {
        case 1: {
            // build left upper corner
            ct = T_L.at( siteIndex ); unfuseI(0,ct); //ct *= CMB.at(vToId(v))[0];
            ct *= C_LU.at( siteIndex );
            ct *= T_U.at( siteIndex ); unfuseI(1,ct); //ct *= CMB.at(vToId(v))[1];
            ct *= c.getSiteRefc(v); 
			ct *= dag(prime(c.getSite(v),AUXLINK,BRAKET_OFFSET));
            ct.prime(ULINK,-1); ct.prime(LLINK,-1);
            break;
        }
        case 2: {
            // build right upper corner
            ct = T_U.at( siteIndex ); unfuseI(1,ct); //ct *= CMB.at(vToId(v))[1];
            ct *= C_RU.at( siteIndex );
            //ct *= sites[siteIndex];
            ct *= T_R.at( siteIndex ); unfuseI(2,ct); //ct *= CMB.at(vToId(v))[2];
            ct *= c.getSiteRefc(v); 
			ct *= dag(prime(c.getSite(v),AUXLINK,BRAKET_OFFSET));
            ct.prime(RLINK,-1); ct.prime(ULINK,1);
            break;
        }
        case 3: {
            // build right lower corner
            ct = T_R.at( siteIndex ); unfuseI(2,ct); //ct *= CMB.at(vToId(v))[2];
            ct *= C_RD.at( siteIndex );
            //ct *= sites[siteIndex];
            ct *= T_D.at( siteIndex ); unfuseI(3,ct); //ct *= CMB.at(vToId(v))[3];
            ct *= c.getSiteRefc(v); 
			ct *= dag(prime(c.getSite(v),AUXLINK,BRAKET_OFFSET));
            ct.prime(DLINK,1); ct.prime(RLINK,1);
            break;
        }
        case 4: {
            // build left lower corner
            ct = T_D.at( siteIndex ); unfuseI(3,ct); //ct *= CMB.at(vToId(v))[3];
            ct *= C_LD.at( siteIndex ); 
            //ct *= sites[siteIndex];
            ct *= T_L.at( siteIndex ); unfuseI(0,ct); //ct *= CMB.at(vToId(v))[0];
            ct *= c.getSiteRefc(v); 
			ct *= dag(prime(c.getSite(v),AUXLINK,BRAKET_OFFSET));
            ct.prime(LLINK,1); ct.prime(DLINK,-1);
            break;
        }
    }
    return ct;
}