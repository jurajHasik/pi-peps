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
	int const BRAKET_OFFSET = 4;

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

 	Index ip, ipt;
 	std::vector<ITensor> P, Pt;
	computeIsometries(direction, c, ip, ipt, P, Pt);

	// copy current C,T,C for each X O(x^2) + O(x^2 D^2)
	// depending on the direction

	// clockwise
	std::vector<Index> I(4);
	std::vector<ITensor> C;
	std::vector<ITensor> T;
	std::vector<ITensor> Ct;
	std::vector<ITensor> * ptr_Taux;
	std::vector<ITensor> * ptr_Tauxt;
	switch (direction) {
		case 1:	{ C = C_LU; ptr_Taux = &T_U; T = T_L; Ct = C_LD; ptr_Tauxt = &T_D; 
			I[0] = I_L; I[1] = I_XH; I[2] = prime(I_U); I[3] = prime(I_D); break; }
		case 2: { C = C_RU; ptr_Taux = &T_R; T = T_U; Ct = C_LU; ptr_Tauxt = &T_L; 
			I[0] = I_U; I[1] = I_XV; I[2] = prime(I_R); I[3] = prime(I_L); break; }  
		case 3: { C = C_RD; ptr_Taux = &T_D; T = T_R; Ct = C_RU; ptr_Tauxt = &T_U; 
			I[0] = I_R; I[1] = prime(I_XH); I[2] = prime(I_D); I[3] = prime(I_U); break; }  
		case 4: { C = C_LD; ptr_Taux = &T_L; T = T_D; Ct = C_RD; ptr_Tauxt = &T_R; 
			I[0] = I_D; I[1] = prime(I_XV); I[2] = prime(I_L); I[3] = prime(I_R); break; } 
	}
	std::vector<ITensor> const& Taux 	= *ptr_Taux;
	std::vector<ITensor> const& Tauxt 	= *ptr_Tauxt;


	// TODO santize this
	// glue that maps from Vertex to position within C,T,Ct,Taux,Tauxt
	auto vToPos = [&c] (Vertex const& v)->int { return c.SI.at(c.vertexToId(v)); };
	// glue that maps from Vertex to aux index of sites
	auto vToAux = [&c] (Vertex const& v)->Index { return c.phys[c.SI.at(c.vertexToId(v))]; };

	// iterate over pairs (Vertex, Id) within elementary cell of cluster
	// Id identifies tensor belonging to Vertex
	for (auto const& v_id : c.vToId) {
		Vertex const& v = v_id.first;
		std::string id  = v_id.second;
		Vertex shifted = v + shift;
		int v_pos = vToPos(v);
		int shifted_pos = vToPos(shifted);

		
		C[shifted_pos] 	= (C[v_pos] * Taux[v_pos]) * P[v_pos];

		T[shifted_pos] 	= (((P[v_pos] * T[v_pos]) * c.getSiteRefc(v)) 
			* dag(prime(c.getSite(v),BRAKET_OFFSET))) * Pt[v_pos];
	
		Ct[shifted_pos] = (Ct[v_pos] * Tauxt[v_pos]) * Pt[v_pos];
	}

	// Post-process the indices of new environment tensors
	for (auto & t : C ) { t.prime(I[2],-1); t *= delta(ip, I_L ); }
	for (auto & t : Ct) { t.prime(I[3],-1); t *= delta(ipt,prime(I_L)); }
	for (auto & t : T ) { t *= delta(ip, I_L ); t *= delta(ipt,prime(I_L)); }
	
	// prepare combiner from aux indices to I_XH or I_XV with appropriate 
	// primelevel
	int pl;
	switch (direction) {
		case 1: { pl = 2; break; }
		case 2: { pl = 3; break; }  
		case 3: { pl = 0; break; }  
		case 4: { pl = 1; break; } 
	}
	for (auto const& v_id : c.vToId) { 
		Vertex const& v = v_id.first;
		std::string id  = v_id.second;
		Vertex shifted = v + shift;
		int v_pos = vToPos(v);
		int shifted_pos = vToPos(shifted);

		T[shifted_pos] *= CMB.at(id)[pl];
	}
}

void CtmEnv::computeIsometries(unsigned int direction, Cluster const& c,
        Index & ip, Index & ipt, 
        std::vector<ITensor> & P, std::vector<ITensor> & Pt) 
{
	int const BRAKET_OFFSET  = 4;

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
	int corner_i, corner_it;
    switch (direction) {								// pl_oi--T--pl_pi--C--pl_pl_pti--T--
		case 1: { corner_i = 1; corner_it = 4; break; } // 2--U--3--L--1--D
		case 2: { corner_i = 2; corner_it = 1; break; } // R--U--L  
		case 3: { corner_i = 3; corner_it = 2; break; } // D--R--U 
		case 4: { corner_i = 4; corner_it = 3; break; } // L--D--R
	}

	int pl_pi, pl_pti, pl_oi;
	std::vector<Index> I(4);
	switch (direction) {
		case 1:	{ I[0] = I_L; I[1] = I_XH; I[2] = I_U; I[3] = I_D; 
			pl_oi = 2; pl_pi = 3; pl_pti = 1; break; }
		case 2: { I[0] = I_U; I[1] = I_XV; I[2] = prime(I_R); I[3] = prime(I_L); 
			pl_oi = 3; pl_pi = 0; pl_pti = 2; break; }  
		case 3: { I[0] = I_R; I[1] = prime(I_XH); I[2] = prime(I_D); I[3] = prime(I_U); 
			pl_oi = 0; pl_pi = 1; pl_pti = 3; break; }  
		case 4: { I[0] = I_D; I[1] = prime(I_XV); I[2] = prime(I_L); I[3] = prime(I_R); 
			pl_oi = 1; pl_pi = 2; pl_pti = 0; break; } 
	}

	// Prepare vectors P, Pt to hold projectors 
	P  = std::vector<ITensor>( c.sites.size() );
	Pt = std::vector<ITensor>( c.sites.size() );
	ip  = Index("AUXP", x);
	ipt = Index("AUXPt",x);

    for(auto const& v_id : c.vToId) {
    	Vertex const& v = v_id.first;
    	std::string id  = v_id.second;
    	auto ai  = c.aux[vToPos(v)];
    	auto dai = delta(prime(ai,pl_pi), prime(ai,pl_pti));


    	ITensor U, S, V;
    	auto R  = build_corner_V2(corner_i,  c, v);
    	auto Rt = build_corner_V2(corner_it, c, v);
  		R *= dai;
  		R *= prime(dai,BRAKET_OFFSET);
    

    	U = ITensor(I[2], prime(ai,pl_oi), prime(ai,pl_oi+BRAKET_OFFSET));
	    svd( R * Rt, U, S, V, solver, argsSVDRRt);
	    if( S.real(S.inds().front()(1),S.inds().back()(1)) > isoMaxElemWarning ||
	        S.real(S.inds().front()(1),S.inds().back()(1)) < isoMinElemWarning ) {
	        std::cout << "WARNING: CTM-Iso3 " << direction << " [col:row]= "<< v 
	    		<<" Max Sing. val.: "<< S.real(S.inds().front()(1),S.inds().back()(1))
	            << std::endl;
	    }

	    // Create pseudo-inverse matrix
    	auto sIU = commonIndex(U,S);
    	auto sIV = commonIndex(S,V);
    	double max_sv = S.real(S.inds().front()(1),S.inds().back()(1));
    	auto oneOverSqrtT = [&max_sv, &loc_psdInvCutoff](Real r) 
        	{ return (r/max_sv > loc_psdInvCutoff) ? 1.0/sqrt(r) : 0.0; };
    	S.apply(oneOverSqrtT);


    	R *= dai;
  		R *= prime(dai,BRAKET_OFFSET);

	    P[vToPos(v)]  = (Rt*V.dag())*S*delta(sIU, ip);
	   	Pt[vToPos(v)] = (R*U.dag())*S*delta(sIV, ipt);
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
	
    
    switch(direction) {
        case 1: {
            // build left upper corner
            ct = T_L.at( siteIndex ); ct *= CMB.at(vToId(v))[0];
            ct *= C_LU.at( siteIndex );
            //ct *= sites[siteIndex];
            ct *= T_U.at( siteIndex ); ct *= CMB.at(vToId(v))[1];
            ct *= c.getSiteRefc(v); 
			ct *= dag(prime(c.getSite(v),BRAKET_OFFSET));
            ct.prime(ULINK,-1); ct.prime(LLINK,-1);
            break;
        }
        case 2: {
            // build right upper corner
            ct = T_U.at( siteIndex ); ct *= CMB.at(vToId(v))[1];
            ct *= C_RU.at( siteIndex );
            //ct *= sites[siteIndex];
            ct *= T_R.at( siteIndex ); ct *= CMB.at(vToId(v))[2];
            ct *= c.getSiteRefc(v); 
			ct *= dag(prime(c.getSite(v),BRAKET_OFFSET));
            ct.prime(RLINK,-1); ct.prime(ULINK,1);
            break;
        }
        case 3: {
            // build right lower corner
            ct = T_R.at( siteIndex ); ct *= CMB.at(vToId(v))[2];
            ct *= C_RD.at( siteIndex );
            //ct *= sites[siteIndex];
            ct *= T_D.at( siteIndex ); ct *= CMB.at(vToId(v))[3];
            ct *= c.getSiteRefc(v); 
			ct *= dag(prime(c.getSite(v),BRAKET_OFFSET));
            ct.prime(DLINK,1); ct.prime(RLINK,1);
            break;
        }
        case 4: {
            // build left lower corner
            ct = T_D.at( siteIndex ); ct *= CMB.at(vToId(v))[3];
            ct *= C_LD.at( siteIndex ); 
            //ct *= sites[siteIndex];
            ct *= T_L.at( siteIndex ); ct *= CMB.at(vToId(v))[0];
            ct *= c.getSiteRefc(v); 
			ct *= dag(prime(c.getSite(v),BRAKET_OFFSET));
            ct.prime(LLINK,1); ct.prime(DLINK,-1);
            break;
        }
    }
    return ct;
}