#include "p-ipeps/full-update.h"
#include "include/LBFGS.h"

using namespace itensor;

Args fullUpdate_CG_full4S(OpNS const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
    auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    auto epsdistf    = args.getReal("epsdistf",1.0e-8);
	auto cg_linesearch_eps = args.getReal("cgLineSearchEps",1.0e-8);
	auto cg_fdistance_eps  = args.getReal("cgFDistanceEps",1.0e-8);
	auto cg_gradientNorm_eps = args.getReal("cgGradientNormEps",1.0e-8);
    auto otNormType = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();
    std::vector<ITensor> orig_tensors = { cls.sites.at(tn[0]), cls.sites.at(tn[1]),
    	cls.sites.at(tn[2]), cls.sites.at(tn[3]) };

	if(dbg) {
		std::cout<<"GATE: ";
		for(int i=0; i<=3; i++) {
			std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
		}
		std::cout<< std::endl;

		// TODO print OpNS if dbg && dbgLvl
		Print(uJ1J2.op);
		for (auto const& i : uJ1J2.pi) Print(i);
	}

	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};

	// read off auxiliary and physical indices of the cluster sites
	// std::array<Index, 4> aux;
	// for (int i=0; i<4; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

	std::array<Index, 4> phys;
	for (int i=0; i<4; i++) phys[i] = cls.mphys.at( tn[i] );
	
	std::array<Index, 4> opPI({
		noprime(uJ1J2.pi[0]),
		noprime(uJ1J2.pi[1]),
		noprime(uJ1J2.pi[2]),
		noprime(uJ1J2.pi[3])
	});
	
	if (dbg) {
		std::cout << "On-site indices:" << std::endl;
		for (int i=0; i<4; i++) {
			// std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
			std::cout << tn[i] <<" : " << phys[i] << std::endl;
		}
	}

	ITensor deltaBra, deltaKet;
	std::vector<ITensor> pc(4); // holds corners T-C-T
	{
		t_begin_int = std::chrono::steady_clock::now();

		// prepare map from on-site tensor aux-indices to half row/column T
		// environment tensors
		std::array<const std::map<std::string, ITensor> * const, 4> iToT(
			{&ctmEnv.T_L, &ctmEnv.T_U, &ctmEnv.T_R ,&ctmEnv.T_D});

		// prepare map from on-site tensor aux-indices pair to half corner T-C-T
		// environment tensors
		const std::map<int, const std::map<std::string, ITensor> * const > iToC(
			{{23, &ctmEnv.C_LU}, {32, &ctmEnv.C_LU},
			{21, &ctmEnv.C_LD}, {12, &ctmEnv.C_LD},
			{3, &ctmEnv.C_RU}, {30, &ctmEnv.C_RU},
			{1, &ctmEnv.C_RD}, {10, &ctmEnv.C_RD}});

		// precompute 4 (proto)corners of 2x2 environment
		for (int s=0; s<=3; s++) {
			// aux-indices connected to sites
			//                         incoming   outgoing 
			std::vector<int> connected({pl[s*2], pl[s*2+1]});
			//                           current-edge         previous-edge
			std::vector<int> tmp_iToE({ (pl[s*2] + 2) % 4, (pl[s*2 + 1] + 2) % 4 });
			tmp_iToE.push_back(pl[s*2]*10+pl[s*2+1]); // identifier for C ENV tensor
			if(dbg) { 
				std::cout <<"primeLevels (pl) of indices connected to ENV - site: "
					<< tn[s] << std::endl;
				std::cout << tmp_iToE[0] <<" "<< tmp_iToE[1] <<" iToC: "<< tmp_iToE[2] << std::endl;
			}

			pc[s] = (*iToT.at(tmp_iToE[0])).at(tn[s])*(*iToC.at(tmp_iToE[2])).at(tn[s])*
				(*iToT.at(tmp_iToE[1])).at(tn[s]);
			if(dbg) Print(pc[s]);
		}

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed proto Corners (without on-site tensors): "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 
	}

	// ***** FORM REDUCED TENSORS **********************************************
	std::vector<ITensor> rX;
	std::vector<ITensor> qX;
	std::vector<Index> iQX;
	{
		// TODO mask ?, cutoff ?
		auto maskS   = [&machine_eps](Real r) { return (r > std::sqrt(10.0*machine_eps)) ? 1.0 : 1.0; };
		auto cutoffS = [&machine_eps](Real r) { return (r > std::sqrt(10.0*machine_eps)) ? r : r; };

		for (int i=0; i<4; i++) {
			ITensor X(cls.AIc(tn[i],pl[2*i]),cls.AIc(tn[i],pl[2*i+1]),phys[i]),Q,SV;

			svd(cls.sites.at(tn[i]),X,SV,Q,{"Truncate",false});

			SV *= 1.0/SV.real(SV.inds()[0](1),SV.inds()[1](1));
			SV.apply(cutoffS);
			auto maskSV = SV;
			maskSV.apply(maskS);
			auto tmpI_SV = commonIndex(SV,X); 

			Index iqx("iqx", commonIndex(SV,Q).m(), AUXLINK, 0);
			X = (X * SV) * delta(commonIndex(SV,Q),iqx);
			Q *= maskSV;
			Q *= delta(tmpI_SV,iqx);

			rX.emplace_back(X);
			qX.emplace_back(Q);
			iQX.emplace_back(iqx);
		}

		for (int i=0; i<4; i++) { pc[i] *= qX[i]; pc[i] *= prime(dag(qX[i]),AUXLINK,4); }
	}
	// ***** FORM REDUCED TENSORS DONE *****************************************

	// ******************************************************************************************** 
	// 	     OPTIMIZE VIA CG                                                                      *
	// ********************************************************************************************

	ITensor op4s = uJ1J2.op;
	for (int i=0; i<4; i++ ) {
		op4s *= delta(opPI[i],phys[i]);
		op4s *= prime(delta(opPI[i],phys[i]));
	}
	Print(op4s);

	t_begin_int = std::chrono::steady_clock::now();
	
	FU4SiteGradMin g4s(ctmEnv, pc, tn, pl, op4s, rX, iQX, args);
	// g4s.minimize();
	
	// prepare init guess by vectorizing vector<ITensor> rX into Vec
	Vec<Real> x;
	resize(x, 4 * std::pow(cls.auxBondDim,4) * cls.physDim);

	auto extractDenseReal = [](Dense<Real> const& d) { return d.store; };
	for (int i=0; i<4; i++) {
		IndexSet inds(rX[i].inds());
		int n = 1;
		for (auto const& ind : inds) { n *= ind.m(); }
		// std::cout<<"Copying rX["<< i <<"] of size: "<< n << " to x["<< i*n <<":"<< (i+1)*n-1<<"]"<< std::endl;
		// PrintData(rX[i]);
		auto rawRX = applyFunc(extractDenseReal,rX[i].store());
		auto VecRX = makeVecRef<Real>(rawRX.data(), rawRX.size());
		auto partX = subVector(x,i*n,(i+1)*n-1);
	
		std::copy(VecRX.data(),VecRX.data()+n,partX.data());

		// Print(VecRX);
		// Print(x);
	}

	LBFGSpp::LBFGSParam<double> param;
    param.m = 6; // default 6 - number of gradients to preserve
    // param.epsilon = epsdistf; // norm of grad based convergence tests
    param.past = 1;
    param.delta = epsdistf;
    param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_WOLFE;

    LBFGSpp::LBFGSSolver<double,Real> solver(param);

    double final_dist;
    int niter = solver.minimize(g4s, x, final_dist);

	t_end_int = std::chrono::steady_clock::now();

	for (int i=0; i<4; i++) cls.sites.at(tn[i]) = rX[i] * qX[i];

	// POST-OPTIMIZATION DIAGNOSTICS ------------------------------------------
	// max element of on-site tensors
	std::string diag_maxElem;
	for (int i=0; i<4; i++) {
		m = 0.;
		// cls.sites.at(tn[i]).visit(max_m);
		m = norm(cls.sites.at(tn[i]) - orig_tensors[i]);
		diag_maxElem = diag_maxElem + tn[i] +" "+ std::to_string(m);
		if (i < 3) diag_maxElem +=  " ";
	}
	std::cout << diag_maxElem << std::endl;

	// normalize updated tensors
	if (otNormType == "BLE") {
		for (int i=0; i<3; i++) {
			m = 0.;
			cls.sites.at(tn[i]).visit(max_m);
			cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / sqrt(m);
		}
	} else if (otNormType == "BALANCE") {
		double iso_tot_mag = 1.0;
	    for ( auto & site_e : cls.sites)  {
	    	m = 0.;
			site_e.second.visit(max_m);
	    	site_e.second = site_e.second / m;
	    	iso_tot_mag = iso_tot_mag * m;
	    }
	    for (auto & site_e : cls.sites) {
	    	site_e.second = site_e.second * std::pow(iso_tot_mag, (1.0/8.0) );
	    }
	} else if (otNormType == "NONE") {
	} else {
		std::cout<<"Unsupported on-site tensor normalisation after full update: "
			<< otNormType << std::endl;
		exit(EXIT_FAILURE);
	}

	// max element of on-site tensors after normalization
    for (int i=0; i<4; i++) {
        m = 0.;
        cls.sites.at(tn[i]).visit(max_m);
        if (i<3) 
        	std::cout << tn[i] <<" "<< std::to_string(m) << " ";
    	else 
    		std::cout << tn[i] <<" "<< std::to_string(m);
    }
    std::cout << std::endl;

	// prepare and return diagnostic data
	diag_data.add("alsSweep", niter); //altlstsquares_iter);

	std::string siteMaxElem_descriptor = "site max_elem site max_elem site max_elem site max_elem";
	diag_data.add("siteMaxElem_descriptor",siteMaxElem_descriptor);
	diag_data.add("siteMaxElem",diag_maxElem);
	diag_data.add("ratioNonSymLE", -1.0); // ratio of largest elements 
	diag_data.add("ratioNonSymFN", -1.0); // ratio of norms
	// diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
	// diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	
	std::string logMinDiag_descriptor = "f_init f_final normalizedf_init normalizedf_final norm(psi')_init norm(psi')_final time[s]";
	std::ostringstream oss;
	// oss << std::scientific << fdist[0] <<" "<< fdist.back() <<" " 
	// 	<< fdistN[0] <<" "<< fdistN.back() <<" "<< vec_normPsi[0] <<" "<< vec_normPsi.back() <<" "<<
	// 	std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 ;
	oss << std::scientific << 0.0 <<" "<< final_dist <<" " 
		<< 0.0 <<" "<< 0.0 <<" "<< 0.0 <<" "<< g4s.inst_normPsi <<" "<<
		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 ;

	
	diag_data.add("locMinDiag_descriptor",logMinDiag_descriptor);
	diag_data.add("locMinDiag", oss.str());
	// if (symmProtoEnv) {
	// 	diag_data.add("diag_protoEnv", diag_protoEnv);
	// 	diag_data.add("diag_protoEnv_descriptor", diag_protoEnv_descriptor);
	// }

	// auto dist0 = overlaps[overlaps.size()-6] - overlaps[overlaps.size()-5] 
	// 	- overlaps[overlaps.size()-4];
	// auto dist1 = overlaps[overlaps.size()-3] - overlaps[overlaps.size()-2] 
	// 	- overlaps[overlaps.size()-1];
	diag_data.add("finalDist0", 0.0); //dist0);
	diag_data.add("finalDist1", 0.0); //dist1);

	// minGapDisc = (minGapDisc < 100.0) ? minGapDisc : -1 ; // whole spectrum taken
	diag_data.add("minGapDisc", 0.0); //minGapDisc);
	diag_data.add("minEvKept", -1); //minEvKept);
	diag_data.add("maxEvDisc", 0.0); //maxEvDisc);

	return diag_data;
}

FU4SiteGradMin::FU4SiteGradMin(
		CtmEnv const& cctmEnv,
		std::vector< ITensor > const& ppc, // protocorners
		std::vector< std::string > const& ttn,      // site IDs
		std::vector< int > const& ppl,              // primelevels of aux indices          
		ITensor const& oop4s,              // four-site operator
		std::vector< ITensor > & rrX,
		std::vector< Index > const& iiQX,
		Args const& aargs) : 

		ctmEnv(cctmEnv), pc(ppc), tn(ttn), pl(ppl), op4s(oop4s), rX(rrX), 
		iQX(iiQX), args(aargs), g(4), xi(4), h(4)
{
	using DIRECTION = CtmEnv::DIRECTION;

	auto l_AIc = [this](std::string id, int dir) { return ctmEnv.p_cluster->AIc(id,dir); };

	auto getEdgeCombiners = [this](ITensor & cmb_v0, ITensor & cmb_v1,
		DIRECTION direction, Vertex const& v0, int dir0, Vertex const& v1, int dir1, 
		std::string edge) {
    	if(edge == "FULL") {
    		cmb_v0 = combiner(
    			ctmEnv.tauxByVertex(direction,v0,dir0),
    			ctmEnv.p_cluster->AIc(v0,dir0),
    			prime(ctmEnv.p_cluster->AIc(v0,dir0), ctmEnv.p_cluster->BRAKET_OFFSET)
    		);
    		cmb_v1 = combiner(
    			ctmEnv.tauxByVertex(direction,v1,dir1),
    			ctmEnv.p_cluster->AIc(v1,dir1),
    			prime(ctmEnv.p_cluster->AIc(v1,dir1), ctmEnv.p_cluster->BRAKET_OFFSET)
    		);
    	}
    	else if (edge == "KET") {
    		cmb_v0 = combiner(
    			ctmEnv.tauxByVertex(direction,v0,dir0),
    			ctmEnv.p_cluster->AIc(v0,dir0)
    		);
    		cmb_v1 = combiner(
    			ctmEnv.tauxByVertex(direction,v1,dir1),
    			ctmEnv.p_cluster->AIc(v1,dir1)
    		);
    	} else {
    		cmb_v0 = combiner(ctmEnv.tauxByVertex(direction,v0,dir0));
    		cmb_v1 = combiner(ctmEnv.tauxByVertex(direction,v1,dir1));
    	}
    };

    auto getEdgeCombiners_fromTnAndPl = [this, &getEdgeCombiners](
    	ITensor & cmb_v0, ITensor & cmb_v1, int s0, int s1, std::string edge){
    	Vertex v0 = ctmEnv.p_cluster->idToV.at(tn[s0]);
    	Vertex v1 = ctmEnv.p_cluster->idToV.at(tn[s1]);

    	DIRECTION dir_outgoing_s0 = toDIRECTION(pl[2*s0+1]);
    	DIRECTION dir_ingoing_s1  = toDIRECTION(pl[2*s1]);

    	DIRECTION edge_dir        = toDIRECTION((pl[2*s0]+2)%4); 

    	getEdgeCombiners(cmb_v0, cmb_v1, 
    		edge_dir, v0, dir_outgoing_s0, v1, dir_ingoing_s1, edge);
    };

	auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    epsdistf = args.getReal("epsdistf",1.0e-8);

	// ***** FORM "PROTO" ENVIRONMENTS FOR K *********************************** 
	std::chrono::steady_clock::time_point t_begin_int, t_end_int;
	t_begin_int = std::chrono::steady_clock::now();

	// Variant ONE - precompute U|psi> surrounded in environment
	{ 
		ITensor temp, cmb0, cmb1, cmb2, cmb3;
		// Variant ONE - precompute U|psi> surrounded in environment
		protoK = pc[0] * rX[0];
		getEdgeCombiners_fromTnAndPl(cmb0,cmb1,0,1,"KET");
		protoK *= cmb0;
		protoK = reindex(protoK, combinedIndex(cmb0), combinedIndex(cmb1));
		protoK *= (( pc[1] * rX[1] ) * cmb1);
		
		temp = pc[2] * rX[2];
		getEdgeCombiners_fromTnAndPl(cmb0,cmb1,2,3,"KET");
		temp *= cmb0;
		temp = reindex(temp, combinedIndex(cmb0), combinedIndex(cmb1));
		temp *= (( pc[3] * rX[3] ) * cmb1);
	
		getEdgeCombiners_fromTnAndPl(cmb1,cmb2,1,2,"KET");
		getEdgeCombiners_fromTnAndPl(cmb3,cmb0,3,0,"KET");

		protoK = (protoK * cmb0) * cmb1;
		temp   = (temp * cmb2) * cmb3;
		protoK = reindex(protoK, combinedIndex(cmb0), combinedIndex(cmb3),
			combinedIndex(cmb1), combinedIndex(cmb2));

		protoK *= temp;
	}
	
	protoK *= op4s;
	protoK.noprime(PHYS);

	if (dbg && dbgLvl >= 3) { Print(protoK); }

	std::cout<<"protoK.scale(): "<< protoK.scale() <<std::endl;
	t_end_int = std::chrono::steady_clock::now();
	std::cout<<"Proto Envs K constructed - T: "<< 
		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
	// ***** FORM "PROTO" ENVIRONMENTS FOR K DONE *****************************

	// computing NORMUPSI <psi|U^dag U|psi> - Variant ONE
	auto NORMUPSI = protoK * dag(op4s);
	NORMUPSI.noprime(PHYS);
	NORMUPSI *= prime(dag(rX[0]), AUXLINK, 4);
	NORMUPSI *= prime(delta(l_AIc(tn[0],pl[1]),l_AIc(tn[1],pl[2])),4);
	NORMUPSI *= prime(dag(rX[1]), AUXLINK, 4);
	NORMUPSI *= prime(delta(l_AIc(tn[1],pl[3]),l_AIc(tn[2],pl[4])),4);
	NORMUPSI *= prime(dag(rX[2]), AUXLINK, 4);
	NORMUPSI *= prime(delta(l_AIc(tn[2],pl[5]),l_AIc(tn[3],pl[6])),4);
	NORMUPSI *= prime(delta(l_AIc(tn[0],pl[0]),l_AIc(tn[3],pl[7])),4);
	NORMUPSI *= prime(dag(rX[3]), AUXLINK, 4);

	if (NORMUPSI.r() > 0) std::cout<<"NORMUPSI rank > 0"<<std::endl;
	normUPsi = sumels(NORMUPSI);
	if (dbg && dbgLvl >=3) { std::cout<<"<psi|U^dag U|psi> = "<< normUPsi << std::endl; }
}

Real FU4SiteGradMin::operator()(Vec<Real> const& x, Vec<Real> & grad) {
	// unwrap Vec x into four tensors rX
	for (int i=0; i<4; i++) {
		IndexSet inds(rX[i].inds()); // copy indexset
		int n = 1;
		for (auto const& ind : inds) { n *= ind.m(); }
		auto rawRX = subVector(x,i*n,(i+1)*n-1); // reference to a part of x
		//std::cout<<"[operator()] making rawRX["<< n <<"] from slice x["<< i*n <<":"<< (i+1)*n-1<<"]"<< std::endl;

		// copy elements into new vector
		Vec<Real> tmpRX;
		resize(tmpRX,n);
		std::copy(rawRX.data(),rawRX.data()+n,tmpRX.data());
		
		rX[i] = ITensor(inds,Dense<Real>{move(tmpRX.storage())});
		// Print(rX[i]);
	}

	// compute gradient
	gradient(g);
	// unwrap vector<ITensor> g into Vec grad
	auto extractDenseReal = [](Dense<Real> const& d) { return d.store; };
	for (int i=0; i<4; i++) {
		IndexSet inds(g[i].inds());
		int n = 1;
		for (auto const& ind : inds) { n *= ind.m(); }
		auto storageGi = applyFunc(extractDenseReal,g[i].store());
		auto VecGi     = makeVecRef<Real>(storageGi.data(), storageGi.size());
		auto part_grad = subVector(grad,i*n,(i+1)*n-1);
	
		std::copy(VecGi.data(),VecGi.data()+n,part_grad.data());
		
		// Print(g[i]);
	}

	// compute and return value of the function
	auto fu4s_data = func();
	double current_dist = fu4s_data[0];
	inst_normPsi = fu4s_data[1];

	// std::cout<<"f= "<< current_dist << " norm(g)= "<< norm(grad) 
		// << " norm(Psi_new)=  "<< inst_normPsi <<std::endl;

	return current_dist;
}

std::vector<double> FU4SiteGradMin::func() const {
	using DIRECTION = CtmEnv::DIRECTION;
	
	auto l_AIc = [this](std::string id, int dir) { return ctmEnv.p_cluster->AIc(id,dir); };

	auto getEdgeCombiners = [this](ITensor & cmb_v0, ITensor & cmb_v1, 
		DIRECTION direction, Vertex const& v0, int dir0, Vertex const& v1, int dir1, 
		std::string edge) {
    	if(edge == "FULL") {
    		cmb_v0 = combiner(
    			ctmEnv.tauxByVertex(direction,v0,dir0),
    			ctmEnv.p_cluster->AIc(v0,dir0),
    			prime(ctmEnv.p_cluster->AIc(v0,dir0), ctmEnv.p_cluster->BRAKET_OFFSET)
    		);
    		cmb_v1 = combiner(
    			ctmEnv.tauxByVertex(direction,v1,dir1),
    			ctmEnv.p_cluster->AIc(v1,dir1),
    			prime(ctmEnv.p_cluster->AIc(v1,dir1), ctmEnv.p_cluster->BRAKET_OFFSET)
    		);
    	}
    	else if (edge == "KET") {
    		cmb_v0 = combiner(
    			ctmEnv.tauxByVertex(direction,v0,dir0),
    			ctmEnv.p_cluster->AIc(v0,dir0)
    		);
    		cmb_v1 = combiner(
    			ctmEnv.tauxByVertex(direction,v1,dir1),
    			ctmEnv.p_cluster->AIc(v1,dir1)
    		);
    	} else {
    		cmb_v0 = combiner(ctmEnv.tauxByVertex(direction,v0,dir0));
    		cmb_v1 = combiner(ctmEnv.tauxByVertex(direction,v1,dir1));
    	}
    };

    auto getEdgeCombiners_fromTnAndPl = [this,&getEdgeCombiners](
    	ITensor & cmb_v0, ITensor & cmb_v1, int s0, int s1, std::string edge){
    	Vertex v0 = ctmEnv.p_cluster->idToV.at(tn[s0]);
    	Vertex v1 = ctmEnv.p_cluster->idToV.at(tn[s1]);

    	DIRECTION dir_outgoing_s0 = toDIRECTION(pl[2*s0+1]);
    	DIRECTION dir_ingoing_s1  = toDIRECTION(pl[2*s1]);

    	DIRECTION edge_dir        = toDIRECTION((pl[2*s0]+2)%4); 

    	getEdgeCombiners(cmb_v0, cmb_v1, 
    		edge_dir, v0, dir_outgoing_s0, v1, dir_ingoing_s1, edge);
    };

	ITensor NORMPSI, OVERLAP;
	ITensor cmb0, cmb1, cmb2, cmb3;

	{
		ITensor temp;

		NORMPSI = (pc[1] * rX[1]) * prime(dag(rX[1]), AUXLINK,4);
		getEdgeCombiners_fromTnAndPl(cmb0,cmb1,1,2,"FULL");
		NORMPSI *= cmb0;
		NORMPSI = reindex(NORMPSI, combinedIndex(cmb0), combinedIndex(cmb1));
		NORMPSI *= (((pc[2] * rX[2]) * prime(dag(rX[2]), AUXLINK,4)) * cmb1);

		temp = (pc[3] * rX[3]) * prime(dag(rX[3]), AUXLINK,4);
		getEdgeCombiners_fromTnAndPl(cmb0,cmb1,3,0,"FULL");
		temp *= cmb0;
		temp = reindex(temp, combinedIndex(cmb0), combinedIndex(cmb1));
		temp *= (((pc[0] * rX[0]) * prime(dag(rX[0]), AUXLINK,4)) * cmb1 );

		getEdgeCombiners_fromTnAndPl(cmb2,cmb3,2,3,"FULL");
		getEdgeCombiners_fromTnAndPl(cmb0,cmb1,0,1,"FULL");

		NORMPSI = (NORMPSI * cmb1) * cmb2;
		temp   = (temp * cmb0) * cmb3;
		NORMPSI = reindex(NORMPSI, combinedIndex(cmb1), combinedIndex(cmb0),
			combinedIndex(cmb2), combinedIndex(cmb3));

		NORMPSI *= temp;
	}

	OVERLAP = prime(dag(rX[1]), AUXLINK,4);
	OVERLAP *= prime(delta( l_AIc(tn[1],pl[3]), l_AIc(tn[2],pl[4])),4);
	OVERLAP *= prime(dag(rX[2]), AUXLINK,4);
	OVERLAP *= prime(delta( l_AIc(tn[2],pl[5]), l_AIc(tn[3],pl[6])),4);
	
	OVERLAP *= protoK;

	OVERLAP *= prime(dag(rX[3]), AUXLINK,4);
	OVERLAP *= prime(delta(	l_AIc(tn[3],pl[7]), l_AIc(tn[0],pl[0])),4);
	OVERLAP *= prime(delta(	l_AIc(tn[1],pl[2]), l_AIc(tn[0],pl[1])),4);
	OVERLAP *= prime(dag(rX[0]), AUXLINK,4);

	if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
	// inst_normPsi = sumels(NORMPSI);
	// inst_overlap = sumels(OVERLAP);
	std::vector<double> v(2);
	v[0] = sumels(NORMPSI) - 2.0 * sumels(OVERLAP) + normUPsi;
	v[1] = sumels(NORMPSI);
	return v;
	// return 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(sumels(NORMPSI) * normUPsi) + 1.0; // normalized
}

void FU4SiteGradMin::gradient(std::vector<ITensor> &grad) {
	using DIRECTION = CtmEnv::DIRECTION;

	// TODO is this under control ?
	// Force the indetical order of indices on both grad and rX tensors
	for (int i=0; i<4; i++) { grad[i] = 0.0 * prime(rX[i],AUXLINK,4); }

	auto l_AIc = [this](std::string id, int dir) { return ctmEnv.p_cluster->AIc(id,dir); };

	auto getEdgeCombiners = [this](ITensor & cmb_v0, ITensor & cmb_v1, 
		DIRECTION direction, Vertex const& v0, int dir0, Vertex const& v1, int dir1, 
		std::string edge) {
    	if(edge == "FULL") {
    		cmb_v0 = combiner(
    			ctmEnv.tauxByVertex(direction,v0,dir0),
    			ctmEnv.p_cluster->AIc(v0,dir0),
    			prime(ctmEnv.p_cluster->AIc(v0,dir0), ctmEnv.p_cluster->BRAKET_OFFSET)
    		);
    		cmb_v1 = combiner(
    			ctmEnv.tauxByVertex(direction,v1,dir1),
    			ctmEnv.p_cluster->AIc(v1,dir1),
    			prime(ctmEnv.p_cluster->AIc(v1,dir1), ctmEnv.p_cluster->BRAKET_OFFSET)
    		);
    	}
    	else if (edge == "KET") {
    		cmb_v0 = combiner(
    			ctmEnv.tauxByVertex(direction,v0,dir0),
    			ctmEnv.p_cluster->AIc(v0,dir0)
    		);
    		cmb_v1 = combiner(
    			ctmEnv.tauxByVertex(direction,v1,dir1),
    			ctmEnv.p_cluster->AIc(v1,dir1)
    		);
    	} else {
    		cmb_v0 = combiner(ctmEnv.tauxByVertex(direction,v0,dir0));
    		cmb_v1 = combiner(ctmEnv.tauxByVertex(direction,v1,dir1));
    	}
    };

    auto getEdgeCombiners_fromTnAndPl = [this,&getEdgeCombiners](
    	ITensor & cmb_v0, ITensor & cmb_v1, int s0, int s1, std::string edge){
    	Vertex v0 = ctmEnv.p_cluster->idToV.at(tn[s0]);
    	Vertex v1 = ctmEnv.p_cluster->idToV.at(tn[s1]);

    	DIRECTION dir_outgoing_s0 = toDIRECTION(pl[2*s0+1]);
    	DIRECTION dir_ingoing_s1  = toDIRECTION(pl[2*s1]);

    	DIRECTION edge_dir        = toDIRECTION((pl[2*s0]+2)%4); 

    	getEdgeCombiners(cmb_v0, cmb_v1,
    		edge_dir, v0, dir_outgoing_s0, v1, dir_ingoing_s1, edge);
    };

	// compute d<psi'|psi'> contributions
	{
		ITensor M;
		{
			ITensor temp, cmb0, cmb1, cmb2, cmb3;
			// Variant ONE - precompute |psi'> surrounded in environment
			M = pc[0] * rX[0];
			getEdgeCombiners_fromTnAndPl(cmb0,cmb1,0,1,"KET");
			M *= cmb0;
			M = reindex(M, combinedIndex(cmb0), combinedIndex(cmb1));
			M *= (( pc[1] * rX[1] ) * cmb1);
			
			temp = pc[2] * rX[2];
			getEdgeCombiners_fromTnAndPl(cmb0,cmb1,2,3,"KET");
			temp *= cmb0;
			temp = reindex(temp, combinedIndex(cmb0), combinedIndex(cmb1));
			temp *= (( pc[3] * rX[3] ) * cmb1);
		
			getEdgeCombiners_fromTnAndPl(cmb1,cmb2,1,2,"KET");
			getEdgeCombiners_fromTnAndPl(cmb3,cmb0,3,0,"KET");

			M = (M * cmb0) * cmb1;
			temp   = (temp * cmb2) * cmb3;
			M = reindex(M, combinedIndex(cmb0), combinedIndex(cmb3),
			combinedIndex(cmb1), combinedIndex(cmb2));

			M *= temp;
		}
	
		for (int i=0; i<4; i++) {
			ITensor temp;

			int j = (i + 1) % 4;
			int k = (j + 1) % 4;
			temp = prime(dag(rX[j]), AUXLINK, 4);
			temp *= prime(delta(l_AIc(tn[j],pl[2*j]), l_AIc(tn[i],pl[2*i+1])), 4);
			temp *= prime(delta(l_AIc(tn[j],pl[2*j+1]), l_AIc(tn[k],pl[2*k])), 4);
			
			j = (j + 1) % 4;
			k = (j + 1) % 4;
			temp *= prime(dag(rX[j]), AUXLINK, 4);
			temp *= prime(delta(l_AIc(tn[j],pl[2*j+1]), l_AIc(tn[k],pl[2*k])), 4);

			temp *= M;
		
			j = (j + 1) % 4;
			k = (j + 1) % 4;
			temp *= prime(dag(rX[j]), AUXLINK, 4);
			temp *= prime(delta(l_AIc(tn[j],pl[2*j+1]), l_AIc(tn[k],pl[2*k])), 4);			
		
			grad[i] += temp; //* (1.0 + 2.0 * (1.0 - std::abs(inst_normPsi)));
			// grad[i] = temp * (0.5 * inst_overlap / (std::sqrt(normUPsi * inst_normPsi) * std::abs(inst_normPsi)) );
		}
	}

	// compute d<psi'|U|psi> contributions
	for (int i=0; i<4; i++) {
		ITensor temp;

		int j = (i + 1) % 4;
		int k = (j + 1) % 4;
		temp = prime(dag(rX[j]), AUXLINK, 4);
		temp *= prime(delta(l_AIc(tn[j],pl[2*j]), l_AIc(tn[i],pl[2*i+1])), 4);
		temp *= prime(delta(l_AIc(tn[j],pl[2*j+1]), l_AIc(tn[k],pl[2*k])), 4);
		
		j = (j + 1) % 4;
		k = (j + 1) % 4;
		temp *= prime(dag(rX[j]), AUXLINK, 4);
		temp *= prime(delta(l_AIc(tn[j],pl[2*j+1]), l_AIc(tn[k],pl[2*k])), 4);

		temp *= protoK;
	
		j = (j + 1) % 4;
		k = (j + 1) % 4;
		temp *= prime(dag(rX[j]), AUXLINK, 4);
		temp *= prime(delta(l_AIc(tn[j],pl[2*j+1]), l_AIc(tn[k],pl[2*k])), 4);			
	
		grad[i] += -temp;
		// grad[i] += -temp / std::sqrt(inst_normPsi * normUPsi);
	}

	for (auto& ten : grad) { 
		ten.prime(AUXLINK, -4); 
	}
}

// void FU4SiteGradMin::minimize() {
// 	const int ITMAX=200;
// 	const double EPS=1.0e-18;
// 	const double GTOL=1.0e-8;
// 	const double FTOL=epsdistf;
// 	double gg,dgg,test,fret,finit;

// 	double max_mag = 0.;
// 	auto maxComp = [&max_mag](double r) {
//   		if(std::fabs(r) > max_mag) max_mag = std::fabs(r);
//   	};

// 	// compute initial function value and gradient and set variables
// 	double fp = func();
// 	finit = fp;
// 	std::cout<<"Init distance: "<< finit << std::endl;
// 	gradient(xi);
// 	for(int j=0; j<4; j++) {
// 		g[j] = -1.0 * xi[j];
// 		xi[j]=h[j]=g[j];
// 	}

// 	for (int its=0;its<ITMAX;its++) {
// 		// perform line minization
// 		fret=linmin(fp, xi);
// 		std::cout<<"its: "<< its << " currentDist: "<< fret << std::endl;

// 		// if (2.0*abs(fret-fp) <= FTOL*(abs(fret)+abs(fp)+EPS)) {
// 		if ( std::abs(fret - fp)/finit <= FTOL  || std::abs(fret) < 1.0e-7 ) {
// 			std::cout << "Frprmn: converged iter="<< its 
// 				<<". ||psi'> - |psi>|^2 = "<< fret << " Diff: " << std::abs(fret - fp)/finit << std::endl;
// 			return;
// 		}
// 		fp=fret;
// 		gradient(xi); //func.df(p,xi)
// 		test=0.0;
// 		// double den=MAX(abs(fp),1.0);
// 		// for (Int j=0;j<4;j++) {

// 		// 	Doub temp=abs(xi[j])*MAX(abs(p[j]),1.0)/den;
// 		// 	if (temp > test) test=temp;
// 		// }
// 		// if (test < GTOL) { 
// 		// 	std::cout << "Frprmn: converged iter="<< its <<". g^2 = "<< test << std::endl;
// 		// 	return p;
// 		// }

// 		dgg=gg=0.0;
// 		for (int j=0; j<4; j++) {
// 			auto temp = norm(g[j]);
// 			gg += temp * temp;
// 			auto tempT = (xi[j] + g[j])*xi[j];
// 			if (tempT.r() > 0) std::cout <<"ERROR: tempT is not a scalar" << std::endl;
// 			dgg += sumels(tempT);
// 		}

// 		// for (Int j=0;j<n;j++) {
// 		// 	gg += g[j]*g[j];
// 		// //	dgg += xi[j]*xi[j];
// 		// 	dgg += (xi[j]+g[j])*xi[j];
// 		// }
// 		if (gg == 0.0) return;
// 		double gam=dgg/gg;
// 		for (int j=0;j<4;j++) {
// 			g[j] = -1.0 * xi[j];
// 			xi[j]=h[j]=g[j];//+gam*h[j];
// 		}
// 	}
// 	throw("Too many iterations in frprmn");
// 	std::cout << "Frprmn: max iterations exceeded. g^2 = "<< test << std::endl;
// }

// double FU4SiteGradMin::linmin(double fxi, std::vector< ITensor > const& g) {
// 	const int MAXIT = 100;
	
// 	auto ngrad = 0.0;
// 	for (int j=0; j<4; j++) { ngrad += std::pow(norm(g[j]),2); }
// 	std::cout<<"Entering LinMin |g|: "<< std::sqrt(ngrad) << std::endl;

// 	std::vector<ITensor> tmpT;
// 	for (int j=0; j<4; j++) { tmpT.emplace_back(rX[j]); }

// 	// double mag = 2 * std::atan(1);
// 	int sgn = 1; 
// 	double mag = fxi * 0.01;
// 	// for (int j=0; j<4; j++) { rX[j] = std::cos(mag) * tmpT[j] + std::sin(mag) * g[j]; }
// 	for (int j=0; j<4; j++) { rX[j] -= mag * g[j]; }
// 	auto tmp_val  = func();
// 	std::cout<<"linmin -mag: "<< tmp_val << std::endl;
// 	if ( tmp_val < fxi ) { sgn = -1; }
// 	for (int j=0; j<4; j++) { rX[j] += (2.0 * mag) * g[j]; }
// 	auto tmp_val2 = func();
// 	std::cout<<"linmin +mag: "<< tmp_val2 << std::endl;
// 	if ( (tmp_val2 < fxi) && (tmp_val2 < tmp_val) ) { sgn = 1; }

// 	for (int j=0; j<4; j++) { rX[j] -= mag * g[j]; }
// 	double fx = 0.0;
// 	double fx_prev = fxi;
// 	for (int i=0; i<MAXIT; i++) {
// 		for (int j=0; j<4; j++) { rX[j] += (sgn * mag) * g[j]; }
// 		// for (int j=0; j<4; j++) { rX[j] = std::cos(mag) * tmpT[j] + std::sin(sgn * mag) * g[j]; }
// 		fx = func();
// 		std::cout<<"linmin its: "<< i << " dist: "<< fx << std::endl;
// 		if (fx > fx_prev) {
// 			// mag = mag * 0.5;
// 			// for (int j=0; j<4; j++) { rX[j] = std::cos(mag) * tmpT[j] + std::sin(sgn * mag) * g[j]; }
// 			for (int j=0; j<4; j++) { rX[j] -= (sgn * mag) * g[j]; }
// 			fx = fx_prev;
// 			break;
// 		} else {
// 			mag = mag * 2.0;
			
// 			fx_prev = fx;
// 		}
// 		// fx_prev = fx;
// 	}
// 	for (int j=0; j<4; j++) {std::cout<<"1: "<< norm(tmpT[j] - rX[j]) <<" "; }
// 	std::cout << std::endl;

// 	return fx;
// }