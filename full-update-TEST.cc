#include "full-update-TEST.h"

using namespace itensor;

Args fullUpdate_COMB_INV(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
    auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    auto symmProtoEnv = args.getBool("symmetrizeProtoEnv",true);
    auto posDefProtoEnv = args.getBool("positiveDefiniteProtoEnv",true);
    auto iso_eps    = args.getReal("isoEpsilon",1.0e-10);
	auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-14);
	auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);
    auto otNormType = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	double lambda = 0.01;
	double lstep  = 0.1;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

	std::cout<<"GATE: ";
	for(int i=0; i<=3; i++) {
		std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
	}
	std::cout<< std::endl;

	if(dbg && (dbgLvl >= 2)) {
		std::cout<< uJ1J2;
		PrintData(uJ1J2.H1);
		PrintData(uJ1J2.H2);
		PrintData(uJ1J2.H3);
	}

	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};

	// map MPOs
	ITensor dummyMPO = ITensor();
	std::array<const ITensor *, 4> mpo({&uJ1J2.H1, &uJ1J2.H2, &uJ1J2.H3, &dummyMPO});

	// find integer identifier of on-site tensors within CtmEnv
	std::vector<int> si;
	for (int i=0; i<=3; i++) {
		si.push_back(std::distance(ctmEnv.siteIds.begin(),
				std::find(std::begin(ctmEnv.siteIds), 
					std::end(ctmEnv.siteIds), tn[i])));
	}
	if(dbg) {
		std::cout << "siteId -> CtmEnv.sites Index" << std::endl;
		for (int i = 0; i <=3; ++i) { std::cout << tn[i] <<" -> "<< si[i] << std::endl; }
	}

	// read off auxiliary and physical indices of the cluster sites
	std::array<Index, 4> aux({
		noprime(findtype(cls.sites.at(tn[0]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[1]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[2]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[3]), AUXLINK)) });

	std::array<Index, 4> auxRT({ aux[0], aux[1], aux[1], aux[2] });
	std::array<int, 4> plRT({ pl[1], pl[2], pl[3], pl[4] });

	std::array<Index, 4> phys({
		noprime(findtype(cls.sites.at(tn[0]), PHYS)),
		noprime(findtype(cls.sites.at(tn[1]), PHYS)),
		noprime(findtype(cls.sites.at(tn[2]), PHYS)),
		noprime(findtype(cls.sites.at(tn[3]), PHYS)) });

	std::array<Index, 3> opPI({
		noprime(findtype(uJ1J2.H1, PHYS)),
		noprime(findtype(uJ1J2.H2, PHYS)),
		noprime(findtype(uJ1J2.H3, PHYS)) });

	// prepare map from on-site tensor aux-indices to half row/column T
	// environment tensors
	std::array<const std::vector<ITensor> * const, 4> iToT(
		{&ctmEnv.T_L, &ctmEnv.T_U, &ctmEnv.T_R ,&ctmEnv.T_D});

	// prepare map from on-site tensor aux-indices pair to half corner T-C-T
	// environment tensors
	const std::map<int, const std::vector<ITensor> * const > iToC(
		{{23, &ctmEnv.C_LU}, {32, &ctmEnv.C_LU},
		{21, &ctmEnv.C_LD}, {12, &ctmEnv.C_LD},
		{3, &ctmEnv.C_RU}, {30, &ctmEnv.C_RU},
		{1, &ctmEnv.C_RD}, {10, &ctmEnv.C_RD}});

	// for every on-site tensor point from primeLevel(index) to ENV index
	// eg. I_XH or I_XV (with appropriate prime level). 
	std::array< std::array<Index, 4>, 4> iToE; // indexToENVIndex => iToE

	// Find for site 0 through 3 which are connected to ENV
	std::vector<int> plOfSite({0,1,2,3}); // aux-indices (primeLevels) of on-site tensor 

	Index iQA, iQD, iQB;
	ITensor QA, eA(prime(aux[0],pl[1]), phys[0]);
	ITensor QD, eD(prime(aux[2],pl[4]), phys[2]);
	ITensor QB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
	
	ITensor eRE, INVeRE;
	ITensor deltaBra, deltaKet;

	{
		// precompute 4 (proto)corners of 2x2 environment
		std::vector<ITensor> pc(4);
		for (int s=0; s<=3; s++) {
			// aux-indices connected to sites
			std::vector<int> connected({pl[s*2], pl[s*2+1]});
			// set_difference gives aux-indices connected to ENV
			std::sort(connected.begin(), connected.end());
			std::vector<int> tmp_iToE;
			std::set_difference(plOfSite.begin(), plOfSite.end(), 
				connected.begin(), connected.end(), 
	            std::inserter(tmp_iToE, tmp_iToE.begin())); 
			tmp_iToE.push_back(pl[s*2]*10+pl[s*2+1]); // identifier for C ENV tensor
			if(dbg) { 
				std::cout <<"primeLevels (pl) of indices connected to ENV - site: "
					<< tn[s] << std::endl;
				std::cout << tmp_iToE[0] <<" "<< tmp_iToE[1] <<" iToC: "<< tmp_iToE[2] << std::endl;
			}

			// Assign indices by which site is connected to ENV
			if( findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK ) ) {
				iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK );
				iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], VSLINK );
			} else {
				iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], VSLINK );
				iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], HSLINK );
			}

			pc[s] = (*iToT.at(tmp_iToE[0]))[si[s]]*(*iToC.at(tmp_iToE[2]))[si[s]]*
				(*iToT.at(tmp_iToE[1]))[si[s]];
			if(dbg) Print(pc[s]);
			// set primeLevel of ENV indices between T's to 0 to be ready for contraction
			pc[s].noprime(LLINK, ULINK, RLINK, DLINK);
		}
		if(dbg) {
			for(int i=0; i<=3; i++) {
				std::cout <<"Site: "<< tn[i] <<" ";
				for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<" ";
				std::cout << std::endl;
			}
		}
		// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 

		// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT ***************************
		t_begin_int = std::chrono::steady_clock::now();

		// C  D
		//    |
		// A--B
		// ITensor eRE;
		// ITensor deltaBra, deltaKet;

		// Decompose A tensor on which the gate is applied
		//ITensor QA, tempSA, eA(prime(aux[0],pl[1]), phys[0]);
		ITensor tempSA;
		svd(cls.sites.at(tn[0]), eA, tempSA, QA);
		//Index iQA("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
		iQA = Index("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
		eA = (eA*tempSA) * delta(commonIndex(QA,tempSA), iQA);
		QA *= delta(commonIndex(QA,tempSA), iQA);

		// Prepare corner of A
		ITensor tempC = pc[0] * getT(QA, iToE[0], (dbg && (dbgLvl >= 3)) );
		if(dbg && (dbgLvl >=3)) Print(tempC);

		deltaKet = delta(prime(aux[0],pl[0]), prime(aux[3],pl[7]));
		deltaBra = prime(deltaKet,4);
		tempC = (tempC * deltaBra) * deltaKet;
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = tempC;

		// Prepare corner of C
		tempC = pc[3] * getT(cls.sites.at(tn[3]), iToE[3], (dbg && (dbgLvl >= 3)));
		if(dbg && (dbgLvl >=3)) Print(tempC);
		
		deltaKet = delta(prime(aux[3],pl[6]), prime(aux[2],pl[5]));
		deltaBra = prime(deltaKet,4);
		tempC = (tempC * deltaBra) * deltaKet;
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = eRE * tempC;

		// Decompose D tensor on which the gate is applied
		//ITensor QD, tempSD, eD(prime(aux[2],pl[4]), phys[2]);
		ITensor tempSD;
		svd(cls.sites.at(tn[2]), eD, tempSD, QD);
		//Index iQD("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
		iQD = Index("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
		eD = (eD*tempSD) * delta(commonIndex(QD,tempSD), iQD);
		QD *= delta(commonIndex(QD,tempSD), iQD);

		// Prepare corner of D
		tempC = pc[2] * getT(QD, iToE[2], (dbg && (dbgLvl >= 3)));
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = eRE * tempC;

		// Decompose B tensor on which the gate is applied
		//ITensor QB, tempSB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
		ITensor tempSB;
		svd(cls.sites.at(tn[1]), eB, tempSB, QB);
		//Index iQB("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
		iQB = Index("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
		eB = (eB*tempSB) * delta(commonIndex(QB,tempSB), iQB);
		QB *= delta(commonIndex(QB,tempSB), iQB);

		tempC = pc[1] * getT(QB, iToE[1], (dbg && (dbgLvl >= 3)));
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = eRE * tempC;

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed reduced Env - T: "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		if(dbg && (dbgLvl >=3)) Print(eRE);
		// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT DONE **********************
	}

	double diag_maxMsymLE, diag_maxMasymLE;
	double diag_maxMsymFN, diag_maxMasymFN;
	if (symmProtoEnv) {
		// ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT ************************
		t_begin_int = std::chrono::steady_clock::now();
		auto cmbKet = combiner(iQA, iQB, iQD);
		auto cmbBra = prime(cmbKet,4);

		eRE = (eRE * cmbKet) * cmbBra;

		ITensor eRE_sym  = 0.5 * (eRE + swapPrime(eRE,0,4));
		ITensor eRE_asym = 0.5 * (eRE - swapPrime(eRE,0,4));

		m = 0.;
	    eRE_sym.visit(max_m);
	    diag_maxMsymLE = m;
	    std::cout<<"eRE_sym max element: "<< m <<std::endl;
		m = 0.;
	    eRE_asym.visit(max_m);
	    diag_maxMasymLE = m;
	    std::cout<<"eRE_asym max element: "<< m <<std::endl;

		diag_maxMsymFN  = norm(eRE_sym);
		diag_maxMasymFN = norm(eRE_asym);
	
		if (posDefProtoEnv) {
			eRE_sym *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet)));
			
			// ##### V3 ######################################################
			ITensor U_eRE, D_eRE, INVD_eRE;
			diagHermitian(eRE_sym, U_eRE, D_eRE);

			double msign = 1.0;
			double mval = 0.;
			std::vector<double> dM_elems, invElems;
			for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
				dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
				if (std::abs(dM_elems.back()) > mval) {
					mval = std::abs(dM_elems.back());
					msign = dM_elems.back()/mval;
				}
			}
			if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// Drop negative EV's
			if(dbg && (dbgLvl >= 1)) {
				std::cout<<"REFINED SPECTRUM"<< std::endl;
				std::cout<<"MAX EV: "<< mval << std::endl;
			}
			for (auto & elem : dM_elems) {
				if (elem < 0.0) {
					if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
					elem = 0.0;
				}
				if (elem > svd_cutoff) 
					invElems.push_back(1.0/elem);
				else
					invElems.push_back(0.0);
			}
			// ##### END V3 ##################################################

			// ##### V4 ######################################################
			// ITensor U_eRE, D_eRE;
			// diagHermitian(eRE_sym, U_eRE, D_eRE);

			// double msign = 1.0;
			// double mval = 0.;
			// std::vector<double> dM_elems;
			// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
			// 	dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
			// 	if (std::abs(dM_elems.back()) > mval) {
			// 		mval = std::abs(dM_elems.back());
			// 		msign = dM_elems.back()/mval;
			// 	}
			// }
			// if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// // Set EV's to ABS Values
			// if(dbg && (dbgLvl >= 1)) {
			// 	std::cout<<"REFINED SPECTRUM"<< std::endl;
			// 	std::cout<<"MAX EV: "<< mval << std::endl;
			// }
			// for (auto & elem : dM_elems) elem = std::fabs(elem);
			// ##### END V4 ##################################################

			// ##### V5 ######################################################
			// eRE *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet))); // 0--eRE--1
			
			// eRE_sym = conj(eRE); // 0--eRE*--1
			// eRE.mapprime(1,2);   // 0--eRE---2
			// eRE_sym = eRE_sym * eRE; // (0--eRE*--1) * (0--eRE--2) = (1--eRE^dag--0) * (0--eRE--2) 
			// eRE_sym.prime(-1);

			// ITensor U_eRE, D_eRE;
			// diagHermitian(eRE_sym, U_eRE, D_eRE);

			// std::vector<double> dM_elems;
			// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) dM_elems.push_back(
			// 		sqrt(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm))) );
			// D_eRE = diagTensor(dM_elems,D_eRE.inds().front(),D_eRE.inds().back());
			// ##### END V5 ##################################################

			// ##### V6 ######################################################
			// ITensor U_eRE, D_eRE;
			// diagHermitian(eRE_sym, U_eRE, D_eRE);

			// double msign = 1.0;
			// double mval = 0.;
			// std::vector<double> dM_elems;
			// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
			// 	dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
			// 	if (std::abs(dM_elems.back()) > mval) {
			// 		mval = std::abs(dM_elems.back());
			// 		msign = dM_elems.back()/mval;
			// 	}
			// }
			// if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// // Drop negative EV's
			// if(dbg && (dbgLvl >= 1)) {
			// 	std::cout<<"REFINED SPECTRUM"<< std::endl;
			// 	std::cout<<"MAX EV: "<< mval << std::endl;
			// }
			// for (auto & elem : dM_elems) {
			// 	if (elem < 0.0 && std::fabs(elem/mval) < svd_cutoff) {
			// 		if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
			// 		elem = 0.0;
			// 	} else if (elem < 0.0) {
			// 		if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< std::fabs(elem) << std::endl;
			// 		elem = std::fabs(elem);
			// 	}
			// }
			// ##### END V6 ##################################################

			
			D_eRE = diagTensor(dM_elems,D_eRE.inds().front(),D_eRE.inds().back());
			INVD_eRE = diagTensor(invElems,D_eRE.inds().front(),D_eRE.inds().back());

			eRE_sym = ((conj(U_eRE)*D_eRE)*prime(U_eRE))
				* delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet)));
			INVeRE = ((conj(U_eRE)*INVD_eRE)*prime(U_eRE))
				* delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet)));
		}

		eRE = (eRE_sym * cmbKet) * cmbBra;
		INVeRE = (INVeRE * cmbKet) * cmbBra;

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Symmetrized reduced env - T: "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		// ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT DONE *******************
	}

	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K ***************************** 
	t_begin_int = std::chrono::steady_clock::now();

	ITensor protoK = (eRE * eA) * delta(prime(aux[0],pl[1]), prime(aux[1],pl[2]));
	protoK = (protoK * eB) * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4]));
	protoK = (protoK * eD);
	if(dbg && (dbgLvl >=2)) Print(protoK);

	protoK = (( protoK * delta(opPI[0],phys[0]) ) * uJ1J2.H1) * prime(delta(opPI[0],phys[0]));
	protoK = (( protoK * delta(opPI[1],phys[1]) ) * uJ1J2.H2) * prime(delta(opPI[1],phys[1]));
	protoK = (( protoK * delta(opPI[2],phys[2]) ) * uJ1J2.H3) * prime(delta(opPI[2],phys[2]));
	protoK.prime(PHYS,-1);
	if(dbg && (dbgLvl >=2)) Print(protoK);
	Print(eRE);

	std::cout<<"eRE.scale(): "<< eRE.scale()<<" protoK.scale(): "<< protoK.scale() <<std::endl;
	t_end_int = std::chrono::steady_clock::now();
	std::cout<<"Proto Envs for M and K constructed - T: "<< 
		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K DONE ************************
	
	std::vector<double> overlaps;
	std::vector<double> rt_diffs, lambdas;
	//int min_indexCutoff = cls.auxBondDim*cls.auxBondDim*uJ1J2.a12.m();
	double minGapDisc = 100.0; // in logscale
	double minEvKept  = svd_cutoff;
	//double maxEvDisc  = 0.0;

	INVeRE *= protoK;
	Print(INVeRE);

	// compute overlap
	eRE *= INVeRE;
	eRE *= prime(INVeRE,AUXLINK,4);
	if (eRE.r() > 0) {
		std::cout <<"ERROR: non-scalar"<< std::endl;
		exit(EXIT_FAILURE);
	}
	overlaps.push_back(sumelsC(eRE).real());

	// reconstruct the sites
	double pw;
	auto pow_T = [&pw](double r) { return std::pow(r,pw); };
	ITensor tempT, tempSV;
	ITensor nEA(iQA, phys[0]);

	svd(INVeRE, nEA, tempSV, tempT);

	pw = 1.0/3.0;
	tempSV.apply(pow_T);
	Index tempInd = commonIndex(nEA,tempSV);
	nEA = (nEA * tempSV) * delta( commonIndex(tempSV,tempT), prime(aux[0],pl[1]) );

	pw = 2.0;
	tempSV.apply(pow_T);
	tempT = (tempT * tempSV) * delta( tempInd, prime(aux[1],pl[2]) );

	ITensor nEB(prime(aux[1],pl[2]), iQB, phys[1]);
	ITensor nED;

	svd(tempT, nEB, tempSV, nED);

	pw = 0.5;
	tempSV.apply(pow_T);
	tempInd = commonIndex(nEB,tempSV);

	nEB = (nEB * tempSV) * delta( commonIndex(tempSV,nED), prime(aux[1],pl[3]) );
	nED = (nED * tempSV) * delta( tempInd, prime(aux[2],pl[4]) );
	// reconstructing sites DONE

	// update on-site tensors of cluster
	cls.sites.at(tn[0]) = QA * nEA;
	cls.sites.at(tn[1]) = QB * nEB;
	cls.sites.at(tn[2]) = QD * nED;

	// max element of on-site tensors
	std::string diag_maxElem;
	for (int i=0; i<4; i++) {
		m = 0.;
		cls.sites.at(tn[i]).visit(max_m);
		diag_maxElem = diag_maxElem + tn[i] +" : "+ std::to_string(m) +" ";
	}
	std::cout << diag_maxElem << std::endl;

	// normalize updated tensors
	if (otNormType == "PTN3") {
		double nn = std::pow(std::abs(overlaps[overlaps.size()-3]), (1.0/6.0));
		for (int i=0; i<3; i++) cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / nn;
	} else if (otNormType == "PTN4") {
		double nn = std::sqrt(std::abs(overlaps[overlaps.size()-3]));
		double ot_norms_tot = 0.0;
		std::vector<double> ot_norms;
		for (int i=0; i<4; i++) 
			{ ot_norms.push_back(norm(cls.sites.at(tn[i]))); ot_norms_tot += ot_norms.back(); } 
		for (int i=0; i<4; i++) cls.sites.at(tn[i]) = 
			cls.sites.at(tn[i]) / std::pow(nn, (ot_norms[i]/ot_norms_tot));
	} else if (otNormType == "BLE") {
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
        std::cout << tn[i] <<" : "<< std::to_string(m) <<" ";
    }
    std::cout << std::endl;

	// prepare and return diagnostic data
	diag_data.add("alsSweep",0);
	diag_data.add("siteMaxElem",diag_maxElem);
	diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
	diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	auto dist0 = overlaps.back();
	// auto dist1 = overlaps[overlaps.size()-3] - overlaps[overlaps.size()-2] 
	// 	- overlaps[overlaps.size()-1];
	diag_data.add("finalDist0",dist0);
	//diag_data.add("finalDist1",dist1);

	//minGapDisc = (minGapDisc < 100.0) ? minGapDisc : -1 ; // whole spectrum taken
	//diag_data.add("minGapDisc",minGapDisc);
	//diag_data.add("minEvKept",minEvKept);
	//diag_data.add("maxEvDisc",maxEvDisc);

	return diag_data;
}

ITensor psInv(ITensor const& M, Args const& args) {
    auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-14);
	auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);
    
    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	double lambda = 0.01;
	double lstep  = 0.1;

	// symmetrize
	auto i0 = M.inds()[0];
	auto i1 = M.inds()[1];
	auto Msym = 0.5*(M + ( (conj(M) * delta(i0,prime(i1,1)))
    	*delta(prime(i0,1),i1) ).prime(-1));

    // check small negative eigenvalues
    Msym = Msym*delta(prime(i0,1), i1);
    ITensor uM, dM, dbg_D;
    auto spec = diagHermitian(Msym, uM, dM);
    dbg_D = dM;
    
    if(dbg && (dbgLvl >= 1)) {
		std::cout<<"ORIGINAL SPECTRUM ";
		Print(dbg_D);
		std::setprecision(std::numeric_limits<long double>::digits10 + 1);
		for(int idm=1; idm<=dbg_D.inds().front().m(); idm++) 
			std::cout << dbg_D.real(dbg_D.inds().front()(idm),dbg_D.inds().back()(idm)) 
			<< std::endl;
	}

	// Find largest (absolute) EV and in the case of msign * mval < 0, multiply
	// dM by -1 as to set the sign of largest (and consecutive) EV to +1
	// SYM SOLUTION
	double msign = 1.0;
	double mval = 0.;
	std::vector<double> dM_elems;
	for (int idm=1; idm<=dM.inds().front().m(); idm++) {  
		dM_elems.push_back(dM.real(dM.inds().front()(idm),dM.inds().back()(idm)));
		if (std::abs(dM_elems.back()) > mval) {
			mval = std::abs(dM_elems.back());
			msign = dM_elems.back()/mval;
		}
	}
	if (msign < 0.0) {
		for (auto & elem : dM_elems) elem = elem*(-1.0);
		std::reverse(dM_elems.begin(),dM_elems.end());
	}

	// In the case of msign < 0.0, for REFINING spectrum we reverse dM_elems
	// Drop small (and negative) EV's
	int index_cutoff;
	std::vector<double> log_dM_e, log_diffs;
	for (int idm=0; idm<dM_elems.size(); idm++) {
		if ( dM_elems[idm] > mval*machine_eps ) {
			log_dM_e.push_back(std::log(dM_elems[idm]));
			log_diffs.push_back(log_dM_e[std::max(idm-1,0)]-log_dM_e[idm]);
		
			// candidate for cutoff
			if ((dM_elems[idm]/mval < svd_cutoff) && 
				(std::fabs(log_diffs.back()) > svd_maxLogGap) ) {
				index_cutoff = idm;

				// log diagnostics
				// if ( minGapDisc > std::fabs(log_diffs.back()) ) {
				// 	minGapDisc = std::fabs(log_diffs.back());
				// 	//min_indexCutoff = std::min(min_indexCutoff, index_cutoff);
				// 	minEvKept = dM_elems[std::max(idm-1,0)];
				// 	//maxEvDisc  = dM_elems[idm];
				// }
				
				for (int iidm=index_cutoff; iidm<dM_elems.size(); iidm++) dM_elems[iidm] = 0.0;

				//Dynamic setting of iso_eps
				//iso_eps = std::min(iso_eps, dM_elems[std::max(idm-1,0)]);

				break;
			}
		} else {
			index_cutoff = idm;
			for (int iidm=index_cutoff; iidm<dM_elems.size(); iidm++) dM_elems[iidm] = 0.0;

			// log diagnostics
			// minEvKept  = dM_elems[std::max(idm-1,0)];

			//Dynamic setting of iso_eps
			//iso_eps = std::min(iso_eps, dM_elems[std::max(idm-1,0)]);
			
			break;
		}
		if (idm == dM_elems.size()-1) {
			index_cutoff = -1;

			// log diagnostics
			// minEvKept  = dM_elems[idm];

			//Dynamic setting of iso_eps
			//iso_eps = std::min(iso_eps, dM_elems[idm]);
		}
	}

	if (msign < 0.0) {
		//for (auto & elem : dM_elems) elem = elem*(-1.0);
		std::reverse(dM_elems.begin(),dM_elems.end());
	}

	dM = diagTensor(dM_elems,dM.inds().front(),dM.inds().back());
	
	if(dbg && (dbgLvl >= 1)) {
		std::cout<<"REFINED SPECTRUM ";
		Print(dM);
		for (int idm=1; idm<=dM.inds().front().m(); idm++) 
			std::cout << dM.real(dM.inds().front()(idm),dM.inds().back()(idm)) 
				<< std::endl;
	}

	// Invert Hermitian matrix Msym
	std::vector<double> elems_regInvDM;
	for (int idm=1; idm<=dM.inds().front().m(); idm++) {
		if (dM.real(dM.inds().front()(idm),dM.inds().back()(idm))/
				dM.real(dM.inds().front()(1),dM.inds().back()(1))  > svd_cutoff) {  
			elems_regInvDM.push_back(msign*1.0/dM.real(dM.inds().front()(idm),
				dM.inds().back()(idm)) );
		} else
			elems_regInvDM.push_back(0.0);
	}
	auto regInvDM = diagTensor(elems_regInvDM, dM.inds().front(),dM.inds().back());
	
	if(dbg && (dbgLvl >= 1)) { std::cout<<"regInvDM.scale(): "<< regInvDM.scale() << std::endl; }
	
	Msym = (conj(uM)*regInvDM)*prime(uM);
	Msym = Msym*delta(prime(i0,1),i1);

	return Msym;
	// END SYM SOLUTION]
}

Args fullUpdate_COMB_CG(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
    auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    auto symmProtoEnv = args.getBool("symmetrizeProtoEnv",true);
    auto posDefProtoEnv = args.getBool("positiveDefiniteProtoEnv",true);
    auto iso_eps    = args.getReal("isoEpsilon",1.0e-10);
	auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-14);
	auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);
    auto otNormType = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	double lambda = 0.01;
	double lstep  = 0.1;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

	std::cout<<"GATE: ";
	for(int i=0; i<=3; i++) {
		std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
	}
	std::cout<< std::endl;

	if(dbg && (dbgLvl >= 2)) {
		std::cout<< uJ1J2;
		PrintData(uJ1J2.H1);
		PrintData(uJ1J2.H2);
		PrintData(uJ1J2.H3);
	}

	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};

	// map MPOs
	ITensor dummyMPO = ITensor();
	std::array<const ITensor *, 4> mpo({&uJ1J2.H1, &uJ1J2.H2, &uJ1J2.H3, &dummyMPO});

	// find integer identifier of on-site tensors within CtmEnv
	std::vector<int> si;
	for (int i=0; i<=3; i++) {
		si.push_back(std::distance(ctmEnv.siteIds.begin(),
				std::find(std::begin(ctmEnv.siteIds), 
					std::end(ctmEnv.siteIds), tn[i])));
	}
	if(dbg) {
		std::cout << "siteId -> CtmEnv.sites Index" << std::endl;
		for (int i = 0; i <=3; ++i) { std::cout << tn[i] <<" -> "<< si[i] << std::endl; }
	}

	// read off auxiliary and physical indices of the cluster sites
	std::array<Index, 4> aux({
		noprime(findtype(cls.sites.at(tn[0]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[1]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[2]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[3]), AUXLINK)) });

	std::array<Index, 4> auxRT({ aux[0], aux[1], aux[1], aux[2] });
	std::array<int, 4> plRT({ pl[1], pl[2], pl[3], pl[4] });

	std::array<Index, 4> phys({
		noprime(findtype(cls.sites.at(tn[0]), PHYS)),
		noprime(findtype(cls.sites.at(tn[1]), PHYS)),
		noprime(findtype(cls.sites.at(tn[2]), PHYS)),
		noprime(findtype(cls.sites.at(tn[3]), PHYS)) });

	std::array<Index, 3> opPI({
		noprime(findtype(uJ1J2.H1, PHYS)),
		noprime(findtype(uJ1J2.H2, PHYS)),
		noprime(findtype(uJ1J2.H3, PHYS)) });

	// prepare map from on-site tensor aux-indices to half row/column T
	// environment tensors
	std::array<const std::vector<ITensor> * const, 4> iToT(
		{&ctmEnv.T_L, &ctmEnv.T_U, &ctmEnv.T_R ,&ctmEnv.T_D});

	// prepare map from on-site tensor aux-indices pair to half corner T-C-T
	// environment tensors
	const std::map<int, const std::vector<ITensor> * const > iToC(
		{{23, &ctmEnv.C_LU}, {32, &ctmEnv.C_LU},
		{21, &ctmEnv.C_LD}, {12, &ctmEnv.C_LD},
		{3, &ctmEnv.C_RU}, {30, &ctmEnv.C_RU},
		{1, &ctmEnv.C_RD}, {10, &ctmEnv.C_RD}});

	// for every on-site tensor point from primeLevel(index) to ENV index
	// eg. I_XH or I_XV (with appropriate prime level). 
	std::array< std::array<Index, 4>, 4> iToE; // indexToENVIndex => iToE

	// Find for site 0 through 3 which are connected to ENV
	std::vector<int> plOfSite({0,1,2,3}); // aux-indices (primeLevels) of on-site tensor 

	Index iQA, iQD, iQB;
	ITensor QA, eA(prime(aux[0],pl[1]), phys[0]);
	ITensor QD, eD(prime(aux[2],pl[4]), phys[2]);
	ITensor QB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
	
	ITensor eRE;
	ITensor deltaBra, deltaKet;

	{
		// precompute 4 (proto)corners of 2x2 environment
		std::vector<ITensor> pc(4);
		for (int s=0; s<=3; s++) {
			// aux-indices connected to sites
			std::vector<int> connected({pl[s*2], pl[s*2+1]});
			// set_difference gives aux-indices connected to ENV
			std::sort(connected.begin(), connected.end());
			std::vector<int> tmp_iToE;
			std::set_difference(plOfSite.begin(), plOfSite.end(), 
				connected.begin(), connected.end(), 
	            std::inserter(tmp_iToE, tmp_iToE.begin())); 
			tmp_iToE.push_back(pl[s*2]*10+pl[s*2+1]); // identifier for C ENV tensor
			if(dbg) { 
				std::cout <<"primeLevels (pl) of indices connected to ENV - site: "
					<< tn[s] << std::endl;
				std::cout << tmp_iToE[0] <<" "<< tmp_iToE[1] <<" iToC: "<< tmp_iToE[2] << std::endl;
			}

			// Assign indices by which site is connected to ENV
			if( findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK ) ) {
				iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK );
				iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], VSLINK );
			} else {
				iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], VSLINK );
				iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], HSLINK );
			}

			pc[s] = (*iToT.at(tmp_iToE[0]))[si[s]]*(*iToC.at(tmp_iToE[2]))[si[s]]*
				(*iToT.at(tmp_iToE[1]))[si[s]];
			if(dbg) Print(pc[s]);
			// set primeLevel of ENV indices between T's to 0 to be ready for contraction
			pc[s].noprime(LLINK, ULINK, RLINK, DLINK);
		}
		if(dbg) {
			for(int i=0; i<=3; i++) {
				std::cout <<"Site: "<< tn[i] <<" ";
				for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<" ";
				std::cout << std::endl;
			}
		}
		// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 

		// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT ***************************
		t_begin_int = std::chrono::steady_clock::now();

		// C  D
		//    |
		// A--B
		// ITensor eRE;
		// ITensor deltaBra, deltaKet;

		// Decompose A tensor on which the gate is applied
		//ITensor QA, tempSA, eA(prime(aux[0],pl[1]), phys[0]);
		ITensor tempSA;
		svd(cls.sites.at(tn[0]), eA, tempSA, QA);
		//Index iQA("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
		iQA = Index("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
		eA = (eA*tempSA) * delta(commonIndex(QA,tempSA), iQA);
		QA *= delta(commonIndex(QA,tempSA), iQA);

		// Prepare corner of A
		ITensor tempC = pc[0] * getT(QA, iToE[0], (dbg && (dbgLvl >= 3)) );
		if(dbg && (dbgLvl >=3)) Print(tempC);

		deltaKet = delta(prime(aux[0],pl[0]), prime(aux[3],pl[7]));
		deltaBra = prime(deltaKet,4);
		tempC = (tempC * deltaBra) * deltaKet;
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = tempC;

		// Prepare corner of C
		tempC = pc[3] * getT(cls.sites.at(tn[3]), iToE[3], (dbg && (dbgLvl >= 3)));
		if(dbg && (dbgLvl >=3)) Print(tempC);
		
		deltaKet = delta(prime(aux[3],pl[6]), prime(aux[2],pl[5]));
		deltaBra = prime(deltaKet,4);
		tempC = (tempC * deltaBra) * deltaKet;
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = eRE * tempC;

		// Decompose D tensor on which the gate is applied
		//ITensor QD, tempSD, eD(prime(aux[2],pl[4]), phys[2]);
		ITensor tempSD;
		svd(cls.sites.at(tn[2]), eD, tempSD, QD);
		//Index iQD("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
		iQD = Index("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
		eD = (eD*tempSD) * delta(commonIndex(QD,tempSD), iQD);
		QD *= delta(commonIndex(QD,tempSD), iQD);

		// Prepare corner of D
		tempC = pc[2] * getT(QD, iToE[2], (dbg && (dbgLvl >= 3)));
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = eRE * tempC;

		// Decompose B tensor on which the gate is applied
		//ITensor QB, tempSB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
		ITensor tempSB;
		svd(cls.sites.at(tn[1]), eB, tempSB, QB);
		//Index iQB("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
		iQB = Index("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
		eB = (eB*tempSB) * delta(commonIndex(QB,tempSB), iQB);
		QB *= delta(commonIndex(QB,tempSB), iQB);

		tempC = pc[1] * getT(QB, iToE[1], (dbg && (dbgLvl >= 3)));
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = eRE * tempC;

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed reduced Env - T: "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		if(dbg && (dbgLvl >=3)) Print(eRE);
		// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT DONE **********************
	}

	double diag_maxMsymLE, diag_maxMasymLE;
	double diag_maxMsymFN, diag_maxMasymFN;
	if (symmProtoEnv) {
		// ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT ************************
		t_begin_int = std::chrono::steady_clock::now();
		auto cmbKet = combiner(iQA, iQB, iQD);
		auto cmbBra = prime(cmbKet,4);

		eRE = (eRE * cmbKet) * cmbBra;

		ITensor eRE_sym  = 0.5 * (eRE + swapPrime(eRE,0,4));
		ITensor eRE_asym = 0.5 * (eRE - swapPrime(eRE,0,4));

		m = 0.;
	    eRE_sym.visit(max_m);
	    diag_maxMsymLE = m;
	    std::cout<<"eRE_sym max element: "<< m <<std::endl;
		m = 0.;
	    eRE_asym.visit(max_m);
	    diag_maxMasymLE = m;
	    std::cout<<"eRE_asym max element: "<< m <<std::endl;

		diag_maxMsymFN  = norm(eRE_sym);
		diag_maxMasymFN = norm(eRE_asym);
	
		if (posDefProtoEnv) {
			eRE_sym *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet)));
			
			// ##### V3 ######################################################
			ITensor U_eRE, D_eRE;
			diagHermitian(eRE_sym, U_eRE, D_eRE);

			double msign = 1.0;
			double mval = 0.;
			std::vector<double> dM_elems;
			for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
				dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
				if (std::abs(dM_elems.back()) > mval) {
					mval = std::abs(dM_elems.back());
					msign = dM_elems.back()/mval;
				}
			}
			if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// Drop negative EV's
			if(dbg && (dbgLvl >= 1)) {
				std::cout<<"REFINED SPECTRUM"<< std::endl;
				std::cout<<"MAX EV: "<< mval << std::endl;
			}
			for (auto & elem : dM_elems) {
				if (elem < 0.0) {
					if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
					elem = 0.0;
				}
			}
			// ##### END V3 ##################################################

			// ##### V4 ######################################################
			// ITensor U_eRE, D_eRE;
			// diagHermitian(eRE_sym, U_eRE, D_eRE);

			// double msign = 1.0;
			// double mval = 0.;
			// std::vector<double> dM_elems;
			// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
			// 	dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
			// 	if (std::abs(dM_elems.back()) > mval) {
			// 		mval = std::abs(dM_elems.back());
			// 		msign = dM_elems.back()/mval;
			// 	}
			// }
			// if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// // Set EV's to ABS Values
			// if(dbg && (dbgLvl >= 1)) {
			// 	std::cout<<"REFINED SPECTRUM"<< std::endl;
			// 	std::cout<<"MAX EV: "<< mval << std::endl;
			// }
			// for (auto & elem : dM_elems) elem = std::fabs(elem);
			// ##### END V4 ##################################################

			// ##### V5 ######################################################
			// eRE *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet))); // 0--eRE--1
			
			// eRE_sym = conj(eRE); // 0--eRE*--1
			// eRE.mapprime(1,2);   // 0--eRE---2
			// eRE_sym = eRE_sym * eRE; // (0--eRE*--1) * (0--eRE--2) = (1--eRE^dag--0) * (0--eRE--2) 
			// eRE_sym.prime(-1);

			// ITensor U_eRE, D_eRE;
			// diagHermitian(eRE_sym, U_eRE, D_eRE);

			// std::vector<double> dM_elems;
			// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) dM_elems.push_back(
			// 		sqrt(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm))) );
			// D_eRE = diagTensor(dM_elems,D_eRE.inds().front(),D_eRE.inds().back());
			// ##### END V5 ##################################################

			// ##### V6 ######################################################
			// ITensor U_eRE, D_eRE;
			// diagHermitian(eRE_sym, U_eRE, D_eRE);

			// double msign = 1.0;
			// double mval = 0.;
			// std::vector<double> dM_elems;
			// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
			// 	dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
			// 	if (std::abs(dM_elems.back()) > mval) {
			// 		mval = std::abs(dM_elems.back());
			// 		msign = dM_elems.back()/mval;
			// 	}
			// }
			// if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// // Drop negative EV's
			// if(dbg && (dbgLvl >= 1)) {
			// 	std::cout<<"REFINED SPECTRUM"<< std::endl;
			// 	std::cout<<"MAX EV: "<< mval << std::endl;
			// }
			// for (auto & elem : dM_elems) {
			// 	if (elem < 0.0 && std::fabs(elem/mval) < svd_cutoff) {
			// 		if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
			// 		elem = 0.0;
			// 	} else if (elem < 0.0) {
			// 		if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< std::fabs(elem) << std::endl;
			// 		elem = std::fabs(elem);
			// 	}
			// }
			// ##### END V6 ##################################################

			
			D_eRE = diagTensor(dM_elems,D_eRE.inds().front(),D_eRE.inds().back());

			eRE_sym = ((conj(U_eRE)*D_eRE)*prime(U_eRE))
				* delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet)));
		}

		eRE = (eRE_sym * cmbKet) * cmbBra;

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Symmetrized reduced env - T: "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		// ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT DONE *******************
	}

	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K ***************************** 
	t_begin_int = std::chrono::steady_clock::now();

	ITensor protoK = (eRE * eA) * delta(prime(aux[0],pl[1]), prime(aux[1],pl[2]));
	protoK = (protoK * eB) * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4]));
	protoK = (protoK * eD);
	if(dbg && (dbgLvl >=2)) Print(protoK);

	protoK = (( protoK * delta(opPI[0],phys[0]) ) * uJ1J2.H1) * prime(delta(opPI[0],phys[0]));
	protoK = (( protoK * delta(opPI[1],phys[1]) ) * uJ1J2.H2) * prime(delta(opPI[1],phys[1]));
	protoK = (( protoK * delta(opPI[2],phys[2]) ) * uJ1J2.H3) * prime(delta(opPI[2],phys[2]));
	protoK.prime(PHYS,-1);
	if(dbg && (dbgLvl >=2)) Print(protoK);
	Print(eRE);

	std::cout<<"eRE.scale(): "<< eRE.scale()<<" protoK.scale(): "<< protoK.scale() <<std::endl;
	t_end_int = std::chrono::steady_clock::now();
	std::cout<<"Proto Envs for M and K constructed - T: "<< 
		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K DONE ************************
	
	// Prepare Alternating Least Squares to maximize the overlap
	auto print_elem = [](double d) {
		std::setprecision(std::numeric_limits<long double>::digits10 + 1);
		std::cout<< d << std::endl;
	};

	int altlstsquares_iter = 0;
	bool converged = false;
	std::vector<double> overlaps;
	std::vector<double> rt_diffs, lambdas;
	//int min_indexCutoff = cls.auxBondDim*cls.auxBondDim*uJ1J2.a12.m();
	double minGapDisc = 100.0; // in logscale
	double minEvKept  = svd_cutoff;
	//double maxEvDisc  = 0.0;

	ITensor dbg_D, dbg_svM;
	
	// ******************************************************************************************** 
	// 	     OPTIMIZE VIA CG                                                                      *
	// ********************************************************************************************

	// intial guess is given by intial eA--eB--eD
	auto tenX = (((eA * delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])) ) * eB ) 
		* delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])) ) * eD;

	// compute constant <psi|psi>
	ITensor NORMPSI = tenX * eRE * prime(tenX, AUXLINK, 4);
	if (NORMPSI.r() > 0) { 
		std::cout<< "ERROR: normpsi tensor is not a scalar" << std::endl;
		exit(EXIT_FAILURE);
	}
	double fconst = sumels(NORMPSI);

	auto cmbX = combiner(iQA, phys[0], iQB, phys[1], iQD, phys[2]);
	auto cmbB = combiner(prime(iQA,4), phys[0], prime(iQB,4), phys[1], prime(iQD,4), phys[2]);

	Print(tenX);
	Print(cmbX);
	Print(cmbB);

	//::vector<double> vecX( combinedIndex(cmbX).m() );
	VecDoub_IO vecX( combinedIndex(cmbX).m() );
	std::vector<double> grad( combinedIndex(cmbB).m() );
	std::vector<double> vecB( combinedIndex(cmbB).m() );

	tenX *= cmbX;
	for(int i=1; i<=combinedIndex(cmbX).m(); i++) vecX[i-1] = tenX.real(combinedIndex(cmbX)(i));
	//PrintData(tenX);
	tenX *= cmbX;

	protoK *= cmbB;
	for(int i=1; i<=combinedIndex(cmbB).m(); i++) vecB[i-1] = protoK.real(combinedIndex(cmbB)(i));
	//PrintData(protoK);
	protoK *= cmbB;

	
	std::cout << "ENTERING CG LOOP" << std::endl;
	

	// while (not converged) {
	Funcd funcd(eRE, cmbX, cmbB, vecB, fconst);
	Frprmn<Funcd> frprmn(funcd, iso_eps, iso_eps, maxAltLstSqrIter);
	vecX = frprmn.minimize(vecX);
				
	// 	altlstsquares_iter++;
	// 	if (altlstsquares_iter >= maxAltLstSqrIter) converged = true;
	// }

	
	tenX *= cmbX;
	for(int i=1; i<=combinedIndex(cmbX).m(); i++ )
		tenX.set(combinedIndex(cmbX)(i), vecX[i-1]);
	tenX *= cmbX;

	// reconstruct the sites
	double pw;
	auto pow_T = [&pw](double r) { return std::pow(r,pw); };
	ITensor tempT, tempSV;
	ITensor nEA(iQA, phys[0]);

	svd(tenX, nEA, tempSV, tempT);

	pw = 1.0/3.0;
	tempSV.apply(pow_T);
	Index tempInd = commonIndex(nEA,tempSV);
	nEA = (nEA * tempSV) * delta( commonIndex(tempSV,tempT), prime(aux[0],pl[1]) );

	pw = 2.0;
	tempSV.apply(pow_T);
	tempT = (tempT * tempSV) * delta( tempInd, prime(aux[1],pl[2]) );

	ITensor nEB(prime(aux[1],pl[2]), iQB, phys[1]);
	ITensor nED;

	svd(tempT, nEB, tempSV, nED);

	pw = 0.5;
	tempSV.apply(pow_T);
	tempInd = commonIndex(nEB,tempSV);

	nEB = (nEB * tempSV) * delta( commonIndex(tempSV,nED), prime(aux[1],pl[3]) );
	nED = (nED * tempSV) * delta( tempInd, prime(aux[2],pl[4]) );
	// reconstructing sites DONE

	// update on-site tensors of cluster
	cls.sites.at(tn[0]) = QA * nEA;
	cls.sites.at(tn[1]) = QB * nEB;
	cls.sites.at(tn[2]) = QD * nED;

	// max element of on-site tensors
	std::string diag_maxElem;
	for (int i=0; i<4; i++) {
		m = 0.;
		cls.sites.at(tn[i]).visit(max_m);
		diag_maxElem = diag_maxElem + tn[i] +" : "+ std::to_string(m) +" ";
	}
	std::cout << diag_maxElem << std::endl;

	// normalize updated tensors
	if (otNormType == "PTN3") {
		double nn = std::pow(std::abs(overlaps[overlaps.size()-3]), (1.0/6.0));
		for (int i=0; i<3; i++) cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / nn;
	} else if (otNormType == "PTN4") {
		double nn = std::sqrt(std::abs(overlaps[overlaps.size()-3]));
		double ot_norms_tot = 0.0;
		std::vector<double> ot_norms;
		for (int i=0; i<4; i++) 
			{ ot_norms.push_back(norm(cls.sites.at(tn[i]))); ot_norms_tot += ot_norms.back(); } 
		for (int i=0; i<4; i++) cls.sites.at(tn[i]) = 
			cls.sites.at(tn[i]) / std::pow(nn, (ot_norms[i]/ot_norms_tot));
	} else if (otNormType == "BLE") {
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
        std::cout << tn[i] <<" : "<< std::to_string(m) <<" ";
    }
    std::cout << std::endl;

	// prepare and return diagnostic data
	diag_data.add("alsSweep",altlstsquares_iter);
	diag_data.add("siteMaxElem",diag_maxElem);
	diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
	diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	// auto dist0 = overlaps[overlaps.size()-6] - overlaps[overlaps.size()-5] 
	// 	- overlaps[overlaps.size()-4];
	// auto dist1 = overlaps[overlaps.size()-3] - overlaps[overlaps.size()-2] 
	// 	- overlaps[overlaps.size()-1];
	//diag_data.add("finalDist0",dist0);
	//diag_data.add("finalDist1",dist1);

	//minGapDisc = (minGapDisc < 100.0) ? minGapDisc : -1 ; // whole spectrum taken
	//diag_data.add("minGapDisc",minGapDisc);
	//diag_data.add("minEvKept",minEvKept);
	//diag_data.add("maxEvDisc",maxEvDisc);

	return diag_data;
}

Funcd::Funcd(ITensor const& NN, ITensor const& ccmbKet, ITensor const& ccmbBra, 
	std::vector<double> const& vvecB, double ffconst) : N(NN), cmbKet(ccmbKet), 
	cmbBra(ccmbBra), vecB(vvecB), fconst(ffconst) {}

Doub Funcd::operator() (VecDoub_I &x) {
	auto tenX = ITensor(combinedIndex(cmbKet));
	auto tenB = ITensor(combinedIndex(cmbBra));

	for(int i=1; i<=combinedIndex(cmbKet).m(); i++ ) tenX.set(combinedIndex(cmbKet)(i), x[i-1]);
	tenX *= cmbKet;

	for(int i=1; i<=combinedIndex(cmbBra).m(); i++) tenB.set(combinedIndex(cmbBra)(i), vecB[i-1]);
	tenB *= cmbBra;

	auto NORM = (tenX * N) * prime(tenX, AUXLINK, 4);
	auto OVERLAP = prime(tenX, AUXLINK, 4) *  tenB;
	
	if (NORM.r() > 0 || OVERLAP.r() > 0) std::cout<<"Funcd() NORM or OVERLAP rank > 0"<<std::endl;

	return fconst + sumels(NORM) - 2.0 * sumels(OVERLAP);
}

void Funcd::df(VecDoub_I &x, VecDoub_O &deriv) {
	auto tenX = ITensor(combinedIndex(cmbKet));

	for(int i=1; i<=combinedIndex(cmbKet).m(); i++ ) tenX.set(combinedIndex(cmbKet)(i), x[i-1]);
	tenX *= cmbKet;

	tenX *= N;
	tenX *= cmbBra;

	for(int i=1; i<=combinedIndex(cmbBra).m(); i++ ) deriv[i-1] = tenX.real(combinedIndex(cmbBra)(i)) - vecB[i-1];
}

Args fullUpdate_CG(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
    auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    auto symmProtoEnv = args.getBool("symmetrizeProtoEnv",true);
    auto posDefProtoEnv = args.getBool("positiveDefiniteProtoEnv",true);
    auto iso_eps    = args.getReal("isoEpsilon",1.0e-10);
	auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-14);
	auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);
    auto otNormType = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	double lambda = 0.01;
	double lstep  = 0.1;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

	std::cout<<"GATE: ";
	for(int i=0; i<=3; i++) {
		std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
	}
	std::cout<< std::endl;

	if(dbg && (dbgLvl >= 2)) {
		std::cout<< uJ1J2;
		PrintData(uJ1J2.H1);
		PrintData(uJ1J2.H2);
		PrintData(uJ1J2.H3);
	}

	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};

	// map MPOs
	ITensor dummyMPO = ITensor();
	std::array<const ITensor *, 4> mpo({&uJ1J2.H1, &uJ1J2.H2, &uJ1J2.H3, &dummyMPO});

	// find integer identifier of on-site tensors within CtmEnv
	std::vector<int> si;
	for (int i=0; i<=3; i++) {
		si.push_back(std::distance(ctmEnv.siteIds.begin(),
				std::find(std::begin(ctmEnv.siteIds), 
					std::end(ctmEnv.siteIds), tn[i])));
	}
	if(dbg) {
		std::cout << "siteId -> CtmEnv.sites Index" << std::endl;
		for (int i = 0; i <=3; ++i) { std::cout << tn[i] <<" -> "<< si[i] << std::endl; }
	}

	// read off auxiliary and physical indices of the cluster sites
	std::array<Index, 4> aux({
		noprime(findtype(cls.sites.at(tn[0]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[1]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[2]), AUXLINK)),
		noprime(findtype(cls.sites.at(tn[3]), AUXLINK)) });

	std::array<Index, 4> auxRT({ aux[0], aux[1], aux[1], aux[2] });
	std::array<int, 4> plRT({ pl[1], pl[2], pl[3], pl[4] });

	std::array<Index, 4> phys({
		noprime(findtype(cls.sites.at(tn[0]), PHYS)),
		noprime(findtype(cls.sites.at(tn[1]), PHYS)),
		noprime(findtype(cls.sites.at(tn[2]), PHYS)),
		noprime(findtype(cls.sites.at(tn[3]), PHYS)) });

	std::array<Index, 3> opPI({
		noprime(findtype(uJ1J2.H1, PHYS)),
		noprime(findtype(uJ1J2.H2, PHYS)),
		noprime(findtype(uJ1J2.H3, PHYS)) });

	// prepare map from on-site tensor aux-indices to half row/column T
	// environment tensors
	std::array<const std::vector<ITensor> * const, 4> iToT(
		{&ctmEnv.T_L, &ctmEnv.T_U, &ctmEnv.T_R ,&ctmEnv.T_D});

	// prepare map from on-site tensor aux-indices pair to half corner T-C-T
	// environment tensors
	const std::map<int, const std::vector<ITensor> * const > iToC(
		{{23, &ctmEnv.C_LU}, {32, &ctmEnv.C_LU},
		{21, &ctmEnv.C_LD}, {12, &ctmEnv.C_LD},
		{3, &ctmEnv.C_RU}, {30, &ctmEnv.C_RU},
		{1, &ctmEnv.C_RD}, {10, &ctmEnv.C_RD}});

	// for every on-site tensor point from primeLevel(index) to ENV index
	// eg. I_XH or I_XV (with appropriate prime level). 
	std::array< std::array<Index, 4>, 4> iToE; // indexToENVIndex => iToE

	// Find for site 0 through 3 which are connected to ENV
	std::vector<int> plOfSite({0,1,2,3}); // aux-indices (primeLevels) of on-site tensor 

	Index iQA, iQD, iQB;
	ITensor QA, eA(prime(aux[0],pl[1]), phys[0]);
	ITensor QD, eD(prime(aux[2],pl[4]), phys[2]);
	ITensor QB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
	
	ITensor eRE;
	ITensor deltaBra, deltaKet;

	{
		// precompute 4 (proto)corners of 2x2 environment
		std::vector<ITensor> pc(4);
		for (int s=0; s<=3; s++) {
			// aux-indices connected to sites
			std::vector<int> connected({pl[s*2], pl[s*2+1]});
			// set_difference gives aux-indices connected to ENV
			std::sort(connected.begin(), connected.end());
			std::vector<int> tmp_iToE;
			std::set_difference(plOfSite.begin(), plOfSite.end(), 
				connected.begin(), connected.end(), 
	            std::inserter(tmp_iToE, tmp_iToE.begin())); 
			tmp_iToE.push_back(pl[s*2]*10+pl[s*2+1]); // identifier for C ENV tensor
			if(dbg) { 
				std::cout <<"primeLevels (pl) of indices connected to ENV - site: "
					<< tn[s] << std::endl;
				std::cout << tmp_iToE[0] <<" "<< tmp_iToE[1] <<" iToC: "<< tmp_iToE[2] << std::endl;
			}

			// Assign indices by which site is connected to ENV
			if( findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK ) ) {
				iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK );
				iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], VSLINK );
			} else {
				iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], VSLINK );
				iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], HSLINK );
			}

			pc[s] = (*iToT.at(tmp_iToE[0]))[si[s]]*(*iToC.at(tmp_iToE[2]))[si[s]]*
				(*iToT.at(tmp_iToE[1]))[si[s]];
			if(dbg) Print(pc[s]);
			// set primeLevel of ENV indices between T's to 0 to be ready for contraction
			pc[s].noprime(LLINK, ULINK, RLINK, DLINK);
		}
		if(dbg) {
			for(int i=0; i<=3; i++) {
				std::cout <<"Site: "<< tn[i] <<" ";
				for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<" ";
				std::cout << std::endl;
			}
		}
		// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 

		// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT ***************************
		t_begin_int = std::chrono::steady_clock::now();

		// C  D
		//    |
		// A--B
		// ITensor eRE;
		// ITensor deltaBra, deltaKet;

		// Decompose A tensor on which the gate is applied
		//ITensor QA, tempSA, eA(prime(aux[0],pl[1]), phys[0]);
		ITensor tempSA;
		svd(cls.sites.at(tn[0]), eA, tempSA, QA);
		//Index iQA("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
		iQA = Index("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
		eA = (eA*tempSA) * delta(commonIndex(QA,tempSA), iQA);
		QA *= delta(commonIndex(QA,tempSA), iQA);

		// Prepare corner of A
		ITensor tempC = pc[0] * getT(QA, iToE[0], (dbg && (dbgLvl >= 3)) );
		if(dbg && (dbgLvl >=3)) Print(tempC);

		deltaKet = delta(prime(aux[0],pl[0]), prime(aux[3],pl[7]));
		deltaBra = prime(deltaKet,4);
		tempC = (tempC * deltaBra) * deltaKet;
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = tempC;

		// Prepare corner of C
		tempC = pc[3] * getT(cls.sites.at(tn[3]), iToE[3], (dbg && (dbgLvl >= 3)));
		if(dbg && (dbgLvl >=3)) Print(tempC);
		
		deltaKet = delta(prime(aux[3],pl[6]), prime(aux[2],pl[5]));
		deltaBra = prime(deltaKet,4);
		tempC = (tempC * deltaBra) * deltaKet;
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = eRE * tempC;

		// Decompose D tensor on which the gate is applied
		//ITensor QD, tempSD, eD(prime(aux[2],pl[4]), phys[2]);
		ITensor tempSD;
		svd(cls.sites.at(tn[2]), eD, tempSD, QD);
		//Index iQD("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
		iQD = Index("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
		eD = (eD*tempSD) * delta(commonIndex(QD,tempSD), iQD);
		QD *= delta(commonIndex(QD,tempSD), iQD);

		// Prepare corner of D
		tempC = pc[2] * getT(QD, iToE[2], (dbg && (dbgLvl >= 3)));
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = eRE * tempC;

		// Decompose B tensor on which the gate is applied
		//ITensor QB, tempSB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
		ITensor tempSB;
		svd(cls.sites.at(tn[1]), eB, tempSB, QB);
		//Index iQB("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
		iQB = Index("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
		eB = (eB*tempSB) * delta(commonIndex(QB,tempSB), iQB);
		QB *= delta(commonIndex(QB,tempSB), iQB);

		tempC = pc[1] * getT(QB, iToE[1], (dbg && (dbgLvl >= 3)));
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = eRE * tempC;

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed reduced Env - T: "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		if(dbg && (dbgLvl >=3)) Print(eRE);
		// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT DONE **********************
	}

	double diag_maxMsymLE, diag_maxMasymLE;
	double diag_maxMsymFN, diag_maxMasymFN;
	if (symmProtoEnv) {
		// ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT ************************
		t_begin_int = std::chrono::steady_clock::now();
		auto cmbKet = combiner(iQA, iQB, iQD);
		auto cmbBra = prime(cmbKet,4);

		eRE = (eRE * cmbKet) * cmbBra;

		ITensor eRE_sym  = 0.5 * (eRE + swapPrime(eRE,0,4));
		ITensor eRE_asym = 0.5 * (eRE - swapPrime(eRE,0,4));

		m = 0.;
	    eRE_sym.visit(max_m);
	    diag_maxMsymLE = m;
	    std::cout<<"eRE_sym max element: "<< m <<std::endl;
		m = 0.;
	    eRE_asym.visit(max_m);
	    diag_maxMasymLE = m;
	    std::cout<<"eRE_asym max element: "<< m <<std::endl;

		diag_maxMsymFN  = norm(eRE_sym);
		diag_maxMasymFN = norm(eRE_asym);
	
		if (posDefProtoEnv) {
			eRE_sym *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet)));
			
			// ##### V3 ######################################################
			ITensor U_eRE, D_eRE;
			diagHermitian(eRE_sym, U_eRE, D_eRE);

			double msign = 1.0;
			double mval = 0.;
			std::vector<double> dM_elems;
			for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
				dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
				if (std::abs(dM_elems.back()) > mval) {
					mval = std::abs(dM_elems.back());
					msign = dM_elems.back()/mval;
				}
			}
			if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// Drop negative EV's
			if(dbg && (dbgLvl >= 1)) {
				std::cout<<"REFINED SPECTRUM"<< std::endl;
				std::cout<<"MAX EV: "<< mval << std::endl;
			}
			for (auto & elem : dM_elems) {
				if (elem < 0.0) {
					if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
					elem = 0.0;
				}
			}
			// ##### END V3 ##################################################

			// ##### V4 ######################################################
			// ITensor U_eRE, D_eRE;
			// diagHermitian(eRE_sym, U_eRE, D_eRE);

			// double msign = 1.0;
			// double mval = 0.;
			// std::vector<double> dM_elems;
			// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
			// 	dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
			// 	if (std::abs(dM_elems.back()) > mval) {
			// 		mval = std::abs(dM_elems.back());
			// 		msign = dM_elems.back()/mval;
			// 	}
			// }
			// if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// // Set EV's to ABS Values
			// if(dbg && (dbgLvl >= 1)) {
			// 	std::cout<<"REFINED SPECTRUM"<< std::endl;
			// 	std::cout<<"MAX EV: "<< mval << std::endl;
			// }
			// for (auto & elem : dM_elems) elem = std::fabs(elem);
			// ##### END V4 ##################################################

			// ##### V5 ######################################################
			// eRE *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet))); // 0--eRE--1
			
			// eRE_sym = conj(eRE); // 0--eRE*--1
			// eRE.mapprime(1,2);   // 0--eRE---2
			// eRE_sym = eRE_sym * eRE; // (0--eRE*--1) * (0--eRE--2) = (1--eRE^dag--0) * (0--eRE--2) 
			// eRE_sym.prime(-1);

			// ITensor U_eRE, D_eRE;
			// diagHermitian(eRE_sym, U_eRE, D_eRE);

			// std::vector<double> dM_elems;
			// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) dM_elems.push_back(
			// 		sqrt(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm))) );
			// D_eRE = diagTensor(dM_elems,D_eRE.inds().front(),D_eRE.inds().back());
			// ##### END V5 ##################################################

			// ##### V6 ######################################################
			// ITensor U_eRE, D_eRE;
			// diagHermitian(eRE_sym, U_eRE, D_eRE);

			// double msign = 1.0;
			// double mval = 0.;
			// std::vector<double> dM_elems;
			// for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
			// 	dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
			// 	if (std::abs(dM_elems.back()) > mval) {
			// 		mval = std::abs(dM_elems.back());
			// 		msign = dM_elems.back()/mval;
			// 	}
			// }
			// if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// // Drop negative EV's
			// if(dbg && (dbgLvl >= 1)) {
			// 	std::cout<<"REFINED SPECTRUM"<< std::endl;
			// 	std::cout<<"MAX EV: "<< mval << std::endl;
			// }
			// for (auto & elem : dM_elems) {
			// 	if (elem < 0.0 && std::fabs(elem/mval) < svd_cutoff) {
			// 		if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
			// 		elem = 0.0;
			// 	} else if (elem < 0.0) {
			// 		if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< std::fabs(elem) << std::endl;
			// 		elem = std::fabs(elem);
			// 	}
			// }
			// ##### END V6 ##################################################

			
			D_eRE = diagTensor(dM_elems,D_eRE.inds().front(),D_eRE.inds().back());

			eRE_sym = ((conj(U_eRE)*D_eRE)*prime(U_eRE))
				* delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet)));
		}

		eRE = (eRE_sym * cmbKet) * cmbBra;

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Symmetrized reduced env - T: "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		// ***** SYMMETRIZE "EFFECTIVE" REDUCED ENVIRONMENT DONE *******************
	}

	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K ***************************** 
	t_begin_int = std::chrono::steady_clock::now();

	ITensor protoK = (eRE * eA) * delta(prime(aux[0],pl[1]), prime(aux[1],pl[2]));
	protoK = (protoK * eB) * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4]));
	protoK = (protoK * eD);
	if(dbg && (dbgLvl >=2)) Print(protoK);

	protoK = (( protoK * delta(opPI[0],phys[0]) ) * uJ1J2.H1) * prime(delta(opPI[0],phys[0]));
	protoK = (( protoK * delta(opPI[1],phys[1]) ) * uJ1J2.H2) * prime(delta(opPI[1],phys[1]));
	protoK = (( protoK * delta(opPI[2],phys[2]) ) * uJ1J2.H3) * prime(delta(opPI[2],phys[2]));
	protoK.prime(PHYS,-1);
	if(dbg && (dbgLvl >=2)) Print(protoK);

	std::cout<<"eRE.scale(): "<< eRE.scale()<<" protoK.scale(): "<< protoK.scale() <<std::endl;
	t_end_int = std::chrono::steady_clock::now();
	std::cout<<"Proto Envs for M and K constructed - T: "<< 
		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K DONE ************************
	
	// Prepare Alternating Least Squares to maximize the overlap
	auto print_elem = [](double d) {
		std::setprecision(std::numeric_limits<long double>::digits10 + 1);
		std::cout<< d << std::endl;
	};

	int altlstsquares_iter = 0;
	bool converged = false;
	std::vector<double> overlaps;

	// ******************************************************************************************** 
	// 	     OPTIMIZE VIA CG                                                                      *
	// ********************************************************************************************

	// intial guess is given by intial eA, eB, eD
	auto cmbX1 = combiner(eA.inds()[0], eA.inds()[1], eA.inds()[2]); 
	auto cmbX2 = combiner(eB.inds()[0], eB.inds()[1], eB.inds()[2], eB.inds()[3]);
	auto cmbX3 = combiner(eD.inds()[0], eD.inds()[1], eD.inds()[2]);

	auto NORMPSI = ( (eRE * (eA * delta(prime(aux[0],pl[1]),prime(aux[1],pl[2])) ) )
		* ( eB * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])) ) ) * eD;
	NORMPSI *= (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	NORMPSI *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	NORMPSI *= prime(conj(eD), AUXLINK, 4);
	if (NORMPSI.r() > 0) { 
		std::cout<< "ERROR: NORMPSI tensor is not a scalar" << std::endl;
		exit(EXIT_FAILURE);
	}
	double fconst = sumels(NORMPSI);

	VecDoub_IO vecX( combinedIndex(cmbX1).m() + 
		combinedIndex(cmbX2).m() + combinedIndex(cmbX3).m() );
	
	Print(eA);
	Print(eB);
	Print(eD);

	eA *= cmbX1;
	eB *= cmbX2;
	eD *= cmbX3;
	for(int i=1; i<= combinedIndex(cmbX1).m(); i++ ) vecX[i-1] = eA.real( combinedIndex(cmbX1)(i) );
	for(int i=1; i<= combinedIndex(cmbX2).m(); i++ ) vecX[combinedIndex(cmbX1).m() + i-1] = eB.real( combinedIndex(cmbX2)(i) );
	for(int i=1; i<= combinedIndex(cmbX3).m(); i++ ) vecX[combinedIndex(cmbX1).m() + combinedIndex(cmbX2).m() + i-1] = eD.real( combinedIndex(cmbX3)(i) );
	eA *= cmbX1;
	eB *= cmbX2;
	eD *= cmbX3;

  	std::cout << "ENTERING CG LOOP" << std::endl;
	FuncCG funcCG(eRE, protoK, cmbX1, cmbX2, cmbX3, aux, pl, fconst);
	Frprmn<FuncCG> frprmn(funcCG, iso_eps, iso_eps, maxAltLstSqrIter);
	vecX = frprmn.minimize(vecX);

	eA *= cmbX1;
	eB *= cmbX2;
	eD *= cmbX3;
	for(int i=1; i<= combinedIndex(cmbX1).m(); i++ ) eA.set( combinedIndex(cmbX1)(i), vecX[i-1]);
	for(int i=1; i<= combinedIndex(cmbX2).m(); i++ ) eB.set( combinedIndex(cmbX2)(i), vecX[combinedIndex(cmbX1).m() + i-1]);
	for(int i=1; i<= combinedIndex(cmbX3).m(); i++ ) eD.set( combinedIndex(cmbX3)(i), vecX[combinedIndex(cmbX1).m() + combinedIndex(cmbX2).m() + i-1]);
	eA *= cmbX1;
	eB *= cmbX2;
	eD *= cmbX3;

	// update on-site tensors of cluster
	cls.sites.at(tn[0]) = QA * eA;
	cls.sites.at(tn[1]) = QB * eB;
	cls.sites.at(tn[2]) = QD * eD;

	// max element of on-site tensors
	std::string diag_maxElem;
	for (int i=0; i<4; i++) {
		m = 0.;
		cls.sites.at(tn[i]).visit(max_m);
		diag_maxElem = diag_maxElem + tn[i] +" : "+ std::to_string(m) +" ";
	}
	std::cout << diag_maxElem << std::endl;

	// normalize updated tensors
	if (otNormType == "PTN3") {
		double nn = std::pow(std::abs(overlaps[overlaps.size()-3]), (1.0/6.0));
		for (int i=0; i<3; i++) cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / nn;
	} else if (otNormType == "PTN4") {
		double nn = std::sqrt(std::abs(overlaps[overlaps.size()-3]));
		double ot_norms_tot = 0.0;
		std::vector<double> ot_norms;
		for (int i=0; i<4; i++) 
			{ ot_norms.push_back(norm(cls.sites.at(tn[i]))); ot_norms_tot += ot_norms.back(); } 
		for (int i=0; i<4; i++) cls.sites.at(tn[i]) = 
			cls.sites.at(tn[i]) / std::pow(nn, (ot_norms[i]/ot_norms_tot));
	} else if (otNormType == "BLE") {
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
        std::cout << tn[i] <<" : "<< std::to_string(m) <<" ";
    }
    std::cout << std::endl;

	// prepare and return diagnostic data
	diag_data.add("alsSweep",altlstsquares_iter);
	diag_data.add("siteMaxElem",diag_maxElem);
	diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
	diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	// auto dist0 = overlaps[overlaps.size()-6] - overlaps[overlaps.size()-5] 
	// 	- overlaps[overlaps.size()-4];
	// auto dist1 = overlaps[overlaps.size()-3] - overlaps[overlaps.size()-2] 
	// 	- overlaps[overlaps.size()-1];
	//diag_data.add("finalDist0",dist0);
	//diag_data.add("finalDist1",dist1);

	//minGapDisc = (minGapDisc < 100.0) ? minGapDisc : -1 ; // whole spectrum taken
	//diag_data.add("minGapDisc",minGapDisc);
	//diag_data.add("minEvKept",minEvKept);
	//diag_data.add("maxEvDisc",maxEvDisc);

	return diag_data;
}

FuncCG::FuncCG(ITensor const& NN, ITensor const& pprotoK, 
	ITensor const& ccmbX1, ITensor const& ccmbX2, ITensor const& ccmbX3, 
	std::array<Index, 4> const& aaux, std::vector<int> const& ppl,
	double ffconst) : N(NN), protoK(pprotoK),
	cmbX1(ccmbX1), cmbX2(ccmbX2), cmbX3(ccmbX3), 
	aux(aaux), pl(ppl), fconst(ffconst) {}

Doub FuncCG::operator() (VecDoub_I &x) {
	auto eA = ITensor(combinedIndex(cmbX1));
	auto eB = ITensor(combinedIndex(cmbX2));
	auto eD = ITensor(combinedIndex(cmbX3));
	for(int i=1; i<= combinedIndex(cmbX1).m(); i++ ) eA.set( combinedIndex(cmbX1)(i), x[i-1]);
	for(int i=1; i<= combinedIndex(cmbX2).m(); i++ ) eB.set( combinedIndex(cmbX2)(i), x[combinedIndex(cmbX1).m() + i-1]);
	for(int i=1; i<= combinedIndex(cmbX3).m(); i++ ) eD.set( combinedIndex(cmbX3)(i), x[combinedIndex(cmbX1).m() + combinedIndex(cmbX2).m() + i-1]);
	eA *= cmbX1;
	eB *= cmbX2;
	eD *= cmbX3;

	auto NORM = ( (N * (eA * delta(prime(aux[0],pl[1]),prime(aux[1],pl[2])) ) )
		* ( eB * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])) ) ) * eD;
	NORM *= (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	NORM *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	NORM *= prime(conj(eD), AUXLINK, 4);

	auto OVERLAP = protoK * (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	OVERLAP *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	OVERLAP *= prime(conj(eD), AUXLINK, 4);
	
	if (NORM.r() > 0 || OVERLAP.r() > 0) std::cout<<"Funcd() NORM or OVERLAP rank > 0"<<std::endl;

	return fconst + sumels(NORM) - 2.0 * sumels(OVERLAP);
}

void FuncCG::df(VecDoub_I &x, VecDoub_O &deriv) {
	auto AUXLINK = aux[0].type();

	auto eA = ITensor(combinedIndex(cmbX1));
	auto eB = ITensor(combinedIndex(cmbX2));
	auto eD = ITensor(combinedIndex(cmbX3));
	for(int i=1; i<= combinedIndex(cmbX1).m(); i++ ) eA.set( combinedIndex(cmbX1)(i), x[i-1]);
	for(int i=1; i<= combinedIndex(cmbX2).m(); i++ ) eB.set( combinedIndex(cmbX2)(i), x[combinedIndex(cmbX1).m() + i-1]);
	for(int i=1; i<= combinedIndex(cmbX3).m(); i++ ) eD.set( combinedIndex(cmbX3)(i), x[combinedIndex(cmbX1).m() + combinedIndex(cmbX2).m() + i-1]);
	eA *= cmbX1;
	eB *= cmbX2;
	eD *= cmbX3;

	auto protoM = ( ( N * (eA * delta( prime(aux[0],pl[1]), prime(aux[1],pl[2]) )) )
		* (eB * delta( prime(aux[1],pl[3]), prime(aux[2],pl[4]) )) ) * eD;

	// compute eA part of gradient
	auto M = protoM * (prime(conj(eD),AUXLINK,4) * delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) ) );
	M *= (prime(conj(eB),AUXLINK,4) *delta(prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4)) );

	auto K = protoK * (prime(conj(eD),AUXLINK,4) * delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) ) );
	K *= (prime(conj(eB),AUXLINK,4) * delta( prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) ) );

	M *= prime(cmbX1,AUXLINK,4);
	K *= prime(cmbX1,AUXLINK,4);
	for(int i=1; i<= combinedIndex(cmbX1).m(); i++ ) deriv[i-1] = M.real( combinedIndex(cmbX1)(i) ) 
		- K.real( combinedIndex(cmbX1)(i) );

	// compute eB part of gradient	   
	M = protoM * (prime(conj(eD),AUXLINK,4) * delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) ) );
	M *= (prime(conj(eA),AUXLINK,4) *delta(prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4)) );

	K = protoK * (prime(conj(eD),AUXLINK,4) * delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) ) );
	K *= (prime(conj(eA),AUXLINK,4) * delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) ) );

	M *= prime(cmbX2,AUXLINK,4);
	K *= prime(cmbX2,AUXLINK,4);
	for(int i=1; i<= combinedIndex(cmbX2).m(); i++ ) deriv[combinedIndex(cmbX1).m() + i-1] = 
		M.real( combinedIndex(cmbX2)(i) ) - K.real( combinedIndex(cmbX2)(i) );

	// compute eD part of gradient	
	M = protoM * (prime(conj(eA),AUXLINK,4) * delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) ) );
	M *= (prime(conj(eB),AUXLINK,4) *delta(prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4)) );

	K = protoK * (prime(conj(eA),AUXLINK,4) * delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) ) );
	K *= (prime(conj(eB),AUXLINK,4) * delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) ) );
	    
	M *= prime(cmbX3,AUXLINK,4);
	K *= prime(cmbX3,AUXLINK,4);
	for(int i=1; i<= combinedIndex(cmbX3).m(); i++ ) deriv[combinedIndex(cmbX1).m() 
		+ combinedIndex(cmbX2).m() + i-1] = M.real( combinedIndex(cmbX3)(i) ) - K.real( combinedIndex(cmbX3)(i) );
}