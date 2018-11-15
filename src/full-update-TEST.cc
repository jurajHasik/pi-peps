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
	std::cout << "psInv: svd_cutoff = " << svd_cutoff << std::endl;

	double lambda = 0.01;
	double lstep  = 0.1;

	// symmetrize
	auto i0 = M.inds()[0];
	auto i1 = M.inds()[1];

	auto Msym = 0.5*(M + ( (conj(M) * delta(i0,prime(i1,1)))
    	*delta(prime(i0,1),i1) ).prime(-1));

	auto i00 = Index("i00",i0.m());
	auto i11 = Index("i11",i1.m());

	Msym = (delta(i00,i0) * Msym ) * delta(i11,i1);

    // check small negative eigenvalues
    Msym = Msym*delta(prime(i00,1), i11);
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
	// for (int idm=0; idm<dM_elems.size(); idm++) {
	// 	if ( dM_elems[idm] > mval*machine_eps ) {
	// 		log_dM_e.push_back(std::log(dM_elems[idm]));
	// 		log_diffs.push_back(log_dM_e[std::max(idm-1,0)]-log_dM_e[idm]);
		
	// 		// candidate for cutoff
	// 		if ((dM_elems[idm]/mval < svd_cutoff) && 
	// 			(std::fabs(log_diffs.back()) > svd_maxLogGap) ) {
	// 			index_cutoff = idm;

	// 			// log diagnostics
	// 			// if ( minGapDisc > std::fabs(log_diffs.back()) ) {
	// 			// 	minGapDisc = std::fabs(log_diffs.back());
	// 			// 	//min_indexCutoff = std::min(min_indexCutoff, index_cutoff);
	// 			// 	minEvKept = dM_elems[std::max(idm-1,0)];
	// 			// 	//maxEvDisc  = dM_elems[idm];
	// 			// }
				
	// 			for (int iidm=index_cutoff; iidm<dM_elems.size(); iidm++) dM_elems[iidm] = 0.0;

	// 			//Dynamic setting of iso_eps
	// 			//iso_eps = std::min(iso_eps, dM_elems[std::max(idm-1,0)]);

	// 			break;
	// 		}
	// 	} else {
	// 		index_cutoff = idm;
	// 		for (int iidm=index_cutoff; iidm<dM_elems.size(); iidm++) dM_elems[iidm] = 0.0;

	// 		// log diagnostics
	// 		// minEvKept  = dM_elems[std::max(idm-1,0)];

	// 		//Dynamic setting of iso_eps
	// 		//iso_eps = std::min(iso_eps, dM_elems[std::max(idm-1,0)]);
			
	// 		break;
	// 	}
	// 	if (idm == dM_elems.size()-1) {
	// 		index_cutoff = -1;

	// 		// log diagnostics
	// 		// minEvKept  = dM_elems[idm];

	// 		//Dynamic setting of iso_eps
	// 		//iso_eps = std::min(iso_eps, dM_elems[idm]);
	// 	}
	// }

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
	int countCTF = 0;
	std::vector<double> elems_regInvDM;
	for (int idm=1; idm<=dM.inds().front().m(); idm++) {
		if (dM.real(dM.inds().front()(idm),dM.inds().back()(idm))/
				dM.real(dM.inds().front()(1),dM.inds().back()(1))  > svd_cutoff) {  
			elems_regInvDM.push_back(msign*1.0/dM.real(dM.inds().front()(idm),
				dM.inds().back()(idm)) );
		} else {
			// elems_regInvDM.push_back(0.0);
			countCTF += 1;
			elems_regInvDM.push_back(1.0);
		}	
	}
	auto regInvDM = diagTensor(elems_regInvDM, dM.inds().front(),dM.inds().back());
	
	if(dbg && (dbgLvl >= 1)) { 
		std::cout<<"regInvDM.scale(): "<< regInvDM.scale() << std::endl; 
		std::cout<<"cutoff/total: "<< countCTF <<" / "<< regInvDM.inds().front().m() << std::endl; 
	}
	
	Msym = (conj(uM)*regInvDM)*prime(uM);
	Msym = Msym*delta(prime(i00,1),i11);
	Msym = (delta(i00,i0) * Msym ) * delta(i11,i1);

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
	auto cg_linesearch_eps = args.getReal("cgLineSearchEps",1.0e-8);
	auto cg_fdistance_eps  = args.getReal("cgFDistanceEps",1.0e-8);
	auto cg_gradientNorm_eps = args.getReal("cgGradientNormEps",1.0e-8);
	auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);
    auto otNormType = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

	if(dbg) {
		std::cout<<"GATE: ";
		for(int i=0; i<=3; i++) {
			std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
		}
		std::cout<< std::endl;

		if (dbgLvl >= 2) {
			std::cout<< uJ1J2;
			PrintData(uJ1J2.H1);
			PrintData(uJ1J2.H2);
			PrintData(uJ1J2.H3);
		}
	}

	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};

	// read off auxiliary and physical indices of the cluster sites
	std::array<Index, 4> aux;
	for (int i=0; i<4; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

	std::array<Index, 4> phys;
	for (int i=0; i<4; i++) phys[i] = cls.phys[ cls.SI.at(tn[i]) ];
	
	std::array<Index, 3> opPI({
		noprime(findtype(uJ1J2.H1, PHYS)),
		noprime(findtype(uJ1J2.H2, PHYS)),
		noprime(findtype(uJ1J2.H3, PHYS)) });
	
	if (dbg) {
		std::cout << "On-site indices:" << std::endl;
		for (int i=0; i<4; i++) {
			std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
		}
	}

	Index iQA, iQD, iQB;
	ITensor QA, eA(prime(aux[0],pl[1]), phys[0]);
	ITensor QD, eD(prime(aux[2],pl[4]), phys[2]);
	ITensor QB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
	
	ITensor eRE;
	ITensor deltaBra, deltaKet;

	{
		t_begin_int = std::chrono::steady_clock::now();

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

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed proto Corners (without on-site tensors): "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
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
	std::string diag_protoEnv;
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
			double nval = 1.0e+16;
			std::vector<double> dM_elems;
			for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
				dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
				if (std::abs(dM_elems.back()) > mval) {
					mval = std::abs(dM_elems.back());
					msign = dM_elems.back()/mval;
				}
				if (std::abs(dM_elems.back()) < nval) nval = std::abs(dM_elems.back());
			}
			if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// Drop negative EV'std
			int countCTF = 0;
			int countNEG = 0;
			for (auto & elem : dM_elems) {
				if (elem < 0.0) {
					if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
					elem = 0.0;
					countNEG += 1;
				} else if (elem < svd_cutoff) {
					countCTF += 1;
					if(dbg && (dbgLvl >= 2)) std::cout<< elem << std::endl;
				} 
			}
			
			diag_protoEnv = std::to_string(mval) + " " +  std::to_string(countCTF) + " " +  
				std::to_string(countNEG) + " " +  std::to_string(dM_elems.size());
			if(dbg && (dbgLvl >= 1)) {
				std::cout<<"REFINED SPECTRUM"<< std::endl;
				std::cout<< std::scientific << "MAX EV: "<< mval << " MIN EV: " << nval <<std::endl;
				std::cout <<"RATIO svd_cutoff/negative/all "<< countCTF <<"/"<< countNEG << "/"
					<< dM_elems.size() << std::endl;
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

	// <psi'|psi'> = <psi|psi>
	auto NORMPSI = ( (eRE * (eA * delta(prime(aux[0],pl[1]),prime(aux[1],pl[2])) ) )
		* ( eB * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])) ) ) * eD;
	NORMPSI *= (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	NORMPSI *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	NORMPSI *= prime(conj(eD), AUXLINK, 4);
	
	// <psi|U^dag U|psi>
	auto NORMUPSI = (( protoK * delta(opPI[0],phys[0]) ) * conj(uJ1J2.H1)) * prime(delta(opPI[0],phys[0]));
	NORMUPSI = (( NORMUPSI * delta(opPI[1],phys[1]) ) * conj(uJ1J2.H2)) * prime(delta(opPI[1],phys[1]));
	NORMUPSI = (( NORMUPSI * delta(opPI[2],phys[2]) ) * conj(uJ1J2.H3)) * prime(delta(opPI[2],phys[2]));
	NORMUPSI.prime(PHYS,-1);
	NORMUPSI *= (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	NORMUPSI *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	NORMUPSI *= prime(conj(eD), AUXLINK, 4);

	// <psi'|U|psi> = <psi|U|psi>
	auto OVERLAP = protoK * (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	OVERLAP *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	OVERLAP *= prime(conj(eD), AUXLINK, 4);
	
	if (NORMPSI.r() > 0 || NORMUPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<
		"NORMPSI or OVERLAP rank > 0"<<std::endl;

	double normUPsi = sumels(NORMUPSI);
	
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

	//double initDist = 2.0 * (1.0 - sumels(OVERLAP) / std::sqrt(normUPsi * sumels(NORMPSI)) );
	double initDist = sumels(NORMPSI) - 2.0 * sumels(OVERLAP) + normUPsi;
	std::cout << "f_init= "<< initDist << std::endl;
  	std::cout << "ENTERING CG LOOP" << std::endl;

	t_begin_int = std::chrono::steady_clock::now();
	//FuncCG funcCG(eRE, protoK, cmbX1, cmbX2, cmbX3, aux, pl, normUPsi, initDist);
	//FrprmnV2<FuncCG> frprmn(funcCG, cg_fdistance_eps, cg_gradientNorm_eps, 
	//	cg_linesearch_eps, maxAltLstSqrIter, initDist);

	FuncCGV2 funcCG(eRE, protoK, cmbX1, cmbX2, cmbX3, aux, pl, normUPsi, initDist);
	FrprmnCG<FuncCGV2> frprmn(funcCG, cg_fdistance_eps, cg_gradientNorm_eps, 
		cg_linesearch_eps, maxAltLstSqrIter, initDist);
	auto locMinData = frprmn.minimize(vecX);
	vecX = std::move(locMinData.final_p);

	t_end_int = std::chrono::steady_clock::now();
	std::cout << "f_final= "<< locMinData.final_f <<" "	
		<<"T: "<< std::chrono::duration_cast<std::chrono::microseconds>
		(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
	std::cout << "d1evalCount: " <<	frprmn.d1evalCount << std::endl;
	std::cout << "log_xmin: " << frprmn.log_xmin << " log_xmax: " << frprmn.log_xmax << std::endl;

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
	diag_data.add("alsSweep", locMinData.iter);
	diag_data.add("siteMaxElem",diag_maxElem);
	diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
	diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	
	std::ostringstream oss;
	//Add double to stream
	oss << std::scientific << " " << locMinData.final_f << " " << locMinData.final_g2
		<< " " << funcCG.evalCount << " " << funcCG.dfCount;

	diag_data.add("locMinDiag", oss.str());
	if (symmProtoEnv) diag_data.add("diag_protoEnv", diag_protoEnv);
	// diag_data.add("locMinDiag", "CG "+ std::to_string(locMinData.final_f)
	// 	+ " " + std::to_string(locMinData.final_g2) );
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
	double ppsiUNorm, double ffinit) : 
	N(NN), protoK(pprotoK),
	cmbX1(ccmbX1), cmbX2(ccmbX2), cmbX3(ccmbX3), aux(aaux), pl(ppl), 
	psiUNorm(ppsiUNorm), finit(ffinit), psiNorm(ppsiUNorm), evalCount(0), dfCount(0) {}

Doub FuncCG::operator() (VecDoub_I &x) {
	evalCount += 1;

	auto eA = ITensor(combinedIndex(cmbX1));
	auto eB = ITensor(combinedIndex(cmbX2));
	auto eD = ITensor(combinedIndex(cmbX3));
	for(int i=1; i<= combinedIndex(cmbX1).m(); i++ ) eA.set( combinedIndex(cmbX1)(i), x[i-1]);
	for(int i=1; i<= combinedIndex(cmbX2).m(); i++ ) eB.set( combinedIndex(cmbX2)(i), x[combinedIndex(cmbX1).m() + i-1]);
	for(int i=1; i<= combinedIndex(cmbX3).m(); i++ ) eD.set( combinedIndex(cmbX3)(i), x[combinedIndex(cmbX1).m() + combinedIndex(cmbX2).m() + i-1]);
	eA *= cmbX1;
	eB *= cmbX2;
	eD *= cmbX3;

	// <psi'|psi'>
	auto NORM = ( (N * (eA * delta(prime(aux[0],pl[1]),prime(aux[1],pl[2])) ) )
		* ( eB * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])) ) ) * eD;
	NORM *= (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	NORM *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	NORM *= prime(conj(eD), AUXLINK, 4);

	// <psi'|U|psi>
	auto OVERLAP = protoK * (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	OVERLAP *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	OVERLAP *= prime(conj(eD), AUXLINK, 4);
	
	if (NORM.r() > 0 || OVERLAP.r() > 0) std::cout<<"Funcd() NORM or OVERLAP rank > 0"<<std::endl;

	psiNorm = sumels(NORM);
	//return 2.0 * ( 1.0 - sumels(OVERLAP)/std::sqrt(psiUNorm * psiNorm) ) / finit;
	return psiNorm - 2.0 * sumels(OVERLAP) + psiUNorm;
}

void FuncCG::df(VecDoub_I &x, VecDoub_O &deriv) {
	dfCount += 1;
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
	for(int i=1; i<= combinedIndex(cmbX1).m(); i++ ) deriv[i-1] = 
		M.real( combinedIndex(cmbX1)(i) )
		- K.real( combinedIndex(cmbX1)(i) );

	// compute eB part of gradient
	M = protoM * (prime(conj(eD),AUXLINK,4) * delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) ) );
	M *= (prime(conj(eA),AUXLINK,4) *delta(prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4)) );

	K = protoK * (prime(conj(eD),AUXLINK,4) * delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) ) );
	K *= (prime(conj(eA),AUXLINK,4) * delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) ) );

	M *= prime(cmbX2,AUXLINK,4);
	K *= prime(cmbX2,AUXLINK,4);
	for(int i=1; i<= combinedIndex(cmbX2).m(); i++ ) deriv[combinedIndex(cmbX1).m() + i-1] = 
		M.real( combinedIndex(cmbX2)(i) )
	    - K.real( combinedIndex(cmbX2)(i) );

	// compute eD part of gradient	
	M = protoM * (prime(conj(eA),AUXLINK,4) * delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) ) );
	M *= (prime(conj(eB),AUXLINK,4) *delta(prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4)) );

	K = protoK * (prime(conj(eA),AUXLINK,4) * delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) ) );
	K *= (prime(conj(eB),AUXLINK,4) * delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) ) );
	    
	M *= prime(cmbX3,AUXLINK,4);
	K *= prime(cmbX3,AUXLINK,4);
	for(int i=1; i<= combinedIndex(cmbX3).m(); i++ ) deriv[combinedIndex(cmbX1).m() 
		+ combinedIndex(cmbX2).m() + i-1] = 
		M.real( combinedIndex(cmbX3)(i) )
	    - K.real( combinedIndex(cmbX3)(i) );
}

//-----------------------------------------------------------------------------
FuncCGV2::FuncCGV2(ITensor const& NN, ITensor const& pprotoK, 
	ITensor const& ccmbX1, ITensor const& ccmbX2, ITensor const& ccmbX3, 
	std::array<Index, 4> const& aaux, std::vector<int> const& ppl,
	double ppsiUNorm, double ffinit) : 
	N(NN), protoK(pprotoK),
	cmbX1(ccmbX1), cmbX2(ccmbX2), cmbX3(ccmbX3), aux(aaux), pl(ppl), 
	psiUNorm(ppsiUNorm), finit(ffinit), psiNorm(ppsiUNorm), evalCount(0), dfCount(0) {}

Doub FuncCGV2::operator() (VecDoub_I &x) {
	evalCount += 1;

	auto eA = ITensor(combinedIndex(cmbX1));
	auto eB = ITensor(combinedIndex(cmbX2));
	auto eD = ITensor(combinedIndex(cmbX3));
	for(int i=1; i<= combinedIndex(cmbX1).m(); i++ ) eA.set( combinedIndex(cmbX1)(i), x[i-1]);
	for(int i=1; i<= combinedIndex(cmbX2).m(); i++ ) eB.set( combinedIndex(cmbX2)(i), x[combinedIndex(cmbX1).m() + i-1]);
	for(int i=1; i<= combinedIndex(cmbX3).m(); i++ ) eD.set( combinedIndex(cmbX3)(i), x[combinedIndex(cmbX1).m() + combinedIndex(cmbX2).m() + i-1]);
	eA *= cmbX1;
	eB *= cmbX2;
	eD *= cmbX3;

	// <psi'|psi'>
	auto NORM = ( (N * (eA * delta(prime(aux[0],pl[1]),prime(aux[1],pl[2])) ) )
		* ( eB * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])) ) ) * eD;
	NORM *= (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	NORM *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	NORM *= prime(conj(eD), AUXLINK, 4);

	// <psi'|U|psi>
	auto OVERLAP = protoK * (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	OVERLAP *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	OVERLAP *= prime(conj(eD), AUXLINK, 4);
	
	if (NORM.r() > 0 || OVERLAP.r() > 0) std::cout<<"Funcd() NORM or OVERLAP rank > 0"<<std::endl;

	psiNorm = sumels(NORM);
	//return 2.0 * ( 1.0 - sumels(OVERLAP)/std::sqrt(psiUNorm * psiNorm) ) / finit;
	return psiNorm - 2.0 * sumels(OVERLAP) + psiUNorm;
}

void FuncCGV2::df(VecDoub_I &x, VecDoub_O &deriv) {
	dfCount += 1;
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
	for(int i=1; i<= combinedIndex(cmbX1).m(); i++ ) deriv[i-1] = 
		M.real( combinedIndex(cmbX1)(i) )
		- K.real( combinedIndex(cmbX1)(i) );

	// compute eB part of gradient
	M = protoM * (prime(conj(eD),AUXLINK,4) * delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) ) );
	M *= (prime(conj(eA),AUXLINK,4) *delta(prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4)) );

	K = protoK * (prime(conj(eD),AUXLINK,4) * delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) ) );
	K *= (prime(conj(eA),AUXLINK,4) * delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) ) );

	M *= prime(cmbX2,AUXLINK,4);
	K *= prime(cmbX2,AUXLINK,4);
	for(int i=1; i<= combinedIndex(cmbX2).m(); i++ ) deriv[combinedIndex(cmbX1).m() + i-1] = 
		M.real( combinedIndex(cmbX2)(i) )
	    - K.real( combinedIndex(cmbX2)(i) );

	// compute eD part of gradient	
	M = protoM * (prime(conj(eA),AUXLINK,4) * delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) ) );
	M *= (prime(conj(eB),AUXLINK,4) *delta(prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4)) );

	K = protoK * (prime(conj(eA),AUXLINK,4) * delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) ) );
	K *= (prime(conj(eB),AUXLINK,4) * delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) ) );
	    
	M *= prime(cmbX3,AUXLINK,4);
	K *= prime(cmbX3,AUXLINK,4);
	for(int i=1; i<= combinedIndex(cmbX3).m(); i++ ) deriv[combinedIndex(cmbX1).m() 
		+ combinedIndex(cmbX2).m() + i-1] = 
		M.real( combinedIndex(cmbX3)(i) )
	    - K.real( combinedIndex(cmbX3)(i) );
}
//-----------------------------------------------------------------------------

// ***** ALS over 3 sites, Non-linear minimization with bare CG
Args fullUpdate_ALS_CG(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
    auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    auto symmProtoEnv = args.getBool("symmetrizeProtoEnv",true);
    auto posDefProtoEnv = args.getBool("positiveDefiniteProtoEnv",true);
    auto iso_eps    = args.getReal("isoEpsilon",1.0e-10);
	auto cg_linesearch_eps = args.getReal("cgLineSearchEps",1.0e-8);
	auto cg_fdistance_eps  = args.getReal("cgFDistanceEps",1.0e-8);
	auto cg_gradientNorm_eps = args.getReal("cgGradientNormEps",1.0e-8);
	auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-14);
	auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);
    auto otNormType = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

	if(dbg) {
		std::cout<<"GATE: ";
		for(int i=0; i<=3; i++) {
			std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
		}
		std::cout<< std::endl;

		if (dbgLvl >= 2) {
			std::cout<< uJ1J2;
			PrintData(uJ1J2.H1);
			PrintData(uJ1J2.H2);
			PrintData(uJ1J2.H3);
		}
	}

	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};

	// read off auxiliary and physical indices of the cluster sites
	std::array<Index, 4> aux;
	for (int i=0; i<4; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

	std::array<Index, 4> phys;
	for (int i=0; i<4; i++) phys[i] = cls.phys[ cls.SI.at(tn[i]) ];
	
	std::array<Index, 3> opPI({
		noprime(findtype(uJ1J2.H1, PHYS)),
		noprime(findtype(uJ1J2.H2, PHYS)),
		noprime(findtype(uJ1J2.H3, PHYS)) });
	
	if (dbg) {
		std::cout << "On-site indices:" << std::endl;
		for (int i=0; i<4; i++) {
			std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
		}
	}

	Index iQA, iQD, iQB;
	ITensor QA, eA(prime(aux[0],pl[1]), phys[0]);
	ITensor QD, eD(prime(aux[2],pl[4]), phys[2]);
	ITensor QB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
	
	ITensor eRE;
	ITensor deltaBra, deltaKet;

	{
		t_begin_int = std::chrono::steady_clock::now();

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

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed proto Corners (without on-site tensors): "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
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
	std::string diag_protoEnv;
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
			double nval = 1.0e+16;
			std::vector<double> dM_elems;
			for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
				dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
				if (std::abs(dM_elems.back()) > mval) {
					mval = std::abs(dM_elems.back());
					msign = dM_elems.back()/mval;
				}
				if (std::abs(dM_elems.back()) < nval) nval = std::abs(dM_elems.back());
			}
			if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// Drop negative EV'std
			int countCTF = 0;
			int countNEG = 0;
			for (auto & elem : dM_elems) {
				if (elem < 0.0) {
					if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
					elem = 0.0;
					countNEG += 1;
				} else if (elem < svd_cutoff) {
					countCTF += 1;
					if(dbg && (dbgLvl >= 2)) std::cout<< elem << std::endl;
				} 
			}
			diag_protoEnv = std::to_string(mval) + " " +  std::to_string(countCTF) + " " +  
				std::to_string(countNEG) + " " +  std::to_string(dM_elems.size());
			if(dbg && (dbgLvl >= 1)) {
				std::cout<<"REFINED SPECTRUM"<< std::endl;
				std::cout<< std::scientific << "MAX EV: "<< mval << " MIN EV: " << nval <<std::endl;
				std::cout <<"RATIO svd_cutoff/negative/all "<< countCTF <<"/"<< countNEG << "/"
					<< dM_elems.size() << std::endl;
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

	ITensor protoK = eRE * (eA * delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])) );
	protoK *= (eB * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])) );
	protoK *= eD;
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
	
	// ******************************************************************************************** 
	// 	     OPTIMIZE VIA CG                                                                      *
	// ********************************************************************************************

	// <psi|U^dag U|psi>
	auto NORMUPSI = (( protoK * delta(opPI[0],phys[0]) ) * conj(uJ1J2.H1)) * prime(delta(opPI[0],phys[0]));
	NORMUPSI = (( NORMUPSI * delta(opPI[1],phys[1]) ) * conj(uJ1J2.H2)) * prime(delta(opPI[1],phys[1]));
	NORMUPSI = (( NORMUPSI * delta(opPI[2],phys[2]) ) * conj(uJ1J2.H3)) * prime(delta(opPI[2],phys[2]));
	NORMUPSI.prime(PHYS,-1);
	NORMUPSI *= (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	NORMUPSI *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	NORMUPSI *= prime(conj(eD), AUXLINK, 4);

	if (NORMUPSI.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;
	double normUPsi = sumels(NORMUPSI);

	auto cmbX1 = combiner(eA.inds()[0], eA.inds()[1], eA.inds()[2]); 
	auto cmbX2 = combiner(eB.inds()[0], eB.inds()[1], eB.inds()[2], eB.inds()[3]);
	auto cmbX3 = combiner(eD.inds()[0], eD.inds()[1], eD.inds()[2]);

	VecDoub_IO vecX1( combinedIndex(cmbX1).m() );
	VecDoub_IO vecX2( combinedIndex(cmbX2).m() );
	
	double normPsi, finit;
	ITensor M, K, NORMPSI, OVERLAP;
	Output_FrprmnV2 locMinData;
	FuncALS_CG funcALS_CG(M, K, cmbX1, normUPsi, finit, normUPsi);
	FrprmnALSCG<FuncALS_CG> frprmn(funcALS_CG, cg_fdistance_eps, cg_gradientNorm_eps, 
		cg_linesearch_eps, maxAltLstSqrIter, 0.0);

  	int altlstsquares_iter = 0;
	bool converged = false;
  	std::vector<double> fdist;
  	std::cout << "ENTERING CG LOOP" << std::endl;
	while (not converged) {
		// Optimizing eA
		// 1) construct matrix M, which is defined as <psi~|psi~> = eA^dag * M * eA

		// BRA
		M = eRE * prime(conj(eD), AUXLINK,4);
		M *= delta(	prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		M *= prime(conj(eB), AUXLINK,4);
		M *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
		if(dbg && (dbgLvl >= 3)) Print(M);

		// KET
		M *= eD;
		M *= delta( prime(aux[2],pl[4]), prime(aux[1],pl[3]) );
		M *= eB;
		M *= delta( prime(aux[1],pl[2]), prime(aux[0],pl[1]) );
		if(dbg && (dbgLvl >= 2)) Print(M);

		// 2) construct vector K, which is defined as <psi~|psi'> = eA^dag * K
		K = protoK * prime(conj(eD), AUXLINK,4);
		K *= delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		K *= prime(conj(eB), AUXLINK,4);
		K *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
		if(dbg && (dbgLvl >= 2)) Print(K);

		// <psi'|psi'>
		NORMPSI = (prime(conj(eA), AUXLINK,4) * M) * eA; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(eA), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
		normPsi = sumels(NORMPSI);
		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;

		fdist.push_back( finit );
		if ( fdist.back() < iso_eps ) { converged = true; break; }
		if ( (fdist.size() > 1) && std::abs(fdist.back() - fdist[fdist.size()-2]) < iso_eps ) { 
			converged = true; break; }

		// ***** SOLVE LINEAR SYSTEM M*eA = K by CG ***************************
		eA *= cmbX1;
		for (int i=1; i<=combinedIndex(cmbX1).m(); i++) vecX1[i-1] = eA.real(combinedIndex(cmbX1)(i));

		std::cout << "f_init= "<< finit << std::endl;
		funcALS_CG.setup(cmbX1, normUPsi, finit, normPsi);
		frprmn.setup(finit);
		//Frprmn<FuncALS_CG> frprmn(funcALS_CG, iso_eps, iso_eps, maxAltLstSqrIter);
		locMinData = frprmn.minimize(vecX1);
		vecX1 = std::move(locMinData.final_p);
		std::cout << "f_final= "<< locMinData.final_f << std::endl;

		ITensor tempEA(combinedIndex(cmbX1)); 
		for (int i=1; i<=combinedIndex(cmbX1).m(); i++) tempEA.set(combinedIndex(cmbX1)(i),vecX1[i-1]);
		eA *= cmbX1;
		tempEA *= cmbX1;

	    // Optimizing eB
		// 1) construct matrix M, which is defined as <psi~|psi~> = eB^dag * M * eB	

		// BRA
		M = eRE * prime(conj(eD), AUXLINK,4);
		M *= delta(	prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		M *= prime(conj(eA), AUXLINK,4);
		M *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
		if(dbg && (dbgLvl >= 3)) Print(M);

		// KET
		M *= eD;
		M *= delta( prime(aux[2],pl[4]), prime(aux[1],pl[3]) );
		M *= eA;
		M *= delta( prime(aux[0],pl[1]), prime(aux[1],pl[2]) );
		if(dbg && (dbgLvl >= 2)) Print(M);

		// 2) construct vector K, which is defined as <psi~|psi'> = eB^dag * K
		K = protoK * prime(conj(eD), AUXLINK,4);
		K *= delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		K *= prime(conj(eA), AUXLINK,4);
		K *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
		if(dbg && (dbgLvl >= 2)) Print(K);

		// <psi'|psi'>
		NORMPSI = (prime(conj(eB), AUXLINK,4) * M) * eB; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(eB), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
		normPsi = sumels(NORMPSI);
		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;

		// ***** SOLVE LINEAR SYSTEM M*eB = K ******************************
		eB *= cmbX2;
		for (int i=1; i<=combinedIndex(cmbX2).m(); i++) vecX2[i-1] = eB.real(combinedIndex(cmbX2)(i));

		std::cout << "f_init= "<< finit << std::endl;
		funcALS_CG.setup(cmbX2, normUPsi, finit, normPsi);
		frprmn.setup(finit);
		//Frprmn<FuncALS_CG> frprmn(funcALS_CG, iso_eps, iso_eps, maxAltLstSqrIter);
		locMinData = frprmn.minimize(vecX2);
		vecX2 = std::move(locMinData.final_p);
		std::cout << "f_final= "<< locMinData.final_f << std::endl;

		ITensor tempEB(combinedIndex(cmbX2));
		for (int i=1; i<=combinedIndex(cmbX2).m(); i++) tempEB.set(combinedIndex(cmbX2)(i),vecX2[i-1]);
		eB *= cmbX2;
		tempEB *= cmbX2;
	    
		// Optimizing eD
		// 1) construct matrix M, which is defined as <psi~|psi~> = eD^dag * M * eD	

		// BRA
		M = eRE * prime(conj(eA), AUXLINK,4);
		M *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
		M *= prime(conj(eB), AUXLINK,4);
		M *= delta(	prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
		if(dbg && (dbgLvl >= 3)) Print(M);

		// KET
		M *= eA;
		M *= delta( prime(aux[0],pl[1]), prime(aux[1],pl[2]) );
		M *= eB;
		M *= delta( prime(aux[1],pl[3]), prime(aux[2],pl[4]) );
		if(dbg && (dbgLvl >= 2)) Print(M);

		// 2) construct vector K, which is defined as <psi~|psi'> = eD^dag * K
		K = protoK * prime(conj(eA), AUXLINK,4);
		K *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
		K *= prime(conj(eB), AUXLINK,4);
		K *= delta(	prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
		if(dbg && (dbgLvl >= 2)) Print(K);

		// <psi'|psi'>
		NORMPSI = (prime(conj(eD), AUXLINK,4) * M) * eD; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(eD), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
		normPsi = sumels(NORMPSI);
		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;

		// ***** SOLVE LINEAR SYSTEM M*eD = K ******************************
		eD *= cmbX3;
		for (int i=1; i<=combinedIndex(cmbX3).m(); i++) vecX1[i-1] = eD.real(combinedIndex(cmbX3)(i));

		std::cout << "f_init= "<< finit << std::endl;
		funcALS_CG.setup(cmbX3, normUPsi, finit, normPsi);
		frprmn.setup(finit);
		//Frprmn<FuncALS_CG> frprmn(funcALS_CG, iso_eps, iso_eps, maxAltLstSqrIter);
		locMinData = frprmn.minimize(vecX1);
		vecX1 = std::move(locMinData.final_p);
		std::cout << "f_final= "<< locMinData.final_f << std::endl;

		ITensor tempED(combinedIndex(cmbX3));
		for (int i=1; i<=combinedIndex(cmbX3).m(); i++) tempED.set(combinedIndex(cmbX3)(i),vecX1[i-1]);
		eD *= cmbX3;
		tempED *= cmbX3;

		eA = tempEA;
		eB = tempEB;
		eD = tempED;

		// TEST CRITERION TO STOP THE ALS procedure
		altlstsquares_iter++;
		if (altlstsquares_iter >= maxAltLstSqrIter) converged = true;
	}

	for (int i=0; i < fdist.size(); i++) std::cout <<"STEP "<< i <<"||psi'>-|psi>|^2: "<< fdist[i] << std::endl;

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
	// if (otNormType == "PTN3") {
	// 	double nn = std::pow(std::abs(overlaps[overlaps.size()-3]), (1.0/6.0));
	// 	for (int i=0; i<3; i++) cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / nn;
	// } else if (otNormType == "PTN4") {
	// 	double nn = std::sqrt(std::abs(overlaps[overlaps.size()-3]));
	// 	double ot_norms_tot = 0.0;
	// 	std::vector<double> ot_norms;
	// 	for (int i=0; i<4; i++) 
	// 		{ ot_norms.push_back(norm(cls.sites.at(tn[i]))); ot_norms_tot += ot_norms.back(); } 
	// 	for (int i=0; i<4; i++) cls.sites.at(tn[i]) = 
	// 		cls.sites.at(tn[i]) / std::pow(nn, (ot_norms[i]/ot_norms_tot));
	// } else if (otNormType == "BLE") {
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
        std::cout << tn[i] <<" : "<< std::to_string(m) <<" ";
    }
    std::cout << std::endl;

	// prepare and return diagnostic data
	diag_data.add("alsSweep",altlstsquares_iter);
	diag_data.add("siteMaxElem",diag_maxElem);
	diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
	diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	
	std::ostringstream oss;
	//Add double to stream
	oss << std::scientific << " " << locMinData.final_f << " " << locMinData.final_g2;

	diag_data.add("locMinDiag", oss.str());
	if (symmProtoEnv) diag_data.add("diag_protoEnv", diag_protoEnv);

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

FuncALS_CG::FuncALS_CG(ITensor const& MM, ITensor & KK, ITensor ccmbKet, 
	double ppsiUNorm, double ffinit, double ppsiNorm) : 
	M(MM), K(KK), cmbKet(ccmbKet), 
	psiUNorm(ppsiUNorm), finit(ffinit), psiNorm(ppsiUNorm) {}

void FuncALS_CG::setup(ITensor ccmbKet, double ppsiUNorm, double ffinit, 
	double ppsiNorm) {
	cmbKet = ccmbKet;
	psiUNorm = ppsiUNorm;
	finit = ffinit;
	psiNorm = ppsiNorm;
}

Doub FuncALS_CG::operator() (VecDoub_I &x) {
	auto tmpX = ITensor(combinedIndex(cmbKet));
	for(int i=1; i<=combinedIndex(cmbKet).m(); i++ ) tmpX.set(combinedIndex(cmbKet)(i), x[i-1]);
	tmpX *= cmbKet;

	auto NORM    = (tmpX * M) * prime(conj(tmpX), AUXLINK, 4);
	auto OVERLAP = prime(conj(tmpX), AUXLINK, 4) *  K;
	
	if (NORM.r() > 0 || OVERLAP.r() > 0) std::cout<<"Funcd() NORM or OVERLAP rank > 0"<<std::endl;

	return psiUNorm + sumels(NORM) - 2.0 * sumels(OVERLAP);
}

void FuncALS_CG::df(VecDoub_I &x, VecDoub_O &deriv) {
	auto tmpX = ITensor(combinedIndex(cmbKet));
	for(int i=1; i<=combinedIndex(cmbKet).m(); i++ ) tmpX.set(combinedIndex(cmbKet)(i), x[i-1]);
	tmpX *= cmbKet;

	tmpX *= M;
	tmpX *= prime(cmbKet,AUXLINK,4);

	K *= prime(cmbKet,AUXLINK,4);
	
	for(int i=1; i<=combinedIndex(cmbKet).m(); i++ ) deriv[i-1] = tmpX.real(combinedIndex(cmbKet)(i)) 
		- K.real(combinedIndex(cmbKet)(i));

	K *= prime(cmbKet,AUXLINK,4);
}


//-----------------------------------------------------------------------------

// ***** PRODCTION ALS over 3 sites, BiCG with ITensor
Args fullUpdate_ALS3S_LSCG_IT(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	LinSysSolver const& ls,
	Args const& args) {
 
	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
    auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    auto symmProtoEnv = args.getBool("symmetrizeProtoEnv",true);
    auto posDefProtoEnv = args.getBool("positiveDefiniteProtoEnv",true);
    auto solver   = args.getString("solver","UNSUPPORTED");
    auto epsdistf = args.getReal("epsdistf",1.0e-8);

 
 //    auto iso_eps    = args.getReal("isoEpsilon",1.0e-10);
	// auto cg_linesearch_eps = args.getReal("cgLineSearchEps",1.0e-8);
	// auto cg_fdistance_eps  = args.getReal("cgFDistanceEps",1.0e-8);
	// auto cg_gradientNorm_eps = args.getReal("cgGradientNormEps",1.0e-8);
	// auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-15);
	// auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);

    auto dynamicEps = args.getBool("dynamicEps",false);   
    auto otNormType = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

	if(dbg) {
		std::cout<<"GATE: ";
		for(int i=0; i<=3; i++) {
			std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
		}
		std::cout<< std::endl;

		if (dbgLvl >= 2) {
			std::cout<< uJ1J2;
			PrintData(uJ1J2.H1);
			PrintData(uJ1J2.H2);
			PrintData(uJ1J2.H3);
		}
	}

	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	double m = 0.;
	auto max_m = [&m](double d) {
		if(std::abs(d) > m) m = std::abs(d);
	};

	// read off auxiliary and physical indices of the cluster sites
	std::array<Index, 4> aux;
	for (int i=0; i<4; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

	std::array<Index, 4> phys;
	for (int i=0; i<4; i++) phys[i] = cls.phys[ cls.SI.at(tn[i]) ];
	
	std::array<Index, 3> opPI({
		noprime(findtype(uJ1J2.H1, PHYS)),
		noprime(findtype(uJ1J2.H2, PHYS)),
		noprime(findtype(uJ1J2.H3, PHYS)) });
	
	if (dbg) {
		std::cout << "On-site indices:" << std::endl;
		for (int i=0; i<4; i++) {
			std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
		}
	}

	Index iQA, iQD, iQB;
	ITensor QA, eA(prime(aux[0],pl[1]), phys[0]);
	ITensor QD, eD(prime(aux[2],pl[4]), phys[2]);
	ITensor QB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
	
	ITensor eRE;
	ITensor deltaBra, deltaKet;

	{
		t_begin_int = std::chrono::steady_clock::now();

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
		
			// Disentangle HSLINK and VSLINK indices into aux-indices of corresponding tensors
			// define combiner
			auto cmb0 = combiner(prime(aux[s],tmp_iToE[0]), prime(aux[s],tmp_iToE[0]+4));
			auto cmb1 = combiner(prime(aux[s],tmp_iToE[1]), prime(aux[s],tmp_iToE[1]+4));
			if(dbg && dbgLvl >= 3) { Print(cmb0); Print(cmb1); }

			pc[s] = (pc[s] * delta(combinedIndex(cmb0), iToE[s][tmp_iToE[0]]) * cmb0) 
				* delta(combinedIndex(cmb1), iToE[s][tmp_iToE[1]]) * cmb1;
		}
		if(dbg) {
			for(int i=0; i<=3; i++) {
				std::cout <<"Site: "<< tn[i] <<" ";
				for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<" ";
				std::cout << std::endl;
			}
		}

		
		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed proto Corners (without on-site tensors): "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
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
		svd(cls.sites.at(tn[0]), eA, tempSA, QA, {"Truncate",false});
		//Index iQA("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
		iQA = Index("auxQA", commonIndex(QA,tempSA).m(), AUXLINK, 0);
		eA = (eA*tempSA) * delta(commonIndex(QA,tempSA), iQA);
		QA *= delta(commonIndex(QA,tempSA), iQA);

		// Prepare corner of A
		// ITensor tempC = pc[0] * getT(QA, iToE[0], (dbg && (dbgLvl >= 3)) );
		ITensor tempC = (pc[0] * QA); 
		tempC *= prime(conj(QA), AUXLINK, 4);

		if(dbg && (dbgLvl >=3)) Print(tempC);

		deltaKet = delta(prime(aux[0],pl[0]), prime(aux[3],pl[7]));
		deltaBra = prime(deltaKet,4);
		tempC = (tempC * deltaBra) * deltaKet;
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE = tempC;

		// Prepare corner of C
		// tempC = pc[3] * getT(cls.sites.at(tn[3]), iToE[3], (dbg && (dbgLvl >= 3)));
		tempC = pc[3] * cls.sites.at(tn[3]);
		tempC *= prime(conj(cls.sites.at(tn[3])), AUXLINK, 4);
		if(dbg && (dbgLvl >=3)) Print(tempC);
		
		deltaKet = delta(prime(aux[3],pl[6]), prime(aux[2],pl[5]));
		deltaBra = prime(deltaKet,4);
		tempC = (tempC * deltaBra) * deltaKet;
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE *= tempC;

		// Decompose D tensor on which the gate is applied
		//ITensor QD, tempSD, eD(prime(aux[2],pl[4]), phys[2]);
		ITensor tempSD;
		svd(cls.sites.at(tn[2]), eD, tempSD, QD, {"Truncate",false});
		//Index iQD("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
		iQD = Index("auxQD", commonIndex(QD,tempSD).m(), AUXLINK, 0);
		eD = (eD*tempSD) * delta(commonIndex(QD,tempSD), iQD);
		QD *= delta(commonIndex(QD,tempSD), iQD);

		// Prepare corner of D
		// tempC = pc[2] * getT(QD, iToE[2], (dbg && (dbgLvl >= 3)));
		tempC = pc[2] * QD;
		tempC *= prime(conj(QD), AUXLINK, 4);
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE *= tempC;

		// Decompose B tensor on which the gate is applied
		//ITensor QB, tempSB, eB(prime(aux[1],pl[2]), prime(aux[1],pl[3]), phys[1]);
		ITensor tempSB;
		svd(cls.sites.at(tn[1]), eB, tempSB, QB, {"Truncate",false});
		//Index iQB("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
		iQB = Index("auxQB", commonIndex(QB,tempSB).m(), AUXLINK, 0);
		eB = (eB*tempSB) * delta(commonIndex(QB,tempSB), iQB);
		QB *= delta(commonIndex(QB,tempSB), iQB);

		// tempC = pc[1] * getT(QB, iToE[1], (dbg && (dbgLvl >= 3)));
		tempC = pc[1] * QB;
		tempC *= prime(conj(QB), AUXLINK, 4);
		if(dbg && (dbgLvl >=3)) Print(tempC);

		eRE *= tempC;

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed reduced Env - T: "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		if(dbg && (dbgLvl >=3)) Print(eRE);
		// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT DONE **********************
	}

	double diag_maxMsymLE, diag_maxMasymLE;
	double diag_maxMsymFN, diag_maxMasymFN;
	std::string diag_protoEnv, diag_protoEnv_descriptor;
	double condNum = 1.0;
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

			// find largest and smallest eigenvalues
			double msign = 1.0;
			double mval = 0.;
			double nval = 1.0e+16;
			std::vector<double> dM_elems;
			for (int idm=1; idm<=D_eRE.inds().front().m(); idm++) {  
				dM_elems.push_back(D_eRE.real(D_eRE.inds().front()(idm),D_eRE.inds().back()(idm)));
				if (std::abs(dM_elems.back()) > mval) {
					mval = std::abs(dM_elems.back());
					msign = dM_elems.back()/mval;
				}
				if (std::abs(dM_elems.back()) < nval) nval = std::abs(dM_elems.back());
			}
			if (msign < 0.0) for (auto & elem : dM_elems) elem = elem*(-1.0);

			// Drop negative EV'std
			int countCTF = 0;
			int countNEG = 0;
			for (auto & elem : dM_elems) {
				if (elem < 0.0) {
					if(dbg && (dbgLvl >= 1)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
					elem = 0.0;
					countNEG += 1;
				//} else if (elem/mval < svd_cutoff) {
				//	countCTF += 1;
					//elem = 0.0;
				//	if(dbg && (dbgLvl >= 2)) std::cout<< elem <<" -> "<< 0.0 << std::endl;
				} 
			}
			
			// estimate codition number
			// condNum = ( std::abs(nval/mval) > svd_cutoff ) ? std::abs(mval/nval) : 1.0/svd_cutoff ;
			condNum = mval / nval;

			std::ostringstream oss;
			oss << std::scientific << mval << " " << condNum << " " << countCTF << " " 
				<< countNEG << " " << dM_elems.size();

			diag_protoEnv_descriptor = "MaxEV condNum EV<CTF EV<0 TotalEV";
			diag_protoEnv = oss.str();
			if(dbg && (dbgLvl >= 1)) {
				std::cout<<"REFINED SPECTRUM"<< std::endl;
				std::cout<< std::scientific << "MAX EV: "<< mval << " MIN EV: " << nval <<std::endl;
				std::cout <<"RATIO svd_cutoff/negative/all "<< countCTF <<"/"<< countNEG << "/"
					<< dM_elems.size() << std::endl;
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

	ITensor protoK = eRE * (eA * delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])) );
	protoK *= (eB * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])) );
	protoK *= eD;
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
	
	// ******************************************************************************************** 
	// 	     OPTIMIZE VIA CG                                                                      *
	// ********************************************************************************************

	// <psi|U^dag U|psi>
	auto NORMUPSI = (( protoK * delta(opPI[0],phys[0]) ) * conj(uJ1J2.H1)) * prime(delta(opPI[0],phys[0]));
	NORMUPSI = (( NORMUPSI * delta(opPI[1],phys[1]) ) * conj(uJ1J2.H2)) * prime(delta(opPI[1],phys[1]));
	NORMUPSI = (( NORMUPSI * delta(opPI[2],phys[2]) ) * conj(uJ1J2.H3)) * prime(delta(opPI[2],phys[2]));
	NORMUPSI.prime(PHYS,-1);
	NORMUPSI *= (prime(conj(eA), AUXLINK, 4) * delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4)) );
	NORMUPSI *= (prime(conj(eB), AUXLINK, 4) * delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4)) );
	NORMUPSI *= prime(conj(eD), AUXLINK, 4);

	if (NORMUPSI.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;
	double normUPsi = sumels(NORMUPSI);

	auto cmbX1 = combiner(eA.inds()[0], eA.inds()[1], eA.inds()[2]); 
	auto cmbX2 = combiner(eB.inds()[0], eB.inds()[1], eB.inds()[2], eB.inds()[3]);
	auto cmbX3 = combiner(eD.inds()[0], eD.inds()[1], eD.inds()[2]);

	double normPsi, finit, finitN;
	double prev_finit = 1.0;
	ITensor M, K, NORMPSI, OVERLAP;
	double ferr;
	int fiter;

  	int altlstsquares_iter = 0;
	bool converged = false;
	std::vector<bool> als_tensor_stop(3, false);
	if (dynamicEps) epsdistf = std::max(epsdistf, condNum * machine_eps);
  	std::vector<double> fdist, fdistN, vec_normPsi;
  	std::cout << "ENTERING ALS LOOP tol: " << epsdistf << " solver: "<< solver << std::endl;
  	t_begin_int = std::chrono::steady_clock::now();
	while (not converged) {
		// Optimizing eA
		// 1) construct matrix M, which is defined as <psi~|psi~> = eA^dag * M * eA

		// BRA
		M = eRE * prime(conj(eD), AUXLINK,4);
		M *= delta(	prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		M *= prime(conj(eB), AUXLINK,4);
		M *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
		if(dbg && (dbgLvl >= 3)) Print(M);

		// KET
		M *= eD;
		M *= delta( prime(aux[2],pl[4]), prime(aux[1],pl[3]) );
		M *= eB;
		M *= delta( prime(aux[1],pl[2]), prime(aux[0],pl[1]) );
		if(dbg && (dbgLvl >= 2)) Print(M);

		// 2) construct vector K, which is defined as <psi~|psi'> = eA^dag * K
		K = protoK * prime(conj(eD), AUXLINK,4);
		K *= delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		K *= prime(conj(eB), AUXLINK,4);
		K *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
		if(dbg && (dbgLvl >= 2)) Print(K);

		// <psi'|psi'>
		NORMPSI = (prime(conj(eA), AUXLINK,4) * M) * eA; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(eA), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;
		normPsi = sumels(NORMPSI);
		finit   = prev_finit = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
		finitN  = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;

		fdist.push_back( finit );
		fdistN.push_back( finitN );
		vec_normPsi.push_back( normPsi );
		//if ( fdist.back() < cg_fdistance_eps ) { converged = true; break; }
		std::cout << "stopCond: " << (fdistN.back() - fdistN[fdistN.size()-2])/fdistN[0] << std::endl;
		if ( (fdistN.size() > 1) && std::abs((fdistN.back() - fdistN[fdistN.size()-2])/fdistN[0]) < epsdistf ) 
			{ converged = true; break; }

		// ***** SOLVE LINEAR SYSTEM M*eA = K by CG ***************************
		auto temp = eA;
		FULSCG_IT fulscg(M,K,eA,cmbX1, combiner(iQA, prime(aux[0],pl[1])), args );
		fulscg.solve(K, eA, fiter, ferr, ls, args);
		
		NORMPSI = (prime(conj(eA), AUXLINK,4) * M) * eA; 
		OVERLAP = prime(conj(eA), AUXLINK,4) * K;
		finitN  = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;
		if ( finitN > fdistN.back() ){//|| fulscg.nres < epsdistf ) { 
			std::cout<<"fdistN increased. Reverting to original tensor"; 
			eA = temp;
			fdist.pop_back();
			fdistN.pop_back();
			vec_normPsi.pop_back();
			als_tensor_stop[0] = true;
		} else {
			als_tensor_stop[0] = false;
		}
		std::cout <<"f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;


	    // Optimizing eB
		// 1) construct matrix M, which is defined as <psi~|psi~> = eB^dag * M * eB	

		// BRA
		M = eRE * prime(conj(eD), AUXLINK,4);
		M *= delta(	prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		M *= prime(conj(eA), AUXLINK,4);
		M *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
		if(dbg && (dbgLvl >= 3)) Print(M);

		// KET
		M *= eD;
		M *= delta( prime(aux[2],pl[4]), prime(aux[1],pl[3]) );
		M *= eA;
		M *= delta( prime(aux[0],pl[1]), prime(aux[1],pl[2]) );
		if(dbg && (dbgLvl >= 2)) Print(M);

		// 2) construct vector K, which is defined as <psi~|psi'> = eB^dag * K
		K = protoK * prime(conj(eD), AUXLINK,4);
		K *= delta( prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		K *= prime(conj(eA), AUXLINK,4);
		K *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
		if(dbg && (dbgLvl >= 2)) Print(K);

		// <psi'|psi'>
		NORMPSI = (prime(conj(eB), AUXLINK,4) * M) * eB; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(eB), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
		normPsi = sumels(NORMPSI);
		prev_finit = finit;
		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
		finitN  = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;

		fdist.push_back( finit );
		fdistN.push_back( finitN );
		vec_normPsi.push_back( normPsi );
		std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
		if ( (fdistN.size() > 1) && std::abs((fdistN.back() - fdistN[fdistN.size()-2])/fdistN[0]) < epsdistf ) 
			{ converged = true; break; }

		// ***** SOLVE LINEAR SYSTEM M*eB = K ******************************
		temp = eB;
		FULSCG_IT fulscgEB(M,K,eB,cmbX2, combiner(iQB, prime(aux[1],pl[2]), prime(aux[1],pl[3])), args );
		fulscgEB.solve(K, eB, fiter, ferr, ls, args);
		
		NORMPSI = (prime(conj(eB), AUXLINK,4) * M) * eB; 
		OVERLAP = prime(conj(eB), AUXLINK,4) * K;
		finitN  = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;
		if ( finitN > fdistN.back() ){//|| fulscgEB.nres < epsdistf ) { 
			std::cout<<"fdistN increased. Reverting to original tensor"; 
			eB = temp;
			fdist.pop_back();
			fdistN.pop_back();
			vec_normPsi.pop_back();
			als_tensor_stop[1] = true;
		} else {
			als_tensor_stop[1] = false;
		}
		std::cout <<"EB f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;

	    
		// Optimizing eD
		// 1) construct matrix M, which is defined as <psi~|psi~> = eD^dag * M * eD	

		// BRA
		M = eRE * prime(conj(eA), AUXLINK,4);
		M *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
		M *= prime(conj(eB), AUXLINK,4);
		M *= delta(	prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
		if(dbg && (dbgLvl >= 3)) Print(M);

		// KET
		M *= eA;
		M *= delta( prime(aux[0],pl[1]), prime(aux[1],pl[2]) );
		M *= eB;
		M *= delta( prime(aux[1],pl[3]), prime(aux[2],pl[4]) );
		if(dbg && (dbgLvl >= 2)) Print(M);

		// 2) construct vector K, which is defined as <psi~|psi'> = eD^dag * K
		K = protoK * prime(conj(eA), AUXLINK,4);
		K *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
		K *= prime(conj(eB), AUXLINK,4);
		K *= delta(	prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
		if(dbg && (dbgLvl >= 2)) Print(K);

		// <psi'|psi'>
		NORMPSI = (prime(conj(eD), AUXLINK,4) * M) * eD; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(eD), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
		normPsi = sumels(NORMPSI);
		prev_finit = finit;
		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
		finitN  = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;

		fdist.push_back( finit );
		fdistN.push_back( finitN );
		vec_normPsi.push_back( normPsi );
		std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
		if ( (fdistN.size() > 1) && std::abs((fdistN.back() - fdistN[fdistN.size()-2])/fdistN[0]) < epsdistf ) 
			{ converged = true; break; }

		// ***** SOLVE LINEAR SYSTEM M*eD = K ******************************
		temp = eD;

		FULSCG_IT fulscgED(M,K,eD,cmbX3, combiner(iQD, prime(aux[2],pl[4])), args );
		fulscgED.solve(K, eD, fiter, ferr, ls, args);

		NORMPSI = (prime(conj(eD), AUXLINK,4) * M) * eD; 
		OVERLAP = prime(conj(eD), AUXLINK,4) * K;
		finitN  = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;
		if ( finitN > fdistN.back() ){//|| fulscgED.nres < epsdistf ) { 
			std::cout<<"fdistN increased. Reverting to original tensor"; 
			eD = temp;
			fdist.pop_back();
			fdistN.pop_back();
			vec_normPsi.pop_back();
			als_tensor_stop[2] = true;
		} else {
			als_tensor_stop[2] = false;
		}
		std::cout <<"f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;

		
		altlstsquares_iter++;
		if (altlstsquares_iter >= maxAltLstSqrIter || 
			(als_tensor_stop[0] && als_tensor_stop[1] && als_tensor_stop[2]) ) converged = true;
	}
	t_end_int = std::chrono::steady_clock::now();

	std::cout <<"STEP f=||psi'>-|psi>|^2 f_normalized <psi'|psi'>" << std::endl;
	for (int i=0; i < fdist.size(); i++) std::cout << i <<" "<< fdist[i] <<" "<< fdistN[i] 
		<<" "<< vec_normPsi[i] << std::endl;

	// update on-site tensors of cluster
	cls.sites.at(tn[0]) = QA * eA;
	cls.sites.at(tn[1]) = QB * eB;
	cls.sites.at(tn[2]) = QD * eD;

	// max element of on-site tensors
	std::string diag_maxElem;
	for (int i=0; i<4; i++) {
		m = 0.;
		cls.sites.at(tn[i]).visit(max_m);
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
	diag_data.add("alsSweep",altlstsquares_iter);

	std::string siteMaxElem_descriptor = "site max_elem site max_elem site max_elem site max_elem";
	diag_data.add("siteMaxElem_descriptor",siteMaxElem_descriptor);
	diag_data.add("siteMaxElem",diag_maxElem);
	diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
	diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	
	std::ostringstream oss;
	oss << std::scientific << fdist[0] <<" "<< fdist.back() <<" " 
		<< fdistN[0] <<" "<< fdistN.back() <<" "<< vec_normPsi[0] <<" "<< vec_normPsi.back() <<" "<<
		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 ;

	std::string logMinDiag_descriptor = "f_init f_final normalizedf_init normalizedf_final norm(psi')_init norm(psi')_final time[s]";
	diag_data.add("locMinDiag_descriptor",logMinDiag_descriptor);
	diag_data.add("locMinDiag", oss.str());
	if (symmProtoEnv) {
		diag_data.add("diag_protoEnv", diag_protoEnv);
		diag_data.add("diag_protoEnv_descriptor", diag_protoEnv_descriptor);
	}

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

FULSCG_IT::FULSCG_IT(ITensor & MM, ITensor & BB, ITensor & AA, 
	ITensor ccmbA, ITensor ccmbKet, Args const& aargs) 
	: M(MM), B(BB), A(AA), cmbA(ccmbA), cmbKet(ccmbKet), args(aargs) {

	dbg         = args.getBool("dbg",false);
	solver      = args.getString("solver","pseudoinverse");
	auto epsreg = args.getReal("epsregularisation",1.0e-8);

	std::vector<Index> iket;
	for (auto const& i : M.inds()) {
		if (i.primeLevel() < 4) iket.emplace_back(i);
	}

	cmbKet = combiner(iket);
	auto cmbBra = prime(cmbKet,4);

	auto i0 = combinedIndex(cmbKet);
	auto i1 = combinedIndex(cmbBra);

	M = (M * cmbKet) * cmbBra;

	// Symmetrize
	//M = 0.5*(M + prime( ((M * delta(i0,prime(i1)) ) * delta(i1,prime(i0)) ), -1) ) ;

	// analyse sparsity
	if(dbg) {
		double min_mag = 1.0e-8;
		int count      = 0;
		int countE7    = 0;
		int countE6	   = 0;
		auto sparseCheck = [&min_mag, &count, &countE7, &countE6](Real r) {
  			double absr = std::fabs(r); 
  			if( absr > min_mag) count += 1;
  			if( absr > min_mag*10.0 ) countE7 += 1;
  			if( absr > min_mag*100.0 ) countE6 += 1;
  		};
		M.visit(sparseCheck);
		std::cout<<"Sparsity e-8: "<< count <<"/"<< i0.m() * i1.m() <<" %: "<< 
			count / ((double) i0.m() * i1.m()) << std::endl; 
		std::cout<<"Sparsity e-7: "<< countE7 <<"/"<< i0.m() * i1.m() <<" %: "<< 
			countE7 / ((double) i0.m() * i1.m()) << std::endl; 
		std::cout<<"Sparsity e-6: "<< countE6 <<"/"<< i0.m() * i1.m() <<" %: "<< 
			countE6 / ((double) i0.m() * i1.m()) << std::endl; 
	}

	// regularize
	for(int i=1; i<= i0.m(); i++) {
		M.set(i0(i),i1(i), M.real(i0(i),i1(i)) + epsreg );
	}

	// Diagonal dominance check
	if(dbg) {
		std::vector<double> diag_elems_mag(i0.m());
		std::vector<double> offdiag_mag_sums(i0.m(),0.0);
		int count_diag_dominant = 0;
		for(int i=1; i<= i0.m(); i++) {
			diag_elems_mag[i-1] = std::abs(M.real(i0(i),i1(i)));
			for(int j=1; j<= i0.m(); j++) {
				offdiag_mag_sums[i-1] += std::abs(M.real(i0(j),i1(i)));
			}
			offdiag_mag_sums[i-1] -= diag_elems_mag[i-1];
			if( diag_elems_mag[i-1] > offdiag_mag_sums[i-1] ) count_diag_dominant += 1; 
		}
		std::cout<<"DiagDominance: "<< count_diag_dominant <<"/"<< i0.m() <<std::endl;
	}

	M = (M * cmbKet) * cmbBra;

	auto RES = M * A - B;
	res  = norm(RES);
	nres = res / norm(B);
	std::cout<<"RES: "<< res <<" N_RES: "<< nres <<" ";
}

void FULSCG_IT::asolve(ITensor const& b, ITensor & x, const Int itrnsp) {
	// Identity "preconditioner"
	if ( findtype(b, AUXLINK).primeLevel() >= 4 )
		x = prime(b, AUXLINK, -4);
	else 
		x = b;

	// Diagonal preconditioner 
	// M = (cmbKet * M) * prime(cmbKet,4);

	// double mval = 0.;
	// std::vector<double> diagMvec(combinedIndex(cmbKet).size());
	// for (int i=0; i<combinedIndex(cmbKet).size(); i++) {
	// 	double elem = M.real(combinedIndex(cmbKet)(i+1), prime(combinedIndex(cmbKet),4)(i+1)); 
	// 	mval = std::max(mval, std::abs(elem));
	// }
	// for (int i=0; i<combinedIndex(cmbKet).size(); i++) { 
	// 	double elem = M.real(combinedIndex(cmbKet)(i+1), prime(combinedIndex(cmbKet),4)(i+1));
	// 	diagMvec[i] = (elem/mval > svd_cutoff) ? 1.0/elem : 0.0 ;
	// }

	// ITensor diagM = diagTensor(diagMvec, combinedIndex(cmbKet), prime(combinedIndex(cmbKet),4));
	
	// x = diagM * (b * cmbKet);
	// x *= prime(cmbKet, combinedIndex(cmbKet), 4);

	// M = (cmbKet * M) * prime(cmbKet,4);
}

void FULSCG_IT::asolve_pinv(ITensor const& b, ITensor & x) {
	M = (cmbKet * M) * prime(cmbKet,4);

	// args:
	//	pseudoInvCutoff (default= 1.0e-15):  
    //  pseudoInvCutoffInsert (default= 0.0): sing.values lower than pseudoInvCutoff
    //		are set to this value
	auto pinvM = pseudoInverse(M, args);

	x = pinvM * (b * prime(cmbKet,AUXLINK,4) );
	x *= prime(cmbKet, combinedIndex(cmbKet), 4);

	M = (cmbKet * M) * prime(cmbKet,4);
}

// void FULSCG_IT::asolve_linsystem(ITensor & x) {
	

// 	std::vector<Index> iket;
// 	for (auto const& i : M.inds()) {
// 		if (i.primeLevel() < 4) iket.emplace_back(i);
// 	}
// 	auto pI = findtype(B.inds(),PHYS);
// 	iket.emplace_back(pI);
		
// 	cmbKet = combiner(iket);

// 	M *= delta(pI,prime(pI,4));
// 	M = (cmbKet * M) * prime(cmbKet,4);

// 	B.prime(PHYS,4);
// 	B *= prime(cmbKet,4);
// 	B.prime(-4);
// 	x *= cmbKet;
// 	x.prime(4);
	

// 	// args:
// 	//	method (default= "LU"): direct solution of linear system by 
// 	//		"LU" decomposition for general M
// 	//		"CHOLESKY" for symmetric/hermitian positive definite M  
// 	linsystem(M, B, x, args);

	
// 	//x.prime(-4);
// 	x *= cmbKet;
// 	B.prime(combinedIndex(cmbKet),4);
// 	B *= prime(cmbKet,4);
// 	B.prime(PHYS,-4);	

// 	M = (cmbKet * M) * prime(cmbKet,4);
	
// 	ITensor temp(pI,prime(pI,4));
// 	temp.fill(0.0);
// 	temp.set(pI(1),prime(pI,4)(1),1.0);
// 	M *= temp;
// }

void FULSCG_IT::atimes(ITensor const& x, ITensor & r, const Int itrnsp) {
	
	if (itrnsp == 0) {
		// direct branch
		r = M * x;
		r.prime(AUXLINK, -4);
	} else {
		// transpose branch
		r = M * prime(x, AUXLINK, 4);
	}
	r.scaleTo(1.0);
}

void FULSCG_IT::solveBiCG(ITensor const& b, ITensor &x, const Int itol, const Doub tol,
	const Int itmax, Int &iter, Doub &err)
{
	Doub ak,akden,bk,bkden=1.0,bknum,bnrm,dxnrm,xnrm,zm1nrm,znrm;
	const Doub EPS=1.0e-14;
	//Int j,n=b.size();
	ITensor p(x.inds()),pp(x.inds()),r(x.inds()),rr(x.inds()),z(x.inds()),zz(x.inds());
	
	p.fill(0.0);
	pp.fill(0.0);
	r.fill(0.0);
	rr.fill(0.0);
	z.fill(0.0);
	zz.fill(0.0);

	iter=0;
	atimes(x,r,0);
	r = prime(b, AUXLINK, -4) - r;
	rr = r;
	atimes(r,rr,0);
	if (itol == 1) {
		bnrm=snrm(b,itol);
		asolve(r,z,0);
	}
	else if (itol == 2) {
		asolve(b,z,0);
		bnrm=snrm(z,itol);
		asolve(r,z,0);
	}
	else if (itol == 3 || itol == 4) {
		asolve(b,z,0);
		bnrm=snrm(z,itol);
		asolve(r,z,0);
		znrm=snrm(z,itol);
	} else throw("illegal itol in linbcg");
	while (iter < itmax) {
		++iter;
		asolve(rr,zz,1);
		//for (bknum=0.0,j=0;j<n;j++) bknum += z[j]*rr[j];
		//bknum = sumelsC(z*rr).real();
		auto tempZRR = z * rr;
		// PrintData(tempZRR);
		// if(  tempZRR.r() > 0) std::cout << "W: rank(tempZPP) > 0" << std::endl;
		// if( isComplex(tempZRR) ) std::cout << "W: z*rr is complex val=" << sumelsC(tempZRR).imag() << std::endl;
		bknum = sumels(tempZRR);
		if (iter == 1) {
			// for (j=0;j<n;j++) {
			// 	p[j]=z[j];
			// 	pp[j]=zz[j];
			// }
			p  = z;
			pp = zz;
		} else {
			bk=bknum/bkden;
			// for (j=0;j<n;j++) {
			// 	p[j]=bk*p[j]+z[j];
			// 	pp[j]=bk*pp[j]+zz[j];
			// }
			p  = bk * p + z;
			pp = bk * pp + zz; 
			p.scaleTo(1.0);
			pp.scaleTo(1.0);
		}
		bkden=bknum;
		atimes(p,z,0);
		//for (akden=0.0,j=0;j<n;j++) akden += z[j]*pp[j];
		//akden = sumelsC(z * pp).real();
		auto tempZPP = z * pp;
		// PrintData(tempZPP);
		// if( tempZPP.r() > 0 ) std::cout << "W: rank(tempC) > 0" << std::endl;
		// if( isComplex(tempZPP) ) {
		// 	std::cout<< "W: z*pp is complex val=" << sumelsC(tempZPP).imag() << std::endl;
		// }
		akden = sumels(tempZPP);
		ak=bknum/akden;
		atimes(pp,zz,1);
		// for (j=0;j<n;j++) {
		// 	x[j] += ak*p[j];
		// 	r[j] -= ak*z[j];
		// 	rr[j] -= ak*zz[j];
		// }
		x += ak * p;
		r -= ak * z;
		rr -= ak * zz;
		x.scaleTo(1.0);
		r.scaleTo(1.0);
		rr.scaleTo(1.0);
		asolve(r,z,0);
		if (itol == 1)
			err=snrm(r,itol)/bnrm;
		else if (itol == 2)
			err=snrm(z,itol)/bnrm;
		else if (itol == 3 || itol == 4) {
			zm1nrm=znrm;
			znrm=snrm(z,itol);
			if (abs(zm1nrm-znrm) > EPS*znrm) {
				dxnrm=abs(ak)*snrm(p,itol);
				err=znrm/abs(zm1nrm-znrm)*dxnrm;
			} else {
				err=znrm/bnrm;
				continue;
			}
			xnrm=snrm(x,itol);
			if (err <= 0.5*xnrm) err /= xnrm;
			else {
				err=znrm/bnrm;
				continue;
			}
		}
		if (err <= tol) break;
	}
}

Doub FULSCG_IT::snrm(ITensor const& sx, const Int itol)
{
	// sx is a on-site type ITensor
	if (itol <= 3) {
		// Euclidean norm
		return norm(sx);
	} else {
		// l_infty norm - max(elements)
		Real max_mag = 0.;
		auto maxComp = [&max_mag](Cplx c) {
  			if(std::fabs(c) > max_mag) max_mag = std::fabs(c);
  		};
  		sx.visit(maxComp);
		return max_mag;
	}
}

void FULSCG_IT::solve(itensor::ITensor const& b, itensor::ITensor & x, Int &iter, Doub &err, 
	Args const& args) {

	if (solver == "pseudoinverse") {
		asolve_pinv(b, x);
	// } else if (solver == "direct") {
	// 	asolve_linsystem(x);
	} else if (solver == "bicg") {
		int itol   = args.getInt("cg_convergence_check",1);
		double tol = args.getReal("cg_gradientNorm_eps",1.0e-8);
		// TODO Better estimation of the default value 
		int itmax  = args.getInt("cg_max_iter",100); 
		solveBiCG(b, x, itol, tol, itmax, iter, err);
	} else {
		std::cout<<"[FULSCG_IT::solve] Unsupported solver: "<< solver <<std::endl;
		exit(EXIT_FAILURE);
	}
}

void FULSCG_IT::solve(itensor::ITensor const& b, itensor::ITensor & x, Int &iter, Doub &err, 
	LinSysSolver const& ls, Args const& args) {

	std::vector<Index> iket;
	for (auto const& i : M.inds()) {
		if (i.primeLevel() < 4) iket.emplace_back(i);
	}
	auto pI = findtype(B,PHYS);
	iket.emplace_back(pI);
	cmbKet = combiner(iket);

	M *= delta(pI,prime(pI,4));
	M = (cmbKet * M) * prime(cmbKet,4);
	
	B.prime(PHYS,4);
	B *= prime(cmbKet,4);
	B.prime(-4);
	
	x *= cmbKet;
	x.prime(4);

	auto bi = B.inds()[0];
	auto xi = x.inds()[0];
	B *= delta(bi,xi);
	x *= delta(xi,bi);
	
	ls.solve(M,B,x,args);

	B *= delta(bi,xi);
	x *= delta(xi,bi);

	x.prime(-4);
	x *= cmbKet;
	
	B.prime(combinedIndex(cmbKet),4);
	B *= prime(cmbKet,4);
	B.prime(PHYS,-4);	
	
	M = (cmbKet * M) * prime(cmbKet,4);

	ITensor temp(pI,prime(pI,4));
	temp.fill(0.0);
	temp.set(pI(1),prime(pI,4)(1),1.0);
	M *= temp;
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------





FULSCG::FULSCG(ITensor & MM, ITensor & BB, ITensor & AA, 
	ITensor ccmbA, ITensor ccmbKet,  double ssvd_cutoff) 
	: M(MM), B(BB), A(AA), cmbA(ccmbA), cmbKet(ccmbKet), svd_cutoff(ssvd_cutoff) {}

void FULSCG::asolve(VecDoub_I &b, VecDoub_O &x, const Int itrnsp) {
	// Identity "preconditioner"
	// for (int i=0; i<b.size(); i++) x[i] = b[i];

	ITensor tmpB(combinedIndex(cmbA));
	for (int i=0; i<b.size(); i++) tmpB.set( combinedIndex(cmbA)(i+1), b[i] );
	tmpB *= prime(cmbA, AUXLINK, 4);
	tmpB *= prime(cmbKet,4);
	//Print(tmpB);

	// Diagonal preconditioner

	// 1) construct diagonal tensor 
	M = (cmbKet * M) * prime(cmbKet,4);
	//Print(M);
	

	std::vector<double> diagMvec(combinedIndex(cmbKet).size());
	for (int i=0; i<combinedIndex(cmbKet).size(); i++) {
		double elem = M.real(combinedIndex(cmbKet)(i+1), prime(combinedIndex(cmbKet),4)(i+1)); 
		diagMvec[i] = (elem > svd_cutoff) ? 1.0 / elem : elem ;
	}

	ITensor diagM = diagTensor(diagMvec, combinedIndex(cmbKet), prime(combinedIndex(cmbKet),4));
	
	auto tmp = diagM * tmpB;
	tmp *= cmbKet;
	tmp *= cmbA;

	M = (cmbKet * M) * prime(cmbKet,4);

	for (int i=0; i<combinedIndex(cmbA).size(); i++) x[i] = tmp.real(combinedIndex(cmbA)(i+1));
}

void FULSCG::atimes(VecDoub_I &x, VecDoub_O &r, const Int itrnsp) {
	// A *= cmbA;
	// for (int i=0; i<x.size(); i++) A.set(combinedIndex(cmbA)(i+1), x[i]);
	// A *= cmbA;
	ITensor tmpA( combinedIndex(cmbA) );
	for (int i=0; i<x.size(); i++) tmpA.set(combinedIndex(cmbA)(i+1), x[i]);
	tmpA *= cmbA;


	ITensor tmp;
	if (itrnsp == 0) {
		// direct branch
		tmp = M * tmpA;
		tmp *= prime(cmbA, AUXLINK, 4);
	} else {
		// transpose branch
		tmp = M * prime(tmpA, AUXLINK, 4);
		tmp *= cmbA;
	}

	for (int i=0; i<x.size(); i++) r[i] = tmp.real(combinedIndex(cmbA)(i+1));
}

//-----------------------------------------------------------------------------

void Linbcg::solve(VecDoub_I &b, VecDoub_IO &x, const Int itol, const Doub tol,
	const Int itmax, Int &iter, Doub &err)
{
	Doub ak,akden,bk,bkden=1.0,bknum,bnrm,dxnrm,xnrm,zm1nrm,znrm;
	const Doub EPS=1.0e-14;
	Int j,n=b.size();
	VecDoub p(n),pp(n),r(n),rr(n),z(n),zz(n);
	iter=0;
	atimes(x,r,0);
	for (j=0;j<n;j++) {
		r[j]=b[j]-r[j];
		rr[j]=r[j];
	}
	//atimes(r,rr,0);
	if (itol == 1) {
		bnrm=snrm(b,itol);
		asolve(r,z,0);
	}
	else if (itol == 2) {
		asolve(b,z,0);
		bnrm=snrm(z,itol);
		asolve(r,z,0);
	}
	else if (itol == 3 || itol == 4) {
		asolve(b,z,0);
		bnrm=snrm(z,itol);
		asolve(r,z,0);
		znrm=snrm(z,itol);
	} else throw("illegal itol in linbcg");
	while (iter < itmax) {
		++iter;
		asolve(rr,zz,1);
		for (bknum=0.0,j=0;j<n;j++) bknum += z[j]*rr[j];
		if (iter == 1) {
			for (j=0;j<n;j++) {
				p[j]=z[j];
				pp[j]=zz[j];
			}
		} else {
			bk=bknum/bkden;
			for (j=0;j<n;j++) {
				p[j]=bk*p[j]+z[j];
				pp[j]=bk*pp[j]+zz[j];
			}
		}
		bkden=bknum;
		atimes(p,z,0);
		for (akden=0.0,j=0;j<n;j++) akden += z[j]*pp[j];
		ak=bknum/akden;
		atimes(pp,zz,1);
		for (j=0;j<n;j++) {
			x[j] += ak*p[j];
			r[j] -= ak*z[j];
			rr[j] -= ak*zz[j];
		}
		asolve(r,z,0);
		if (itol == 1)
			err=snrm(r,itol)/bnrm;
		else if (itol == 2)
			err=snrm(z,itol)/bnrm;
		else if (itol == 3 || itol == 4) {
			zm1nrm=znrm;
			znrm=snrm(z,itol);
			if (abs(zm1nrm-znrm) > EPS*znrm) {
				dxnrm=abs(ak)*snrm(p,itol);
				err=znrm/abs(zm1nrm-znrm)*dxnrm;
			} else {
				err=znrm/bnrm;
				continue;
			}
			xnrm=snrm(x,itol);
			if (err <= 0.5*xnrm) err /= xnrm;
			else {
				err=znrm/bnrm;
				continue;
			}
		}
		if (err <= tol) break;
	}
}

Doub Linbcg::snrm(VecDoub_I &sx, const Int itol)
{
	Int i,isamax,n=sx.size();
	Doub ans;
	if (itol <= 3) {
		ans = 0.0;
		for (i=0;i<n;i++) ans += SQR(sx[i]);
		return sqrt(ans);
	} else {
		isamax=0;
		for (i=0;i<n;i++) {
			if (abs(sx[i]) > abs(sx[isamax])) isamax=i;
		}
		return abs(sx[isamax]);
	}
}

//-----------------------------------------------------------------------------

// ***** ALS over 4 sites while applying a 3 site gate
// Args fullUpdate_CG_IT(MPO_3site const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
// 	std::vector<std::string> tn, std::vector<int> pl,
// 	Args const& args) {
 
// 	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
//     auto dbg = args.getBool("fuDbg",false);
//     auto dbgLvl = args.getInt("fuDbgLevel",0);
//     auto symmProtoEnv = args.getBool("symmetrizeProtoEnv",true);
//     auto posDefProtoEnv = args.getBool("positiveDefiniteProtoEnv",true);
//     auto iso_eps    = args.getReal("isoEpsilon",1.0e-10);
// 	auto cg_linesearch_eps = args.getReal("cgLineSearchEps",1.0e-8);
// 	auto cg_fdistance_eps  = args.getReal("cgFDistanceEps",1.0e-8);
// 	auto cg_gradientNorm_eps = args.getReal("cgGradientNormEps",1.0e-8);
// 	auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-15);
// 	auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);
//     auto otNormType = args.getString("otNormType");

//     double machine_eps = std::numeric_limits<double>::epsilon();
// 	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

// 	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

//     // prepare to hold diagnostic data
//     Args diag_data = Args::global();

// 	if(dbg) {
// 		std::cout<<"GATE: ";
// 		for(int i=0; i<=3; i++) {
// 			std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
// 		}
// 		std::cout<< std::endl;

// 		if (dbgLvl >= 2) {
// 			std::cout<< uJ1J2;
// 			PrintData(uJ1J2.H1);
// 			PrintData(uJ1J2.H2);
// 			PrintData(uJ1J2.H3);
// 		}
// 	}

// 	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
// 	double m = 0.;
// 	auto max_m = [&m](double d) {
// 		if(std::abs(d) > m) m = std::abs(d);
// 	};

// 	// read off auxiliary and physical indices of the cluster sites
// 	std::array<Index, 4> aux;
// 	for (int i=0; i<4; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

// 	std::array<Index, 4> phys;
// 	for (int i=0; i<4; i++) phys[i] = cls.phys[ cls.SI.at(tn[i]) ];
	
// 	std::array<Index, 3> opPI({
// 		noprime(findtype(uJ1J2.H1, PHYS)),
// 		noprime(findtype(uJ1J2.H2, PHYS)),
// 		noprime(findtype(uJ1J2.H3, PHYS)) });
	
// 	if (dbg) {
// 		std::cout << "On-site indices:" << std::endl;
// 		for (int i=0; i<4; i++) {
// 			std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
// 		}
// 	}

// 	ITensor deltaBra, deltaKet;
// 	std::vector<ITensor> pc(4); // holds corners T-C-T
// 	{
// 		t_begin_int = std::chrono::steady_clock::now();

// 		// find integer identifier of on-site tensors within CtmEnv
// 		std::vector<int> si;
// 		for (int i=0; i<=3; i++) {
// 			si.push_back(std::distance(ctmEnv.siteIds.begin(),
// 					std::find(std::begin(ctmEnv.siteIds), 
// 						std::end(ctmEnv.siteIds), tn[i])));
// 		}
// 		if(dbg) {
// 			std::cout << "siteId -> CtmEnv.sites Index" << std::endl;
// 			for (int i = 0; i <=3; ++i) { std::cout << tn[i] <<" -> "<< si[i] << std::endl; }
// 		}

// 		// prepare map from on-site tensor aux-indices to half row/column T
// 		// environment tensors
// 		std::array<const std::vector<ITensor> * const, 4> iToT(
// 			{&ctmEnv.T_L, &ctmEnv.T_U, &ctmEnv.T_R ,&ctmEnv.T_D});

// 		// prepare map from on-site tensor aux-indices pair to half corner T-C-T
// 		// environment tensors
// 		const std::map<int, const std::vector<ITensor> * const > iToC(
// 			{{23, &ctmEnv.C_LU}, {32, &ctmEnv.C_LU},
// 			{21, &ctmEnv.C_LD}, {12, &ctmEnv.C_LD},
// 			{3, &ctmEnv.C_RU}, {30, &ctmEnv.C_RU},
// 			{1, &ctmEnv.C_RD}, {10, &ctmEnv.C_RD}});

// 		// for every on-site tensor point from primeLevel(index) to ENV index
// 		// eg. I_XH or I_XV (with appropriate prime level). 
// 		std::array< std::array<Index, 4>, 4> iToE; // indexToENVIndex => iToE

// 		// Find for site 0 through 3 which are connected to ENV
// 		std::vector<int> plOfSite({0,1,2,3}); // aux-indices (primeLevels) of on-site tensor 

// 		// precompute 4 (proto)corners of 2x2 environment
// 		for (int s=0; s<=3; s++) {
// 			// aux-indices connected to sites
// 			std::vector<int> connected({pl[s*2], pl[s*2+1]});
// 			// set_difference gives aux-indices connected to ENV
// 			std::sort(connected.begin(), connected.end());
// 			std::vector<int> tmp_iToE;
// 			std::set_difference(plOfSite.begin(), plOfSite.end(), 
// 				connected.begin(), connected.end(), 
// 	            std::inserter(tmp_iToE, tmp_iToE.begin())); 
// 			tmp_iToE.push_back(pl[s*2]*10+pl[s*2+1]); // identifier for C ENV tensor
// 			if(dbg) { 
// 				std::cout <<"primeLevels (pl) of indices connected to ENV - site: "
// 					<< tn[s] << std::endl;
// 				std::cout << tmp_iToE[0] <<" "<< tmp_iToE[1] <<" iToC: "<< tmp_iToE[2] << std::endl;
// 			}

// 			// Assign indices by which site is connected to ENV
// 			if( findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK ) ) {
// 				iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], HSLINK );
// 				iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], VSLINK );
// 			} else {
// 				iToE[s][tmp_iToE[0]] = findtype( (*iToT.at(tmp_iToE[0]))[si[s]], VSLINK );
// 				iToE[s][tmp_iToE[1]] = findtype( (*iToT.at(tmp_iToE[1]))[si[s]], HSLINK );
// 			}

// 			pc[s] = (*iToT.at(tmp_iToE[0]))[si[s]]*(*iToC.at(tmp_iToE[2]))[si[s]]*
// 				(*iToT.at(tmp_iToE[1]))[si[s]];
// 			if(dbg) Print(pc[s]);
// 			// set primeLevel of ENV indices between T's to 0 to be ready for contraction
// 			pc[s].noprime(LLINK, ULINK, RLINK, DLINK);
		
// 			// Disentangle HSLINK and VSLINK indices into aux-indices of corresponding tensors
// 			// define combiner
// 			auto cmb0 = combiner(prime(aux[s],tmp_iToE[0]), prime(aux[s],tmp_iToE[0]+4));
// 			auto cmb1 = combiner(prime(aux[s],tmp_iToE[1]), prime(aux[s],tmp_iToE[1]+4));
// 			if(dbg && dbgLvl >= 3) { Print(cmb0); Print(cmb1); }

// 			pc[s] = (pc[s] * delta(combinedIndex(cmb0), iToE[s][tmp_iToE[0]]) * cmb0) 
// 				* delta(combinedIndex(cmb1), iToE[s][tmp_iToE[1]]) * cmb1;
// 		}
// 		if(dbg) {
// 			for(int i=0; i<=3; i++) {
// 				std::cout <<"Site: "<< tn[i] <<" ";
// 				for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<" ";
// 				std::cout << std::endl;
// 			}
// 		}

// 		t_end_int = std::chrono::steady_clock::now();
// 		std::cout<<"Constructed proto Corners (without on-site tensors): "<< 
// 			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
// 		// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 
// 	}

// 	// ***** FORM "PROTO" ENVIRONMENTS FOR K *********************************** 
// 	t_begin_int = std::chrono::steady_clock::now();

// 	// TODO ?

// 	ITensor protoK;
// 	// 	// Variant ONE - precompute U|psi> surrounded in environment
// 	// 	protoK = pc[0] * cls.sites.at(tn[0]);
// 	// 	protoK = protoK * delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])); 
// 	// 	protoK = protoK * ( pc[1] * cls.sites.at(tn[1]) );
// 	// 	protoK = protoK * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])); 
		
// 	// 	protoK = protoK * ( pc[2] * cls.sites.at(tn[2]) );
// 	// 	protoK = protoK * delta(prime(aux[2],pl[5]), prime(aux[3],pl[6]));
// 	// 	protoK = protoK * delta(prime(aux[0],pl[0]), prime(aux[3],pl[7])); 
// 	// 	protoK = protoK * ( pc[3] * cls.sites.at(tn[3]) );
// 	{ 
// 		ITensor temp;
// 		// Variant ONE - precompute U|psi> surrounded in environment
// 		protoK = pc[0] * cls.sites.at(tn[0]);
// 		protoK *= delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])); 
// 		protoK *= ( pc[1] * cls.sites.at(tn[1]) );
// 		protoK *= delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])); 
// 		protoK *= delta(prime(aux[0],pl[0]), prime(aux[3],pl[7])); 
		
// 		temp = pc[2] * cls.sites.at(tn[2]);
// 		temp *= delta(prime(aux[2],pl[5]), prime(aux[3],pl[6]));
// 		temp *= ( pc[3] * cls.sites.at(tn[3]) );
	
// 		protoK *= temp;
// 	}

// 	// multiply by 4-site operator
// 	// protoK = (( protoK * delta(opPI[0],phys[0]) ) * uJ1J2.H1 * delta(prime(opPI[0]),phys[0]));
// 	// protoK = (( protoK * delta(opPI[1],phys[1]) ) * uJ1J2.H2 * delta(prime(opPI[1]),phys[1]));
// 	// protoK = (( protoK * delta(opPI[2],phys[2]) ) * uJ1J2.H3 * delta(prime(opPI[2]),phys[2]));
// 	{
// 		ITensor temp;
// 		temp = uJ1J2.H1 * uJ1J2.H2 * uJ1J2.H3;
// 		temp = ((temp * delta(opPI[0],phys[0])) * delta(opPI[1],phys[1])) * delta(opPI[2],phys[2]);
		
// 		protoK *= temp;
// 		protoK = ((protoK * delta(prime(opPI[0]),phys[0])) * delta(prime(opPI[1]),phys[1]) )
// 			* delta(prime(opPI[2]),phys[2]);
// 	}
// 	// TODO For now, assume the operator does not act on 4th site - just a syntactic filler

// 	if (dbg && dbgLvl >= 3) { Print(protoK); }

// 	std::cout<<"protoK.scale(): "<< protoK.scale() <<std::endl;
// 	t_end_int = std::chrono::steady_clock::now();
// 	std::cout<<"Proto Envs K constructed - T: "<< 
// 		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
// 	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K DONE ************************
	
// 	// ******************************************************************************************** 
// 	// 	     OPTIMIZE VIA CG                                                                      *
// 	// ********************************************************************************************

// 	// <psi|U^dag U|psi> - Variant ONE
// 	auto NORMUPSI = (( protoK * delta(opPI[0],phys[0]) ) * conj(uJ1J2.H1)) * prime(delta(opPI[0],phys[0]));
// 	NORMUPSI = (( NORMUPSI * delta(opPI[1],phys[1]) ) * conj(uJ1J2.H2)) * prime(delta(opPI[1],phys[1]));
// 	NORMUPSI = (( NORMUPSI * delta(opPI[2],phys[2]) ) * conj(uJ1J2.H3)) * prime(delta(opPI[2],phys[2]));
// 	// TODO For now, assume the operator does not act on 4th site - just a syntactic filler
// 	NORMUPSI = NORMUPSI * delta(phys[3], prime(phys[3]));

// 	NORMUPSI.prime(PHYS,-1);
// 	NORMUPSI *= prime(conj(cls.sites.at(tn[0])), AUXLINK, 4);
// 	NORMUPSI *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
// 	NORMUPSI *= prime(conj(cls.sites.at(tn[1])), AUXLINK, 4);
// 	NORMUPSI *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
// 	NORMUPSI *= prime(conj(cls.sites.at(tn[2])), AUXLINK, 4);
// 	NORMUPSI *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
// 	NORMUPSI *= delta(prime(aux[0],pl[0]+4),prime(aux[3],pl[7]+4));
// 	NORMUPSI *= prime(conj(cls.sites.at(tn[3])), AUXLINK, 4);

// 	if (NORMUPSI.r() > 0) std::cout<<"NORMUPSI rank > 0"<<std::endl;
// 	double normUPsi = sumels(NORMUPSI);

// 	ITensor M, K, NORMPSI, OVERLAP;
// 	std::vector<ITensor> pcS(4);
// 	for (int i=0; i<4; i++) {
// 		pcS[i] = (pc[i] * cls.sites.at(tn[i])) * prime(conj(cls.sites.at(tn[i])), AUXLINK,4);
// 	}
// 	double normPsi, finit, finitN;
// 	double prev_finit = 1.0;
// 	double ferr;
// 	int fiter;
// 	int itol = 1;

//   	int altlstsquares_iter = 0;
// 	bool converged = false;
// 	// cg_gradientNorm_eps = std::max(cg_gradientNorm_eps, condNum * machine_eps);
// 	// // cg_fdistance_eps    = std::max(cg_fdistance_eps, condNum * machine_eps);
//   	std::vector<double> fdist, fdistN, vec_normPsi;
//   	std::cout << "ENTERING CG LOOP tol: " << cg_gradientNorm_eps << std::endl;
//   	t_begin_int = std::chrono::steady_clock::now();
// 	while (not converged) {
	
// 		// Optimizing A
// 		// 1) construct matrix M, which is defined as <psi~|psi~> = A^dag * M * A

// 		// M = (pc[1] * cls.sites.at(tn[1])) * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
// 		// M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
// 		// M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
// 		// M = ((M * pc[2]) * cls.sites.at(tn[2])) * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
// 		// M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
// 		// M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
// 		// M = ((M * pc[3]) * cls.sites.at(tn[3])) * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
// 		// M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
// 		// M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
// 		// M *= delta(prime(aux[1],pl[2]),  prime(aux[0],pl[1]));
// 		// M *= delta(prime(aux[1],pl[2]+4),prime(aux[0],pl[1]+4));
// 		// M = M * pc[0];

// 		{
// 			ITensor temp;
	
// 			M = pcS[1];
// 			M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
// 			M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
// 			M *= pcS[2];
// 			M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
// 			M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
// 			M *= delta(prime(aux[1],pl[2]),  prime(aux[0],pl[1]));
// 			M *= delta(prime(aux[1],pl[2]+4),prime(aux[0],pl[1]+4));
			
// 			temp = pcS[3];
// 			temp *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
// 			temp *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
// 			temp *= pc[0];

// 			M *= temp;
// 		}


// 		if(dbg && (dbgLvl >= 3)) Print(M);

// 		//  2) construct vector K, which is defined as <psi~|U|psi> = A^dag * K
// 		// K = protoK * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
// 		// K *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
// 		// K = K * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
// 		// K *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
// 		// K = K * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
// 		// K *= delta(	prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
// 		// K *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
		
// 		{
// 			ITensor temp;

// 			temp = prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
// 			temp *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
// 			temp *= prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
// 			temp *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
// 			temp *= prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
// 			temp *= delta(	prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
// 			temp *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
		
// 			K = protoK * temp;
// 		}

// 		if(dbg && (dbgLvl >= 3)) Print(K);

// 		// <psi'|psi'>
// 		NORMPSI = (prime(conj(cls.sites.at(tn[0])), AUXLINK,4) * M) * cls.sites.at(tn[0]); 
// 		// <psi'|U|psi>
// 		OVERLAP = prime(conj(cls.sites.at(tn[0])), AUXLINK,4) * K;

// 		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
// 		normPsi = sumels(NORMPSI);
// 		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
// 		finitN  = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;
// 		prev_finit = finit;

// 		fdist.push_back( finit );
// 		fdistN.push_back( finitN );
// 		vec_normPsi.push_back( normPsi );
// 		//if ( fdist.back() < cg_fdistance_eps ) { converged = true; break; }
		
// 		if ( (fdist.size() > 1) && std::abs((fdist.back() - fdist[fdist.size()-2])/fdist[0]) < cg_fdistance_eps ) 
// 		{ 
// 			std::cout << "stopCond: " << (fdist.back() - fdist[fdist.size()-2])/fdist[0] << std::endl;
// 			converged = true; break; 
// 		} else {
// 			std::cout << "stopCond: " << (fdist.back() - fdist[fdist.size()-2])/fdist[0] << std::endl;
// 		}

// 		// ***** SOLVE LINEAR SYSTEM M*A = K by CG ***************************
// 		ITensor dummyComb;
// 		int tensorDim = cls.physDim * cls.auxBondDim * cls.auxBondDim * cls.auxBondDim * cls.auxBondDim;
// 		FULSCG_IT fulscg(M, K, cls.sites.at(tn[0]), dummyComb, dummyComb, svd_cutoff );
// 		fulscg.solveIT(K, cls.sites.at(tn[0]), itol, cg_gradientNorm_eps, tensorDim, fiter, ferr);
// 		std::cout <<"A f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
// 		pcS[0] = (pc[0] * cls.sites.at(tn[0])) * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);

// 	    // Optimizing B
// 		// 1) construct matrix M, which is defined as <psi~|psi~> = B^dag * M * B	
		
// 		// M = (pc[2] * cls.sites.at(tn[2])) * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
// 		// M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
// 		// M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
// 		// M = ((M * pc[3]) * cls.sites.at(tn[3])) * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
// 		// M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
// 		// M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
// 		// M = ((M * pc[0]) * cls.sites.at(tn[0])) * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
// 		// M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
// 		// M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
// 		// M *= delta(prime(aux[2],pl[4]),  prime(aux[1],pl[3]));
// 		// M *= delta(prime(aux[2],pl[4]+4),prime(aux[1],pl[3]+4));
// 		// M = M * pc[1];

// 		{
// 			ITensor temp;

// 			M = pcS[2];
// 			M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
// 			M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
// 			M *= pcS[3];
// 			M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
// 			M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
// 			M *= delta(prime(aux[2],pl[4]),  prime(aux[1],pl[3]));
// 			M *= delta(prime(aux[2],pl[4]+4),prime(aux[1],pl[3]+4));
			
// 			temp = pcS[0];
// 			temp *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
// 			temp *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
// 			temp *= pc[1];

// 			M *= temp;
// 		}

// 		if(dbg && (dbgLvl >= 3)) Print(M);

// 		// 2) construct vector K, which is defined as <psi~|psi'> = eB^dag * K
// 		// K = protoK * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
// 		// K *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
// 		// K = K * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
// 		// K *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
// 		// K = K * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
// 		// K *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
// 		// K *= delta(	prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		
// 		{
// 			ITensor temp;

// 			temp = prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
// 			temp *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
// 			temp *= prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
// 			temp *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
// 			temp *= prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
// 			temp *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
// 			temp *= delta(	prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		
// 			K = protoK * temp;
// 		}	

// 		if(dbg && (dbgLvl >= 3)) Print(K);

// 		// <psi'|psi'>
// 		NORMPSI = (prime(conj(cls.sites.at(tn[1])), AUXLINK,4) * M) * cls.sites.at(tn[1]); 
// 		// <psi'|U|psi>
// 		OVERLAP = prime(conj(cls.sites.at(tn[1])), AUXLINK,4) * K;

// 		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
// 		normPsi = sumels(NORMPSI);
// 		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;

// 		// stopCond
// 		if ( (fdist.size() > 1) && std::abs((finit - prev_finit)/fdist[0]) < cg_fdistance_eps ) 
// 		{ 
// 			std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
// 			converged = true; break; 
// 		} else {
// 			std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
// 		}

// 		prev_finit = finit;

// 		// ***** SOLVE LINEAR SYSTEM M*B = K ******************************
// 		ferr = 1.0;
// 		int accfiter = 0;
// 		//while ( (ferr > cg_gradientNorm_eps) && (accfiter < tensorDim * 10 ) )  {
// 			FULSCG_IT fulscgEB(M,K,cls.sites.at(tn[1]),dummyComb, dummyComb, svd_cutoff );
// 			fulscgEB.solveIT(K, cls.sites.at(tn[1]), itol, cg_gradientNorm_eps, tensorDim, fiter, ferr);
// 			accfiter += fiter;
// 			std::cout <<"B f_err= "<< ferr <<" f_iter= "<< accfiter << std::endl;
// 		//}
// 		pcS[1] = (pc[1] * cls.sites.at(tn[1])) * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	    
// 		// Optimizing D
// 		// 1) construct matrix M, which is defined as <psi~|psi~> = D^dag * M * D	

// 		// M = (pc[3] * cls.sites.at(tn[3])) * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
// 		// M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
// 		// M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
// 		// M = ((M * pc[0]) * cls.sites.at(tn[0])) * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
// 		// M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
// 		// M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
// 		// M = ((M * pc[1]) * cls.sites.at(tn[1])) * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
// 		// M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
// 		// M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
// 		// M *= delta(prime(aux[3],pl[6]),  prime(aux[2],pl[5]));
// 		// M *= delta(prime(aux[3],pl[6]+4),prime(aux[2],pl[5]+4));
// 		// M = M * pc[2];

// 		{
// 			ITensor temp;

// 			M = pcS[3];
// 			M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
// 			M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
// 			M *= pcS[0];
// 			M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
// 			M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
// 			M *= delta(prime(aux[3],pl[6]),  prime(aux[2],pl[5]));
// 			M *= delta(prime(aux[3],pl[6]+4),prime(aux[2],pl[5]+4));
			
// 			temp = pcS[1];
// 			temp *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
// 			temp *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
// 			temp *= pc[2];

// 			M *= temp;
// 		}

// 		if(dbg && (dbgLvl >= 3)) Print(M);

// 		// 2) construct vector K, which is defined as <psi~|psi'> = D^dag * K
// 		// K = protoK * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
// 		// K *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
// 		// K = K * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
// 		// K *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
// 		// K = K * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
// 		// K *= delta(	prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
// 		// K *= delta(	prime(aux[3],pl[6]+4), prime(aux[2],pl[5]+4) );
		
// 		{
// 			ITensor temp;

// 			temp = prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
// 			temp *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
// 			temp *= prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
// 			temp *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
// 			temp *= prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
// 			temp *= delta(	prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
// 			temp *= delta(	prime(aux[3],pl[6]+4), prime(aux[2],pl[5]+4) );
		
// 			K = protoK * temp;
// 		}

// 		if(dbg && (dbgLvl >= 3)) Print(K);

// 		// <psi'|psi'>
// 		NORMPSI = (prime(conj(cls.sites.at(tn[2])), AUXLINK,4) * M) * cls.sites.at(tn[2]); 
// 		// <psi'|U|psi>
// 		OVERLAP = prime(conj(cls.sites.at(tn[2])), AUXLINK,4) * K;

// 		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
// 		normPsi = sumels(NORMPSI);
// 		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;

// 		// stopCond
// 		if ( (fdist.size() > 1) && std::abs((finit - prev_finit)/fdist[0]) < cg_fdistance_eps ) 
// 		{ 
// 			std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
// 			converged = true; break; 
// 		} else {
// 			std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
// 		}

// 		prev_finit = finit;

// 		// ***** SOLVE LINEAR SYSTEM M*eD = K ******************************
// 		FULSCG_IT fulscgED(M,K,cls.sites.at(tn[2]),dummyComb, dummyComb, svd_cutoff );
// 		fulscgED.solveIT(K, cls.sites.at(tn[2]), itol, cg_gradientNorm_eps, tensorDim, fiter, ferr);

// 		std::cout <<"D f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
// 		pcS[2] = (pc[2] * cls.sites.at(tn[2])) * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);

// 		// Optimizing C
// 		// 1) construct matrix M, which is defined as <psi~|psi~> = C^dag * M * C	

// 		// M = (pc[0] * cls.sites.at(tn[0])) * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
// 		// M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
// 		// M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
// 		// M = ((M * pc[1]) * cls.sites.at(tn[1])) * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
// 		// M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
// 		// M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
// 		// M = ((M * pc[2]) * cls.sites.at(tn[2])) * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
// 		// M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
// 		// M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
// 		// M *= delta(prime(aux[0],pl[0]),  prime(aux[3],pl[7]));
// 		// M *= delta(prime(aux[0],pl[0]+4),prime(aux[3],pl[7]+4));
// 		// M = M * pc[3];

// 		{
// 			ITensor temp;

// 			M = pcS[0];
// 			M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
// 			M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
// 			M *= pcS[1];
// 			M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
// 			M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
// 			M *= delta(prime(aux[0],pl[0]),  prime(aux[3],pl[7]));
// 			M *= delta(prime(aux[0],pl[0]+4),prime(aux[3],pl[7]+4));
			
// 			temp = pcS[2];
// 			temp *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
// 			temp *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
// 			temp *= pc[3];

// 			M *= temp;
// 		}

// 		if(dbg && (dbgLvl >= 3)) Print(M);

// 		// 2) construct vector K, which is defined as <psi~|psi'> = C^dag * K
// 		// K = protoK * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
// 		// K *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
// 		// K = K * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
// 		// K *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
// 		// K = K * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
// 		// K *= delta(	prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
// 		// K *= delta(	prime(aux[0],pl[0]+4), prime(aux[3],pl[7]+4) );
		
// 		{
// 			ITensor temp;

// 			temp = prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
// 			temp *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
// 			temp *= prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
// 			temp *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
// 			temp *= prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
// 			temp *= delta(	prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
// 			temp *= delta(	prime(aux[0],pl[0]+4), prime(aux[3],pl[7]+4) );
		
// 			K = protoK * temp;
// 		}

// 		if(dbg && (dbgLvl >= 3)) Print(K);

// 		// <psi'|psi'>
// 		NORMPSI = (prime(conj(cls.sites.at(tn[3])), AUXLINK,4) * M) * cls.sites.at(tn[3]); 
// 		// <psi'|U|psi>
// 		OVERLAP = prime(conj(cls.sites.at(tn[3])), AUXLINK,4) * K;

// 		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
// 		normPsi = sumels(NORMPSI);
// 		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;

// 		// stopCond
// 		if ( (fdist.size() > 1) && std::abs((finit - prev_finit)/fdist[0]) < cg_fdistance_eps ) 
// 		{ 
// 			std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
// 			converged = true; break; 
// 		} else {
// 			std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
// 		}

// 		prev_finit = finit;

// 		// ***** SOLVE LINEAR SYSTEM M*C = K ******************************
// 		FULSCG_IT fulscgEC(M,K,cls.sites.at(tn[3]),dummyComb, dummyComb, svd_cutoff );
// 		fulscgEC.solveIT(K, cls.sites.at(tn[3]), itol, cg_gradientNorm_eps, tensorDim, fiter, ferr);

// 		std::cout <<"C f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
// 		pcS[3] = (pc[3] * cls.sites.at(tn[3])) * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);

// 		// TEST CRITERION TO STOP THE ALS procedure
// 		altlstsquares_iter++;
// 		if (altlstsquares_iter >= maxAltLstSqrIter) converged = true;
// 	}
// 	t_end_int = std::chrono::steady_clock::now();

// 	std::cout <<"STEP f=||psi'>-|psi>|^2 f_normalized <psi'|psi'>" << std::endl;
// 	for (int i=0; i < fdist.size(); i++) std::cout << i <<" "<< fdist[i] <<" "<< fdistN[i] 
// 		<<" "<< vec_normPsi[i] << std::endl;

// 	// // update on-site tensors of cluster
// 	// cls.sites.at(tn[0]) = QA * eA;
// 	// cls.sites.at(tn[1]) = QB * eB;
// 	// cls.sites.at(tn[2]) = QD * eD;

// 	// max element of on-site tensors
// 	std::string diag_maxElem;
// 	for (int i=0; i<4; i++) {
// 		m = 0.;
// 		cls.sites.at(tn[i]).visit(max_m);
// 		diag_maxElem = diag_maxElem + tn[i] +" "+ std::to_string(m);
// 		if (i < 3) diag_maxElem +=  " ";
// 	}
// 	std::cout << diag_maxElem << std::endl;

// 	// normalize updated tensors
// 	if (otNormType == "BLE") {
// 		for (int i=0; i<3; i++) {
// 			m = 0.;
// 			cls.sites.at(tn[i]).visit(max_m);
// 			cls.sites.at(tn[i]) = cls.sites.at(tn[i]) / sqrt(m);
// 		}
// 	} else if (otNormType == "BALANCE") {
// 		double iso_tot_mag = 1.0;
// 	    for ( auto & site_e : cls.sites)  {
// 	    	m = 0.;
// 			site_e.second.visit(max_m);
// 	    	site_e.second = site_e.second / m;
// 	    	iso_tot_mag = iso_tot_mag * m;
// 	    }
// 	    for (auto & site_e : cls.sites) {
// 	    	site_e.second = site_e.second * std::pow(iso_tot_mag, (1.0/8.0) );
// 	    }
// 	} else if (otNormType == "NONE") {
// 	} else {
// 		std::cout<<"Unsupported on-site tensor normalisation after full update: "
// 			<< otNormType << std::endl;
// 		exit(EXIT_FAILURE);
// 	}

// 	// max element of on-site tensors after normalization
//     for (int i=0; i<4; i++) {
//         m = 0.;
//         cls.sites.at(tn[i]).visit(max_m);
//         if (i<3) 
//         	std::cout << tn[i] <<" "<< std::to_string(m) << " ";
//     	else 
//     		std::cout << tn[i] <<" "<< std::to_string(m);
//     }
//     std::cout << std::endl;

// 	// prepare and return diagnostic data
// 	diag_data.add("alsSweep",altlstsquares_iter);

// 	std::string siteMaxElem_descriptor = "site max_elem site max_elem site max_elem site max_elem";
// 	diag_data.add("siteMaxElem_descriptor",siteMaxElem_descriptor);
// 	diag_data.add("siteMaxElem",diag_maxElem);
// 	diag_data.add("ratioNonSymLE", -1.0); // ratio of largest elements 
// 	diag_data.add("ratioNonSymFN", -1.0); // ratio of norms
// 	// diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
// 	// diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	
// 	std::ostringstream oss;
// 	oss << std::scientific << fdist[0] <<" "<< fdist.back() <<" " 
// 		<< fdistN[0] <<" "<< fdistN.back() <<" "<< vec_normPsi[0] <<" "<< vec_normPsi.back() <<" "<<
// 		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 ;

// 	std::string logMinDiag_descriptor = "f_init f_final normalizedf_init normalizedf_final norm(psi')_init norm(psi')_final time[s]";
// 	diag_data.add("locMinDiag_descriptor",logMinDiag_descriptor);
// 	diag_data.add("locMinDiag", oss.str());
// 	// if (symmProtoEnv) {
// 	// 	diag_data.add("diag_protoEnv", diag_protoEnv);
// 	// 	diag_data.add("diag_protoEnv_descriptor", diag_protoEnv_descriptor);
// 	// }

// 	// auto dist0 = overlaps[overlaps.size()-6] - overlaps[overlaps.size()-5] 
// 	// 	- overlaps[overlaps.size()-4];
// 	// auto dist1 = overlaps[overlaps.size()-3] - overlaps[overlaps.size()-2] 
// 	// 	- overlaps[overlaps.size()-1];
// 	//diag_data.add("finalDist0",dist0);
// 	//diag_data.add("finalDist1",dist1);

// 	//minGapDisc = (minGapDisc < 100.0) ? minGapDisc : -1 ; // whole spectrum taken
// 	//diag_data.add("minGapDisc",minGapDisc);
// 	//diag_data.add("minEvKept",minEvKept);
// 	//diag_data.add("maxEvDisc",maxEvDisc);

// 	return diag_data;
// }

// ***** PRODUCTION LEVEL ALS over 4 sites applying 4 site operator, BiCG with ITensor
Args fullUpdate_ALS4S_LSCG_IT(OpNS const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
    auto dbg              = args.getBool("fuDbg",false);
    auto dbgLvl           = args.getInt("fuDbgLevel",0);
	auto epsdistf         = args.getReal("epsdistf",1.0e-8);
    auto solver           = args.getString("solver","UNSUPPORTED");
    auto otNormType       = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

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
	std::array<Index, 4> aux;
	for (int i=0; i<4; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

	std::array<Index, 4> phys;
	for (int i=0; i<4; i++) phys[i] = cls.phys[ cls.SI.at(tn[i]) ];
	
	std::array<Index, 4> opPI({
		noprime(uJ1J2.pi[0]),
		noprime(uJ1J2.pi[1]),
		noprime(uJ1J2.pi[2]),
		noprime(uJ1J2.pi[3])
	});
	
	if (dbg) {
		std::cout << "On-site indices:" << std::endl;
		for (int i=0; i<4; i++) {
			std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
		}
	}

	ITensor deltaBra, deltaKet;
	std::vector<ITensor> pc(4); // holds corners T-C-T
	{
		t_begin_int = std::chrono::steady_clock::now();

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

		// precompute 4 (proto)corners of 2x2 environment
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
		
			// Disentangle HSLINK and VSLINK indices into aux-indices of corresponding tensors
			// define combiner
			auto cmb0 = combiner(prime(aux[s],tmp_iToE[0]), prime(aux[s],tmp_iToE[0]+4));
			auto cmb1 = combiner(prime(aux[s],tmp_iToE[1]), prime(aux[s],tmp_iToE[1]+4));
			if(dbg && dbgLvl >= 3) { Print(cmb0); Print(cmb1); }

			pc[s] = (pc[s] * delta(combinedIndex(cmb0), iToE[s][tmp_iToE[0]]) * cmb0) 
				* delta(combinedIndex(cmb1), iToE[s][tmp_iToE[1]]) * cmb1;
		}
		if(dbg) {
			for(int i=0; i<=3; i++) {
				std::cout <<"Site: "<< tn[i] <<" ";
				for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<" ";
				std::cout << std::endl;
			}
		}

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed proto Corners (without on-site tensors): "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 
	}

	// ***** FORM "PROTO" ENVIRONMENTS FOR K *********************************** 
	t_begin_int = std::chrono::steady_clock::now();

	
	// for ( auto const& s : tn) { 
	// 	cls.sites.at(s) /= norm(cls.sites.at(s));
	// 	std::cout<<"norm(Site["<< s <<"]): "<< norm(cls.sites.at(s)) << std::endl;
	// };

	std::vector<ITensor> rX;
	std::vector<ITensor> qX;
	std::vector<Index> iQX;
	for (int i=0; i<4; i++) {
		ITensor X(prime(aux[i],pl[2*i]),prime(aux[i],pl[2*i+1]),phys[i]),Q,SV;

		svd(cls.sites.at(tn[i]),X,SV,Q,{"Truncate",false});

		Index iqx("iqx", commonIndex(SV,Q).m(), AUXLINK, 0);
		X = (X * SV) * delta(commonIndex(SV,Q),iqx);
		Q *= delta(commonIndex(SV,Q),iqx);

		rX.emplace_back(X);
		qX.emplace_back(Q);
		iQX.emplace_back(iqx);
	}

	for (int i=0; i<4; i++) { pc[i] *= qX[i]; pc[i] *= prime(conj(qX[i]),AUXLINK,4); }

	ITensor protoK;
	{ 
		ITensor temp;
		// Variant ONE - precompute U|psi> surrounded in environment
		protoK = pc[0] * rX[0];
		protoK *= delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])); 
		protoK *= ( pc[1] * rX[1] );
		protoK *= delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])); 
		protoK *= delta(prime(aux[0],pl[0]), prime(aux[3],pl[7])); 
		
		temp = (pc[2] * rX[2] );
		temp *= delta(prime(aux[2],pl[5]), prime(aux[3],pl[6]));
		temp *= ( pc[3] * rX[3] );
	
		protoK *= temp;
	}

	{
		ITensor temp = uJ1J2.op;
		for (int i=0; i<4; i++ ) {
			temp *= delta(opPI[i],phys[i]);
			temp *= prime(delta(opPI[i],phys[i]));
		}
		Print(temp);

		protoK *= temp;
		protoK.noprime(PHYS);
	}

	if (dbg && dbgLvl >= 3) { Print(protoK); }

	std::cout<<"protoK.scale(): "<< protoK.scale() <<std::endl;
	t_end_int = std::chrono::steady_clock::now();
	std::cout<<"Proto Envs K constructed - T: "<< 
		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
	// ***** FORM "PROTO" ENVIRONMENTS FOR M and K DONE ************************
	
	// ******************************************************************************************** 
	// 	     OPTIMIZE VIA CG                                                                      *
	// ********************************************************************************************

	// <psi|U^dag U|psi> - Variant ONE
	ITensor NORMUPSI;
	{
		ITensor temp = uJ1J2.op;
		for (int i=0; i<4; i++ ) {
			temp *= delta(opPI[i],phys[i]);
			temp *= prime(delta(opPI[i],phys[i]));
	 	}

	 	NORMUPSI = protoK * conj(temp);
	 	NORMUPSI.noprime(PHYS);
	}
	
	NORMUPSI *= prime(conj(rX[0]), AUXLINK, 4);
	NORMUPSI *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
	NORMUPSI *= prime(conj(rX[1]), AUXLINK, 4);
	NORMUPSI *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
	NORMUPSI *= prime(conj(rX[2]), AUXLINK, 4);
	NORMUPSI *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
	NORMUPSI *= delta(prime(aux[0],pl[0]+4),prime(aux[3],pl[7]+4));
	NORMUPSI *= prime(conj(rX[3]), AUXLINK, 4);

	if (NORMUPSI.r() > 0) std::cout<<"NORMUPSI rank > 0"<<std::endl;
	double normUPsi = sumels(NORMUPSI);

	ITensor M, K, NORMPSI, OVERLAP;
	std::vector<ITensor> pcS(4);
	for (int i=0; i<4; i++) {
		// pcS[i] = (pc[i] * cls.sites.at(tn[i])) * prime(conj(cls.sites.at(tn[i])), AUXLINK,4);
		pcS[i] = (pc[i] * rX[i]) * prime(conj(rX[i]), AUXLINK,4);
	}

	double normPsi, finit, finitN;
	double prev_finit = 1.0;
	double prev_finitN = 1.0;
	double ferr;
	int fiter;

  	int altlstsquares_iter = 0;
	bool converged = false;
  	std::vector<double> fdist, fdistN, vec_normPsi;
  	ITensor dummyComb;
  	std::vector<bool> fdistN_growth(4,false);
  	std::cout << "ENTERING ALS LOOP tol: " << epsdistf << " solver: "<< solver << std::endl;
  	t_begin_int = std::chrono::steady_clock::now();
	while (not converged) {
	
		// ***** Optimizing A *************************************************
		// 1) construct matrix M, which is defined as <psi~|psi~> = A^dag * M * A

		{
			ITensor temp;
	
			M = pcS[1];
			M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
			M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
			M *= pcS[2];
			M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
			M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
			M *= delta(prime(aux[1],pl[2]),  prime(aux[0],pl[1]));
			M *= delta(prime(aux[1],pl[2]+4),prime(aux[0],pl[1]+4));
			
			temp = pcS[3];
			temp *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
			temp *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
			temp *= pc[0];

			M *= temp;
		}

		if(dbg && (dbgLvl >= 3)) Print(M);
		{
			std::vector<Index> iket;
			for (auto const& i : M.inds()) {
				if (i.primeLevel() < 4) iket.emplace_back(i);
			}

			auto cmbKet = combiner(iket);
			auto cmbBra = prime(cmbKet,4);

			auto i0 = combinedIndex(cmbKet);
			auto i1 = combinedIndex(cmbBra);

			M = (M * cmbKet) * cmbBra;
			M *= delta(i1,prime(i0,1));
			ITensor U, D;
			diagHermitian(M, U, D);
			std::vector<double> delems;
			for (int i=1; i<= D.inds().front().m(); i++) {
				delems.emplace_back( D.real(D.inds().front()(i),D.inds().back()(i)) );
				if (delems.back() < 0.0) std::cout<< delems.back() <<" ";
			}

			for (auto & e : delems) {
				e = std::max(e,0.0);
			}
		
			auto reg_D = diagTensor( delems, D.inds().front(), D.inds().back() );

			M = conj(U)*reg_D*prime(U);
			M = ((M * cmbKet) * delta(prime(i0,1),i1) ) * cmbBra;
		}

		// 2) construct vector K, which is defined as <psi~|psi'> = A^dag * K
		{
			ITensor temp;

			temp = prime(conj(rX[1]), AUXLINK,4);
			temp *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
			temp *= prime(conj(rX[2]), AUXLINK,4);
			temp *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
			temp *= prime(conj(rX[3]), AUXLINK,4);
			temp *= delta(	prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
			temp *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
		
			K = protoK * temp;
		}

		if(dbg && (dbgLvl >= 3)) Print(K);

		//rX[0] /= norm(rX[0]);
		// <psi'|psi'>
		NORMPSI = (prime(conj(rX[0]), AUXLINK,4) * M) * rX[0];
		// <psi'|U|psi>
		OVERLAP = prime(conj(rX[0]), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
		normPsi = sumels(NORMPSI);
		finit   = prev_finit  = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
		finitN  = prev_finitN = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;

		fdist.push_back( finit );
		fdistN.push_back( finitN );
		vec_normPsi.push_back( normPsi );
		
		std::cout << "distfnorm: "<< finitN <<" ";
		std::cout << "stopCond: " << (fdist.back() - fdist[fdist.size()-2])/fdist[0] << " "; //std::endl;
		// if ( (fdist.size() > 1) && std::abs((fdist.back() - fdist[fdist.size()-2])/fdist[0]) < epsdistf ) 
		// if ( (fdist.size() > 1) && (std::abs( fdistN.back() - fdistN[fdistN.size()-2] ) < epsdistf || finitN < epsdistf ) ) 
		if ( (fdist.size() > 1) && (std::abs( fdistN.back() - fdistN[fdistN.size()-2] ) < epsdistf ) ) 
			{ converged = true; break; }
		if ( fdistN_growth[0] && fdistN_growth[1] && fdistN_growth[2] && fdistN_growth[3] ) { converged = true; break; }  

		// ***** SOLVE LINEAR SYSTEM M*A = K by CG ***************************
		ITensor tempM = M;
		ITensor tempK = K;
		auto oldSite = rX[0];
		FULSCG_IT fulscg(M, K, rX[0], dummyComb, dummyComb, args );
		fulscg.solve(K, rX[0], fiter, ferr, args);

		// CHECK AFTER
		
		std::cout <<"Norm(init): "<< norm(oldSite) <<" Norm(final): "<< norm(rX[0]) << std::endl;
		// rX[0] /= norm(rX[0]);
		// double tmpnorm = norm(rX[0]);
		// rX[0] = rX[0] * (1.0/tmpnorm);
		// tempK /= tmpnorm;
		// <psi'|psi'>
		NORMPSI = (prime(conj(rX[0]), AUXLINK,4) * tempM) * rX[0]; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(rX[0]), AUXLINK,4) * tempK;
		normPsi = sumels(NORMPSI);
		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;

		finitN = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;
		std::cout << "distfnorm: "<< finitN <<" ";
		std::cout << "stopCondN: "<< finitN - prev_finitN << std::endl;
		std::cout << "Norm drx: "<< norm(oldSite - rX[0]) << std::endl;
		std::cout << "Norm dM: "<< norm(M - tempM) << std::endl;
		std::cout << "Norm dK: "<< norm(K - tempK) << std::endl; 

		std::cout << "RES: "<< norm( tempM  * rX[0] - tempK) << std::endl;
		fdistN_growth[0] = false;
		if ( ((finitN - prev_finitN) > 0.0) || (finitN < 0.0) ) {
			std::vector<Index> iket;
			for (auto const& i : M.inds()) {
				if (i.primeLevel() < 4) iket.emplace_back(i);
			}

			auto cmbKet = combiner(iket);
			auto cmbBra = prime(cmbKet,4);

			auto i0 = combinedIndex(cmbKet);
			auto i1 = combinedIndex(cmbBra);

			M = (M * cmbKet) * cmbBra;
			M *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet),1));
			ITensor U, D;
			diagHermitian(M, U, D);
			std::cout << "spec(M): ";
			for (int i=1; i<= D.inds().front().m(); i++) {
				std::cout << D.real(D.inds().front()(i),D.inds().back()(i)) << " ";
			}
			std::cout << std::endl;
		
			std::cout << "TAKING OLD SITE"<<std::endl;
			rX[0] = oldSite;
			fdist.pop_back();
			fdistN.pop_back();
			vec_normPsi.pop_back();
			fdistN_growth[0] = true;
		}
		// END CHECK AFTER

		std::cout <<"A f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
		pcS[0] = (pc[0] * rX[0]) * prime(conj(rX[0]), AUXLINK,4);

	    // ***** Optimizing B *************************************************
		// 1) construct matrix M, which is defined as <psi~|psi~> = B^dag * M * B		
		{
			ITensor temp;

			M = pcS[2];
			M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
			M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
			M *= pcS[3];
			M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
			M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
			M *= delta(prime(aux[2],pl[4]),  prime(aux[1],pl[3]));
			M *= delta(prime(aux[2],pl[4]+4),prime(aux[1],pl[3]+4));
			
			temp = pcS[0];
			temp *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
			temp *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
			temp *= pc[1];

			M *= temp;
		}

		if(dbg && (dbgLvl >= 3)) Print(M);
		// {
		// 	std::vector<Index> iket;
		// 	for (auto const& i : M.inds()) {
		// 		if (i.primeLevel() < 4) iket.emplace_back(i);
		// 	}

		// 	auto cmbKet = combiner(iket);
		// 	auto cmbBra = prime(cmbKet,4);

		// 	auto i0 = combinedIndex(cmbKet);
		// 	auto i1 = combinedIndex(cmbBra);

		// 	M = (M * cmbKet) * cmbBra;
		// 	M *= delta(i1,prime(i0,1));
		// 	ITensor U, D;
		// 	diagHermitian(M, U, D);
		// 	std::vector<double> delems;
		// 	for (int i=1; i<= D.inds().front().m(); i++) {
		// 		delems.emplace_back( D.real(D.inds().front()(i),D.inds().back()(i)) );
		// 		if (delems.back() < 0.0) std::cout<< delems.back() <<" ";
		// 	}

		// 	for (auto & e : delems) {
		// 		e = std::max(e,0.0);
		// 	}
		
		// 	auto reg_D = diagTensor( delems, D.inds().front(), D.inds().back() );

		// 	M = conj(U)*reg_D*prime(U);
		// 	M = ((M * cmbKet) * delta(prime(i0,1),i1) ) * cmbBra;
		// }

		// 2) construct vector K, which is defined as <psi~|psi'> = B^dag * K		
		{
			ITensor temp;

			temp = prime(conj(rX[2]), AUXLINK,4);
			temp *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
			temp *= prime(conj(rX[3]), AUXLINK,4);
			temp *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
			temp *= prime(conj(rX[0]), AUXLINK,4);
			temp *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
			temp *= delta(	prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		
			K = protoK * temp;
		}	

		if(dbg && (dbgLvl >= 3)) Print(K);

		//rX[1] /= norm(rX[1]);
		// <psi'|psi'>
		NORMPSI = (prime(conj(rX[1]), AUXLINK,4) * M) * rX[1]; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(rX[1]), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
		normPsi = sumels(NORMPSI);
		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
		finitN  = prev_finitN = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;

		fdist.push_back( finit );
		fdistN.push_back( finitN );
		vec_normPsi.push_back( normPsi );

		// stopCond
		std::cout << "distfnorm: "<< finitN <<" ";
		std::cout << "stopCond: " << (fdist.back() - fdist[fdist.size()-2])/fdist[0] << " "; //std::endl;
		// if ( (fdist.size() > 1) && std::abs((fdist.back() - fdist[fdist.size()-2])/fdist[0]) < epsdistf ) 
		//if ( (fdist.size() > 1) && (std::abs( fdistN.back() - fdistN[fdistN.size()-2] ) < epsdistf || finitN < epsdistf ) ) 	
		if ( (fdist.size() > 1) && (std::abs( fdistN.back() - fdistN[fdistN.size()-2] ) < epsdistf ) ) 
			{ converged = true; break; }
		if ( fdistN_growth[0] && fdistN_growth[1] && fdistN_growth[2] && fdistN_growth[3] ) { converged = true; break; }  

		prev_finit = finit;

		// ***** SOLVE LINEAR SYSTEM M*B = K ******************************
		oldSite = rX[1];
		FULSCG_IT fulscgEB(M,K,rX[1],dummyComb, dummyComb, args );
		fulscgEB.solve(K, rX[1], fiter, ferr, args);

		// CHECK AFTER
		//rX[1] /= norm(rX[1]);
		// <psi'|psi'>
		NORMPSI = (prime(conj(rX[1]), AUXLINK,4) * M) * rX[1]; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(rX[1]), AUXLINK,4) * K;
		normPsi = sumels(NORMPSI);

		finitN = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;
		std::cout << "distfnorm: "<< finitN <<" ";
		std::cout << "stopCondN: "<< finitN - prev_finitN << std::endl;
		// std::cout << "RES: "<< norm( M * rX[1] - K ) << std::endl;
		fdistN_growth[1] = false;
		if ( ((finitN - prev_finitN) > 0.0) || (finitN < 0.0) ) {
			std::vector<Index> iket;
			for (auto const& i : M.inds()) {
				if (i.primeLevel() < 4) iket.emplace_back(i);
			}

			auto cmbKet = combiner(iket);
			auto cmbBra = prime(cmbKet,4);

			auto i0 = combinedIndex(cmbKet);
			auto i1 = combinedIndex(cmbBra);

			M = (M * cmbKet) * cmbBra;
			M *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet),1));
			ITensor U, D;
			diagHermitian(M, U, D);
			std::cout << "spec(M): ";
			for (int i=1; i<= D.inds().front().m(); i++) {
				std::cout << D.real(D.inds().front()(i),D.inds().back()(i)) << " ";
			}
			std::cout << std::endl;

			std::cout << "TAKING OLD SITE"<<std::endl;
			rX[1] = oldSite;
			fdist.pop_back();
			fdistN.pop_back();
			vec_normPsi.pop_back();
			fdistN_growth[1] = true;
		}
		// END CHECK AFTER

		std::cout <<"B f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
		pcS[1] = (pc[1] * rX[1]) * prime(conj(rX[1]), AUXLINK,4);
	    
		// Optimizing D
		// 1) construct matrix M, which is defined as <psi~|psi~> = D^dag * M * D	
		{
			ITensor temp;

			M = pcS[3];
			M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
			M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
			M *= pcS[0];
			M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
			M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
			M *= delta(prime(aux[3],pl[6]),  prime(aux[2],pl[5]));
			M *= delta(prime(aux[3],pl[6]+4),prime(aux[2],pl[5]+4));
			
			temp = pcS[1];
			temp *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
			temp *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
			temp *= pc[2];

			M *= temp;
		}

		if(dbg && (dbgLvl >= 3)) Print(M);
		// {
		// 	std::vector<Index> iket;
		// 	for (auto const& i : M.inds()) {
		// 		if (i.primeLevel() < 4) iket.emplace_back(i);
		// 	}

		// 	auto cmbKet = combiner(iket);
		// 	auto cmbBra = prime(cmbKet,4);

		// 	auto i0 = combinedIndex(cmbKet);
		// 	auto i1 = combinedIndex(cmbBra);

		// 	M = (M * cmbKet) * cmbBra;
		// 	M *= delta(i1,prime(i0,1));
		// 	ITensor U, D;
		// 	diagHermitian(M, U, D);
		// 	std::vector<double> delems;
		// 	for (int i=1; i<= D.inds().front().m(); i++) {
		// 		delems.emplace_back( D.real(D.inds().front()(i),D.inds().back()(i)) );
		// 		if (delems.back() < 0.0) std::cout<< delems.back() <<" ";
		// 	}

		// 	for (auto & e : delems) {
		// 		e = std::max(e,0.0);
		// 	}
		
		// 	auto reg_D = diagTensor( delems, D.inds().front(), D.inds().back() );

		// 	M = conj(U)*reg_D*prime(U);
		// 	M = ((M * cmbKet) * delta(prime(i0,1),i1) ) * cmbBra;
		// }

		// 2) construct vector K, which is defined as <psi~|psi'> = D^dag * K
		{
			ITensor temp;

			temp = prime(conj(rX[3]), AUXLINK,4);
			temp *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
			temp *= prime(conj(rX[0]), AUXLINK,4);
			temp *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
			temp *= prime(conj(rX[1]), AUXLINK,4);
			temp *= delta(	prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
			temp *= delta(	prime(aux[3],pl[6]+4), prime(aux[2],pl[5]+4) );
		
			K = protoK * temp;
		}

		if(dbg && (dbgLvl >= 3)) Print(K);

		//rX[2] /= norm(rX[2]);
		// <psi'|psi'>
		NORMPSI = (prime(conj(rX[2]), AUXLINK,4) * M) * rX[2]; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(rX[2]), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
		normPsi = sumels(NORMPSI);
		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
		finitN  = prev_finitN = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;

		fdist.push_back( finit );
		fdistN.push_back( finitN );
		vec_normPsi.push_back( normPsi );

		std::cout << "distfnorm: "<< finitN <<" ";
		std::cout << "stopCond: " << (fdist.back() - fdist[fdist.size()-2])/fdist[0] << " "; //std::endl;
		// if ( (fdist.size() > 1) && std::abs((fdist.back() - fdist[fdist.size()-2])/fdist[0]) < epsdistf ) 
		// if ( (fdist.size() > 1) && (std::abs( fdistN.back() - fdistN[fdistN.size()-2] ) < epsdistf || finitN < epsdistf ) ) 
		if ( (fdist.size() > 1) && (std::abs( fdistN.back() - fdistN[fdistN.size()-2] ) < epsdistf ) ) 
			{ converged = true; break; }
		if ( fdistN_growth[0] && fdistN_growth[1] && fdistN_growth[2] && fdistN_growth[3] ) { converged = true; break; }  

		prev_finit = finit;

		// ***** SOLVE LINEAR SYSTEM M*eD = K ******************************
		oldSite = rX[2];
		FULSCG_IT fulscgED(M,K,rX[2],dummyComb, dummyComb, args );
		fulscgED.solve(K, rX[2], fiter, ferr, args);

		// CHECK AFTER
		//rX[2] /= norm(rX[2]);
		// <psi'|psi'>
		NORMPSI = (prime(conj(rX[2]), AUXLINK,4) * M) * rX[2]; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(rX[2]), AUXLINK,4) * K;
		normPsi = sumels(NORMPSI);

		finitN = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;
		std::cout << "distfnorm: "<< finitN <<" ";
		std::cout << "stopCondN: "<< finitN - prev_finitN << std::endl;
		// std::cout << "RES: "<< norm( M * rX[2] - K ) << std::endl;
		fdistN_growth[2] = false;
		if ( ((finitN - prev_finitN) > 0.0) || (finitN < 0.0) ) {
			std::vector<Index> iket;
			for (auto const& i : M.inds()) {
				if (i.primeLevel() < 4) iket.emplace_back(i);
			}

			auto cmbKet = combiner(iket);
			auto cmbBra = prime(cmbKet,4);

			auto i0 = combinedIndex(cmbKet);
			auto i1 = combinedIndex(cmbBra);

			M = (M * cmbKet) * cmbBra;
			M *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet),1));
			ITensor U, D;
			diagHermitian(M, U, D);
			std::cout << "spec(M): ";
			for (int i=1; i<= D.inds().front().m(); i++) {
				std::cout << D.real(D.inds().front()(i),D.inds().back()(i)) << " ";
			}
			std::cout << std::endl;

			std::cout << "TAKING OLD SITE"<<std::endl;
			rX[2] = oldSite;
			fdist.pop_back();
			fdistN.pop_back();
			vec_normPsi.pop_back();
			fdistN_growth[2] = true;
		}
		// END CHECK AFTER

		std::cout <<"D f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
		pcS[2] = (pc[2] * rX[2]) * prime(conj(rX[2]), AUXLINK,4);

		// ***** Optimizing C *************************************************
		// 1) construct matrix M, which is defined as <psi~|psi~> = C^dag * M * C	
		{
			ITensor temp;

			M = pcS[0];
			M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
			M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
			M *= pcS[1];
			M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
			M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
			M *= delta(prime(aux[0],pl[0]),  prime(aux[3],pl[7]));
			M *= delta(prime(aux[0],pl[0]+4),prime(aux[3],pl[7]+4));
			
			temp = pcS[2];
			temp *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
			temp *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
			temp *= pc[3];

			M *= temp;
		}

		if(dbg && (dbgLvl >= 3)) Print(M);
		// {
		// 	std::vector<Index> iket;
		// 	for (auto const& i : M.inds()) {
		// 		if (i.primeLevel() < 4) iket.emplace_back(i);
		// 	}

		// 	auto cmbKet = combiner(iket);
		// 	auto cmbBra = prime(cmbKet,4);

		// 	auto i0 = combinedIndex(cmbKet);
		// 	auto i1 = combinedIndex(cmbBra);

		// 	M = (M * cmbKet) * cmbBra;
		// 	M *= delta(i1,prime(i0,1));
		// 	ITensor U, D;
		// 	diagHermitian(M, U, D);
		// 	std::vector<double> delems;
		// 	for (int i=1; i<= D.inds().front().m(); i++) {
		// 		delems.emplace_back( D.real(D.inds().front()(i),D.inds().back()(i)) );
		// 		if (delems.back() < 0.0) std::cout<< delems.back() <<" ";
		// 	}

		// 	for (auto & e : delems) {
		// 		e = std::max(e,0.0);
		// 	}
		
		// 	auto reg_D = diagTensor( delems, D.inds().front(), D.inds().back() );

		// 	M = conj(U)*reg_D*prime(U);
		// 	M = ((M * cmbKet) * delta(prime(i0,1),i1) ) * cmbBra;
		// }

		// 2) construct vector K, which is defined as <psi~|psi'> = C^dag * K
		{
			ITensor temp;

			temp = prime(conj(rX[0]), AUXLINK,4);
			temp *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
			temp *= prime(conj(rX[1]), AUXLINK,4);
			temp *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
			temp *= prime(conj(rX[2]), AUXLINK,4);
			temp *= delta(	prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
			temp *= delta(	prime(aux[0],pl[0]+4), prime(aux[3],pl[7]+4) );
		
			K = protoK * temp;
		}

		if(dbg && (dbgLvl >= 3)) Print(K);

		//rX[3] /= norm(rX[3]);
		// <psi'|psi'>
		NORMPSI = (prime(conj(rX[3]), AUXLINK,4) * M) * rX[3]; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(rX[3]), AUXLINK,4) * K;

		if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
		normPsi = sumels(NORMPSI);
		finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
		finitN  = prev_finitN = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;

		fdist.push_back( finit );
		fdistN.push_back( finitN );
		vec_normPsi.push_back( normPsi );

		std::cout << "distfnorm: "<< finitN <<" ";
		std::cout << "stopCond: " << (fdist.back() - fdist[fdist.size()-2])/fdist[0] << " "; //std::endl;
		// if ( (fdist.size() > 1) && std::abs((fdist.back() - fdist[fdist.size()-2])/fdist[0]) < epsdistf ) 
		// if ( (fdist.size() > 1) && (std::abs( fdistN.back() - fdistN[fdistN.size()-2] ) < epsdistf || finitN < epsdistf ) ) 		
		if ( (fdist.size() > 1) && (std::abs( fdistN.back() - fdistN[fdistN.size()-2] ) < epsdistf ) ) 		
			{ converged = true; break; }
		if ( fdistN_growth[0] && fdistN_growth[1] && fdistN_growth[2] && fdistN_growth[3] ) { converged = true; break; }  
		
		prev_finit = finit;

		// ***** SOLVE LINEAR SYSTEM M*C = K ******************************
		oldSite = rX[3];
		FULSCG_IT fulscgEC(M,K,rX[3],dummyComb, dummyComb, args );
		fulscgEC.solve(K, rX[3], fiter, ferr, args);
		
		// CHECK AFTER
		// rX[3] /= norm(rX[3]);
		// <psi'|psi'>
		NORMPSI = (prime(conj(rX[3]), AUXLINK,4) * M) * rX[3]; 
		// <psi'|U|psi>
		OVERLAP = prime(conj(rX[3]), AUXLINK,4) * K;
		normPsi = sumels(NORMPSI);

		finitN = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;
		std::cout << "distfnorm: "<< finitN <<" ";
		std::cout << "stopCondN: "<< finitN - prev_finitN << std::endl;
		// std::cout << "RES: "<< norm( M * rX[3] - K ) << std::endl;
		fdistN_growth[3] = false;
		if ( ((finitN - prev_finitN) > 0.0) || (finitN < 0.0) ) {
			std::vector<Index> iket;
			for (auto const& i : M.inds()) {
				if (i.primeLevel() < 4) iket.emplace_back(i);
			}

			auto cmbKet = combiner(iket);
			auto cmbBra = prime(cmbKet,4);

			auto i0 = combinedIndex(cmbKet);
			auto i1 = combinedIndex(cmbBra);

			M = (M * cmbKet) * cmbBra;
			M *= delta(combinedIndex(cmbBra),prime(combinedIndex(cmbKet),1));
			ITensor U, D;
			diagHermitian(M, U, D);
			std::cout << "spec(M): ";
			for (int i=1; i<= D.inds().front().m(); i++) {
				std::cout << D.real(D.inds().front()(i),D.inds().back()(i)) << " ";
			}
			std::cout << std::endl;

			std::cout << "TAKING OLD SITE"<<std::endl;
			rX[3] = oldSite;
			fdist.pop_back();
			fdistN.pop_back();
			vec_normPsi.pop_back();
			fdistN_growth[3] = true;
		}
		// END CHECK AFTER

		std::cout <<"C f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
		pcS[3] = (pc[3] * rX[3]) * prime(conj(rX[3]), AUXLINK,4);

		// TEST CRITERION TO STOP THE ALS procedure
		altlstsquares_iter++;
		if (altlstsquares_iter >= maxAltLstSqrIter) converged = true;
	}
	t_end_int = std::chrono::steady_clock::now();

	std::cout <<"STEP f=||psi'>-|psi>|^2 f_normalized <psi'|psi'>" << std::endl;
	for (int i=0; i < fdist.size(); i++) std::cout << i <<" "<< fdist[i] <<" "<< fdistN[i] 
		<<" "<< vec_normPsi[i] << std::endl;

	for (int i=0; i<4; i++) cls.sites.at(tn[i]) = rX[i] * qX[i];

	// max element of on-site tensors
	std::string diag_maxElem;
	for (int i=0; i<4; i++) {
		m = 0.;
		cls.sites.at(tn[i]).visit(max_m);
		diag_maxElem = diag_maxElem + tn[i] +" "+ std::to_string(m);
		if (i < 3) diag_maxElem +=  " ";
	}
	std::cout << diag_maxElem << std::endl;

	// normalize updated tensors
	if (otNormType == "BLE") {
		for (int i=0; i<4; i++) {
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
	diag_data.add("alsSweep",altlstsquares_iter);

	std::string siteMaxElem_descriptor = "site max_elem site max_elem site max_elem site max_elem";
	diag_data.add("siteMaxElem_descriptor",siteMaxElem_descriptor);
	diag_data.add("siteMaxElem",diag_maxElem);
	diag_data.add("ratioNonSymLE", -1.0); // ratio of largest elements 
	diag_data.add("ratioNonSymFN", -1.0); // ratio of norms
	// diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
	// diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	
	std::ostringstream oss;
	oss << std::scientific << fdist[0] <<" "<< fdist.back() <<" " 
		<< fdistN[0] <<" "<< fdistN.back() <<" "<< vec_normPsi[0] <<" "<< vec_normPsi.back() <<" "<<
		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 ;

	std::string logMinDiag_descriptor = "f_init f_final normalizedf_init normalizedf_final norm(psi')_init norm(psi')_final time[s]";
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
	//diag_data.add("finalDist0",dist0);
	//diag_data.add("finalDist1",dist1);

	//minGapDisc = (minGapDisc < 100.0) ? minGapDisc : -1 ; // whole spectrum taken
	//diag_data.add("minGapDisc",minGapDisc);
	//diag_data.add("minEvKept",minEvKept);
	//diag_data.add("maxEvDisc",maxEvDisc);

	return diag_data;
}

//-----------------------------------------------------------------------------
Args fullUpdate_CG_full4S(OpNS const& uJ1J2, Cluster & cls, CtmEnv const& ctmEnv,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
	auto maxAltLstSqrIter = args.getInt("maxAltLstSqrIter",50);
    auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    auto symmProtoEnv = args.getBool("symmetrizeProtoEnv",true);
    auto posDefProtoEnv = args.getBool("positiveDefiniteProtoEnv",true);
    auto iso_eps    = args.getReal("isoEpsilon",1.0e-10);
	auto cg_linesearch_eps = args.getReal("cgLineSearchEps",1.0e-8);
	auto cg_fdistance_eps  = args.getReal("cgFDistanceEps",1.0e-8);
	auto cg_gradientNorm_eps = args.getReal("cgGradientNormEps",1.0e-8);
	auto svd_cutoff = args.getReal("pseudoInvCutoff",1.0e-14);
	auto svd_maxLogGap = args.getReal("pseudoInvMaxLogGap",0.0);
    auto otNormType = args.getString("otNormType");

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

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
	std::array<Index, 4> aux;
	for (int i=0; i<4; i++) aux[i] = cls.aux[ cls.SI.at(tn[i]) ];

	std::array<Index, 4> phys;
	for (int i=0; i<4; i++) phys[i] = cls.phys[ cls.SI.at(tn[i]) ];
	
	std::array<Index, 4> opPI({
		noprime(uJ1J2.pi[0]),
		noprime(uJ1J2.pi[1]),
		noprime(uJ1J2.pi[2]),
		noprime(uJ1J2.pi[3])
	});
	
	if (dbg) {
		std::cout << "On-site indices:" << std::endl;
		for (int i=0; i<4; i++) {
			std::cout << tn[i] <<" : "<< aux[i] << " " << phys[i] << std::endl;
		}
	}

	ITensor deltaBra, deltaKet;
	std::vector<ITensor> pc(4); // holds corners T-C-T
	{
		t_begin_int = std::chrono::steady_clock::now();

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

		// precompute 4 (proto)corners of 2x2 environment
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
		
			// Disentangle HSLINK and VSLINK indices into aux-indices of corresponding tensors
			// define combiner
			auto cmb0 = combiner(prime(aux[s],tmp_iToE[0]), prime(aux[s],tmp_iToE[0]+4));
			auto cmb1 = combiner(prime(aux[s],tmp_iToE[1]), prime(aux[s],tmp_iToE[1]+4));
			if(dbg && dbgLvl >= 3) { Print(cmb0); Print(cmb1); }

			pc[s] = (pc[s] * delta(combinedIndex(cmb0), iToE[s][tmp_iToE[0]]) * cmb0) 
				* delta(combinedIndex(cmb1), iToE[s][tmp_iToE[1]]) * cmb1;
		}
		if(dbg) {
			for(int i=0; i<=3; i++) {
				std::cout <<"Site: "<< tn[i] <<" ";
				for (auto const& ind : iToE[i]) if(ind) std::cout<< ind <<" ";
				std::cout << std::endl;
			}
		}

		t_end_int = std::chrono::steady_clock::now();
		std::cout<<"Constructed proto Corners (without on-site tensors): "<< 
			std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
		// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 
	}

	// // ***** FORM "PROTO" ENVIRONMENTS FOR K *********************************** 
	// t_begin_int = std::chrono::steady_clock::now();

	// // TODO ?

	// ITensor protoK;
	// // 	// Variant ONE - precompute U|psi> surrounded in environment
	// // 	protoK = pc[0] * cls.sites.at(tn[0]);
	// // 	protoK = protoK * delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])); 
	// // 	protoK = protoK * ( pc[1] * cls.sites.at(tn[1]) );
	// // 	protoK = protoK * delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])); 
		
	// // 	protoK = protoK * ( pc[2] * cls.sites.at(tn[2]) );
	// // 	protoK = protoK * delta(prime(aux[2],pl[5]), prime(aux[3],pl[6]));
	// // 	protoK = protoK * delta(prime(aux[0],pl[0]), prime(aux[3],pl[7])); 
	// // 	protoK = protoK * ( pc[3] * cls.sites.at(tn[3]) );
	// { 
	// 	ITensor temp;
	// 	// Variant ONE - precompute U|psi> surrounded in environment
	// 	protoK = pc[0] * cls.sites.at(tn[0]);
	// 	protoK *= delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])); 
	// 	protoK *= ( pc[1] * cls.sites.at(tn[1]) );
	// 	protoK *= delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])); 
	// 	protoK *= delta(prime(aux[0],pl[0]), prime(aux[3],pl[7])); 
		
	// 	temp = pc[2] * cls.sites.at(tn[2]);
	// 	temp *= delta(prime(aux[2],pl[5]), prime(aux[3],pl[6]));
	// 	temp *= ( pc[3] * cls.sites.at(tn[3]) );
	
	// 	protoK *= temp;
	// }

	// // multiply by 4-site operator
	// // protoK = (( protoK * delta(opPI[0],phys[0]) ) * uJ1J2.H1 * delta(prime(opPI[0]),phys[0]));
	// // protoK = (( protoK * delta(opPI[1],phys[1]) ) * uJ1J2.H2 * delta(prime(opPI[1]),phys[1]));
	// // protoK = (( protoK * delta(opPI[2],phys[2]) ) * uJ1J2.H3 * delta(prime(opPI[2]),phys[2]));
	// {
	// 	ITensor temp;
	// 	temp = uJ1J2.H1 * uJ1J2.H2 * uJ1J2.H3;
	// 	temp = ((temp * delta(opPI[0],phys[0])) * delta(opPI[1],phys[1])) * delta(opPI[2],phys[2]);
		
	// 	protoK *= temp;
	// 	protoK = ((protoK * delta(prime(opPI[0]),phys[0])) * delta(prime(opPI[1]),phys[1]) )
	// 		* delta(prime(opPI[2]),phys[2]);
	// }
	// // TODO For now, assume the operator does not act on 4th site - just a syntactic filler

	// if (dbg && dbgLvl >= 3) { Print(protoK); }

	// std::cout<<"protoK.scale(): "<< protoK.scale() <<std::endl;
	// t_end_int = std::chrono::steady_clock::now();
	// std::cout<<"Proto Envs K constructed - T: "<< 
	// 	std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 <<" [sec]"<<std::endl;
	// // ***** FORM "PROTO" ENVIRONMENTS FOR M and K DONE ************************
	
	// // ******************************************************************************************** 
	// // 	     OPTIMIZE VIA CG                                                                      *
	// // ********************************************************************************************

	ITensor op4s = uJ1J2.op;
	for (int i=0; i<4; i++ ) {
		op4s *= delta(opPI[i],phys[i]);
		op4s *= prime(delta(opPI[i],phys[i]));
	}
	Print(op4s);

	t_begin_int = std::chrono::steady_clock::now();
	
	CG4S_IT cg4s(pc, aux, tn, pl, op4s, cls, args);
	cg4s.minimize();
	
	t_end_int = std::chrono::steady_clock::now();
	
	// // <psi|U^dag U|psi> - Variant ONE
	// auto NORMUPSI = (( protoK * delta(opPI[0],phys[0]) ) * conj(uJ1J2.H1)) * prime(delta(opPI[0],phys[0]));
	// NORMUPSI = (( NORMUPSI * delta(opPI[1],phys[1]) ) * conj(uJ1J2.H2)) * prime(delta(opPI[1],phys[1]));
	// NORMUPSI = (( NORMUPSI * delta(opPI[2],phys[2]) ) * conj(uJ1J2.H3)) * prime(delta(opPI[2],phys[2]));
	// // TODO For now, assume the operator does not act on 4th site - just a syntactic filler
	// NORMUPSI = NORMUPSI * delta(phys[3], prime(phys[3]));

	// NORMUPSI.prime(PHYS,-1);
	// NORMUPSI *= prime(conj(cls.sites.at(tn[0])), AUXLINK, 4);
	// NORMUPSI *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
	// NORMUPSI *= prime(conj(cls.sites.at(tn[1])), AUXLINK, 4);
	// NORMUPSI *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
	// NORMUPSI *= prime(conj(cls.sites.at(tn[2])), AUXLINK, 4);
	// NORMUPSI *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
	// NORMUPSI *= delta(prime(aux[0],pl[0]+4),prime(aux[3],pl[7]+4));
	// NORMUPSI *= prime(conj(cls.sites.at(tn[3])), AUXLINK, 4);

	// if (NORMUPSI.r() > 0) std::cout<<"NORMUPSI rank > 0"<<std::endl;
	// double normUPsi = sumels(NORMUPSI);

	// ITensor M, K, NORMPSI, OVERLAP;
	// std::vector<ITensor> pcS(4);
	// for (int i=0; i<4; i++) {
	// 	pcS[i] = (pc[i] * cls.sites.at(tn[i])) * prime(conj(cls.sites.at(tn[i])), AUXLINK,4);
	// }
	// double normPsi, finit, finitN;
	// double prev_finit = 1.0;
	// double ferr;
	// int fiter;
	// int itol = 1;

 //  	int altlstsquares_iter = 0;
	// bool converged = false;
	// // cg_gradientNorm_eps = std::max(cg_gradientNorm_eps, condNum * machine_eps);
	// // // cg_fdistance_eps    = std::max(cg_fdistance_eps, condNum * machine_eps);
 //  	std::vector<double> fdist, fdistN, vec_normPsi;
 //  	std::cout << "ENTERING CG LOOP tol: " << cg_gradientNorm_eps << std::endl;
 //  	t_begin_int = std::chrono::steady_clock::now();
	// while (not converged) {
	
	// 	// Optimizing A
	// 	// 1) construct matrix M, which is defined as <psi~|psi~> = A^dag * M * A

	// 	// M = (pc[1] * cls.sites.at(tn[1])) * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	// 	// M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
	// 	// M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
	// 	// M = ((M * pc[2]) * cls.sites.at(tn[2])) * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	// 	// M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
	// 	// M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
	// 	// M = ((M * pc[3]) * cls.sites.at(tn[3])) * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	// 	// M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
	// 	// M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
	// 	// M *= delta(prime(aux[1],pl[2]),  prime(aux[0],pl[1]));
	// 	// M *= delta(prime(aux[1],pl[2]+4),prime(aux[0],pl[1]+4));
	// 	// M = M * pc[0];

	// 	{
	// 		ITensor temp;
	
	// 		M = pcS[1];
	// 		M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
	// 		M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
	// 		M *= pcS[2];
	// 		M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
	// 		M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
	// 		M *= delta(prime(aux[1],pl[2]),  prime(aux[0],pl[1]));
	// 		M *= delta(prime(aux[1],pl[2]+4),prime(aux[0],pl[1]+4));
			
	// 		temp = pcS[3];
	// 		temp *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
	// 		temp *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
	// 		temp *= pc[0];

	// 		M *= temp;
	// 	}


	// 	if(dbg && (dbgLvl >= 3)) Print(M);

	// 	//  2) construct vector K, which is defined as <psi~|U|psi> = A^dag * K
	// 	// K = protoK * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	// 	// K *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
	// 	// K = K * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	// 	// K *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
	// 	// K = K * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	// 	// K *= delta(	prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
	// 	// K *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
		
	// 	{
	// 		ITensor temp;

	// 		temp = prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	// 		temp *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
	// 		temp *= prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	// 		temp *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
	// 		temp *= prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	// 		temp *= delta(	prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
	// 		temp *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
		
	// 		K = protoK * temp;
	// 	}

	// 	if(dbg && (dbgLvl >= 3)) Print(K);

	// 	// <psi'|psi'>
	// 	NORMPSI = (prime(conj(cls.sites.at(tn[0])), AUXLINK,4) * M) * cls.sites.at(tn[0]); 
	// 	// <psi'|U|psi>
	// 	OVERLAP = prime(conj(cls.sites.at(tn[0])), AUXLINK,4) * K;

	// 	if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
	// 	normPsi = sumels(NORMPSI);
	// 	finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;
	// 	finitN  = 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(normUPsi * normPsi) + 1.0;
	// 	prev_finit = finit;

	// 	fdist.push_back( finit );
	// 	fdistN.push_back( finitN );
	// 	vec_normPsi.push_back( normPsi );
	// 	//if ( fdist.back() < cg_fdistance_eps ) { converged = true; break; }
		
	// 	if ( (fdist.size() > 1) && std::abs((fdist.back() - fdist[fdist.size()-2])/fdist[0]) < cg_fdistance_eps ) 
	// 	{ 
	// 		std::cout << "stopCond: " << (fdist.back() - fdist[fdist.size()-2])/fdist[0] << std::endl;
	// 		converged = true; break; 
	// 	} else {
	// 		std::cout << "stopCond: " << (fdist.back() - fdist[fdist.size()-2])/fdist[0] << std::endl;
	// 	}

	// 	// ***** SOLVE LINEAR SYSTEM M*A = K by CG ***************************
	// 	ITensor dummyComb;
	// 	int tensorDim = cls.physDim * cls.auxBondDim * cls.auxBondDim * cls.auxBondDim * cls.auxBondDim;
	// 	FULSCG_IT fulscg(M, K, cls.sites.at(tn[0]), dummyComb, dummyComb, svd_cutoff );
	// 	fulscg.solveIT(K, cls.sites.at(tn[0]), itol, cg_gradientNorm_eps, tensorDim, fiter, ferr);
	// 	std::cout <<"A f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
	// 	pcS[0] = (pc[0] * cls.sites.at(tn[0])) * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);

	//     // Optimizing B
	// 	// 1) construct matrix M, which is defined as <psi~|psi~> = B^dag * M * B	
		
	// 	// M = (pc[2] * cls.sites.at(tn[2])) * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	// 	// M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
	// 	// M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
	// 	// M = ((M * pc[3]) * cls.sites.at(tn[3])) * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	// 	// M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
	// 	// M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
	// 	// M = ((M * pc[0]) * cls.sites.at(tn[0])) * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
	// 	// M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
	// 	// M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
	// 	// M *= delta(prime(aux[2],pl[4]),  prime(aux[1],pl[3]));
	// 	// M *= delta(prime(aux[2],pl[4]+4),prime(aux[1],pl[3]+4));
	// 	// M = M * pc[1];

	// 	{
	// 		ITensor temp;

	// 		M = pcS[2];
	// 		M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
	// 		M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
	// 		M *= pcS[3];
	// 		M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
	// 		M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
	// 		M *= delta(prime(aux[2],pl[4]),  prime(aux[1],pl[3]));
	// 		M *= delta(prime(aux[2],pl[4]+4),prime(aux[1],pl[3]+4));
			
	// 		temp = pcS[0];
	// 		temp *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
	// 		temp *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
	// 		temp *= pc[1];

	// 		M *= temp;
	// 	}

	// 	if(dbg && (dbgLvl >= 3)) Print(M);

	// 	// 2) construct vector K, which is defined as <psi~|psi'> = eB^dag * K
	// 	// K = protoK * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	// 	// K *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
	// 	// K = K * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	// 	// K *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
	// 	// K = K * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
	// 	// K *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
	// 	// K *= delta(	prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		
	// 	{
	// 		ITensor temp;

	// 		temp = prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	// 		temp *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
	// 		temp *= prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	// 		temp *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
	// 		temp *= prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
	// 		temp *= delta(	prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
	// 		temp *= delta(	prime(aux[2],pl[4]+4), prime(aux[1],pl[3]+4) );
		
	// 		K = protoK * temp;
	// 	}	

	// 	if(dbg && (dbgLvl >= 3)) Print(K);

	// 	// <psi'|psi'>
	// 	NORMPSI = (prime(conj(cls.sites.at(tn[1])), AUXLINK,4) * M) * cls.sites.at(tn[1]); 
	// 	// <psi'|U|psi>
	// 	OVERLAP = prime(conj(cls.sites.at(tn[1])), AUXLINK,4) * K;

	// 	if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
	// 	normPsi = sumels(NORMPSI);
	// 	finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;

	// 	std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
	// 	prev_finit = finit;

	// 	// ***** SOLVE LINEAR SYSTEM M*B = K ******************************
	// 	ferr = 1.0;
	// 	int accfiter = 0;
	// 	while ( (ferr > cg_gradientNorm_eps) && (accfiter < tensorDim * 10 ) )  {
	// 		FULSCG_IT fulscgEB(M,K,cls.sites.at(tn[1]),dummyComb, dummyComb, svd_cutoff );
	// 		fulscgEB.solveIT(K, cls.sites.at(tn[1]), itol, cg_gradientNorm_eps, tensorDim, fiter, ferr);
	// 		accfiter += fiter;
	// 		std::cout <<"B f_err= "<< ferr <<" f_iter= "<< accfiter << std::endl;
	// 	}
	// 	pcS[1] = (pc[1] * cls.sites.at(tn[1])) * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	    
	// 	// Optimizing D
	// 	// 1) construct matrix M, which is defined as <psi~|psi~> = D^dag * M * D	

	// 	// M = (pc[3] * cls.sites.at(tn[3])) * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	// 	// M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
	// 	// M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
	// 	// M = ((M * pc[0]) * cls.sites.at(tn[0])) * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
	// 	// M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
	// 	// M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
	// 	// M = ((M * pc[1]) * cls.sites.at(tn[1])) * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	// 	// M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
	// 	// M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
	// 	// M *= delta(prime(aux[3],pl[6]),  prime(aux[2],pl[5]));
	// 	// M *= delta(prime(aux[3],pl[6]+4),prime(aux[2],pl[5]+4));
	// 	// M = M * pc[2];

	// 	{
	// 		ITensor temp;

	// 		M = pcS[3];
	// 		M *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
	// 		M *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
	// 		M *= pcS[0];
	// 		M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
	// 		M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
	// 		M *= delta(prime(aux[3],pl[6]),  prime(aux[2],pl[5]));
	// 		M *= delta(prime(aux[3],pl[6]+4),prime(aux[2],pl[5]+4));
			
	// 		temp = pcS[1];
	// 		temp *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
	// 		temp *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
	// 		temp *= pc[2];

	// 		M *= temp;
	// 	}

	// 	if(dbg && (dbgLvl >= 3)) Print(M);

	// 	// 2) construct vector K, which is defined as <psi~|psi'> = D^dag * K
	// 	// K = protoK * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	// 	// K *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
	// 	// K = K * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
	// 	// K *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
	// 	// K = K * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	// 	// K *= delta(	prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
	// 	// K *= delta(	prime(aux[3],pl[6]+4), prime(aux[2],pl[5]+4) );
		
	// 	{
	// 		ITensor temp;

	// 		temp = prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	// 		temp *= delta( prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
	// 		temp *= prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
	// 		temp *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
	// 		temp *= prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	// 		temp *= delta(	prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
	// 		temp *= delta(	prime(aux[3],pl[6]+4), prime(aux[2],pl[5]+4) );
		
	// 		K = protoK * temp;
	// 	}

	// 	if(dbg && (dbgLvl >= 3)) Print(K);

	// 	// <psi'|psi'>
	// 	NORMPSI = (prime(conj(cls.sites.at(tn[2])), AUXLINK,4) * M) * cls.sites.at(tn[2]); 
	// 	// <psi'|U|psi>
	// 	OVERLAP = prime(conj(cls.sites.at(tn[2])), AUXLINK,4) * K;

	// 	if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
	// 	normPsi = sumels(NORMPSI);
	// 	finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;

	// 	std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
	// 	prev_finit = finit;

	// 	// ***** SOLVE LINEAR SYSTEM M*eD = K ******************************
	// 	FULSCG_IT fulscgED(M,K,cls.sites.at(tn[2]),dummyComb, dummyComb, svd_cutoff );
	// 	fulscgED.solveIT(K, cls.sites.at(tn[2]), itol, cg_gradientNorm_eps, tensorDim, fiter, ferr);

	// 	std::cout <<"D f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
	// 	pcS[2] = (pc[2] * cls.sites.at(tn[2])) * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);

	// 	// Optimizing C
	// 	// 1) construct matrix M, which is defined as <psi~|psi~> = C^dag * M * C	

	// 	// M = (pc[0] * cls.sites.at(tn[0])) * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
	// 	// M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
	// 	// M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
	// 	// M = ((M * pc[1]) * cls.sites.at(tn[1])) * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	// 	// M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
	// 	// M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
	// 	// M = ((M * pc[2]) * cls.sites.at(tn[2])) * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	// 	// M *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
	// 	// M *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
	// 	// M *= delta(prime(aux[0],pl[0]),  prime(aux[3],pl[7]));
	// 	// M *= delta(prime(aux[0],pl[0]+4),prime(aux[3],pl[7]+4));
	// 	// M = M * pc[3];

	// 	{
	// 		ITensor temp;

	// 		M = pcS[0];
	// 		M *= delta(prime(aux[0],pl[1]),  prime(aux[1],pl[2]));
	// 		M *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
	// 		M *= pcS[1];
	// 		M *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
	// 		M *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
	// 		M *= delta(prime(aux[0],pl[0]),  prime(aux[3],pl[7]));
	// 		M *= delta(prime(aux[0],pl[0]+4),prime(aux[3],pl[7]+4));
			
	// 		temp = pcS[2];
	// 		temp *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
	// 		temp *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
	// 		temp *= pc[3];

	// 		M *= temp;
	// 	}

	// 	if(dbg && (dbgLvl >= 3)) Print(M);

	// 	// 2) construct vector K, which is defined as <psi~|psi'> = C^dag * K
	// 	// K = protoK * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
	// 	// K *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
	// 	// K = K * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	// 	// K *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
	// 	// K = K * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	// 	// K *= delta(	prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
	// 	// K *= delta(	prime(aux[0],pl[0]+4), prime(aux[3],pl[7]+4) );
		
	// 	{
	// 		ITensor temp;

	// 		temp = prime(conj(cls.sites.at(tn[0])), AUXLINK,4);
	// 		temp *= delta( prime(aux[0],pl[1]+4), prime(aux[1],pl[2]+4) );
	// 		temp *= prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	// 		temp *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
	// 		temp *= prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	// 		temp *= delta(	prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
	// 		temp *= delta(	prime(aux[0],pl[0]+4), prime(aux[3],pl[7]+4) );
		
	// 		K = protoK * temp;
	// 	}

	// 	if(dbg && (dbgLvl >= 3)) Print(K);

	// 	// <psi'|psi'>
	// 	NORMPSI = (prime(conj(cls.sites.at(tn[3])), AUXLINK,4) * M) * cls.sites.at(tn[3]); 
	// 	// <psi'|U|psi>
	// 	OVERLAP = prime(conj(cls.sites.at(tn[3])), AUXLINK,4) * K;

	// 	if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
	// 	normPsi = sumels(NORMPSI);
	// 	finit   = normPsi - 2.0 * sumels(OVERLAP) + normUPsi;

	// 	std::cout << "stopCond: " << (finit - prev_finit)/fdist[0] << std::endl;
	// 	prev_finit = finit;

	// 	// ***** SOLVE LINEAR SYSTEM M*C = K ******************************
	// 	FULSCG_IT fulscgEC(M,K,cls.sites.at(tn[3]),dummyComb, dummyComb, svd_cutoff );
	// 	fulscgEC.solveIT(K, cls.sites.at(tn[3]), itol, cg_gradientNorm_eps, tensorDim, fiter, ferr);

	// 	std::cout <<"C f_err= "<< ferr <<" f_iter= "<< fiter << std::endl;
	// 	pcS[3] = (pc[3] * cls.sites.at(tn[3])) * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);

	// 	// TEST CRITERION TO STOP THE ALS procedure
	// 	altlstsquares_iter++;
	// 	if (altlstsquares_iter >= maxAltLstSqrIter) converged = true;
	// }
	// t_end_int = std::chrono::steady_clock::now();

	// std::cout <<"STEP f=||psi'>-|psi>|^2 f_normalized <psi'|psi'>" << std::endl;
	// for (int i=0; i < fdist.size(); i++) std::cout << i <<" "<< fdist[i] <<" "<< fdistN[i] 
	// 	<<" "<< vec_normPsi[i] << std::endl;

	// POST-OPTIMIZATION DIAGNOSTICS ------------------------------------------
	// max element of on-site tensors
	std::string diag_maxElem;
	for (int i=0; i<4; i++) {
		m = 0.;
		cls.sites.at(tn[i]).visit(max_m);
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
	diag_data.add("alsSweep", 0); //altlstsquares_iter);

	std::string siteMaxElem_descriptor = "site max_elem site max_elem site max_elem site max_elem";
	diag_data.add("siteMaxElem_descriptor",siteMaxElem_descriptor);
	diag_data.add("siteMaxElem",diag_maxElem);
	diag_data.add("ratioNonSymLE", -1.0); // ratio of largest elements 
	diag_data.add("ratioNonSymFN", -1.0); // ratio of norms
	// diag_data.add("ratioNonSymLE",diag_maxMasymLE/diag_maxMsymLE); // ratio of largest elements 
	// diag_data.add("ratioNonSymFN",diag_maxMasymFN/diag_maxMsymFN); // ratio of norms
	
	std::ostringstream oss;
	// oss << std::scientific << fdist[0] <<" "<< fdist.back() <<" " 
	// 	<< fdistN[0] <<" "<< fdistN.back() <<" "<< vec_normPsi[0] <<" "<< vec_normPsi.back() <<" "<<
	// 	std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 ;
	oss << std::scientific << 0.0 <<" "<< 0.0 <<" " 
		<< 0.0 <<" "<< 0.0 <<" "<< 0.0 <<" "<< cg4s.inst_normPsi <<" "<<
		std::chrono::duration_cast<std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 ;

	std::string logMinDiag_descriptor = "f_init f_final normalizedf_init normalizedf_final norm(psi')_init norm(psi')_final time[s]";
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

CG4S_IT::CG4S_IT(
		std::vector< ITensor > const& ppc, // protocorners 
		std::array< Index, 4 > const& aaux,  // aux indices
		std::vector< std::string > const& ttn,      // site IDs
		std::vector< int > const& ppl,              // primelevels of aux indices          
		ITensor const& oop4s,              // four-site operator
		Cluster & ccls,
		Args const& aargs) : 

		pc(ppc), aux(aaux), tn(ttn), pl(ppl), op4s(oop4s), cls(ccls), args(aargs),
		g(4), xi(4), h(4) 
{

	auto dbg = args.getBool("fuDbg",false);
    auto dbgLvl = args.getInt("fuDbgLevel",0);
    epsdistf = args.getReal("epsdistf",1.0e-8);

	// ***** FORM "PROTO" ENVIRONMENTS FOR K *********************************** 
	std::chrono::steady_clock::time_point t_begin_int, t_end_int;
	t_begin_int = std::chrono::steady_clock::now();

	// 	// Variant ONE - precompute U|psi> surrounded in environment
	{ 
		ITensor temp;
		// Variant ONE - precompute U|psi> surrounded in environment
		protoK = pc[0] * cls.sites.at(tn[0]);
		protoK *= delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])); 
		protoK *= ( pc[1] * cls.sites.at(tn[1]) );
		protoK *= delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])); 
		protoK *= delta(prime(aux[0],pl[0]), prime(aux[3],pl[7])); 
		
		temp = pc[2] * cls.sites.at(tn[2]);
		temp *= delta(prime(aux[2],pl[5]), prime(aux[3],pl[6]));
		temp *= ( pc[3] * cls.sites.at(tn[3]) );
	
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
	auto NORMUPSI = protoK * conj(op4s);
	NORMUPSI.noprime(PHYS);
	NORMUPSI *= prime(conj(cls.sites.at(tn[0])), AUXLINK, 4);
	NORMUPSI *= delta(prime(aux[0],pl[1]+4),prime(aux[1],pl[2]+4));
	NORMUPSI *= prime(conj(cls.sites.at(tn[1])), AUXLINK, 4);
	NORMUPSI *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
	NORMUPSI *= prime(conj(cls.sites.at(tn[2])), AUXLINK, 4);
	NORMUPSI *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
	NORMUPSI *= delta(prime(aux[0],pl[0]+4),prime(aux[3],pl[7]+4));
	NORMUPSI *= prime(conj(cls.sites.at(tn[3])), AUXLINK, 4);

	if (NORMUPSI.r() > 0) std::cout<<"NORMUPSI rank > 0"<<std::endl;
	normUPsi = sumels(NORMUPSI);
	if (dbg && dbgLvl >=3) { std::cout<<"<psi|U^dag U|psi> = "<< normUPsi << std::endl; }
}

void CG4S_IT::minimize() {
	const int ITMAX=200;
	const double EPS=1.0e-18;
	const double GTOL=1.0e-8;
	const double FTOL=epsdistf;
	double gg,dgg,test,fret,finit;
	
	// double max_mag = 0.;
	// auto maxComp = [&max_mag](double r) {
 //  		if(std::fabs(r) > max_mag) max_mag = std::fabs(r);
 //  	};

	// compute initial function value and gradient and set variables
	double fp = func();
	finit = fp;
	std::cout<<"Init distance: "<< finit << std::endl;
	grad(xi);
	for(int j=0; j<4; j++) {
		g[j] = -1.0 * xi[j];
		xi[j]=h[j]=g[j];
	}

	for (int its=0;its<ITMAX;its++) {
		// perform line minization
		fret=linmin(fp, xi);
		std::cout<<"its: "<< its << " currentDist: "<< fret << std::endl;

		// if (2.0*abs(fret-fp) <= FTOL*(abs(fret)+abs(fp)+EPS)) {
		if ( std::abs(fret - fp)/finit <= FTOL  || std::abs(fret) < 1.0e-7 ) {
			std::cout << "Frprmn: converged iter="<< its 
				<<". ||psi'> - |psi>|^2 = "<< fret << " Diff: " << std::abs(fret - fp)/finit << std::endl;
			return;
		}
		fp=fret;
		grad(xi); //func.df(p,xi)
		test=0.0;
		// double den=MAX(abs(fp),1.0);
		// for (Int j=0;j<4;j++) {

		// 	Doub temp=abs(xi[j])*MAX(abs(p[j]),1.0)/den;
		// 	if (temp > test) test=temp;
		// }
		// if (test < GTOL) { 
		// 	std::cout << "Frprmn: converged iter="<< its <<". g^2 = "<< test << std::endl;
		// 	return p;
		// }

		dgg=gg=0.0;
		for (int j=0; j<4; j++) {
			auto temp = norm(g[j]);
			gg += temp * temp;
			auto tempT = (xi[j] + g[j])*xi[j];
			if (tempT.r() > 0) std::cout <<"ERROR: tempT is not a scalar" << std::endl;
			dgg += sumels(tempT);
		}

		// for (Int j=0;j<n;j++) {
		// 	gg += g[j]*g[j];
		// //	dgg += xi[j]*xi[j];
		// 	dgg += (xi[j]+g[j])*xi[j];
		// }
		if (gg == 0.0) return;
		double gam=dgg/gg;
		for (int j=0;j<4;j++) {
			g[j] = -1.0 * xi[j];
			xi[j]=h[j]=g[j];//+gam*h[j];
		}
	}
	// throw("Too many iterations in frprmn");
	std::cout << "Frprmn: max iterations exceeded. g^2 = "<< test << std::endl;
}

double CG4S_IT::func() {
	ITensor NORMPSI, OVERLAP;

	{
		ITensor temp;

		NORMPSI = (pc[1] * cls.sites.at(tn[1])) * prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
		NORMPSI *= delta(prime(aux[1],pl[3]),  prime(aux[2],pl[4]));
		NORMPSI *= delta(prime(aux[1],pl[3]+4),prime(aux[2],pl[4]+4));
		NORMPSI = ((NORMPSI * pc[2]) * cls.sites.at(tn[2])) * prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
		NORMPSI *= delta(prime(aux[2],pl[5]),  prime(aux[3],pl[6]));
		NORMPSI *= delta(prime(aux[2],pl[5]+4),prime(aux[3],pl[6]+4));
		NORMPSI *= delta(prime(aux[1],pl[2]),  prime(aux[0],pl[1]));
		NORMPSI *= delta(prime(aux[1],pl[2]+4),prime(aux[0],pl[1]+4));

		temp = (pc[3] * cls.sites.at(tn[3])) * prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
		temp *= delta(prime(aux[3],pl[7]),  prime(aux[0],pl[0]));
		temp *= delta(prime(aux[3],pl[7]+4),prime(aux[0],pl[0]+4));
		temp *= (pc[0] * cls.sites.at(tn[0])) * prime(conj(cls.sites.at(tn[0])), AUXLINK,4);

		NORMPSI *= temp;
	}

	OVERLAP = prime(conj(cls.sites.at(tn[1])), AUXLINK,4);
	OVERLAP *= delta( prime(aux[1],pl[3]+4), prime(aux[2],pl[4]+4) );
	OVERLAP *= prime(conj(cls.sites.at(tn[2])), AUXLINK,4);
	OVERLAP *= delta( prime(aux[2],pl[5]+4), prime(aux[3],pl[6]+4) );
	
	OVERLAP *= protoK;

	OVERLAP *= prime(conj(cls.sites.at(tn[3])), AUXLINK,4);
	OVERLAP *= delta(	prime(aux[3],pl[7]+4), prime(aux[0],pl[0]+4) );
	OVERLAP *= delta(	prime(aux[1],pl[2]+4), prime(aux[0],pl[1]+4) );
	OVERLAP *= prime(conj(cls.sites.at(tn[0])), AUXLINK,4);

	if (NORMPSI.r() > 0 || OVERLAP.r() > 0) std::cout<<"NORMPSI or OVERLAP rank > 0"<<std::endl;	
	inst_normPsi = sumels(NORMPSI);
	inst_overlap = sumels(OVERLAP);
	// return sumels(NORMPSI) - 2.0 * sumels(OVERLAP) + normUPsi;
	return 1.0 - 2.0 * sumels(OVERLAP)/std::sqrt(sumels(NORMPSI) * normUPsi) + 1.0;
}

double CG4S_IT::linmin(double fxi, std::vector< ITensor > const& g) {
	const int MAXIT = 5;
	
	auto ngrad = 0.0;
	for (int j=0; j<4; j++) { ngrad += std::pow(norm(g[j]),2); }
	std::cout<<"Entering LinMin |g|: "<< std::sqrt(ngrad) << std::endl;

	std::vector<ITensor> tmpT;
	for (int j=0; j<4; j++) { tmpT.emplace_back(cls.sites.at(tn[j])); }

	// double mag = 2 * std::atan(1);
	int sgn = 1; 
	double mag = fxi * 0.01;
	// for (int j=0; j<4; j++) { cls.sites.at(tn[j]) = std::cos(mag) * tmpT[j] + std::sin(mag) * g[j]; }
	for (int j=0; j<4; j++) { cls.sites.at(tn[j]) -= mag * g[j]; }
	auto tmp_val  = func();
	std::cout<<"linmin -mag: "<< tmp_val << std::endl;
	if ( tmp_val < fxi ) { sgn = -1; }
	for (int j=0; j<4; j++) { cls.sites.at(tn[j]) += (2.0 * mag) * g[j]; }
	auto tmp_val2 = func();
	std::cout<<"linmin +mag: "<< tmp_val2 << std::endl;
	if ( (tmp_val2 < fxi) && (tmp_val2 < tmp_val) ) { sgn = 1; }

	for (int j=0; j<4; j++) { cls.sites.at(tn[j]) -= mag * g[j]; }
	double fx = 0.0;
	double fx_prev = fxi;
	for (int i=0; i<MAXIT; i++) {
		for (int j=0; j<4; j++) { cls.sites.at(tn[j]) += (sgn * mag) * g[j]; }
		// for (int j=0; j<4; j++) { cls.sites.at(tn[j]) = std::cos(mag) * tmpT[j] + std::sin(sgn * mag) * g[j]; }
		fx = func();
		std::cout<<"linmin its: "<< i << " dist: "<< fx << std::endl;
		if (fx > fx_prev) {
			// mag = mag * 0.5;
			// for (int j=0; j<4; j++) { cls.sites.at(tn[j]) = std::cos(mag) * tmpT[j] + std::sin(sgn * mag) * g[j]; }
			for (int j=0; j<4; j++) { cls.sites.at(tn[j]) -= (sgn * mag) * g[j]; }
			fx = fx_prev;
			break;
		} else {
			mag = mag * 2.0;
			
			fx_prev = fx;
		}
		// fx_prev = fx;
	}
	for (int j=0; j<4; j++) {std::cout<<"1: "<< norm(tmpT[j] - cls.sites.at(tn[j])) <<" "; }
	std::cout << std::endl;

	return fx;
}

void CG4S_IT::grad(std::vector<ITensor> &grad) {
	// compute d<psi'|psi'> contributions
	{
		ITensor M;
		{
			ITensor temp;
			// Variant ONE - precompute |psi'> surrounded in environment
			M = pc[0] * cls.sites.at(tn[0]);
			M *= delta(prime(aux[0],pl[1]), prime(aux[1],pl[2])); 
			M *= ( pc[1] * cls.sites.at(tn[1]) );
			M *= delta(prime(aux[1],pl[3]), prime(aux[2],pl[4])); 
			M *= delta(prime(aux[0],pl[0]), prime(aux[3],pl[7])); 
			
			temp = pc[2] * cls.sites.at(tn[2]);
			temp *= delta(prime(aux[2],pl[5]), prime(aux[3],pl[6]));
			temp *= ( pc[3] * cls.sites.at(tn[3]) );
		
			M *= temp;
		}
	
		for (int i=0; i<4; i++) {
			ITensor temp;

			int j = (i + 1) % 4;
			int k = (j + 1) % 4;
			temp = prime(conj(cls.sites.at(tn[j])), AUXLINK, 4);
			temp *= prime(delta(prime(aux[j],pl[2*j]), prime(aux[i],pl[2*i+1])), 4);
			temp *= prime(delta(prime(aux[j],pl[2*j+1]), prime(aux[k],pl[2*k])), 4);
			
			j = (j + 1) % 4;
			k = (j + 1) % 4;
			temp *= prime(conj(cls.sites.at(tn[j])), AUXLINK, 4);
			temp *= prime(delta(prime(aux[j],pl[2*j+1]), prime(aux[k],pl[2*k])), 4);

			temp *= M;
		
			j = (j + 1) % 4;
			k = (j + 1) % 4;
			temp *= prime(conj(cls.sites.at(tn[j])), AUXLINK, 4);
			temp *= prime(delta(prime(aux[j],pl[2*j+1]), prime(aux[k],pl[2*k])), 4);			
		
			grad[i] = temp; //* (1.0 + 2.0 * (1.0 - std::abs(inst_normPsi)));
			// grad[i] = temp * (0.5 * inst_overlap / (std::sqrt(normUPsi * inst_normPsi) * std::abs(inst_normPsi)) );
		}
	}

	// compute d<psi'|U|psi> contributions
	for (int i=0; i<4; i++) {
		ITensor temp;

		int j = (i + 1) % 4;
		int k = (j + 1) % 4;
		temp = prime(conj(cls.sites.at(tn[j])), AUXLINK, 4);
		temp *= prime(delta(prime(aux[j],pl[2*j]), prime(aux[i],pl[2*i+1])), 4);
		temp *= prime(delta(prime(aux[j],pl[2*j+1]), prime(aux[k],pl[2*k])), 4);
		
		j = (j + 1) % 4;
		k = (j + 1) % 4;
		temp *= prime(conj(cls.sites.at(tn[j])), AUXLINK, 4);
		temp *= prime(delta(prime(aux[j],pl[2*j+1]), prime(aux[k],pl[2*k])), 4);

		temp *= protoK;
	
		j = (j + 1) % 4;
		k = (j + 1) % 4;
		temp *= prime(conj(cls.sites.at(tn[j])), AUXLINK, 4);
		temp *= prime(delta(prime(aux[j],pl[2*j+1]), prime(aux[k],pl[2*k])), 4);			
	
		grad[i] += -temp;
		// grad[i] += -temp / std::sqrt(inst_normPsi * normUPsi);
	}

	for (auto& ten : grad) { 
		ten.prime(AUXLINK, -4); 
	}
}
