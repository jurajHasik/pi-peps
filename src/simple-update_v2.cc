#include "simple-update_v2.h"

using namespace itensor;

// 2 SITE OPS #########################################################

void applyH_T1_L_T2(MPO_2site const& mpo2s, 
	ITensor & T1, ITensor & T2, ITensor & L, ITensor & LI, bool dbg) {
	if(dbg) {std::cout <<">>>>> applyH_12_T1_L_T2 called <<<<<"<< std::endl;
		std::cout << mpo2s;
		PrintData(mpo2s.H1);
    	PrintData(mpo2s.H2);}

    const size_t auxd     = commonIndex(T1,L).m();
	const Real svCutoff   = 1.0e-14;

	/*
	 * Applying 2-site MPO leads to a new tensor network of the form
	 * 
	 *    \ |    __               
	 *   --|1|~~|H1|~~s1           
	 *      |     |               s1       s2
	 *      L     |               |_       |_ 
	 *    \ |     |       ==   --|  |-----|  |--
	 *   --|2|~~|H2|~~s2  ==   --|1~|     |2~|--  
	 *      |             ==   --|__|--L--|__|--
     *
	 * Indices s1,s2 are relabeled back to physical indices of 
	 * original sites 1 and 2 after applying MPO.
	 *
	 */

	if(dbg) {std::cout <<"----- Initial |12> -----"<< std::endl;
		Print(T1);
		Print(L);
		Print(T2);}
	auto ipT1 = findtype(T1, PHYS);
	auto ipT2 = findtype(T2, PHYS);  
	auto iT1_L = commonIndex(T1, L);
	auto iL_T2 = commonIndex(L, T2);

	/*
	 * Extract reduced tensors from on-site tensor to apply 2-site 
	 * gate 
	 *
	 *        | /        | /       =>     |      /     /    |
	 *     --|1*|-------|2*|--     =>  --|1X|--|1R|--|2R|--|2X|-- 
	 *        |          |         =>     |                 |
	 *
	 */
	ITensor T1R(ipT1, iT1_L);
	ITensor T2R(ipT2, iL_T2);
	ITensor T1X, T2X, sv1XR, sv2XR;
	auto spec = svd( T1, T1R, sv1XR, T1X);
	if(dbg) Print(spec);
	spec = svd( T2, T2R, sv2XR, T2X);
	if(dbg) Print(spec);
	T1R = T1R * sv1XR;
	T2R = T2R * sv2XR;

	// D^2 x s x auxD_mpo3s
	auto kd_phys1 = delta(ipT1, mpo2s.Is1);
	T1R = (T1R * kd_phys1) * mpo2s.H1;
	T1R = (T1R * kd_phys1.prime()).prime(PHYS,-1);
	// D^2 x s x auxD_mpo3s^2
	auto kd_phys2 = delta(ipT2, mpo2s.Is2);
	T2R = (T2R * kd_phys2 ) * mpo2s.H2;
	T2R = (T2R * kd_phys2.prime()).prime(PHYS,-1);

	if(dbg) {std::cout <<"----- Appyling H1-H2 to |1R--2R> -----"<< std::endl;
		Print(T1R);
    	Print(T2R);}

	/*
	 * Perform SVD of new on-site tensors |1R~| and |2R~| by contrating them
	 * along diagonal matrix with weights
	 *
	 *       _______               s1                       s2
	 *  s1~~|       |~~s2           |                        |
	 *    --| 1~ 2~ |--    ==>      |                        |
	 *      |_______|      ==>  --|1~~|++a1++|SV_L12|++a2++|2~~|--
	 *
	 * where 1~~ and 2~~ are now holding singular vectors wrt
	 * to SVD and SV_L12 holds a new weights
	 * We keep only auxBondDim largest singular values
	 *
	 */

	if(dbg) std::cout <<"----- Perform SVD along link12 -----"<< std::endl;
	ITensor SV_L12;
	spec = svd(T1R*L*T2R, T1R, SV_L12, T2R, {"Maxm", auxd, "svCutoff",svCutoff});
	if(dbg) {Print(T1R);
		Print(spec);
		Print(T2R);}

	// Set proper indices to resulting tensors from SVD routine
	Index n1 = commonIndex(T1R, SV_L12);
	Index n2 = commonIndex(SV_L12, T2R);

	T1 = T1R * T1X * delta(n1, iT1_L);
	
	// Prepare results
	SV_L12 = SV_L12 / norm(SV_L12);
	for (size_t i=1; i<=auxd; ++i) {
		if(i <= n1.m()) {
			L.set(iT1_L(i),iL_T2(i), SV_L12.real(n1(i),n2(i)));
			LI.set(iT1_L(i),iL_T2(i), 1.0/SV_L12.real(n1(i),n2(i)));
		} else {
			L.set(iT1_L(i),iL_T2(i), 0.0);
			LI.set(iT1_L(i),iL_T2(i), 0.0);
		}
	}

	T2 = T2R * T2X * delta(n2, iL_T2);

	if(dbg) {Print(T1);
		PrintData(L);
		Print(T2);}
}

void applyH_T1_L_T2_v2(MPO_2site const& mpo2s, 
	ITensor & T1, ITensor & T2, ITensor & L, bool dbg) {
	if(dbg) {
		std::cout <<">>>>> applyH_12_T1_L_T2 called <<<<<"<< std::endl;
		PrintData(mpo2s.H1);
    	PrintData(mpo2s.H2);
    }

    auto sqrtT = [](double r) { return std::sqrt(r); };
	/*
	 * Applying 2-site MPO leads to a new tensor network of the form
	 * 
	 *    \ |    __               
	 *   --|1|~~|H1|~~s1           
	 *      |     |               s1       s2
	 *      L     |               |_       |_ 
	 *    \ |     |       ==   --|  |-----|  |--
	 *   --|2|~~|H2|~~s2  ==   --|1~|     |2~|--  
	 *      |             ==   --|__|--L--|__|--
     *
	 * Indices s1,s2 are relabeled back to physical indices of 
	 * original sites 1 and 2 after applying MPO.
	 *
	 */

	if(dbg) {
		std::cout <<"----- Initial |12> -----"<< std::endl;
		Print(T1);
		Print(L);
		Print(T2);
	}
	auto ipT1 = findtype(T1, PHYS);
	auto ipT2 = findtype(T2, PHYS);  
	auto iT1_L = commonIndex(T1, L);
	auto iL_T2 = commonIndex(L, T2);
	L.apply(sqrtT);

	/*
	 * Extract reduced tensors from on-site tensor to apply 2-site 
	 * gate 
	 *
	 *        | /        | /       =>     |      /     /    |
	 *  --|1*L^1/2|--|2*L^1/2|--   =>  --|1X|--|1R|--|2R|--|2X|-- 
	 *        |          |         =>     |                 |
	 *
	 */
	ITensor T1R(ipT1, iT1_L);
	ITensor T2R(ipT2, iL_T2);
	ITensor T1X, T2X, sv1XR, sv2XR;
	auto spec = svd( (T1*L)*delta(iT1_L, iL_T2), T1R, sv1XR, T1X, {"Truncate",false});
	if(dbg) Print(spec);
	spec = svd( (T2*L)*delta(iT1_L, iL_T2), T2R, sv2XR, T2X, {"Truncate",false});
	if(dbg) Print(spec);
	T1R = T1R * sv1XR;
	T2R = T2R * sv2XR;

	// D^2 x s x auxD_mpo3s
	auto kd_phys1 = delta(ipT1, mpo2s.Is1);
	T1R = (T1R * kd_phys1) * mpo2s.H1;
	T1R = (T1R * kd_phys1.prime()).prime(PHYS,-1);
	// D^2 x s x auxD_mpo3s^2
	auto kd_phys2 = delta(ipT2, mpo2s.Is2);
	T2R = (T2R * kd_phys2 ) * mpo2s.H2;
	T2R = (T2R * kd_phys2.prime()).prime(PHYS,-1);

	if(dbg) {
		std::cout <<"----- Appyling H1-H2 to |1R--2R> -----"<< std::endl;
		Print(T1R);
    	Print(T2R);
    }

	/*
	 * Perform SVD of new on-site tensors |1R~| and |2R~| by contrating them
	 * along diagonal matrix with weights
	 *
	 *       _______               s1                       s2
	 *  s1~~|       |~~s2           |                        |
	 *    --| 1~ 2~ |--    ==>      |                        |
	 *      |_______|      ==>  --|1~~|++a1++|SV_L12|++a2++|2~~|--
	 *
	 * where 1~~ and 2~~ are now holding singular vectors wrt
	 * to SVD and SV_L12 holds a new weights
	 * We keep only auxBondDim largest singular values
	 *
	 */

	if(dbg) std::cout <<"----- Perform SVD along link12 -----"<< std::endl;
	ITensor SV_L12;
	spec = svd(T1R*delta(iT1_L, iL_T2)*T2R, T1R, SV_L12, T2R, 
		{"Maxm", iT1_L.m(), "Minm", iT1_L.m()});
	if(dbg) {
		Print(T1R);
		Print(spec);
		Print(T2R);
	}

	// Set proper indices to resulting tensors from SVD routine
	Index iT1_SV_L12 = commonIndex(T1R, SV_L12);
	Index iSV_L12_T2 = commonIndex(SV_L12, T2R);

	T1 = (T1R * delta(iT1_SV_L12, iT1_L)) * T1X;
	
	std::vector< double > elemsL(iT1_L.m());
	for (int i=1; i<=iT1_L.m(); i++) {
		elemsL[i-1] = SV_L12.real(iT1_SV_L12(i),iSV_L12_T2(i));
	}
	L = diagTensor(elemsL, iT1_L, iL_T2);
	L = L / norm(L);

	T2 = (T2R * delta(iSV_L12_T2, iL_T2)) * T2X;

	if(dbg) {
		Print(T1);
		PrintData(L);
		Print(T2);
	}
}

// 3 SITE OPS #########################################################

Args simpleUpdate(MPO_3site const& u123, Cluster & cls,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
    auto dbg = args.getBool("suDbg",false);
    auto dbgLvl = args.getInt("suDbgLevel",0);

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

	std::cout<<"GATE: ";
	for(int i=0; i<=3; i++) {
		std::cout<<">-"<<pl[2*i]<<"-> "<<tn[i]<<" >-"<<pl[2*i+1]<<"->"; 
	}
	std::cout<< std::endl;

	if(dbg && (dbgLvl >= 2)) {
		std::cout << u123;
		PrintData(u123.H1);
		PrintData(u123.H2);
		PrintData(u123.H3);
	}
	if(dbg && (dbgLvl >= 2)) {
		Print(cls.sites.at(tn[0]));
		Print(cls.sites.at(tn[1]));
		Print(cls.sites.at(tn[2]));	
	}


	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 

	// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT ***************************
	t_begin_int = std::chrono::steady_clock::now();

	std::vector< ITensor > tmpT;
	ITensor l12, l23;

	for ( const auto& lw : cls.siteToWeights.at(tn[0]) )
		if (lw.dirs[0] == pl[1]) l12 = cls.weights.at(lw.wId);
	for ( const auto& lw : cls.siteToWeights.at(tn[1]) )
		if (lw.dirs[0] == pl[3]) l23 = cls.weights.at(lw.wId);

	tmpT.push_back(cls.sites.at(tn[0]));
	for ( const auto& lw : cls.siteToWeights.at(tn[0]) )
		if (lw.dirs[0] != pl[1]) tmpT.back() *= cls.weights.at(lw.wId);
	
	tmpT.push_back(cls.sites.at(tn[1]));
	for ( const auto& lw : cls.siteToWeights.at(tn[1]) )
		if ( (lw.dirs[0] != pl[2]) && (lw.dirs[0] != pl[3]) ) 
			tmpT.back() *= cls.weights.at(lw.wId);
	
	tmpT.push_back(cls.sites.at(tn[2]));
	for ( const auto& lw : cls.siteToWeights.at(tn[2]) )
		if (lw.dirs[0] != pl[4]) tmpT.back() *= cls.weights.at(lw.wId);

	applyH_123_v2(u123, tmpT[0], tmpT[1], tmpT[2], l12, l23, args);

	for ( const auto& lw : cls.siteToWeights.at(tn[0]) )
		if (lw.dirs[0] != pl[1]) tmpT[0] *= getInvDiagT(cls.weights.at(lw.wId));

	for ( const auto& lw : cls.siteToWeights.at(tn[1]) )
		if ( (lw.dirs[0] != pl[2]) && (lw.dirs[0] != pl[3]) ) 
			tmpT[1] *= getInvDiagT(cls.weights.at(lw.wId));

	for ( const auto& lw : cls.siteToWeights.at(tn[2]) )
		if (lw.dirs[0] != pl[4]) tmpT[2] *= getInvDiagT(cls.weights.at(lw.wId));

	for (int i=0; i<3; i++) cls.sites[tn[i]] = tmpT[i];

	for ( const auto& lw : cls.siteToWeights.at(tn[0]) )
		if (lw.dirs[0] == pl[1]) cls.weights[lw.wId] = l12;
	for ( const auto& lw : cls.siteToWeights.at(tn[1]) )
		if (lw.dirs[0] == pl[3]) cls.weights[lw.wId] = l23;

	return diag_data;
}

Args simpleUpdate(MPO_2site const& u12, Cluster & cls,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
    auto dbg = args.getBool("suDbg",false);
    auto dbgLvl = args.getInt("suDbgLevel",0);

    double machine_eps = std::numeric_limits<double>::epsilon();
	if(dbg && (dbgLvl >= 1)) std::cout<< "M EPS: " << machine_eps << std::endl;

	std::chrono::steady_clock::time_point t_begin_int, t_end_int;

    // prepare to hold diagnostic data
    Args diag_data = Args::global();

	if(dbg && dbgLvl >=2 ) {
		std::cout<<"GATE: ";
		std::cout<< tn[0] <<" >- "<<pl[0]<<" -> "<<pl[1]<<" >- "<<tn[1]<< std::endl; 

		PrintData(u12.H1);
		PrintData(u12.H2);
	
		Print(cls.sites.at(tn[0]));
		Print(cls.sites.at(tn[1]));
	}


	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS ************************
	// ***** SET UP NECESSARY MAPS AND CONSTANT TENSORS DONE ******************* 

	// ***** COMPUTE "EFFECTIVE" REDUCED ENVIRONMENT ***************************
	t_begin_int = std::chrono::steady_clock::now();

	std::vector< ITensor > tmpT;
	ITensor l12;

	for ( const auto& lw : cls.siteToWeights.at(tn[0]) )
		if (lw.dirs[0] == pl[0]) l12 = cls.weights.at(lw.wId);

	tmpT.push_back(cls.sites.at(tn[0]));
	for ( const auto& lw : cls.siteToWeights.at(tn[0]) )
		if (lw.dirs[0] != pl[0]) tmpT.back() *= cls.weights.at(lw.wId);
	
	tmpT.push_back(cls.sites.at(tn[1]));
	for ( const auto& lw : cls.siteToWeights.at(tn[1]) )
		if ( lw.dirs[0] != pl[1] ) tmpT.back() *= cls.weights.at(lw.wId);

	applyH_T1_L_T2_v2(u12, tmpT[0], tmpT[1], l12, dbg && (dbgLvl>=3) );

	for ( const auto& lw : cls.siteToWeights.at(tn[0]) )
		if (lw.dirs[0] != pl[0]) tmpT[0] *= getInvDiagT(cls.weights.at(lw.wId));

	for ( const auto& lw : cls.siteToWeights.at(tn[1]) )
		if ( lw.dirs[0] != pl[1] ) tmpT[1] *= getInvDiagT(cls.weights.at(lw.wId));

	for (int i=0; i<2; i++) cls.sites[tn[i]] = tmpT[i];

	for ( const auto& lw : cls.siteToWeights.at(tn[0]) )
		if (lw.dirs[0] == pl[0]) cls.weights[lw.wId] = l12;

	return diag_data;
}

Args simpleUpdate(OpNS const& u12, Cluster & cls,
	std::vector<std::string> tn, std::vector<int> pl,
	Args const& args) {
 
	return Args::global();
}


void applyH_123_v1(MPO_3site const& mpo3s, 
	ITensor & T1, ITensor & T2, ITensor & T3, ITensor & l12, ITensor & l23,
	bool dbg) {
	/* Input is assumed to define following TN
	 *
	 *        s1                 s2                 s3 
	 *     | /                | /                | /
     *  --|T1|--a1--l12--a2--|T2|--a3--l23--a4--|T3|
     *     |                  |                  |   
     *
     */
    if(dbg) { Print(T1);
	    Print(T2);
	    Print(T3); } 

    Index s1 = noprime(findtype(T1, PHYS));
    Index s2 = noprime(findtype(T2, PHYS));
    Index s3 = noprime(findtype(T3, PHYS));
    Index a1 = commonIndex(T1,l12);
    Index a2 = commonIndex(l12,T2);
    Index a3 = commonIndex(T2,l23);
    Index a4 = commonIndex(l23,T3);
    if(dbg) {Print(a1);
	    Print(a4);

		std::cout <<">>>>> applyH_123_v1 called <<<<<"<< std::endl;
		std::cout << mpo3s;
		Print(mpo3s.H1);
	    Print(mpo3s.H2);
	    Print(mpo3s.H3);

		Print(l12);
		Print(l23); }

    // STEP 2 Decompose T1, T2, T3 to get subtensors upon which we act
	/*
	 * First we decompose the on-site tensors T1, T2, T3 to simpler 
	 * objects containing only the links over which H1--H2--H3 acts
	 * 
	 * 
	 *
	 *      s1                          s1
	 *    | /            |             / 
	 * --|T1|--a1 => --|rT1|--<SV1>--|mT1|--a1
	 *    |              |
	 *
	 *         s2              s2
	 *      | /               /
	 * a1--|T2|--a4 => a1--|mT2|--a2 
	 *      |                | 
     *                    <SV2>
     *                       | /
     *                     |rT2|
     *                      /
     *                
     *         s3           s3
     *      | /            /             |
     * a4--|T3|-- => a4--|mT3|--<SV3>--|rT3|--
     *      |                            |
     *
     */

    ITensor rT1, mT1, svT1, rT2, mT2, svT2, rT3, mT3, svT3, sv1, sv2; 
    mT1 = ITensor(s1, a1);
    mT2 = ITensor(s2, a2, a3);
    mT3 = ITensor(s3, a4);
    svd(T1, mT1, svT1, rT1);
	svd(T2, mT2, svT2, rT2);
	svd(T3, mT3, svT3, rT3);
	mT1 = mT1 * svT1;
	mT2 = mT2 * svT2;
	mT3 = mT3 * svT3;

	Index am1 = commonIndex(rT1, mT1);
	Index am2 = commonIndex(rT2, mT2);
	Index am3 = commonIndex(rT3, mT3);

	if(dbg) {Print(mT1);
		Print(mT2);
		Print(mT3); }

	/*
	 * Applying 3-site MPO leads to a new tensor network of the form
	 *
	 *        am1 
	 *         |     __                   s1 s2  s3
	 *       |mT1|~~|H1|~~s1             __|__|__|__
	 *         |     |        ==        |           |
	 *  am2--|mT2|~~|H2|~~s2  ==   am1--|1~  2~  3~ |--am3  
	 *         |     |        ==        |___________|
	 *       |mT3|~~|H3|~~s3                 |
	 *         |                            am2
	 *        am3
	 *
	 * Indices s1,s2,s3 are relabeled back to physical indices of 
	 * original sites 1,2 and 3 after applying MPO.
	 *
	 */
	// std::cout <<"----- Appyling H1-H2-H3 to |123> -----"<< std::endl;
	ITensor res = (mT1*delta(s1,mpo3s.Is1))*l12*(mT2*delta(s2,mpo3s.Is2))
		*l23*(mT3*delta(s3,mpo3s.Is3))*mpo3s.H1*mpo3s.H2*mpo3s.H3;
	//Print(res);
	res = ((res.noprime(PHYS)*delta(s1,mpo3s.Is1))*delta(s2,mpo3s.Is2))
		*delta(s3,mpo3s.Is3);
	//Print(res);
	ITensor orig = res;

	/*
	 * Perform SVD to extract the tensor associated to s1 and truncate
	 * the resulting SVD matrices back to auxBond dimension 
	 *
	 *          s1 s2 s3
	 *        __|__|__|__                   s1                 s2  s3
	 *       |           |      ==>         |                  |   |
	 *  am1--|1~  2~  3~ |--am3 ==>  am1--|mT1|--n1--l12--n2--|2~ 3~|--am3  
	 *       |___________|      ==>                            |
	 *             |                                          am2 
	 *            am2
	 *
	 */
	mT1 = ITensor(s1,am1);
	svd(res, mT1, sv1, res, {"Maxm", a1.m(), "Minm", a1.m()});
	Index n1 = commonIndex(mT1,sv1);
	Index n2 = commonIndex(sv1,res);

	if(dbg) {Print(mT1);
		Print(sv1); }

	/*
	 * Perform SVD of tensor res from previous SVD and again reduce
	 * the resulting SV matrix
	 *
	 *            s2  s3                    s2                 s3
	 *           __|___|_                   |                  |
	 *      n2--| 2~  3~ |--am3  ==>  n2--|mT2|--n3--l23--n4--|mT3|--am3
     *            |                         |
	 *           am2                       am2
	 *
	 */
	mT2 = ITensor(n2,s2,am2);
	svd(res, mT2, sv2, mT3, {"Maxm", a1.m(), "Minm", a1.m()});
	Index n3 = commonIndex(mT2,sv2);
	Index n4 = commonIndex(sv2,mT3);

	if(dbg) {Print(mT2);
		Print(sv2);
		Print(mT3); }

	// Reconstruct on-site tensors by contraction with remainders
	T1 = (rT1*mT1) *delta(n1,a1);
	T2 = ((rT2*mT2) *delta(n2,a2)) *delta(n3,a3);
	T3 = (rT3*mT3) *delta(n4,a4);

	for (int i=1; i<=a1.m(); i++) {
		l12.set(a1(i),a2(i), sv1.real(n1(i),n2(i)));
		l23.set(a3(i),a4(i), sv2.real(n3(i),n4(i)));
	}
	l12 = l12 / norm(l12);
	l23 = l23 / norm(l23);

	if(dbg) {Print(T1);
		Print(T2);
		Print(T3);
		Print(l12);
		Print(l23); }
}

void applyH_123_v2(MPO_3site const& mpo3s, ITensor & T1, ITensor & T2, ITensor & T3,
	ITensor & l12, ITensor & l23, Args const& args) {

	auto dbg = args.getBool("suDbg",false);
    auto dbgLvl = args.getInt("suDbgLevel",0);

	const size_t auxd     = commonIndex(T1,l12).m();
	const Real svCutoff   = 1.0e-14;

	/* Input is assumed to define following TN
	 *
	 *        s1                 s2                 s3 
	 *     | /                | /                | /
     *  --|T1|--a1--l12--a2--|T2|--a3--l23--a4--|T3|
     *     |                  |                  |   
     *
     */
    Index s1 = noprime(findtype(T1, PHYS));
    Index s2 = noprime(findtype(T2, PHYS));
    Index s3 = noprime(findtype(T3, PHYS));
    Index a1 = commonIndex(T1,l12);
    Index a2 = commonIndex(l12,T2);
    Index a3 = commonIndex(T2,l23);
    Index a4 = commonIndex(l23,T3);
    
    if(dbg) { 
    	std::cout <<">>>>> applyH_123_v2 called <<<<<"<< std::endl;
		std::cout << mpo3s;

		Print(T1);
		Print(T2);
		Print(T3);

		Print(l12);
		Print(l23);

    	Print(a1);
    	Print(a2);
    	Print(a3);
	    Print(a4); 
	}

    // STEP 2 Decompose T1, T2, T3 to get subtensors upon which we act
	/*
	 * First we decompose the on-site tensors T1, T2, T3 to simpler 
	 * objects containing only the links over which H1--H2--H3 acts
	 * 
	 * 
	 *
	 *      s1                          s1
	 *    | /            |             / 
	 * --|T1|--a1 => --|rT1|--<SV1>--|mT1|--a1
	 *    |              |
	 *
	 *         s2              s2
	 *      | /               /
	 * a1--|T2|--a4 => a1--|mT2|--a2 
	 *      |                | 
     *                    <SV2>
     *                       | /
     *                     |rT2|
     *                      /
     *                
     *         s3           s3
     *      | /            /             |
     * a4--|T3|-- => a4--|mT3|--<SV3>--|rT3|--
     *      |                            |
     *
     */

	ITensor rT1, mT1, svT1, rT2, mT2, svT2, rT3, mT3, svT3, sv1, sv2;  
    mT1 = ITensor(s1, a1);
    mT2 = ITensor(s2, a2, a3);
    mT3 = ITensor(s3, a4);
    svd(T1, mT1, svT1, rT1);
	svd(T2, mT2, svT2, rT2);
	svd(T3, mT3, svT3, rT3);
	mT1 = mT1 * svT1;
	mT2 = mT2 * svT2;
	mT3 = mT3 * svT3;

	Index am1 = commonIndex(rT1, mT1);
	Index am2 = commonIndex(rT2, mT2);
	Index am3 = commonIndex(rT3, mT3);

	if(dbg) { 
		Print(mT1);
		Print(mT2);
		Print(mT3); 
	}

	/*
	 * Applying 3-site MPO leads to a new tensor network of the form
	 *
	 *        am1 
	 *         |     __                   s1 s2  s3
	 *       |mT1|~~|H1|~~s1             __|__|__|__
	 *         |     |        ==        |           |
	 *  am2--|mT2|~~|H2|~~s2  ==   am1--|1~  2~  3~ |--am3  
	 *         |     |        ==        |___________|
	 *       |mT3|~~|H3|~~s3                 |
	 *         |                            am2
	 *        am3
	 *
	 * Indices s1,s2,s3 are relabeled back to physical indices of 
	 * original sites 1,2 and 3 after applying MPO.
	 *
	 */
	// std::cout <<"----- Appyling H1-H2-H3 to |123> -----"<< std::endl;
	ITensor res = (mT1*delta(s1,mpo3s.Is1))*l12*(mT2*delta(s2,mpo3s.Is2))
		*l23*(mT3*delta(s3,mpo3s.Is3))*mpo3s.H1*mpo3s.H2*mpo3s.H3;
	if(dbg) Print(res);
	res = ((res.noprime(PHYS)*delta(s1,mpo3s.Is1))*delta(s2,mpo3s.Is2))
		*delta(s3,mpo3s.Is3);
	if(dbg) Print(res);

	/*
	 * Perform SVD to extract the tensor associated to s1 and truncate
	 * the resulting SVD matrices back to auxBond dimension 
	 *
	 *          s1 s2 s3
	 *        __|__|__|__                   s1                 s2  s3
	 *       |           |      ==>         |                  |   |
	 *  am1--|1~  2~  3~ |--am3 ==>  am1--|mT1|--n1--l12--n2--|2~ 3~|--am3  
	 *       |___________|      ==>                            |
	 *             |                                          am2 
	 *            am2
	 *
	 */
	mT1 = ITensor(s1,am1);
	svd(res, mT1, sv1, res, {"Maxm", auxd, "Cutoff", svCutoff});
	Index n1 = commonIndex(mT1,sv1);
	Index n2 = commonIndex(sv1,res);

	// Normalize sv1
	sv1 = sv1 / norm(sv1);
	if(dbg) { Print(mT1);
		PrintData(sv1); }

	//PRB 82, 245119, 2010
	/*
	 * Perform SVD of tensor obtained by contrating result from previous SVD
	 * with matrix of singular values and again reduce the resulting SV matrix
	 *
	 *            s2  s3                    s2                 s3
	 *        ____|___|__                   |                  |
	 *   n1--|l12 2~  3~ |--am3  ==>  n1--|mT2|--n3--l23--n4--|mT3|--am3
     *            |                         |
	 *           am2                       am2
	 *
	 */
	mT2 = ITensor(n1,s2,am2);
	svd(res*sv1, mT2, sv2, mT3, {"Maxm", auxd, "Cutoff", svCutoff});
	Index n3 = commonIndex(mT2,sv2);
	Index n4 = commonIndex(sv2,mT3);

	// Normalize sv2
	sv2 = sv2 / norm(sv2);
	if(dbg) { Print(mT2);
		PrintData(sv1);
		PrintData(sv2);
		Print(mT3); }

	// Prepare results
	std::vector<double> tmpD(a1.m(), 0.0);
	for (size_t i=1; i<=n1.m(); ++i) tmpD[i-1] = sv1.real(n1(i),n2(i));
	l12 = diagTensor(tmpD, a1, a2);

	tmpD = std::vector<double>(a3.m(), 0.0);
	for (size_t i=1; i<=n3.m(); ++i) tmpD[i-1] = sv2.real(n3(i),n4(i));
	l23 = diagTensor(tmpD, a3, a4);

	auto inverseT = [](Real r) { return 1.0/r; };
	sv1.apply(inverseT);

	// Reconstruct on-site tensors by contraction with remainders
	T1 = (rT1*mT1) *delta(n1,a1);
	T2 = (((rT2*mT2) *sv1) *delta(n2,a2)) *delta(n3,a3);
	T3 = (rT3*mT3) *delta(n4,a4);

	if(dbg) { Print(T1);
		Print(T2);
		Print(T3);
		Print(l12);
		Print(l23); }
}

void applyH_123_v3(MPO_3site const& mpo3s, 
	ITensor & T1, ITensor & T2, ITensor & T3, ITensor & l12, ITensor & l23,
	ITensor & l12I, ITensor & l23I, bool dbg) {

	const size_t auxd     = commonIndex(T1,l12).m();
	const Real svCutoff   = 1.0e-14;

	/* Input is assumed to define following TN
	 *
	 *        s1                 s2                 s3 
	 *     | /                | /                | /
     *  --|T1|--a1--l12--a2--|T2|--a3--l23--a4--|T3|
     *     |                  |                  |   
     *
     */
    if(dbg) { Print(T1);
	    Print(T2);
	    Print(T3); } 

    Index s1 = noprime(findtype(T1, PHYS));
    Index s2 = noprime(findtype(T2, PHYS));
    Index s3 = noprime(findtype(T3, PHYS));
    Index a1 = commonIndex(T1,l12);
    Index a2 = commonIndex(l12,T2);
    Index a3 = commonIndex(T2,l23);
    Index a4 = commonIndex(l23,T3);
    if(dbg) {Print(a1);
	    Print(a4);

		std::cout <<">>>>> applyH_123_v1 called <<<<<"<< std::endl;
		std::cout << mpo3s;
		Print(mpo3s.H1);
	    Print(mpo3s.H2);
	    Print(mpo3s.H3);

		Print(l12);
		Print(l23); }	

	double pw;
	auto pow_T = [&pw](double r) { return std::pow(r,pw); };

	// STEP 1 Absorb sqrt of l12 and l23 to tensor T1, T2, T3
	pw = 0.5;
	l12.apply(pow_T);
	l23.apply(pow_T);

	T1 = (T1 * l12) *delta(a2, a1); //--a1
	T2 = (l12 * T2 * l23); // a1-- --a4
	T3 = (T3 * l23) *delta(a3, a4); //--a4
	
	if(dbg) {Print(T1);
		Print(T2);
		Print(T3); }

    // STEP 2 Decompose T1, T2, T3 to get subtensors upon which we act
	/*
	 * First we decompose the on-site tensors T1, T2, T3 to simpler 
	 * objects containing only the links over which H1--H2--H3 acts
	 * 
	 * 
	 *
	 *      s1                          s1
	 *    | /            |             / 
	 * --|T1|--a1 => --|rT1|--<SV1>--|mT1|--a1
	 *    |              |
	 *
	 *         s2              s2
	 *      | /               /
	 * a1--|T2|--a4 => a1--|mT2|--a2 
	 *      |                | 
     *                    <SV2>
     *                       | /
     *                     |rT2|
     *                      /
     *                
     *         s3           s3
     *      | /            /             |
     * a4--|T3|-- => a4--|mT3|--<SV3>--|rT3|--
     *      |                            |
     *
     */

    ITensor rT1, mT1, svT1, rT2, mT2, svT2, rT3, mT3, svT3, sv1, sv2;
    mT1 = ITensor(s1, a1);
    mT2 = ITensor(s2, a1, a4);
    mT3 = ITensor(s3, a4);
    svd(T1, mT1, svT1, rT1);
	svd(T2, mT2, svT2, rT2);
	svd(T3, mT3, svT3, rT3);
	mT1 = mT1 * svT1;
	mT2 = mT2 * svT2;
	mT3 = mT3 * svT3;

	Index am1 = commonIndex(rT1, mT1);
	Index am2 = commonIndex(rT2, mT2);
	Index am3 = commonIndex(rT3, mT3);

	if(dbg) {Print(mT1);
		Print(mT2);
		Print(mT3); }

	/*
	 * Applying 3-site MPO leads to a new tensor network of the form
	 *
	 *        am1 
	 *         |     __                   s1 s2  s3
	 *       |mT1|~~|H1|~~s1             __|__|__|__
	 *         |     |        ==        |           |
	 *  am2--|mT2|~~|H2|~~s2  ==   am1--|1~  2~  3~ |--am3  
	 *         |     |        ==        |___________|
	 *       |mT3|~~|H3|~~s3                 |
	 *         |                            am2
	 *        am3
	 *
	 * Indices s1,s2,s3 are relabeled back to physical indices of 
	 * original sites 1,2 and 3 after applying MPO.
	 *
	 */
	// std::cout <<"----- Appyling H1-H2-H3 to |123> -----"<< std::endl;
	ITensor res = (mT1*delta(s1,mpo3s.Is1))*(mT2*delta(s2,mpo3s.Is2))
		*(mT3*delta(s3,mpo3s.Is3))*mpo3s.H1*mpo3s.H2*mpo3s.H3;
	res = ((res.noprime(PHYS)*delta(s1,mpo3s.Is1))*delta(s2,mpo3s.Is2))
		*delta(s3,mpo3s.Is3);
	ITensor orig = res;

	/*
	 * Perform SVD to extract the tensor associated to s1 and truncate
	 * the resulting SVD matrices back to auxBond dimension 
	 *
	 *          s1 s2 s3
	 *        __|__|__|__                   s1                 s2  s3
	 *       |           |      ==>         |                  |   |
	 *  am1--|1~  2~  3~ |--am3 ==>  am1--|mT1|--n1--l12--n2--|2~ 3~|--am3  
	 *       |___________|      ==>                            |
	 *             |                                          am2 
	 *            am2
	 *
	 */
	mT1 = ITensor(s1,am1);
	svd(res, mT1, sv1, res, {"Maxm", a1.m(), "Cutoff", svCutoff});
	Index n1 = commonIndex(mT1,sv1);
	Index n2 = commonIndex(sv1,res);

	sv1 = sv1 / norm(sv1);
	if(dbg) {Print(mT1);
		Print(sv1); }

	//PRB 82, 245119, 2010
	/*
	 * Perform SVD of tensor obtained by contrating result from previous SVD
	 * with matrix of singular values and again reduce the resulting SV matrix
	 *
	 *            s2  s3                    s2                 s3
	 *        ____|___|__                   |                  |
	 *   n1--|l12 2~  3~ |--am3  ==>  n1--|mT2|--n3--l23--n4--|mT3|--am3
     *            |                         |
	 *           am2                       am2
	 *
	 */
	mT2 = ITensor(n1,s2,am2);
	svd(res*sv1, mT2, sv2, mT3, {"Maxm", a1.m(), "Cutoff", svCutoff});
	Index n3 = commonIndex(mT2,sv2);
	Index n4 = commonIndex(sv2,mT3);

	sv2 = sv2 / norm(sv2);
	if(dbg) {Print(mT2);
		Print(sv2);
		Print(mT3); }
	
	// Prepare results
	for (size_t i=1; i<=a1.m(); ++i) {
		if(i <= n1.m()) {
			l12.set(a1(i),a2(i), sv1.real(n1(i),n2(i)));
			l12I.set(a1(i),a2(i), 1.0/sv1.real(n1(i),n2(i)));
		} else {
			l12.set(a1(i),a2(i), 0.0);
			l12I.set(a1(i),a2(i), 0.0);
		}
	}

	for (size_t i=1; i<=a3.m(); ++i) {
		if(i <= n3.m()) {
			l23.set(a3(i),a4(i), sv2.real(n3(i),n4(i)));
			l23I.set(a3(i),a4(i), 1.0/sv2.real(n3(i),n4(i)));
		} else {
			l23.set(a3(i),a4(i), 0.0);
			l23I.set(a3(i),a4(i), 0.0);
		}
	}

	auto inverseT = [](Real r) { return 1.0/r; };
	sv1.apply(inverseT);

	// Reconstruct on-site tensors by contraction with remainders
	T1 = (rT1*mT1) *delta(n1,a1);
	T2 = (((rT2*mT2)* sv1) *delta(n2,a2)) *delta(n3,a3);
	T3 = (rT3*mT3) *delta(n4,a4);

	if(dbg) {Print(T1);
		Print(T2);
		Print(T3);
		Print(l12);
		Print(l23); }
}

ITensor getInvDiagT(ITensor const& t) {
	double machine_eps = std::numeric_limits<double>::epsilon();

	std::vector<double> tmpD;

	double elem;
	for(int i=1; i<=t.inds()[0].m(); i++) {
		elem = t.real(t.inds()[0](i),t.inds()[1](i));
		if ( std::abs(elem) > machine_eps) 
			tmpD.push_back( 1.0 / elem );
		else
			tmpD.push_back(0.0);

	}

	return diagTensor(tmpD, t.inds()[0], t.inds()[1]);
}

std::ostream& 
operator<<(std::ostream& s, MPO_2site const& mpo2s) {
	s <<"----- BEGIN MPO_2site "<< std::string(50,'-') << std::endl;
	s << mpo2s.Is1 <<" "<< mpo2s.Is2 << std::endl;
	s <<"H1 "<< mpo2s.H1 << std::endl;
	s <<"H2 "<< mpo2s.H2;
	s <<"----- END MPO_2site "<< std::string(52,'-') << std::endl;
	return s; 
}