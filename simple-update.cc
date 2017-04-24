#include "simple-update.h"

using namespace itensor;

// TODO the multiplication by negative) scalar is not uniquely defined
MPO_3site operator*(double scalar, MPO_3site const& mpo3s ) {
	MPO_3site result;
	if (scalar > 0.0) {
		// use "symmetric" variant
		result.H1 = mpo3s.H1 * std::pow(scalar,1.0/3.0);
		result.H2 = mpo3s.H2 * std::pow(scalar,1.0/3.0);
		result.H3 = mpo3s.H3 * std::pow(scalar,1.0/3.0);
	} else {
		// not implemented, throw error
		std::cout <<"neg.scalar*MPO_3site not supported"<< std::endl;
		exit(EXIT_FAILURE);
	}
	result.Is1 = mpo3s.Is1;
	result.Is2 = mpo3s.Is2;
	result.Is3 = mpo3s.Is3;
	return result;
}

MPO_3site getMPO3s_Id(int physDim) {
	MPO_3site mpo3s;
	
	// Define physical indices
	mpo3s.Is1 = Index(TAG_MPO3S_PHYS1,physDim,PHYS);
	mpo3s.Is2 = Index(TAG_MPO3S_PHYS2,physDim,PHYS);
	mpo3s.Is3 = Index(TAG_MPO3S_PHYS3,physDim,PHYS);

    ITensor idpI1(mpo3s.Is1,prime(mpo3s.Is1,1));
    ITensor idpI2(mpo3s.Is2,prime(mpo3s.Is2,1));
    ITensor idpI3(mpo3s.Is3,prime(mpo3s.Is3,1));
    for (int i=1;i<=physDim;i++) {
        idpI1.set(mpo3s.Is1(i),prime(mpo3s.Is1,1)(i),1.0);
        idpI2.set(mpo3s.Is2(i),prime(mpo3s.Is2,1)(i),1.0);
        idpI3.set(mpo3s.Is3(i),prime(mpo3s.Is3,1)(i),1.0);
    }

    ITensor id3s = idpI1*idpI2*idpI3;
    //PrintData(id3s);

    mpo3s.H1 = ITensor(mpo3s.Is1,prime(mpo3s.Is1));
    ITensor SV_12,temp;
    svd(id3s,mpo3s.H1,SV_12,temp);
    
    PrintData(mpo3s.H1);
    PrintData(SV_12);
    Print(temp);

    //Index cmnI = commonIndex(mpo3s.H1,SV_12);
    // Define aux indices linking the on-site MPOs
	//Index iMPO3s12(TAG_MPO3S_12LINK,cmnI.m(),MPOLINK);
	//Index iMPO3s23(TAG_MPO3S_23LINK,1,MPOLINK);
    //mpo3s.H1 *= delta(cmnI,iMPO3s12)

	Index a2 = commonIndex(SV_12,temp);
	mpo3s.H2 = ITensor(mpo3s.Is2,prime(mpo3s.Is2,1),a2);
	ITensor SV_23;
    svd(temp,mpo3s.H2,SV_23,mpo3s.H3);

    PrintData(mpo3s.H1);
    PrintData(SV_12);
    PrintData(mpo3s.H2);
    PrintData(SV_23);
    PrintData(mpo3s.H3);

    /*
     *   |                      |                      |
     *  |H1|--a1--<SV_12>--a2--|H2|--a3--<SV_23>--a4--|H3|
     *   |                      |                      |
     *
     */

	//create a lambda function
	//which returns the square of its argument
	auto sqrt_T = [](Real r) { return sqrt(r); };

	SV_12.apply(sqrt_T);
	SV_23.apply(sqrt_T);

	Index a1 = commonIndex(mpo3s.H1,SV_12);
	// Define aux indices linking the on-site MPOs
	Index iMPO3s12(TAG_MPO3S_12LINK,a2.m(),MPOLINK);
	mpo3s.H1 = (mpo3s.H1 * SV_12) * delta(a2,iMPO3s12);
	
	Index a3 = commonIndex(mpo3s.H2,SV_23);
	Index a4 = commonIndex(SV_23,mpo3s.H3);
	Index iMPO3s23(TAG_MPO3S_23LINK,a3.m(),MPOLINK);
	mpo3s.H2 = delta(iMPO3s12,a1)*(SV_12 * mpo3s.H2 * SV_23)
		*delta(a4,iMPO3s23);
	
	mpo3s.H3 = (mpo3s.H3 * SV_23) * delta(a3,iMPO3s23);

	PrintData(mpo3s.H1);
	PrintData(mpo3s.H2);
	PrintData(mpo3s.H3);

	PrintData(mpo3s.H2*mpo3s.H3);

	svd(mpo3s.H2*mpo3s.H3,mpo3s.H3,SV_23,mpo3s.H2);
	PrintData(mpo3s.H2);
    PrintData(SV_23);
    PrintData(mpo3s.H3);

    SV_23.apply(sqrt_T);
    
    a3 = commonIndex(mpo3s.H2,SV_23);
    a4 = commonIndex(SV_23,mpo3s.H3);
    iMPO3s23 = Index(TAG_MPO3S_23LINK,a3.m(),MPOLINK);
    mpo3s.H2 = (mpo3s.H2 * SV_23) * delta(a4,iMPO3s23);
	mpo3s.H3 = (mpo3s.H3 * SV_23) * delta(a3,iMPO3s23);

	PrintData(mpo3s.H1);
	PrintData(mpo3s.H2);
	PrintData(mpo3s.H3);

	return mpo3s;
}

void applyH_123(MPO_3site const& mpo3s, 
	ITensor & T1, ITensor & T2, ITensor & T3, 
	std::pair<Index, Index> const& link12, 
	std::pair<Index, Index> const& link23) {

	std::cout << mpo3s;
	PrintData(mpo3s.H1);
    PrintData(mpo3s.H2);
    PrintData(mpo3s.H3);

	std::cout <<"link12 "<< link12.first <<" "<< link12.second << std::endl;
	std::cout <<"link23 "<< link23.first <<" "<< link23.second << std::endl;

	/*
	 * Applying 3-site MPO leads to a new tensor network of the form
	 * 
	 *    \ |    __               s1    s2    s3
	 *   --|1|~~|H1|~~s1          |_    |_    |_
	 *    \ |     |       ==   --|  |  |  |  |  |--
	 *   --|2|~~|H2|~~s2  ==   --|1~|==|2~|==|3~|--  
	 *    \ |     |       ==   --|__|  |__|  |__|--
	 *   --|3|~~|H3|~~s3                ||
	 *      |                       
	 *
	 * Indices s1,s2,s3 are relabeled back to physical indices of 
	 * original sites 1,2 and 3 after applying MPO.
	 * Now auxiliary index linking sites 1,2 & 3 have dimension increased 
	 * from D to D*auxDim of applied 3-site MPO. To obtain the network 
	 * of the original size, we have to reduce dimensions of these links 
	 * back to D by performing SVDs along links 12 and 23.
	 *
	 */

	std::cout <<"----- Initial |123> -----"<< std::endl;
	Print(T1);
	Print(T2);
	Print(T3);

	// D^4 x Ds x auxD_mpo3s
	ITensor kd_phys1 = delta(findtype(T1.inds(),PHYS), mpo3s.Is1);
	T1 *= mpo3s.H1*kd_phys1;
	T1 = (T1*kd_phys1.prime()).prime(PHYS,-1);
	// D^4 x Ds x auxD_mpo3s^2
	ITensor kd_phys2 = delta(findtype(T2.inds(),PHYS), mpo3s.Is2);
	T2 *= mpo3s.H2*kd_phys2;
	T2 = (T2*kd_phys2.prime()).prime(PHYS,-1);
	// D^4 x Ds x auxD_mpo3s
	ITensor kd_phys3 = delta(findtype(T3.inds(),PHYS), mpo3s.Is3);
	T3 *= mpo3s.H3*kd_phys3;
	T3 = (T3*kd_phys3.prime()).prime(PHYS,-1);

	std::cout <<"----- Appyling H1-H2-H3 to |123> -----"<< std::endl;
	PrintData(T1);
    PrintData(T2);
    PrintData(T3);

	/*
	 * Perform SVD of new on-site tensors |1~| and |2~| by contrating them
	 * along common index
	 *
	 *       _______               s1               s2
	 *  s1~~|       |~~s2           |                |
	 *    --| 1~ 2~ |==    ==>  --|   |            |   |
	 *    --|       |--    ==>  --|1~~|++|SV_L12|++|2~~|==
	 *    --|_______|--         --|___|            |___|
	 *                                               ||
	 *
	 * where 1~~ and 2~~ are now holding singular vectors wrt
	 * to SVD done along link between sites 1 and 2
	 *
	 */
	std::cout <<"----- Perform SVD along link12 -----"<< std::endl;
	ITensor SV_L12;
	svd(T1*delta(link12.first, link12.second)*T2, T1, SV_L12, T2);

	Print(T1);
	PrintData(SV_L12);
	Print(T2);

	// Take the square-root of SV's
	auto sqrtT = [](Real r) { return sqrt(r); };

	std::cout <<"----- (NOT)Reducing dimension of link12 -----"<< std::endl;
	SV_L12.apply(sqrtT);
	T1 *= SV_L12;
	ITensor dR_12L = delta( commonIndex(T1,SV_L12), link12.first );
	T1 *= dR_12L;

	T2 *= SV_L12;
	ITensor dR_12R = delta( commonIndex(T2,SV_L12), link12.second );
	T2 *= dR_12R;

	PrintData(SV_L12);
	PrintData(T1);
	Print(T2);

	/*
	 * Perform SVD of new on-site tensors |2~~| and |3~| by contrating them
	 * along common index
	 *
	 *       _______               s2               s3
	 *  s2~~|       |~~s3           |                |
	 *    ++| 2~~ 3~|--    ==>    |    |            |   |--
	 *    --|       |--    ==>  ++|2~~~|++|SV_L23|++|3~~|--
	 *    --|_______|--           |____|            |___|--
	 *                              ||
	 *
	 * where 1~~ and 2~~ are now holding singular vectors wrt
	 * to SVD done along link between sites 1 and 2
	 *
	 */
	std::cout <<"----- Perform SVD along link23 -----"<< std::endl;
	ITensor SV_L23;
	svd(T2*delta(link23.first, link23.second)*T3, T2, SV_L23, T3);

	Print(T2);
	PrintData(SV_L23);
	Print(T3);

	/*
	 * We absorb SV matrices into new on-site tensors in symmetric fashion,
	 * absorbing square root of SV matrix of the link to both on-site
	 * tensors which it connects. Then we discard the excess SV values
	 * to reduce back to the auxBond dimension D
	 *
	 * ++|SV_L12|++ => ++|SV_L12^1/2|++|SV_L12^1/2|++
	 *
	 *     s1
	 *     |
	 *  --|   |                                     ==>    \ |
	 *  --|1~~|++|SV_L12^1/2|++|dR1|--link12.first  ==>  --|1n|~~s1
	 *  --|___|                                     ==>      |
	 *
	 *  where dR is a reduction tensor, which discards all the SV values
	 *  except the first D. Analogous operation is performed on the |2~~~|
	 *  on-site tensor
	 *
	 *                                        s2
     *                                        |       
	 *                                      |    |    ==>   \ |
	 *  link12.second--|dR2|++|SV_L12^1/2|++|2~~~|++  ==> --|2n|~~s2
	 *                                      |____|    ==>     |
	 *                                        ||
	 */

	std::cout <<"----- (NOT)Reducing dimension of link23 -----"<< std::endl;
	SV_L23.apply(sqrtT);
	T2 *= SV_L23;
	ITensor dR_23L = delta( commonIndex(T2,SV_L23), link23.first );
	T2 *= dR_23L;

	T3 *= SV_L23;
	ITensor dR_23R = delta( commonIndex(T3,SV_L23), link23.second );
	T3 *= dR_23R;

	PrintData(SV_L23);
	PrintData(T2);
	PrintData(T3);
}

std::ostream& 
operator<<(std::ostream& s, MPO_3site const& mpo3s) {
	s <<"----- BEGIN MPO_3site "<< std::string(50,'-') << std::endl;
	s << mpo3s.Is1 <<" "<< mpo3s.Is2 <<" "<< mpo3s.Is3 << std::endl;
	s <<"H1 "<< mpo3s.H1 << std::endl;
	s <<"H2 "<< mpo3s.H2 << std::endl;
	s <<"H3 "<< mpo3s.H3;
	s <<"----- END MPO_3site "<< std::string(52,'-') << std::endl;
	return s; 
} 