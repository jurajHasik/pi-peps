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

	// Define aux indices linking the on-site MPOs
	Index iMPO3s12(TAG_MPO3S_12LINK,1,MPOLINK);
	Index iMPO3s23(TAG_MPO3S_23LINK,1,MPOLINK);

	mpo3s.H1 = ITensor(iMPO3s12,mpo3s.Is1,prime(mpo3s.Is1));
	mpo3s.H2 = ITensor(iMPO3s12,iMPO3s23,mpo3s.Is2,prime(mpo3s.Is2));
	mpo3s.H3 = ITensor(iMPO3s23,mpo3s.Is3,prime(mpo3s.Is3));

	// Define tensor elements
	for (int i=1;i<=physDim;i++) {
		mpo3s.H1.set( iMPO3s12(1), mpo3s.Is1(i), prime(mpo3s.Is1)(i), 1.0 );
		mpo3s.H2.set( iMPO3s12(1), iMPO3s23(1), mpo3s.Is2(i), 
			prime(mpo3s.Is2)(i), 1.0);
		mpo3s.H3.set( iMPO3s23(1), mpo3s.Is3(i), prime(mpo3s.Is3)(i), 1.0 );
	}

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

	// Take the square-root of SV's
	auto sqrtT = [](Real r) { return sqrt(r); };

	std::cout <<"----- Reducing dimension of link12 -----"<< std::endl;
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

	std::cout <<"----- Reducing dimension of link23 -----"<< std::endl;
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