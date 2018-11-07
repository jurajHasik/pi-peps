#include <chrono>
#include <algorithm>
#include "ctm-cluster-io.h"
#include "cluster-ev-builder.h"
#include "simple-update.h"

using namespace itensor;

// obtain weights for each of 3 sites T1,T2,T3 without weights on links
// connecting T1--l12--T2 and T2--l23--T3
// p3s: {&T1,&l12,&T2,&l23,&T3} holds pointers to on-site tensors and
//      weights on connecting links 
const std::map< ITensor *, const std::vector< ITensor *> > get3siteTN(
    std::map< ITensor *, const std::vector< ITensor *> > const& tn,
    std::vector< ITensor* > const& p3s) 
{

    std::vector<ITensor*> mock_l12 = {p3s[1]};
    std::vector<ITensor*> mock_l12l23 = {p3s[1],p3s[3]};
    std::sort(mock_l12l23.begin(), mock_l12l23.end());
    std::vector<ITensor*> mock_l23 = {p3s[3]};
    
    std::vector<ITensor*> wt1({}), wt2({}), wt3({});
    
    // std::cout << "res: ";
    // for (const auto &p : res) {
    //     std::cout << "[" << p.first << "] => ";
    //     for(const auto &e : p.second) {
    //         std::cout << e << " ";
    //     }
    //     std::cout << std::endl;
    // }

    std::set_difference(tn.at(p3s[0]).begin(),tn.at(p3s[0]).end(),
        mock_l12.begin(),mock_l12.end(), std::back_inserter(wt1));

    std::set_difference(tn.at(p3s[2]).begin(),tn.at(p3s[2]).end(),
        mock_l12l23.begin(),mock_l12l23.end(), std::back_inserter(wt2));

    std::set_difference(tn.at(p3s[4]).begin(),tn.at(p3s[4]).end(),
        mock_l23.begin(),mock_l23.end(), std::back_inserter(wt3));

    const std::map<ITensor*, const std::vector< ITensor *> > res = 
        {{p3s[0],wt1},{p3s[2],wt2},{p3s[4],wt3}};

    return res;
}

// Prepare cluster - by contracting the weights with on-site tensors
//      
//      l5     l7               The site aux indices are as follows
//       |      | 
// --l1--A--l2--B--l1--                    auxI1 
//       |      |                           | 
//      l6     l8                   auxI0-- A --auxI2
//       |      |                           | 
// --l3--C--l4--D--l3--                    auxI3
//       |      | 
//      l5     l7
void formCluster(Cluster & cls, std::vector< ITensor > const& ts,
    std::vector< ITensor > ls) {

    // Set new sites to cluster
    auto sqrtT = [](double r) { return sqrt(r); };

    // for(size_t i=0; i<=7; ++i) {
    //     ls[i].apply(sqrtT);
    // }
    ls[1].apply(sqrtT);
    ls[3].apply(sqrtT);
    ls[5].apply(sqrtT);
    ls[7].apply(sqrtT);

    auto aIA = noprime(findtype(ts[0].inds(), AUXLINK));
    auto aIB = noprime(findtype(ts[1].inds(), AUXLINK));
    auto aIC = noprime(findtype(ts[2].inds(), AUXLINK));
    auto aID = noprime(findtype(ts[3].inds(), AUXLINK));

    auto tA = ts[0] * ls[0] * ls[1] * ls[4] * ls[5];
    tA = ( ( (tA*delta(aIA, prime(aIB,2))) *delta(prime(aIA,1), prime(aIC,3))) 
            *delta(prime(aIA,2), aIB) ) *delta(prime(aIA,3), prime(aIC,1));

    auto tB = ts[1] * ls[1] * ls[0] * ls[6] * ls[7];
    tB = ( ( (tB*delta(aIB, prime(aIA,2))) *delta(prime(aIB,1), prime(aID,3)))
            *delta(prime(aIB,2), aIA)) *delta(prime(aIB,3), prime(aID,1));

    auto tC = ts[2] * ls[2] * ls[3] * ls[5] * ls[4];
    tC = ( ( (tC*delta(aIC, prime(aID,2))) *delta(prime(aIC,1), prime(aIA,3)))
            *delta(prime(aIC,2), aID)) *delta(prime(aIC,3), prime(aIA,1));

    auto tD = ts[3] * ls[3] * ls[2] * ls[7] * ls[6];
    tD = ( ( (tD*delta(aID, prime(aIC,2))) *delta(prime(aID,1), prime(aIB,3)))
            *delta(prime(aID,2), aIC)) *delta(prime(aID,3), prime(aIB,1));

    // Build new cluster
    cls.sites = {{"A", tA}, {"B", tB}, {"C", tC}, {"D", tD}};
    //std::cout << cls; //DBG
}

// Compute expectation value of 2site OP over 2x2 supercell {{A,B},{C,D}}
//
// <bra|    A    B   
//        C |  D | 
//        | |  | |
//       | OP2s ||
//        | |  | |  
//        | A  | B
//        C    D    |ket>
//
std::vector< std::complex<double> > getEV2Site(Cluster const& cls) {

    auto aIA  = noprime(findtype(cls.sites.at("A").inds(), AUXLINK));
    auto aIB  = noprime(findtype(cls.sites.at("B").inds(), AUXLINK));
    auto aIC  = noprime(findtype(cls.sites.at("C").inds(), AUXLINK));
    auto aID  = noprime(findtype(cls.sites.at("D").inds(), AUXLINK));

    auto pIA  = noprime(findtype(cls.sites.at("A").inds(), PHYS));
    auto pIB  = noprime(findtype(cls.sites.at("B").inds(), PHYS));
    auto pIC  = noprime(findtype(cls.sites.at("C").inds(), PHYS));
    auto pID  = noprime(findtype(cls.sites.at("D").inds(), PHYS));

    // NN terms - very small objects (phys dim)^4
    auto H_nnhAB = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIA, pIB);
    auto H_nnhAC = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIA, pIC);
    auto H_nnhBD = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIB, pID);
    auto H_nnhCD = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIC, pID);
    // NNN terms
    auto H_nnnhAD = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIA, pID);
    auto H_nnnhBC = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIB, pIC);

    // form density matrix for phys DoFs of the cluster
    auto dnmat = clusterDenMat(cls);
    // Print(dnmat);

    double cls_norm = norm(dnmat * delta(pIA,prime(pIA,1)) * delta(pIB,prime(pIB,1))
        * delta(pIC,prime(pIC,1)) * delta(pID,prime(pID,1)));
    
    //Print(Bra);

    auto KetAB = dnmat * H_nnhAB.first * H_nnhAB.second;
    KetAB = KetAB * delta(pIC,prime(pIC,1)) * delta(pID,prime(pID,1));

    // pA1, pB1, pB2, pA2 => op2s => pA0, pB1, pB3, pA2
    auto KetAC = dnmat * H_nnhAC.first * H_nnhAC.second;
    KetAC = KetAC * delta(pIB,prime(pIB,1)) * delta(pID,prime(pID,1));

    // pA1, pB1, pB2, pA2 => op2s => pA1, pB1, pB3, pA3
    auto KetCD = dnmat * H_nnhCD.first * H_nnhCD.second;
    KetCD = KetCD * delta(pIA,prime(pIA,1)) * delta(pIB,prime(pIB,1));

    // pA1, pB1, pB2, pA2 => op2s => pA1, pB0, pB2, pA3
    auto KetBD = dnmat * H_nnhBD.first * H_nnhBD.second; 
    KetBD = KetBD * delta(pIA,prime(pIA,1)) * delta(pIC,prime(pIC,1));

    auto KetAD = dnmat * H_nnnhAD.first * H_nnnhAD.second;
    KetAD = KetAD * delta(pIB,prime(pIB,1)) * delta(pIC,prime(pIC,1));

    auto KetBC = dnmat * H_nnnhBC.first * H_nnnhBC.second;
    KetBC = KetBC * delta(pIA,prime(pIA,1)) * delta(pID,prime(pID,1));

    if( KetBD.r() > 0 || KetAB.r() > 0 || KetCD.r() >0 || KetAC.r() > 0 
        || KetAD.r() > 0 || KetBC.r() > 0 ) {
        std::cout <<"Expectation value not a tensor rank 0"<< std::endl;
        exit(EXIT_FAILURE);    
    }

    std::vector< std::complex<double> > evs = {
        sumels(KetAB)/cls_norm, sumels(KetAC)/cls_norm,
        sumels(KetCD)/cls_norm, sumels(KetBD)/cls_norm,
        sumels(KetAD)/cls_norm, sumels(KetBC)/cls_norm };

    std::cout <<"NN: "<< evs[0].real() <<", "<< evs[1].real() <<", "
        << evs[2].real() <<", "<< evs[3].real() <<" NNN: "<< evs[4].real()
        <<", "<< evs[5].real() << std::endl;

    return evs;
}

// perform 3-site simple update over sites T1,T2,T3 connected by links
// with weights T1--l12--T2 and T2--l23--T3
// invWs: {&l1-&l1I, &l2->&l2I,...} map from pointers to weights to pointers 
//     to inverse weights
// p3s: {&T1,&l12,&T2,&l23,&T3} holds pointers to on-site tensors and
//      weights on connecting links
// pT1, pT2, pT3: {&l1,&l2,&l3}, {&l4,&l5}, {...} hold pointers to weights
//      surrounding the on-site tensors T1,T2,T3
void simpUp(MPO_3site const& uJ1J2, 
    std::map<ITensor*,ITensor*> const& invWs,
    std::vector<ITensor*> const& pT1, 
    std::vector<ITensor*> const& pT2, 
    std::vector<ITensor*> const& pT3,
    std::vector<ITensor*> const& p3s)
{
    (*p3s[0]) = (((*p3s[0]) * (*pT1[0])) * (*pT1[1])) * (*pT1[2]);
    (*p3s[2]) = ((*p3s[2]) * (*pT2[0])) * (*pT2[1]);
    (*p3s[4]) = (((*p3s[4]) * (*pT3[0])) * (*pT3[1])) * (*pT3[2]);

    applyH_123_v2(uJ1J2, *p3s[0], *p3s[2], *p3s[4], *p3s[1], *p3s[3],
        *invWs.at(p3s[1]), *invWs.at(p3s[3]));

    (*p3s[0]) = (*p3s[0]) * (*invWs.at(pT1[0])) * (*invWs.at(pT1[1])) * (*invWs.at(pT1[2]));
    (*p3s[2]) = (*p3s[2]) * (*invWs.at(pT2[0])) * (*invWs.at(pT2[1]));
    (*p3s[4]) = (*p3s[4]) * (*invWs.at(pT3[0])) * (*invWs.at(pT3[1])) * (*invWs.at(pT3[2]));
}

// perform 3-site simple update over sites T1,T2,T3 connected by weights l12,l23 as
// follows T1--l12--T2 and T2--l23--T3 
// tn: {&T1->{&l1,&l2,&l3,&4},&T2->{...},...} map from pointer to sites T1...T4 to
//     vectors holding pointers to weights l1...l8
// invWs: {&l1-&l1I, &l2->&l2I,...} map from pointers to weights to pointers 
//     to inverse weights  
// p3s: {&T1,&l12,&T2,&23,&T3} vector of pointers to sites T1,T2,T3 
//     and weights l12,l23 
//
void simpUp(MPO_3site const& uJ1J2, 
    std::map< ITensor *, const std::vector< ITensor *> > const& tn,
    std::map< ITensor *, ITensor * > const& invWs,
    std::vector< ITensor* > const& p3s) 
{

    // obtain map of weight for sites T1,...,T3 without weights l12,l23
    // {&T1->{&l1,&l2,&l3,&l4}/l12, &T2->{...}/{&l12,&l23},...} 
    auto p3w = get3siteTN(tn, p3s);

    simpUp(uJ1J2, invWs, p3w.at(p3s[0]), p3w.at(p3s[2]), p3w.at(p3s[4]), p3s);
}

int main( int argc, char *argv[] ) {
    // ########################################################################
    // Handle command-line arguments

    //OP2S_TYPE arg_op2s_type;
    //F_MPO3S arg_f_mpo3s;
    std::string arg_initType, arg_inClusterFile, arg_outClusterFile;
    int arg_auxBondDim, arg_nIter;
    double arg_tau, arg_J1, arg_J2;
    
    std::string metaInfo;

    arg_initType = std::string(argv[1]);
    if( (arg_initType == "FILE") && (argc >= 9) ) {
        arg_inClusterFile  = argv[2];
        arg_outClusterFile = argv[3];
        arg_auxBondDim     = std::stoi(argv[4]);
        arg_nIter          = std::stoi(argv[5]);
        arg_tau            = std::stod(argv[6]);
        arg_J1             = std::stod(argv[7]);
        arg_J2             = std::stod(argv[8]);

        std::cout <<"Initializing from File: "<< arg_inClusterFile << std::endl;
        metaInfo.append("Init "+arg_initType+" "+arg_inClusterFile+";");
    // otherwise we start with random cluster 
    } else if( ((arg_initType == "RANDOM") || (arg_initType == "RND_AB")
        || (arg_initType == "AFM")) && (argc >= 8) ) {
        arg_outClusterFile = argv[2];
        arg_auxBondDim     = std::stoi(argv[3]);
        arg_nIter          = std::stoi(argv[4]);
        arg_tau            = std::stod(argv[5]);
        arg_J1             = std::stod(argv[6]);
        arg_J2             = std::stod(argv[7]);
        metaInfo.append("Init by: "+arg_initType+";");    
    } else {
        std::cout <<"Invalid amount of Agrs (< 7)"<< std::endl;
        exit(EXIT_FAILURE);
    }

    double eps_threshold = 1.0e-8;
    double tau_threshold = 1.0e-10;

    std::cout.precision( std::numeric_limits< double >::max_digits10 );
    std::cout <<"Simulation parameters"<< std::endl;
    std::cout <<"auxBondDim   : "<< arg_auxBondDim << std::endl;
    std::cout <<"imag time tau: "<< arg_tau << std::endl;
    std::cout <<"J1           : "<< arg_J1 << std::endl;
    std::cout <<"J2           : "<< arg_J2 << std::endl;
    std::cout <<"nIterations  : "<< arg_nIter << std::endl;
    //std::cout <<"eps_threshold: "<< eps_threshold << std::endl;
    //std::cout <<"tau_threshold: "<< tau_threshold << std::endl;
    metaInfo.append("nIterations: "+std::to_string(arg_nIter)
        +";tau "+std::to_string(arg_tau)
        +";J1 "+std::to_string(arg_J1)
        +";J2 "+std::to_string(arg_J2)+";");

    Cluster cls;
    Index aIA, aIB, pIA, pIB, aIC, aID, pIC, pID;
    ITensor A, B, C, D;

    if( arg_initType == "FILE" ) {
        cls = readCluster(arg_inClusterFile);
        cls.auxBondDim = arg_auxBondDim;

        A = cls.sites[cls.cToS.at(std::make_pair(0,0))];
        B = cls.sites[cls.cToS.at(std::make_pair(1,0))];
        C = cls.sites[cls.cToS.at(std::make_pair(0,1))];
        D = cls.sites[cls.cToS.at(std::make_pair(1,1))];

        pIA = noprime(findtype(A.inds(), PHYS));
        pIB = noprime(findtype(B.inds(), PHYS));
        pIC = noprime(findtype(C.inds(), PHYS));
        pID = noprime(findtype(D.inds(), PHYS));
        aIA = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIB = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIC = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aID = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        
        // relabel original indices to take up desired bond dimensions
        auto taIA = noprime(findtype(A.inds(), AUXLINK));
        auto taIB = noprime(findtype(B.inds(), AUXLINK));
        auto taIC = noprime(findtype(C.inds(), AUXLINK));
        auto taID = noprime(findtype(D.inds(), AUXLINK));
        
        auto D_I = delta(taIA,aIA);
        A = A*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
        D_I = delta(taIB,aIB);
        B = B*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
        D_I = delta(taIC,aIC);
        C = C*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
        D_I = delta(taID,aID);
        D = D*D_I*prime(D_I,1)*prime(D_I,2)*prime(D_I,3);
    } else {
        // ----- DEFINE BLANK CLUSTER ----------------------------------
        cls = Cluster();
        cls.sizeN = 2;
        cls.sizeM = 2;
        cls.auxBondDim = arg_auxBondDim;
        cls.physDim    = 2;

        cls.siteIds = std::vector< std::string >(4);
        cls.siteIds = { "A", "B", "C", "D" };

        cls.cToS  = {
            {std::make_pair(0,0),"A"},
            {std::make_pair(1,0),"B"},
            {std::make_pair(0,1),"C"},
            {std::make_pair(1,1),"D"}
        };

        aIA = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIB = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aIC = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        aID = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
        pIA = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pIB = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pIC = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pID = Index(TAG_I_PHYS, cls.physDim, PHYS);

        A = ITensor(aIA, prime(aIA,1), prime(aIA,2), prime(aIA,3), pIA);
        B = ITensor(aIB, prime(aIB,1), prime(aIB,2), prime(aIB,3), pIB);
        C = ITensor(aIC, prime(aIC,1), prime(aIC,2), prime(aIC,3), pIC);
        D = ITensor(aID, prime(aID,1), prime(aID,2), prime(aID,3), pID);

        if (arg_initType == "RND_AB") {
            std::cout <<"Initializing by RANDOM TENSORS A,B,C=B,D=A"<< std::endl;
            randomize(A);
            randomize(B);
            C = B * delta(pIB, pIC);
            D = A * delta(pIA, pID);
            for (int i=0; i<=3; ++i) {
                C = C * delta(prime(aIB,i), prime(aIC,i));
                D = D * delta(prime(aIA,i), prime(aID,i));
            }
        } else if(arg_initType == "RANDOM") {
            std::cout <<"Initializing by RANDOM TENSORS"<< std::endl;
            // Randomize

            randomize(A);
            randomize(B);
            randomize(C);
            randomize(D);

            // auto shift05 = [](Real r){ return r-0.5; };
            // A.apply(shift05);
            // B.apply(shift05);
            // C.apply(shift05);
            // D.apply(shift05);

        } else if (arg_initType == "AFM") {
            std::cout <<"Initializing by AFM order A=down, B=up"<< std::endl;
            // Spin DOWN on site A, spin   UP on site B
            // Spin UP   on site C, spin DOWN on site D
            A.set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
                pIA(2), 1.0);
            B.set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
                pIB(1), 1.0);
            C.set(aIC(1), prime(aIC,1)(1), prime(aIC,2)(1), prime(aIC,3)(1),
                pIC(1), 1.0);
            D.set(aID(1), prime(aID,1)(1), prime(aID,2)(1), prime(aID,3)(1),
                pID(2), 1.0);
        }

        cls.sites = {{"A", A}, {"B", B}, {"C",C}, {"D",D}};
        // ----- END DEFINE CLUSTER ------------------------------------
    }
    cls.metaInfo = metaInfo;
    std::cout << cls; //DBG

    // Prepare nnH Hamiltonian as 2-site MPO, with indices 
    // H_nnh.first[pIA,pIA'] and H_nnh.second[pIB,pIB']
    // auto H_nnh = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIA, pIB);

    // Get Exp of 3-site operator u_123 - building block of Trotter Decomposition
    MPO_3site uJ1J2(getMPO3s_Uj1j2_v2(arg_tau, arg_J1, arg_J2));

    // Prepare Sites (from cluster) + weights to begin optimization 
    // procedure
    // We assume we have cluster of the form   AB
    //                                         CD
    // Adding the weights results in the following tensor network
    //
    //      l5     l7               The site aux indices are as follows
    //       |      | 
    // --l1--A--l2--B--l1--                    auxI1 
    //       |      |                           | 
    //      l6     l8                   auxI0-- A --auxI2
    //       |      |                           | 
    // --l3--C--l4--D--l3--                    auxI3
    //       |      | 
    //      l5     l7

    // define 

    // horizontal
    auto l1 = ITensor(prime(aIB,2), aIA); 
    auto l2 = ITensor(prime(aIA,2), aIB);
    auto l3 = ITensor(prime(aID,2), aIC);
    auto l4 = ITensor(prime(aIC,2), aID);
    // vertical
    auto l5 = ITensor(prime(aIC,3), prime(aIA,1));
    auto l6 = ITensor(prime(aIA,3), prime(aIC,1));
    auto l7 = ITensor(prime(aID,3), prime(aIB,1));
    auto l8 = ITensor(prime(aIB,3), prime(aID,1));

    // Initially set all weights to 1
    for (int i=1; i<=aIA.m(); i++) {
        l1.set(prime(aIB,2)(i), aIA(i), 1.0);
        l2.set(prime(aIA,2)(i), aIB(i), 1.0);
        l3.set(prime(aID,2)(i), aIC(i), 1.0); 
        l4.set(prime(aIC,2)(i), aID(i), 1.0);
        l5.set(prime(aIC,3)(i), prime(aIA,1)(i), 1.0);
        l6.set(prime(aIA,3)(i), prime(aIC,1)(i), 1.0);
        l7.set(prime(aID,3)(i), prime(aIB,1)(i), 1.0);
        l8.set(prime(aIB,3)(i), prime(aID,1)(i), 1.0);
    }

    // Define set of inverse diag weights
    auto l1I = l1;
    auto l2I = l2;
    auto l3I = l3;
    auto l4I = l4;
    auto l5I = l5;
    auto l6I = l6;
    auto l7I = l7;
    auto l8I = l8;

    // Define weights sets for individual sites
    std::vector< ITensor * > pwA = {&l1, &l2, &l5, &l6};
    std::vector< ITensor * > pwB = {&l1, &l2, &l7, &l8};
    std::vector< ITensor * > pwC = {&l3, &l4, &l5, &l6};
    std::vector< ITensor * > pwD = {&l3, &l4, &l7, &l8};

    for (auto &v : {&pwA, &pwB, &pwC, &pwD}) {
        std::sort((*v).begin(), (*v).end());
        std::cout << std::is_sorted((*v).begin(),(*v).end()) << " ";
    }
    std::cout << std::endl;

    // Define map from sites to weight sets
    const std::map< ITensor *, const std::vector<ITensor * > > tn = 
        {{&A, pwA}, {&B, pwB}, {&C, pwC}, {&D, pwD}};
    // Define map form weight tensors to inverse weight tensors
    const std::map< ITensor *, ITensor * > invWs = {
        {&l1,&l1I}, {&l2,&l2I}, {&l3,&l3I}, {&l4,&l4I},
        {&l5,&l5I}, {&l6,&l6I}, {&l7,&l7I}, {&l8,&l8I}
    };

    // for (const auto &p : tn) {
    //     std::cout << "[" << p.first << "] => ";
    //     for(const auto &e : p.second) {
    //         std::cout << e << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // auto res = get3siteTN(tn, {&A,&l2,&B,&l8,&D});

    // for (const auto &p : res) {
    //     std::cout << "[" << p.first << "] => ";
    //     for(const auto &e : p.second) {
    //         std::cout << e << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Compute Initial value of APPROX(!) energy
    formCluster(cls, {A,B,C,D}, {l1, l2, l3, l4, l5, l6, l7, l8});
    auto e_nnh = getEV2Site(cls);
    auto avgE = 2.0*(e_nnh[0] + e_nnh[1] + e_nnh[2] + e_nnh[3])/4.0;
    auto e_avgE_prev = avgE;

    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;
    t_iso_begin = std::chrono::steady_clock::now();
    
    std::vector< std::vector<ITensor*> > opt_seq = {
        //A-D link variant GREEN
        {&A,&l2,&B,&l8,&D}, //{&D,&l8,&B,&l2,&A},
        {&A,&l6,&C,&l4,&D}, //{&D,&l4,&C,&l6,&A},

        //A-D link variant GREY
        {&D,&l7,&B,&l1,&A}, //{&A,&l1,&B,&l7,&D},
        {&D,&l3,&C,&l5,&A}, //{&A,&l5,&C,&l3,&D},         

        //A-D link variant RED 
        {&A,&l6,&C,&l3,&D}, //{&D,&l3,&C,&l6,&A},
        {&A,&l1,&B,&l8,&D}, //{&D,&l8,&B,&l1,&A},
         
        //A-D link variant BLUE
        {&D,&l4,&C,&l5,&A}, //{&A,&l5,&C,&l4,&D},
        {&D,&l7,&B,&l2,&A}, //{&A,&l2,&B,&l7,&D},

        //####################################

        //B-C link variant GREEN
        {&B,&l8,&D,&l4,&C}, //{&C,&l4,&D,&l8,&B},
        {&B,&l2,&A,&l6,&C}, //{&C,&l6,&A,&l2,&B},

        //B-C link variant GREY
        {&C,&l5,&A,&l1,&B}, //{&B,&l1,&A,&l5,&C}, 
        {&C,&l3,&D,&l7,&B}, //{&B,&l7,&D,&l3,&C},
        
        //B-C link variant RED
        {&B,&l1,&A,&l6,&C}, //{&C,&l6,&A,&l1,&B},
        {&B,&l8,&D,&l3,&C}, //{&C,&l3,&D,&l8,&B},

        //B-C link variant BLUE
        {&C,&l4,&D,&l7,&B}, //{&B,&l7,&D,&l4,&C},
        {&C,&l5,&A,&l2,&B}, //{&B,&l2,&A,&l5,&C}

    };

    for (int nStep=1; nStep<=arg_nIter; nStep++) {
        
        for(auto it = opt_seq.end(); it-- != opt_seq.begin(); ) {
            simpUp(uJ1J2, tn, invWs, *it); 
        }
        
        for(size_t i=0; i<opt_seq.size(); ++i) {
            simpUp(uJ1J2, tn, invWs, opt_seq[i]);    
        }


        if ( nStep % 1000 == 0 ) { 
            t_iso_end = std::chrono::steady_clock::now();
            std::cout <<"STEP "<< nStep <<" T= "<< std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0 
            << std::endl;
        
            formCluster(cls, {A,B,C,D}, {l1, l2, l3, l4, l5, l6, l7, l8});
            e_nnh = getEV2Site(cls);
            avgE = 2.0*(e_nnh[0] + e_nnh[1] + e_nnh[2] + e_nnh[3])/4.0;
            
    //         std::cout <<"E: "<< e_nnh.real() <<" + "<< e_nnh.imag() 
    //             << std::endl;
            
    //         // Check difference against previous value of energy
    //         //if ( abs(e_nnh.real() - e_nnh_prev.real()) < eps_threshold ) {
    //         if ( (e_nnh_prev.real() - e_nnh.real()) < eps_threshold ) {
    //             std::cout << "Energy difference < "<< eps_threshold 
    //                 << std::endl;
    //             std::cout << "Changing tau -> tau/2: "<< arg_tau <<" -> "
    //                 << arg_tau/2.0 << std::endl;
    //             arg_tau = arg_tau/2.0;

    //             if (arg_tau < tau_threshold) {
    //                 std::cout << "tau too small - stopping optimization" 
    //                     << std::endl;
    //                 break;
    //             }    

    //             // Get new evolution op with decreased arg_tau
    //             //nnh = getMPO2s_NNH(4, arg_tau, arg_J, arg_h);
    //             nnh = getMPO2s_NNHstagh(4, arg_tau, arg_J, arg_h);
    //         }
                e_avgE_prev = avgE;

                t_iso_begin = std::chrono::steady_clock::now();
        }
    }

    // Set new sites to cluster
    auto sqrtT = [](double r) { return sqrt(r); };

    l1.apply(sqrtT);
    l2.apply(sqrtT);
    l3.apply(sqrtT);
    l4.apply(sqrtT);
    l5.apply(sqrtT);
    l6.apply(sqrtT);
    l7.apply(sqrtT);
    l8.apply(sqrtT);

    A = A * l1 * l2 * l5 * l6;
    A = ( ( (A*delta(aIA, prime(aIB,2))) *delta(prime(aIA,1), prime(aIC,3))) 
            *delta(prime(aIA,2), aIB) ) *delta(prime(aIA,3), prime(aIC,1));

    B = B * l2 * l1 * l7 * l8;
    B = ( ( (B*delta(aIB, prime(aIA,2))) *delta(prime(aIB,1), prime(aID,3)))
            *delta(prime(aIB,2), aIA)) *delta(prime(aIB,3), prime(aID,1));

    C = C * l3 * l4 * l6 * l5;
    C = ( ( (C*delta(aIC, prime(aID,2))) *delta(prime(aIC,1), prime(aIA,3)))
            *delta(prime(aIC,2), aID)) *delta(prime(aIC,3), prime(aIA,1));

    D = D * l4 * l3 * l8 * l7;
    D = ( ( (D*delta(aID, prime(aIC,2))) *delta(prime(aID,1), prime(aIB,3)))
            *delta(prime(aID,2), aIC)) *delta(prime(aID,3), prime(aIB,1));

    auto normalizeBLE_T = [](ITensor& t)
    {
        double m = 0.;
        auto max_m = [&m](double d)
        {
            if(std::abs(d) > m) m = std::abs(d);
        };

        t.visit(max_m);
        std::cout << "MAX elem = "<< m << std::endl;
        t /= m;
    };

    normalizeBLE_T(A);
    normalizeBLE_T(B);
    normalizeBLE_T(C);
    normalizeBLE_T(D);

    PrintData(l1);
    PrintData(l2);
    PrintData(l3);
    PrintData(l4);
    PrintData(l5);
    PrintData(l6);
    PrintData(l7);
    PrintData(l8);

    // Build new cluster
    cls.sites = {{"A",A}, {"B",B}, {"C",C}, {"D",D}};
    //std::cout << cls; //DBG

    writeCluster(arg_outClusterFile, cls);
}