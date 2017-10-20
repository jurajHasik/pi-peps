#include "ctm-cluster-io.h"
#include "cluster-ev-builder.h"
#include "simple-update.h"
#include <algorithm>
#include <chrono>

using namespace itensor;


// Prepare cluster - by contracting the weights with on-site tensors A, B
// and creating new (identical) tensors C=B, D=A and independent aux and physical
// indices of C, D (not equal to indices of A or B)
void formCluster(Cluster & cls, ITensor const& A, ITensor const& B,
    std::vector< ITensor > ls) {

    auto sqrtT = [](double r) { return sqrt(r); };

    ls[0].apply(sqrtT);
    ls[1].apply(sqrtT);
    ls[2].apply(sqrtT);
    ls[3].apply(sqrtT);

    // Extract the IndexSet of individual weight tensors l1 ... l4
    std::vector<IndexSet> wInds;
    for(size_t i=0; i < ls.size(); ++i) wInds.push_back(ls[i].inds());

    // contract with square roots of weights
    cls.sites["A"] = A * ls[0] * ls[1] * ls[2] * ls[3];
    for(auto const& is : wInds) cls.sites["A"] *= delta(is[0], is[1]);
    cls.sites["B"] = B * ls[0] * ls[1] * ls[2] * ls[3];
    for(auto const& is : wInds) cls.sites["B"] *= delta(is[0], is[1]);

    auto pIA  = noprime(findtype(cls.sites["A"].inds(), PHYS));
    auto pIB  = noprime(findtype(cls.sites["B"].inds(), PHYS));
    auto pIC = Index(TAG_I_PHYS, cls.physDim, PHYS);
    auto pID = Index(TAG_I_PHYS, cls.physDim, PHYS);

    auto aIA  = noprime(findtype(cls.sites["A"].inds(), AUXLINK));
    auto aIB  = noprime(findtype(cls.sites["B"].inds(), AUXLINK));
    auto aIC = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
    auto aID = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);

    cls.sites["C"] = cls.sites["B"] * delta(pIB,pIC);
    cls.sites["D"] = cls.sites["A"] * delta(pIA,pID);

    cls.sites["A"] = cls.sites["A"] * ls[0] * delta(wInds[0][0], wInds[0][1]) 
        * ls[2]*delta(wInds[2][0], wInds[2][1]);
    cls.sites["B"] = cls.sites["B"] * ls[3] * delta(wInds[3][0], wInds[3][1]) 
        * ls[0]*delta(wInds[0][0], wInds[0][1]);
    cls.sites["C"] = cls.sites["C"] * ls[1] * delta(wInds[1][0], wInds[1][1]) 
        * ls[2]*delta(wInds[2][0], wInds[2][1]);
    cls.sites["D"] = cls.sites["D"] * ls[1] * delta(wInds[1][0], wInds[1][1]) 
        * ls[3]*delta(wInds[3][0], wInds[3][1]);

    for(size_t i=0; i<=3; ++i) {
        cls.sites["C"] *= delta(prime(aIB,i), prime(aIC,i));
        cls.sites["D"] *= delta(prime(aIA,i), prime(aID,i));
    }

    // Build new cluster
    // cls.sites = {{"A", tA}, {"B", tB}, {"C", tC}, {"D", tD}};
    // std::cout << cls; //DBG
}

// Compupte expectation value of 2site OP over 2x2 supercell {{A,B},{B,A}}
//
// <bra|    A    B   
//        B |  A | 
//        | |  | |
//       | OP2s ||
//        | |  | |  
//        | A  | B
//        B    A    |ket>
//
std::vector< std::complex<double> > getEV2Site(Cluster const& cls) {

    auto pIA  = noprime(findtype(cls.sites.at("A").inds(), PHYS));
    auto pIB  = noprime(findtype(cls.sites.at("B").inds(), PHYS));
    auto pIC  = noprime(findtype(cls.sites.at("C").inds(), PHYS));
    auto pID  = noprime(findtype(cls.sites.at("D").inds(), PHYS));

    auto H_nnhAB = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIA, pIB);
    auto H_nnhAC = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIA, pIC);
    auto H_nnhBD = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIB, pID);
    auto H_nnhCD = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIC, pID);

    auto bra = contractCluster(cls);
    Print(bra);

    auto dnmat = bra*prime(conj(bra),PHYS,1);
    Print(dnmat);

    double cls_norm = norm(dnmat * delta(pIA,prime(pIA,1)) * delta(pIB,prime(pIB,1))
        * delta(pIC,prime(pIC,1)) * delta(pID,prime(pID,1)));

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

    if( KetBD.r() > 0 || KetAB.r() > 0 || KetCD.r() >0 || KetAC.r() > 0 ) {
        std::cout <<"Expectation value not a tensor rank 0"<< std::endl;
        exit(EXIT_FAILURE);    
    }

    std::vector< std::complex<double> > evs = {
        sumels(KetAB)/cls_norm, sumels(KetAC)/cls_norm,
        sumels(KetCD)/cls_norm, sumels(KetBD)/cls_norm };

    std::cout << evs[0].real() <<", "<< evs[1].real() <<", "
        << evs[2].real() <<", "<< evs[3].real() << std::endl;

    return evs;
}

// Apply simple update over sites A,B connected by link with
// weight l1. The weights l2,...,l4 are used to construct on-site tensors A,B 
void simpUp(
    const std::map< ITensor *, const std::vector<ITensor * > > tn,
    const std::map< ITensor *, ITensor * > invWs,
    MPO_2site const& nnh, const std::vector<ITensor*> p2s) {
 
    std::vector<ITensor*> mockA = {p2s[1]};
    std::vector<ITensor*> mockB = {p2s[1]};
    std::vector<ITensor*> wtA({}), wtB({});

    std::set_difference(tn.at(p2s[0]).begin(),tn.at(p2s[0]).end(),
        mockA.begin(),mockA.end(), std::back_inserter(wtA));
    std::set_difference(tn.at(p2s[2]).begin(),tn.at(p2s[2]).end(),
        mockB.begin(),mockB.end(), std::back_inserter(wtB));

    for (const auto& w : wtA) (*p2s[0]) *= (*w);
    for (const auto& w : wtB) (*p2s[2]) *= (*w);
    applyH_T1_L_T2(nnh, *p2s[0], *p2s[2], *p2s[1], *invWs.at(p2s[1]));
    for (const auto& w : wtA) (*p2s[0]) *= (*invWs.at(w));
    for (const auto& w : wtB) (*p2s[2]) *= (*invWs.at(w));
}

int main( int argc, char *argv[] ) {
    // ########################################################################
    // Handle command-line arguments

    //OP2S_TYPE arg_op2s_type;
    //F_MPO3S arg_f_mpo3s;
    std::string arg_initType, arg_inClusterFile, arg_outClusterFile;
    int arg_auxBondDim, arg_nIter;
    double arg_tau, arg_J, arg_h;

    std::string metaInfo;
    
    arg_initType = std::string(argv[1]);
    if( (arg_initType == "FILE") && (argc >= 8) ) {
        arg_inClusterFile  = argv[2];
        arg_outClusterFile = argv[3];
        arg_nIter          = std::stoi(argv[4]);
        arg_tau            = std::stod(argv[5]);
        arg_J              = std::stod(argv[6]);
        arg_h              = std::stod(argv[7]);

        std::cout <<"Initializing from File: "<< arg_inClusterFile << std::endl;
        metaInfo.append("Init "+arg_initType+" "+arg_inClusterFile+";");
    // otherwise we start with random cluster 
    } else if( ((arg_initType == "RANDOM") || (arg_initType == "AFM")) 
        && (argc >= 8) ) {
        arg_outClusterFile = argv[2];
        arg_auxBondDim     = std::stoi(argv[3]);
        arg_nIter          = std::stoi(argv[4]);
        arg_tau            = std::stod(argv[5]);
        arg_J              = std::stod(argv[6]);
        arg_h              = std::stod(argv[7]);

        metaInfo.append("Init by: "+arg_initType+";");
        std::cout <<"Initializing by "+arg_initType << std::endl;     
    } else {
        std::cout <<"Invalid amount of Agrs (< 7)"<< std::endl;
        exit(EXIT_FAILURE);
    }
    
    double eps_threshold = 1.0e-8;
    double tau_threshold = 1.0e-10;

    std::cout.precision( std::numeric_limits< double >::max_digits10 );
    std::cout <<"Simulation parameters"<< std::endl;
    std::cout <<"imag time tau: "<< arg_tau << std::endl;
    std::cout <<"J            : "<< arg_J << std::endl;
    std::cout <<"h            : "<< arg_h << std::endl;
    std::cout <<"nIterations  : "<< arg_nIter << std::endl;
    //std::cout <<"eps_threshold: "<< eps_threshold << std::endl;
    //std::cout <<"tau_threshold: "<< tau_threshold << std::endl;
    metaInfo.append("nIterations: "+std::to_string(arg_nIter)
        +";tau "+std::to_string(arg_tau)
        +";J "+std::to_string(arg_J)
        +";h "+std::to_string(arg_h)+";");

    Cluster cls;
    Index aIA, aIB, pIA, pIB;
    ITensor A, B;

    if( arg_initType == "FILE" ) {
        cls = readCluster(arg_inClusterFile);

        A = cls.sites[cls.cToS.at(std::make_pair(0,0))];
        B = cls.sites[cls.cToS.at(std::make_pair(1,0))];

        aIA = noprime(findtype(A.inds(), AUXLINK));
        aIB = noprime(findtype(B.inds(), AUXLINK));
        pIA = noprime(findtype(A.inds(), PHYS));
        pIB = noprime(findtype(B.inds(), PHYS));
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
        pIA = Index(TAG_I_PHYS, cls.physDim, PHYS);
        pIB = Index(TAG_I_PHYS, cls.physDim, PHYS);

        A = ITensor(aIA, prime(aIA,1), prime(aIA,2), prime(aIA,3), pIA);
        B = ITensor(aIB, prime(aIB,1), prime(aIB,2), prime(aIB,3), pIB);

        if (arg_initType == "RANDOM") {
            // Randomize
            randomize(A);
            randomize(B);
        } else if (arg_initType == "AFM") {
            // Spin DOWN on site A, spin UP on site B
            A.set(aIA(1), prime(aIA,1)(1), prime(aIA,2)(1), prime(aIA,3)(1),
                pIA(2), 1.0);
            B.set(aIB(1), prime(aIB,1)(1), prime(aIB,2)(1), prime(aIB,3)(1),
                pIB(1), 1.0);
        }

        cls.sites = {{"A", A}, {"B", B}, {"C",B}, {"D",A}};
        // ----- END DEFINE CLUSTER ------------------------------------
    }
    std::cout << cls; //DBG

    // Get Exp of 2-site operator - building block of Trotter Decomposition
    // NN-Heisenberg in external (uniform or staggered) mag-field
    //MPO_2site nnh(getMPO2s_NNH(4, arg_tau, arg_J, arg_h));        // uniform mag field
    MPO_2site nnh(getMPO2s_NNHstagh(4, arg_tau, arg_J, arg_h));   // staggered mag field
    MPO_2site nnh2(getMPO2s_NNHstagh(4, 2.0*arg_tau, arg_J, arg_h));

    // Prepare Sites (from cluster) + weights to begin optimization 
    // procedure
    // We assume we have cluster of the form   AB
    //                                         BA
    // Adding the weights results in the following tensor network
    //
    //      l3     l4               The site aux indices are as follows
    //       |      | 
    // --l1--A--l2--B--l1--                    auxI1 
    //       |      |                           | 
    //      l4     l3                   auxI0-- A --auxI2
    //       |      |                           | 
    // --l2--B--l1--A--l2--                    auxI3
    //       |      | 
    //      l3     l4

    ITensor l1(prime(aIB,2), aIA);
    ITensor l2(prime(aIA,2), aIB);
    ITensor l3(prime(aIB,3), prime(aIA,1));
    ITensor l4(prime(aIA,3), prime(aIB,1));

    // Initially set all weights to 1
    for (int i=1; i<=aIA.m(); i++) {
        l1.set(prime(aIB,2)(i), aIA(i), 1.0);
        l2.set(prime(aIA,2)(i), aIB(i), 1.0);
        l3.set(prime(aIB,3)(i), prime(aIA,1)(i), 1.0);
        l4.set(prime(aIA,3)(i), prime(aIB,1)(i), 1.0);
    }
    Print(l1);
    Print(l2);
    Print(l3);
    Print(l4);

    // Define set of inverse diag weights
    auto l1I = l1;
    auto l2I = l2;
    auto l3I = l3;
    auto l4I = l4;

    std::vector< ITensor * > pwA = {&l1, &l2, &l3, &l4};
    std::sort(pwA.begin(), pwA.end());

    // Define map from sites to weight sets
    const std::map< ITensor *, const std::vector<ITensor * > > tn = 
        {{&A,pwA},{&B,pwA}};
    // Define map form weight tensors to inverse weight tensors
    const std::map< ITensor *, ITensor * > invWs = 
        {{&l1,&l1I},{&l2,&l2I},{&l3,&l3I},{&l4,&l4I}};

    // Compute Initial value of APPROX(!) energy
    formCluster(cls, A, B, {l1, l2, l3, l4});
    auto e_nnh = getEV2Site(cls);
    auto avgE = (e_nnh[0] + e_nnh[1] + e_nnh[2] + e_nnh[3])/2.0;
    std::cout <<"E: "<< avgE.real() <<" + "<< avgE.imag() << std::endl;
    auto e_avgE_prev = avgE;

    std::cout.precision(10);
    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;
    t_iso_begin = std::chrono::steady_clock::now();

    std::vector< std::vector<ITensor*> > opt_seq = {
        {&A, &l2, &B}, {&A, &l1, &B}, {&A, &l4, &B}, {&A, &l3, &B}
    };

    for (int nStep=1; nStep<=arg_nIter; nStep++) {
    
        for(size_t i=0; i<opt_seq.size(); ++i) {
            simpUp(tn, invWs, nnh, opt_seq[i]);    
        }
        for(auto it = opt_seq.end(); it-- != opt_seq.begin(); ) {
            simpUp(tn, invWs, nnh, *it); 
        }
        
        if ( nStep % 1000 == 0 ) { 
            t_iso_end = std::chrono::steady_clock::now();
            std::cout <<"STEP "<< nStep <<" T= "<< std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0 
            << std::endl;
            //PrintData(l1);
            //PrintData(l2);
            //PrintData(l3);
            //PrintData(l4);
        
            formCluster(cls, A, B, {l1, l2, l3, l4});
            e_nnh = getEV2Site(cls);
            avgE = (e_nnh[0] + e_nnh[1] + e_nnh[2] + e_nnh[3])/2.0;
            // auto aniDist = abs(avgE-e_nnh[0])+abs(avgE-e_nnh[1])+abs(avgE-e_nnh[2])
            //     +abs(avgE-e_nnh[3]);

            std::cout <<"E: "<< avgE.real() <<" + "<< avgE.imag() 
                << std::endl;//<<" ani: "<< aniDist << std::endl;
            
            // Check difference against previous value of energy
            // if ( (e_avgE_prev.real() - avgE.real()) < eps_threshold ) {
            //     std::cout << "Energy difference < "<< eps_threshold 
            //         << std::endl;
            //     std::cout << "Changing tau -> tau/2: "<< arg_tau <<" -> "
            //         << arg_tau/2.0 << std::endl;
            //     arg_tau = arg_tau/2.0;

            //     if (arg_tau < tau_threshold) {
            //         std::cout << "tau too small - stopping optimization" 
            //             << std::endl;
            //         break;
            //     }    

            //     // Get new evolution op with decreased arg_tau
            //     //nnh = getMPO2s_NNH(4, arg_tau, arg_J, arg_h);
            //     nnh = getMPO2s_NNHstagh(4, arg_tau, arg_J, arg_h);
            // }
            e_avgE_prev = avgE;

            t_iso_begin = std::chrono::steady_clock::now();
        }
    }

    auto sqrtT = [](double r) { return sqrt(r); };

    l1.apply(sqrtT);
    l2.apply(sqrtT);
    l3.apply(sqrtT);
    l4.apply(sqrtT);

    std::vector<IndexSet> wInds = {l1.inds(), l2.inds(), l3.inds(), l4.inds()};

    auto tA = A * l1 * l2 * l3 * l4;
    for(auto const& is : wInds) tA = tA * delta(is[0], is[1]);
    auto tB = B * l1 * l2 * l3 * l4;
    for(auto const& is : wInds) tB = tB * delta(is[0], is[1]);

    auto pIC = Index(TAG_I_PHYS, cls.physDim, PHYS);
    auto pID = Index(TAG_I_PHYS, cls.physDim, PHYS);

    auto aIC = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);
    auto aID = Index(TAG_I_AUX, cls.auxBondDim, AUXLINK);

    auto tC = tB * delta(pIB,pIC);
    auto tD = tA * delta(pIA,pID);
    for(int i=0; i<=3; i++) {
        tC = tC * delta(prime(aIB,i), prime(aIC,i));
        tD = tD * delta(prime(aIA,i), prime(aID,i));
    }

    // Build new cluster
    cls.sites = {{"A", tA}, {"B", tB}, {"C", tC}, {"D", tD}};
    //std::cout << cls;

    writeCluster(arg_outClusterFile, cls);
}