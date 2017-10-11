#include "ctm-cluster-io.h"
#include "cluster-ev-builder.h"
#include "simple-update.h"
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

    auto tA = A * ls[0] * ls[1] * ls[2] * ls[3];
    tA = swapPrime(tA, 0, 2);
    tA = swapPrime(tA, 1, 3);
    tA.noprime(PHYS);

    auto tB = B * ls[0] * ls[1] * ls[2] * ls[3];
    tB = swapPrime(tB, 0, 2);
    tB = swapPrime(tB, 1, 3);
    tB.noprime(PHYS);

    auto pIA  = noprime(findtype(tA.inds(), PHYS));
    auto pIB  = noprime(findtype(tB.inds(), PHYS));
    auto pIC = Index(TAG_I_PHYS, cls.physDim, PHYS);
    auto pID = Index(TAG_I_PHYS, cls.physDim, PHYS);

    auto aIA  = noprime(findtype(tA.inds(), AUXLINK));
    auto aIB  = noprime(findtype(tB.inds(), AUXLINK));
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

    auto aIA  = noprime(findtype(cls.sites.at("A").inds(), AUXLINK));
    auto aIB  = noprime(findtype(cls.sites.at("B").inds(), AUXLINK));
    auto aIC  = noprime(findtype(cls.sites.at("C").inds(), AUXLINK));
    auto aID  = noprime(findtype(cls.sites.at("D").inds(), AUXLINK));

    auto pIA  = noprime(findtype(cls.sites.at("A").inds(), PHYS));
    auto pIB  = noprime(findtype(cls.sites.at("B").inds(), PHYS));
    auto pIC  = noprime(findtype(cls.sites.at("C").inds(), PHYS));
    auto pID  = noprime(findtype(cls.sites.at("D").inds(), PHYS));

    auto H_nnhAB = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIA, pIB);
    auto H_nnhAC = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIA, pIC);
    auto H_nnhBD = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIB, pID);
    auto H_nnhCD = EVBuilder::get2SiteSpinOP(EVBuilder::OP2S_SS, pIC, pID);

    auto Bra = contractCluster(cls);
    // apply PBC
    Bra = (Bra * delta(aIA, prime(aIB,2))) * delta(aIC, prime(aID,2));
    Bra = (Bra * delta(prime(aIA,1), prime(aIC,3))) * delta(prime(aIB,1), prime(aID,3));

    Print(Bra);

    double cls_norm = std::pow(norm(Bra),2.0);
    
    //Print(Bra);

    auto Ket = prime(conj(Bra),1);

    auto KetAB = Ket * H_nnhAB.first * H_nnhAB.second;
    KetAB.mapprime(PHYS,1,0);
    KetAB = KetAB * Bra;

    // pA1, pB1, pB2, pA2 => op2s => pA0, pB1, pB3, pA2
    auto KetAC = Ket * H_nnhAC.first * H_nnhAC.second; 
    KetAC.mapprime(PHYS,1,0);
    KetAC = KetAC * Bra;

    // pA1, pB1, pB2, pA2 => op2s => pA1, pB1, pB3, pA3
    auto KetCD = Ket * H_nnhCD.first * H_nnhCD.second; 
    KetCD.mapprime(PHYS,1,0);
    KetCD = KetCD * Bra;

    // pA1, pB1, pB2, pA2 => op2s => pA1, pB0, pB2, pA3
    auto KetBD = Ket * H_nnhBD.first * H_nnhBD.second; 
    KetBD.mapprime(PHYS,1,0);
    KetBD = KetBD * Bra;

    if( KetBD.r() > 0 || KetAB.r() > 0 || KetCD.r() >0 || KetAC.r() > 0 ) {
        std::cout <<"Expectation value not a tensor rank 0"<< std::endl;
        exit(EXIT_FAILURE);    
    }

    std::vector< std::complex<double> > evs = {
        sumels(KetAB)/cls_norm, sumels(KetCD)/cls_norm,
        sumels(KetAC)/cls_norm, sumels(KetBD)/cls_norm };

    std::cout << evs[0] <<" "<< evs[1] <<" "<< evs[2] <<" "<< evs[3] << std::endl;

    return evs;
}

// Apply simple update over sites A,B connected by link with
// weight l1. The weights l2,...,l4 are used to construct on-site tensors A,B 
void simpUp(MPO_2site const& nnh, ITensor & A, ITensor & B,
    ITensor & l1, ITensor const& l2, ITensor const& l3, ITensor const& l4,
    ITensor & l1I, ITensor const& l2I, ITensor const& l3I, ITensor const& l4I) {
    
    // Define "regulator" function to cut-off large values after inversion
    // of weight matrices
    // auto regT = [](double r) { 
    //     return ((abs(r) > 1.0e10) ? 0.0 : r); };

    A = A*l2*l3*l4;
    B = B*l2*l3*l4;
    applyH_T1_L_T2(nnh, A, B, l1);
    A = A*l2I*l3I*l4I;
    B = B*l2I*l3I*l4I;

    auto l1IS = l1.inds();
    for (int i=1; i<=l1IS[0].m(); i++ ) {
        l1I.set(l1IS[0](i), l1IS[1](i), 1.0/l1.real(l1IS[0](i), l1IS[0](i)));
    }
    //l2I.apply(regT);
}

int main( int argc, char *argv[] ) {
    // ########################################################################
    // Handle command-line arguments

    //OP2S_TYPE arg_op2s_type;
    //F_MPO3S arg_f_mpo3s;
    std::string arg_initType, arg_inClusterFile, arg_outClusterFile;
    int arg_auxBondDim, arg_nIter;
    double arg_tau, arg_J, arg_h;
    
    arg_initType = std::string(argv[1]);
    if( (arg_initType == "FILE") && (argc >= 8) ) {
        arg_inClusterFile  = argv[2];
        arg_outClusterFile = argv[3];
        arg_nIter          = std::stoi(argv[4]);
        arg_tau            = std::stod(argv[5]);
        arg_J              = std::stod(argv[6]);
        arg_h              = std::stod(argv[7]);

        std::cout <<"Initializing from File: "<< arg_inClusterFile << std::endl;
    // otherwise we start with random cluster 
    } else if( (arg_initType == "RANDOM") && (argc >= 8) ) {
        arg_outClusterFile = argv[2];
        arg_auxBondDim     = std::stoi(argv[3]);
        arg_nIter          = std::stoi(argv[4]);
        arg_tau            = std::stod(argv[5]);
        arg_J              = std::stod(argv[6]);
        arg_h              = std::stod(argv[7]);

        std::cout <<"Initializing by RANDOM TENSORS"<< std::endl;
    } else if( (arg_initType == "AFM") && (argc >= 8) ) {
        arg_outClusterFile = argv[2];
        arg_auxBondDim     = std::stoi(argv[3]);
        arg_nIter          = std::stoi(argv[4]);
        arg_tau            = std::stod(argv[5]);
        arg_J              = std::stod(argv[6]);
        arg_h              = std::stod(argv[7]);

        std::cout <<"Initializing by AFM order A=down, B=up"<< std::endl;        
    } else {
        std::cout <<"Invalid amount of Agrs (< 7)"<< std::endl;
        exit(EXIT_FAILURE);
    }
    
    double eps_threshold = 1.0e-8;
    double tau_threshold = 1.0e-10;

    std::cout <<"Simulation parameters"<< std::endl;
    std::cout <<"imag time tau: "<< arg_tau << std::endl;
    std::cout <<"J            : "<< arg_J << std::endl;
    std::cout <<"h            : "<< arg_h << std::endl;
    std::cout <<"nIterations  : "<< arg_nIter << std::endl;
    std::cout <<"eps_threshold: "<< eps_threshold << std::endl;
    std::cout <<"tau_threshold: "<< tau_threshold << std::endl;

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

    // Compute Initial value of APPROX(!) energy
    formCluster(cls, A, B, {l1, l2, l3, l4});
    auto e_nnh = getEV2Site(cls);
    auto avgE = 2.0*(e_nnh[0] + e_nnh[1] + e_nnh[2] + e_nnh[3])/4.0;
    std::cout <<"E: "<< avgE.real() <<" + "<< avgE.imag() << std::endl;
    auto e_avgE_prev = avgE;

    // Define "regulator" function to cut-off large values after inversion
    // of weight matrices
    auto regT = [](double r) { 
        return ((abs(r) > 1.0e10) ? 0.0 : r); };

    std::cout.precision(10);
    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;
    t_iso_begin = std::chrono::steady_clock::now();
    for (int nStep=1; nStep<=arg_nIter; nStep++) {
        
        // Apply 2-site op along bond A--l2--B
        //simpUp(nnh, A, B, l2, l1,l3,l4, l2I, l1I,l3I,l4I);
        // A = A*l1*l3*l4;
        // B = B*l4*l1*l3;
        // applyH_T1_L_T2(nnh, A, B, l2);
        // A = A*l1I*l3I*l4I;
        // B = B*l4I*l1I*l3I;

        // for (int i=1; i<=aIA.m(); i++ ) {
        //     l2I.set(prime(aIA,2)(i), aIB(i), 1.0/l2.real(prime(aIA,2)(i), aIB(i)));
        // }
        // l2I.apply(regT);
        //PrintData(l2I);
        // std::cout << "##### APPLIED nnh along A--l2--B #####" << std::endl;
        // Print(A);
        // Print(l2);
        // Print(B);

        // Apply 2-site op along bond B--l1--A
        //simpUp(nnh, A, B, l1, l3,l2,l4, l1I, l3I,l2I,l4I);
        // A = A*l3*l2*l4;
        // B = B*l2*l4*l3;
        // applyH_T1_L_T2(nnh, A, B, l1);
        // A = A*l3I*l2I*l4I;
        // B = B*l2I*l4I*l3I;

        // for (int i=1; i<=aIA.m(); i++ ) {
        //     l1I.set(prime(aIB,2)(i), aIA(i), 1.0/l1.real(prime(aIB,2)(i), aIA(i)));
        // }
        // l1I.apply(regT);
        //PrintData(l1I);
        // std::cout << "##### APPLIED nnh along B--l1--A #####" << std::endl;
        // Print(B);
        // Print(l1);
        // Print(A);

        // Apply 2-site op along bond A--l4--B
        //simpUp(nnh, A, B, l4, l1,l3,l2, l4I, l1I,l3I,l2I);
        // A = A*l1*l3*l2;
        // B = B*l1*l3*l2;
        // applyH_T1_L_T2(nnh, A, B, l4);
        // A = A*l1I*l3I*l2I;
        // B = B*l1I*l3I*l2I;

        // for (int i=1; i<=aIA.m(); i++ ) {
        //     l4I.set(prime(aIA,3)(i), prime(aIB,1)(i), 
        //         1.0/l4.real(prime(aIA,3)(i), prime(aIB,1)(i)));
        // }
        // l4I.apply(regT);
        //PrintData(l4I);
        // std::cout << "##### APPLIED nnh along A--l4--B #####" << std::endl;
        // Print(A);
        // Print(l4);
        // Print(B);

        // Apply 2-site op along bond B--l3--A
        //simpUp(nnh, A, B, l3, l2,l4,l1, l3I, l2I,l4I,l1I);
        // A = A*l2*l4*l1;
        // B = B*l2*l4*l1;
        // applyH_T1_L_T2(nnh, A, B, l3);
        // A = A*l2I*l4I*l1I;
        // B = B*l2I*l4I*l1I;

        // for (int i=1; i<=aIA.m(); i++ ) {
        //     l3I.set(prime(aIB,3)(i), prime(aIA,1)(i), 
        //         1.0/l3.real(prime(aIB,3)(i), prime(aIA,1)(i)));
        // }
        // l3I.apply(regT);
        //PrintData(l3I);
        // std::cout << "##### APPLIED nnh along B--l3--A #####" << std::endl;
        // Print(B);
        // Print(l3);
        // Print(A);
    
        simpUp(nnh, A, B, l2, l1,l3,l4, l2I, l1I,l3I,l4I); //A--l2--B
        simpUp(nnh, A, B, l1, l3,l2,l4, l1I, l3I,l2I,l4I); //B--l1--A
        simpUp(nnh, A, B, l4, l1,l3,l2, l4I, l1I,l3I,l2I); //A--l4--B    

        //simpUp(nnh2, A, B, l3, l2,l4,l1, l3I, l2I,l4I,l1I); //B--l3--A
        simpUp(nnh, A, B, l3, l2,l4,l1, l3I, l2I,l4I,l1I); //B--l3--A
        simpUp(nnh, A, B, l3, l2,l4,l1, l3I, l2I,l4I,l1I); //B--l3--A

        simpUp(nnh, A, B, l4, l1,l3,l2, l4I, l1I,l3I,l2I); //A--l4--B
        simpUp(nnh, A, B, l1, l3,l2,l4, l1I, l3I,l2I,l4I); //B--l1--A
        simpUp(nnh, A, B, l2, l1,l3,l4, l2I, l1I,l3I,l4I); //A--l2--B
        

        if ( nStep % 1000 == 0 ) { 
            t_iso_end = std::chrono::steady_clock::now();
            std::cout <<"STEP "<< nStep <<" T= "<< std::chrono::duration_cast
            <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0 
            << std::endl;
            //PrintData(l1);
            //PrintData(l2);
            //PrintData(l3);
            //PrintData(l4);
            t_iso_begin = std::chrono::steady_clock::now();
        
            formCluster(cls, A, B, {l1, l2, l3, l4});
            e_nnh = getEV2Site(cls);
            avgE = 2.0*(e_nnh[0] + e_nnh[1] + e_nnh[2] + e_nnh[3])/4.0;
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
        }
    }

    formCluster(cls, A, B, {l1, l2, l3, l4});
    writeCluster(arg_outClusterFile, cls);
}