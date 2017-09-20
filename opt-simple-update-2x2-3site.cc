#include "ctm-cluster-io.h"
#include "cluster-ev-builder.h"
#include "simple-update.h"
#include <chrono>

using namespace itensor;

// Compute expectation value of 2site OP over 2x2 supercell {{A,B},{B,A}} 
// with weights lambda1..lambda4
//
// <bra|    A    B   
//        B |  A | 
//        | |  | |
//       | OP2s ||
//        | |  | |  
//        | A  | B
//        B    A    |ket>
//
// std::complex< double > getEV2Site(std::pair< ITensor, ITensor > const& op2s, 
//     ITensor const& A, ITensor const& B, ITensor const& l1, ITensor const& l2,
//     ITensor const& l3, ITensor const& l4) {

//     // Set new sites to cluster
//     auto squareT = [](double r) { return r*r; };

//     auto aIA = noprime(findtype(A.inds(), AUXLINK));
//     auto aIB = noprime(findtype(B.inds(), AUXLINK));
//     //auto pIA = noprime(findtype(A.inds(), PHYS));
//     //auto pIB = noprime(findtype(B.inds(), PHYS));

//     /*
//      * A--l2--B
//      */
//     auto Bra = A * l2 * B;  
//     //Print(Bra);

//     auto lTemp = l1;
//     lTemp.apply(squareT);

//     //PrintData(l1);
//     //PrintData(lTemp);

//     /*
//      * l1--A--l2--B--l1
//      */
//     Bra = (Bra * lTemp).prime(AUXLINK,4);
//     //Print(Bra);

//     /*
//      * l1--A--l2--B--l1
//      *            |
//      *            l3
//      *            |
//      *            A
//      *  
//      */
//     Bra = Bra * (l3*delta(prime(aIB,7),prime(aIB,3))) * prime(A, PHYS, 1);
//     //Print(Bra);

//     lTemp = l4;
//     lTemp.apply(squareT);

//     /*
//      *            l4 
//      *            |
//      * l1--A--l2--B--l1
//      *            |
//      *            l3
//      *            |
//      *            A
//      *            |
//      *            l4
//      */
//     Bra = Bra * (lTemp*delta(prime(aIB,1),prime(aIB,5)));
//     //Print(Bra);


    
//      *            l4 
//      *            |
//      * l1--A--l2--B--l1
//      *     |      |
//      *     l4     l3
//      *            |
//      *        l1--A
//      *            |
//      *            l4
     
//     Bra = (Bra * l1) * (l4*delta(prime(aIA,7),prime(aIA,3)));
//     //Print(Bra);

//     /*
//      *            l4 
//      *            |
//      * l1--A--l2--B--l1
//      *     |      |
//      *     l4     l3
//      *     |      |
//      *     B--l1--A
//      *            |
//      *            l4
//      */
//     Bra = Bra * prime(B, PHYS, 1);
//     //Print(Bra);

//     lTemp = l2;
//     lTemp.apply(squareT);

//     /*
//      *            l4 
//      *            |
//      * l1--A--l2--B--l1
//      *     |      |
//      *     l4     l3
//      *     |      |
//      * l2--B--l1--A--l2
//      *            |
//      *            l4
//      */
//     Bra = Bra * lTemp;
//     //Print(Bra);

//     lTemp = l3;
//     lTemp.apply(squareT);

//     /*
//      *     l3     l4 
//      *     |      |
//      * l1--A--l2--B--l1
//      *     |      |
//      *     l4     l3
//      *     |      |
//      * l2--B--l1--A--l2
//      *     |      |
//      *     l3     l4
//      */
//     Bra = Bra * (lTemp*delta(prime(aIA,5),prime(aIA,1)));
//     //Print(Bra);

//     auto Ket = prime(conj(Bra),1);

//     //Print(iPs1);
//     //Print(iPs2);

//     Ket = Ket * op2s.first * op2s.second;
//     //Print(Ket);

//     Ket = Ket.mapprime(PHYS,0,1);
//     Ket = Ket.prime(-1);
//     //Print(Ket);

//     double cls_norm = norm(Bra);
//     auto ev_op2s    = sumels(Ket * Bra);

//     return ev_op2s/cls_norm;
// }

int main( int argc, char *argv[] ) {
    // ########################################################################
    // Handle command-line arguments

    //OP2S_TYPE arg_op2s_type;
    //F_MPO3S arg_f_mpo3s;
    std::string arg_initType, arg_inClusterFile, arg_outClusterFile;
    int arg_auxBondDim, arg_nIter;
    double arg_tau, arg_J1, arg_J2;
    
    arg_initType = std::string(argv[1]);
    if( (arg_initType == "FILE") && (argc >= 8) ) {
        arg_inClusterFile  = argv[2];
        arg_outClusterFile = argv[3];
        arg_nIter          = std::stoi(argv[4]);
        arg_tau            = std::stod(argv[5]);
        arg_J1             = std::stod(argv[6]);
        arg_J2             = std::stod(argv[7]);

        std::cout <<"Initializing from File: "<< arg_inClusterFile << std::endl;
    // otherwise we start with random cluster 
    } else if( (arg_initType == "RANDOM") && (argc >= 8) ) {
        arg_outClusterFile = argv[2];
        arg_auxBondDim     = std::stoi(argv[3]);
        arg_nIter          = std::stoi(argv[4]);
        arg_tau            = std::stod(argv[5]);
        arg_J1             = std::stod(argv[6]);
        arg_J2             = std::stod(argv[7]);

        std::cout <<"Initializing by RANDOM TENSORS"<< std::endl;
    } else if( (arg_initType == "AFM") && (argc >= 8) ) {
        arg_outClusterFile = argv[2];
        arg_auxBondDim     = std::stoi(argv[3]);
        arg_nIter          = std::stoi(argv[4]);
        arg_tau            = std::stod(argv[5]);
        arg_J1             = std::stod(argv[6]);
        arg_J2             = std::stod(argv[7]);

        std::cout <<"Initializing by AFM order A=down, B=up"<< std::endl;        
    } else {
        std::cout <<"Invalid amount of Agrs (< 7)"<< std::endl;
        exit(EXIT_FAILURE);
    }
    
    double eps_threshold = 1.0e-8;
    double tau_threshold = 1.0e-10;

    std::cout <<"Simulation parameters"<< std::endl;
    std::cout <<"imag time tau: "<< arg_tau << std::endl;
    std::cout <<"J1           : "<< arg_J1 << std::endl;
    std::cout <<"h2           : "<< arg_J2 << std::endl;
    std::cout <<"nIterations  : "<< arg_nIter << std::endl;
    std::cout <<"eps_threshold: "<< eps_threshold << std::endl;
    std::cout <<"tau_threshold: "<< tau_threshold << std::endl;

    Cluster cls;
    Index aIA, aIB, pIA, pIB, aIC, aID, pIC, pID;
    ITensor A, B, C, D;

    if( arg_initType == "FILE" ) {
        cls = readCluster(arg_inClusterFile);

        A = cls.sites[cls.cToS.at(std::make_pair(0,0))];
        B = cls.sites[cls.cToS.at(std::make_pair(1,0))];
        C = cls.sites[cls.cToS.at(std::make_pair(0,1))];
        D = cls.sites[cls.cToS.at(std::make_pair(1,1))];

        aIA = noprime(findtype(A.inds(), AUXLINK));
        aIB = noprime(findtype(B.inds(), AUXLINK));
        aIC = noprime(findtype(C.inds(), AUXLINK));
        aID = noprime(findtype(D.inds(), AUXLINK));
        pIA = noprime(findtype(A.inds(), PHYS));
        pIB = noprime(findtype(B.inds(), PHYS));
        pIC = noprime(findtype(C.inds(), PHYS));
        pID = noprime(findtype(D.inds(), PHYS));
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

        if (arg_initType == "RANDOM") {
            // Randomize
            randomize(A);
            randomize(B);
            randomize(C);
            randomize(D);
        } else if (arg_initType == "AFM") {
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

    // horizontal
    ITensor l1(prime(aIB,2), aIA);
    ITensor l2(prime(aIA,2), aIB);
    ITensor l3(prime(aID,2), aIC); 
    ITensor l4(prime(aIC,2), aID);
    // vertical
    ITensor l5(prime(aIC,3), prime(aIA,1));
    ITensor l6(prime(aIA,3), prime(aIC,1));
    ITensor l7(prime(aID,3), prime(aIB,1));
    ITensor l8(prime(aIB,3), prime(aID,1));

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

    // Compute Initial value of APPROX(!) energy
    //auto e_nnh = getEV2Site(H_nnh, A, B, l1, l2, l3, l4);
    //std::cout <<"E: "<< e_nnh.real() <<" + "<< e_nnh.imag() << std::endl;
    //auto e_nnh_prev = e_nnh;

    // Define "regulator" function to cut-off large values after inversion
    // of weight matrices
    auto regT = [](double r) { 
        return ((abs(r) > 1.0e10) ? 0.0 : r); };

    std::chrono::steady_clock::time_point t_iso_begin, t_iso_end;
    t_iso_begin = std::chrono::steady_clock::now();
    
    for (int nStep=1; nStep<=arg_nIter; nStep++) {
        // Apply 3-site op along bond A--l2--B--l8--D
        std::cout << "##### APPLYING U123 A--l2--B--l8--D #####" << std::endl;
        A = A*l1*l5*l6;
        B = B*l7*l1;
        D = D*l3*l7*l4;
        applyH_123_X(uJ1J2, A, B, D, l2, l8);
        A = A*l1I*l5I*l6I;
        B = B*l7I*l1I;
        D = D*l3I*l7I*l4I;

        for (int i=1; i<=aIA.m(); i++ ) {
            l2I.set(prime(aIA,2)(i), aIB(i), 1.0/l2.real(prime(aIA,2)(i), aIB(i)));
            l8I.set(prime(aIB,3)(i), prime(aID,1)(i), 
                1.0/l8.real(prime(aIB,3)(i), prime(aID,1)(i)));
        }
        l2I.apply(regT);
        l8I.apply(regT);
        std::cout << "##### DONE A--l2--B--l8--D #####" << std::endl;
        Print(A);
        Print(B);
        Print(D);

        // Apply 3-site op along bond A--l6--C--l4--D
        std::cout << "##### APPLYING U123 A--l6--C--l4--D #####" << std::endl;
        A = A*l1*l5*l2;
        C = C*l3*l5;
        D = D*l8*l3*l7;
        applyH_123_X(uJ1J2, A, C, D, l6, l4);
        A = A*l1I*l5I*l2I;
        C = C*l3I*l5I;
        D = D*l8I*l3I*l7I;

        for (int i=1; i<=aIA.m(); i++ ) {
            l6I.set(prime(aIA,3)(i), prime(aIC,1)(i), 
                1.0/l6.real(prime(aIA,3)(i), prime(aIC,1)(i)));
            l4I.set(prime(aIC,2)(i), aID(i), 1.0/l4.real(prime(aIC,2)(i), aID(i)));
        }
        l6I.apply(regT);
        l4I.apply(regT);
        std::cout << "##### DONE A--l6--C--l4--D #####" << std::endl;
        Print(A);
        Print(C);
        Print(D);

        // Apply 3-site op along bond C--l5--A--l1--B
        std::cout << "##### APPLYING U123 C--l5--A--l1--B #####" << std::endl;
        C = C*l3*l6*l4;
        A = A*l2*l6;
        B = B*l2*l7*l8;
        applyH_123_X(uJ1J2, C, A, B, l5, l1);
        C = C*l3I*l6I*l4I;
        A = A*l2I*l6I;
        B = B*l2I*l7I*l8I;

        for (int i=1; i<=aIA.m(); i++ ) {
            l5I.set(prime(aIC,3)(i), prime(aIA,1)(i), 
                1.0/l5.real(prime(aIC,3)(i), prime(aIA,1)(i)));
            l1I.set(prime(aIB,3)(i), aIA(i), 1.0/l1.real(prime(aIB,3)(i), aIA(i)));
        }
        l5I.apply(regT);
        l1I.apply(regT);
        std::cout << "##### DONE C--l5--A--l1--B #####" << std::endl;
        Print(C);
        Print(A);
        Print(B);

        // Apply 3-site op along bond C--l3--D--l7--B
        std::cout << "##### APPLYING U123 C--l3--D--l7--B #####" << std::endl;
        C = C*l6*l4*l5;
        D = D*l4*l8;
        B = B*l1*l8*l2;
        applyH_123_X(uJ1J2, C, D, B, l3, l7);
        C = C*l6I*l4I*l5I;
        D = D*l4I*l8I;
        B = B*l1I*l8I*l2I;

        for (int i=1; i<=aIA.m(); i++ ) {
            l3I.set(prime(aID,2)(i), aIC(i), 1.0/l3.real(prime(aID,2)(i), aIC(i)));
            l7I.set(prime(aID,3)(i), prime(aIB,1)(i),
                1.0/l7.real(prime(aID,3)(i), prime(aIB,1)(i)));
        }
        l3I.apply(regT);
        l7I.apply(regT);
        std::cout << "##### DONE C--l3--D--l7--B #####" << std::endl;
        Print(C);
        Print(D);
        Print(B);

    //     if ( nStep % 1000 == 0 ) { 
    //         t_iso_end = std::chrono::steady_clock::now();
    //         std::cout <<"STEP "<< nStep <<" T= "<< std::chrono::duration_cast
    //         <std::chrono::microseconds>(t_iso_end - t_iso_begin).count()/1000.0 
    //         << std::endl;
    //         //PrintData(l1);
    //         //PrintData(l2);
    //         //PrintData(l3);
    //         //PrintData(l4);
    //         t_iso_begin = std::chrono::steady_clock::now();
        
    //         auto e_nnh = getEV2Site(H_nnh, A, B, l1, l2, l3, l4);
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
    //         e_nnh_prev = e_nnh;
    //     }
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

    // Build new cluster
    cls.sites = {{"A",A}, {"B",B}, {"C",C}, {"D",D}};
    std::cout << cls; //DBG

    writeCluster(arg_outClusterFile, cls);
}