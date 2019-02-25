#include "pi-peps/config.h"
#include <iostream>
DISABLE_WARNINGS
#include "itensor/all.h"
ENABLE_WARNINGS
#include <chrono>

using namespace itensor;

int main( int argc, char *argv[] ) {
    std::cout << "Hello" << std::endl;

    Index K  = Index("k",5);
    Index Kp = prime(K);

    ITensor A(K,Kp);
    A.set(K(1),Kp(1), 6.80);
    A.set(K(1),Kp(2),-6.05);
    A.set(K(1),Kp(3),-0.45);
    A.set(K(1),Kp(4), 8.32);
    A.set(K(1),Kp(5),-9.67); 

    A.set(K(2),Kp(1),-2.11);
    A.set(K(2),Kp(2),-3.30);
    A.set(K(2),Kp(3), 2.58);
    A.set(K(2),Kp(4), 2.71);
    A.set(K(2),Kp(5),-5.14); 

    A.set(K(3),Kp(1), 5.66);
    A.set(K(3),Kp(2), 5.36);
    A.set(K(3),Kp(3),-2.70);
    A.set(K(3),Kp(4), 4.35);
    A.set(K(3),Kp(5),-7.26); 

    A.set(K(4),Kp(1), 5.97);
    A.set(K(4),Kp(2),-4.44);
    A.set(K(4),Kp(3), 0.27);
    A.set(K(4),Kp(4),-7.17);
    A.set(K(4),Kp(5), 6.08); 

    A.set(K(5),Kp(1), 8.23);
    A.set(K(5),Kp(2), 1.08);
    A.set(K(5),Kp(3), 9.04);
    A.set(K(5),Kp(4), 2.14);
    A.set(K(5),Kp(5),-6.87); 

    ITensor U1(K),S1,V1;

    svd(A,U1,S1,V1,{"SVDThreshold",1E-16});

    PrintData(U1);
    PrintData(S1);
    PrintData(V1);

    ITensor U(K),S,V;

    svd_dd(A,U,S,V,{"SVD_METHOD","gesdd"});

    PrintData(U);
    PrintData(S);
    PrintData(V);

    svd_dd(A,U,S,V,{"SVD_METHOD","rsvd","Maxm",5});

    PrintData(U);
    PrintData(S);
    PrintData(V);

    // ITensor U2(K),S2,V2;

    // svd_dd(A,U2,S2,V2,{"SVD_METHOD","redsvd"});

    // PrintData(U2);
    // PrintData(S2);
    // PrintData(V2);

    // std::cout << "TRUNCATED" << std::endl;

    // svd(A,U1,S1,V1,{"Maxm",2,"SVDThreshold",1E-16});

    // PrintData(U1);
    // PrintData(S1);
    // PrintData(V1);

    // svd_dd(A,U,S,V,{"Maxm",2,"SVD_METHOD","gesdd"});

    // PrintData(U);
    // PrintData(S);
    // PrintData(V);

    int maxm = 70;
    Index iM("m",49*70);
    Index iMp = prime(iM);

    ITensor largeT(iM,iMp);
    randomize(largeT);

    // auto largeT2 = largeT;
    // auto largeT3 = largeT;
    // auto largeT4 = largeT;

    // auto largeT5 = largeT;
    // auto largeT6 = largeT;

    std::cout << "ITensor svd implementation: "<< std::endl;
    ITensor U_l1(iM), S_l1, V_l1;

    // Start timing iteration loop
    std::chrono::steady_clock::time_point t_begin_int, t_end_int; 
    t_begin_int = std::chrono::steady_clock::now();

    svd_dd(largeT,U_l1,S_l1,V_l1);

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;    

    std::cout << "ITensor svd_dd implementation: "<< std::endl;

    // Start timing iteration loop
    t_begin_int = std::chrono::steady_clock::now();

    svd_dd(largeT,U_l1,S_l1,V_l1);

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;    
    
    //  t_begin_int = std::chrono::steady_clock::now();

    // svd(largeT,U_l1,S_l1,V_l1,{"SVDThreshold",1E-16});

    // t_end_int = std::chrono::steady_clock::now();
    // std::cout <<"T: "<< std::chrono::duration_cast
    //     <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
    //     <<" [sec]"<< std::endl;
    // for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;

    // std::cout << "TRUNCATE" << std::endl;
    // svd(largeT,U_l1,S_l1,V_l1,{"Maxm",5});
    // for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;    

    // std::cout << "svd_dd itensor full" << std::endl;
    // svd_dd(largeT,U_l1,S_l1,V_l1, {"SVD_METHOD","itensor"});
    // for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;    

    // std::cout << "svd_dd itensor truncated" << std::endl;
    // svd_dd(largeT,U_l1,S_l1,V_l1, {"SVD_METHOD","itensor"});
    // for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;    


    std::cout << "dgesdd implementation: "<< std::endl;
    ITensor U_l(iM), S_l, V_l;
    
    auto args = Args::global();
    args.add("SVD_METHOD","gesdd");
    t_begin_int = std::chrono::steady_clock::now();

    svd_dd(largeT,U_l,S_l,V_l,args);

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l.real(S_l.inds().front()(i),S_l.inds().back()(i)) << std::endl;



    // std::cout << "redsvd implementation: "<< std::endl;
    // ITensor U_l3(iM), S_l3, V_l3;
    
    // args = Args::global();
    // args.add("SVD_METHOD","redsvd");
    // t_begin_int = std::chrono::steady_clock::now();

    // svd_dd(largeT,U_l3,S_l3,V_l3,args);

    // t_end_int = std::chrono::steady_clock::now();
    // std::cout <<"T: "<< std::chrono::duration_cast
    //     <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
    //     <<" [sec]"<< std::endl;
    // for (int i=1;i<=5;i++) std::cout<< S_l3.real(S_l3.inds().front()(i),S_l3.inds().back()(i)) << std::endl;

    // args = Args::global();
    // args.add("SVD_METHOD","gesvdx");

    // t_begin_int = std::chrono::steady_clock::now();

    // svd_dd(largeT3,U_l,S_l,V_l,args);    

    // t_end_int = std::chrono::steady_clock::now();
    // std::cout <<"T: "<< std::chrono::duration_cast
    //     <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
    //     <<" [sec]"<< std::endl;
    // for (int i=1;i<=5;i++) std::cout<< S_l.real(S_l.inds().front()(i),S_l.inds().back()(i)) << std::endl;

    // args.add("GESVDX_IL",1);
    // args.add("GESVDX_IU",70);

    // t_begin_int = std::chrono::steady_clock::now();

    // svd_dd(largeT4,U_l,S_l,V_l,args);  

    // t_end_int = std::chrono::steady_clock::now();
    // std::cout <<"T: "<< std::chrono::duration_cast
    //     <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
    //     <<" [sec]"<< std::endl;
    // for (int i=1;i<=5;i++) std::cout<< S_l.real(S_l.inds().front()(i),S_l.inds().back()(i)) << std::endl;

    // std::cout << "TRUNCATE" << std::endl;

    // std::cout << "ITensor implementation: "<< std::endl;
    // svd(largeT5,U_l1,S_l1,V_l1,{"Maxm",5});
    // for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;

    // std::cout << "GESDD implementation: "<< std::endl;
    // largeT2 = largeT;
    // svd_dd(largeT6,U_l,S_l,V_l,{"Maxm",5,"SVD_METHOD","gesdd"});
    // for (int i=1;i<=5;i++) std::cout<< S_l.real(S_l.inds().front()(i),S_l.inds().back()(i)) << std::endl;  

    std::cout << "TRUNCATE ITensor svd implementation: "<< std::endl;

    // Start timing iteration loop
    t_begin_int = std::chrono::steady_clock::now();

    svd(largeT,U_l1,S_l1,V_l1,{"Maxm",maxm});

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;

    // PrintData(U_l1);
    // PrintData(S_l1);
    // PrintData(V_l1);

    std::cout << "TRUNCATE ITensor svd_dd implementation: "<< std::endl;

    // Start timing iteration loop
    t_begin_int = std::chrono::steady_clock::now();

    svd_dd(largeT,U_l1,S_l1,V_l1,{"Maxm",maxm});

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;    
    

    std::cout << "TRUNCATE dgesdd implementation: "<< std::endl;

    args = Args::global();
    args.add("SVD_METHOD","gesdd");
    args.add("Maxm",maxm);
    t_begin_int = std::chrono::steady_clock::now();

    svd_dd(largeT,U_l,S_l,V_l,args);

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l.real(S_l.inds().front()(i),S_l.inds().back()(i)) << std::endl;


    std::cout << "TRUNCATE rsvd implementation: "<< std::endl;

    args = Args::global();
    args.add("SVD_METHOD","rsvd");
    args.add("Maxm",maxm);
    t_begin_int = std::chrono::steady_clock::now();

    svd_dd(largeT,U_l,S_l,V_l,args);

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l.real(S_l.inds().front()(i),S_l.inds().back()(i)) << std::endl;

    std::cout<<"Norm largeT: "<< norm(largeT) << std::endl;
    std::cout<<"Norm Approx itensor svd: "<< norm(U_l1*S_l1*V_l1) << std::endl;
    std::cout<<"Diff-norm itensor svd: "<< norm(largeT - U_l1*S_l1*V_l1) << std::endl;
    std::cout<<"Norm Approx rsvd: "<< norm(U_l*S_l*V_l) << std::endl;
    std::cout<<"Diff-norm itensor svd: "<< norm(largeT - U_l*S_l*V_l) << std::endl;

    // PrintData(U_l);
    // PrintData(S_l);
    // PrintData(V_l);

    // std::cout << "TRUNCATE redsvd implementation: "<< std::endl;
    
    // args = Args::global();
    // args.add("SVD_METHOD","redsvd");
    // args.add("Maxm",64);
    // t_begin_int = std::chrono::steady_clock::now();

    // svd_dd(largeT,U_l3,S_l3,V_l3,args);

    // t_end_int = std::chrono::steady_clock::now();
    // std::cout <<"T: "<< std::chrono::duration_cast
    //     <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
    //     <<" [sec]"<< std::endl;
    // for (int i=1;i<=5;i++) std::cout<< S_l3.real(S_l3.inds().front()(i),S_l3.inds().back()(i)) << std::endl;
}
