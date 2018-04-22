#include <iostream>
#include "itensor/all.h"
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

    svd(A,U1,S1,V1);

    PrintData(U1);
    PrintData(S1);
    PrintData(V1);

    ITensor U(K),S,V;

    svd_dd(A,U,S,V);

    PrintData(U);
    PrintData(S);
    PrintData(V);

    Index iM("m",3430);
    Index iMp = prime(iM);

    ITensor largeT(iM,iMp);
    randomize(largeT);

    auto largeT2 = largeT;
    auto largeT3 = largeT;
    auto largeT4 = largeT;

    ITensor U_l1(iM), S_l1, V_l1;

    // Start timing iteration loop
    std::chrono::steady_clock::time_point t_begin = 
        std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t_begin_int, t_end_int; 
    t_begin_int = std::chrono::steady_clock::now();

    svd(largeT,U_l1,S_l1,V_l1);

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l1.real(S_l1.inds().front()(i),S_l1.inds().back()(i)) << std::endl;    
    
    ITensor U_l(iM), S_l, V_l;
    t_begin_int = std::chrono::steady_clock::now();

    svd_dd(largeT2,U_l,S_l,V_l);

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l.real(S_l.inds().front()(i),S_l.inds().back()(i)) << std::endl;

    auto args = Args::global();
    args.add("SVD_METHOD","gesvdx");

    t_begin_int = std::chrono::steady_clock::now();

    svd_dd(largeT3,U_l,S_l,V_l,args);    

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l.real(S_l.inds().front()(i),S_l.inds().back()(i)) << std::endl;

    args.add("GESVDX_IL",1);
    args.add("GESVDX_IU",70);

    t_begin_int = std::chrono::steady_clock::now();

    svd_dd(largeT4,U_l,S_l,V_l,args);  

    t_end_int = std::chrono::steady_clock::now();
    std::cout <<"T: "<< std::chrono::duration_cast
        <std::chrono::microseconds>(t_end_int - t_begin_int).count()/1000000.0 
        <<" [sec]"<< std::endl;
    for (int i=1;i<=5;i++) std::cout<< S_l.real(S_l.inds().front()(i),S_l.inds().back()(i)) << std::endl;

}