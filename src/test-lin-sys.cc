#include <iostream>
#include "itensor/all.h"

using namespace itensor;

int main( int argc, char *argv[] ) {
    std::cout << "Hello" << std::endl;

    int dim = 3;
    Index I  = Index("i",dim);
    Index Ip = prime(I,1);

    ITensor A = ITensor(I,Ip);

    for (int i=1; i<=dim; i++) {
        A.set(I(i),Ip(i),1.0*i);
    }
    PrintData(A);

    ITensor B = ITensor(I);

    for (int i=1; i<=dim; i++) {
        B.set(I(i),2.0);
    }
    PrintData(B);

    ITensor X;

    linsystem(A,B,X,{"plDiff",1,"dbg",true});

    PrintData(X);


    dim = 2;
    Index J  = Index("i",dim);
    Index Jp = prime(J,2);

    ITensor AA = ITensor(J,Jp);
    for (int i=1; i<=dim; i++) {
        AA.set(J(i),Jp(i),1.0*i);
    }
    AA = AA*prime(AA,1);
    PrintData(AA);

    ITensor BB = ITensor(J);
    for (int i=1; i<=dim; i++) {
        BB.set(J(i),2.0);
    }
    BB = BB*prime(BB,1);
    PrintData(BB);

    ITensor Y;

    linsystem(AA,BB,Y,{"plDiff",2,"dbg",true});

    PrintData(Y);

    Index K  = Index("k",5);
    Index Kp = prime(K);

    A = ITensor(K,Kp);
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

    B = ITensor(Kp);
    B.set(Kp(1), 4.02);
    B.set(Kp(2), 6.19);
    B.set(Kp(3),-8.22);
    B.set(Kp(4),-7.57);
    B.set(Kp(5),-3.03);

    ITensor Z;
    linsystem(A,B,Z,{"plDiff",1,"dbg",true});

    PrintData(Z);
}