#include <iostream>
#include "itensor/all.h"


using namespace itensor;

int main( int argc, char *argv[] ) {
    //####################################
    std::cout << "TESTING CholeskySolver" << std::endl;

    CholeskySolver linsysSolver = CholeskySolver();
 
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

    linsystem(A,B,X,linsysSolver,{"plDiff",1,"dbg",true});

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

    
    linsystem(AA,BB,Y,linsysSolver,{"plDiff",2,"dbg",true});

    PrintData(Y);

    std::cout << "TESTING CholeskySolver A is REAL" << std::endl;
    Index K  = Index("k",5);
    Index Kp = prime(K);

    A = ITensor(K,Kp);
    A.set(K(1),Kp(1), 3.14);
    A.set(K(1),Kp(2), 0.17);
    A.set(K(1),Kp(3),-0.90);
    A.set(K(1),Kp(4), 1.65);
    A.set(K(1),Kp(5),-0.72); 

    A.set(K(2),Kp(1), 0.17);
    A.set(K(2),Kp(2), 0.79);
    A.set(K(2),Kp(3), 0.83);
    A.set(K(2),Kp(4),-0.65);
    A.set(K(2),Kp(5), 0.28); 

    A.set(K(3),Kp(1),-0.90);
    A.set(K(3),Kp(2), 0.83);
    A.set(K(3),Kp(3), 4.53);
    A.set(K(3),Kp(4),-3.70);
    A.set(K(3),Kp(5), 1.60); 

    A.set(K(4),Kp(1), 1.65);
    A.set(K(4),Kp(2),-0.65);
    A.set(K(4),Kp(3),-3.70);
    A.set(K(4),Kp(4), 5.32);
    A.set(K(4),Kp(5),-1.37); 

    A.set(K(5),Kp(1),-0.72);
    A.set(K(5),Kp(2), 0.28);
    A.set(K(5),Kp(3), 1.60);
    A.set(K(5),Kp(4),-1.37);
    A.set(K(5),Kp(5), 1.98); 

    B = ITensor(Kp);
    B.set(Kp(1),-7.29);
    B.set(Kp(2), 9.25);
    B.set(Kp(3), 5.99);
    B.set(Kp(4),-1.94);
    B.set(Kp(5),-8.30);

    ITensor Z;
    linsystem(A,B,Z,linsysSolver,{"plDiff",1,"dbg",true});

//     Solution
// *  -6.02
// *  15.62
// *   3.02
// *   3.25
// *  -8.78

    PrintData(Z);

    std::cout << "TESTING CholeskySolver A is COMPLEX" << std::endl;
    K  = Index("k",4);
    Kp = prime(K);

    A = ITensor(K,Kp);
    A.set(K(1),Kp(1), 5.96+0.00_i);
    A.set(K(1),Kp(2), 0.40-1.19_i);
    A.set(K(1),Kp(3),-0.83-0.48_i);
    A.set(K(1),Kp(4),-0.57+0.40_i);

    A.set(K(2),Kp(1), 0.40+1.19_i);
    A.set(K(2),Kp(2), 7.95+0.00_i);
    A.set(K(2),Kp(3), 0.33+0.09_i);
    A.set(K(2),Kp(4), 0.22+0.74_i);

    A.set(K(3),Kp(1),-0.83+0.48_i);
    A.set(K(3),Kp(2), 0.33-0.09_i);
    A.set(K(3),Kp(3), 4.43+0.00_i);
    A.set(K(3),Kp(4),-1.09+0.32_i);

    A.set(K(4),Kp(1),-0.57-0.40_i);
    A.set(K(4),Kp(2), 0.22-0.74_i);
    A.set(K(4),Kp(3),-1.09-0.32_i);
    A.set(K(4),Kp(4), 3.46+0.00_i);

    B = ITensor(K);
    B.set(K(1),-2.94+5.79_i);
    B.set(K(2), 8.12-9.12_i);
    B.set(K(3), 9.09-5.03_i);
    B.set(K(4), 7.36+6.77_i);

    ITensor W(Kp);
    linsystem(A,B,W,clinsysSolver,{"plDiff",1,"dbg",true});

    PrintData(W);

//   Solution
// * (  0.80,  1.62)
// * (  1.26, -1.78)
// * (  3.38, -0.29)
// * (  3.46,  2.92)

    PrintData(W);
    std::cout << "END TESTING CholeskySolver" << std::endl;
    //####################################

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
