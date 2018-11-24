#include <gtest/gtest.h>
#include <iostream>
#include "itensor/all.h"
#include "linsyssolvers-lapack.h"

using namespace itensor;

// Solves Ax = B for A = diag(1,2,3), B = (2,2,2)
TEST(LinearSystemCholesky0, Default_cotr){
    double eps = 1.0e-08;
    int dim = 3;
    Index I  = Index("i",dim);
    Index Ip = prime(I,1);

    ITensor A = ITensor(I,Ip);
    for (int i=1; i<=dim; i++) {
        A.set(I(i),Ip(i),1.0*i);
    }

    ITensor B = ITensor(I);
    for (int i=1; i<=dim; i++) {
        B.set(I(i),2.0);
    }

    CholeskySolver linsysSolver = CholeskySolver();
    ITensor X(Ip);
    linsystem(A,B,X,linsysSolver,{"dbg",false});

    ITensor Y(Ip);
    for (int i=1; i<=dim; i++) Y.set(Ip(i),2.0/(1.0*i));

    EXPECT_TRUE( norm(X-Y) < eps );
}

// Solves Ax = B for A = diag(1,2,2,4), B = (4,4,4,4)
TEST(LinearSystemCholesky1, Default_cotr) {
    double eps = 1.0e-08;
    int dim = 2;
    Index J  = Index("i",dim);
    Index Jp = prime(J,2);

    ITensor A = ITensor(J,Jp);
    for (int i=1; i<=dim; i++) {
        A.set(J(i),Jp(i),1.0*i);
    }
    A = A*prime(A,1); // 0,1--A--2,3 <- 0--A--2 * 1--A--3

    ITensor B = ITensor(J);
    for (int i=1; i<=dim; i++) {
        B.set(J(i),2.0);
    }
    B = B*prime(B,1); // 0,1--B <- 0--B * 1--B

    CholeskySolver linsysSolver = CholeskySolver();
    ITensor X(prime(J,2),prime(J,3));
    linsystem(A,B,X,linsysSolver,{"dbg",true,"dpl",2});

    ITensor XX(prime(J,3),prime(J,2));
    linsystem(A,B,XX,linsysSolver,{"dbg",true,"dpl",2});

    ITensor Y(prime(J,2),prime(J,3));
    Y.set(prime(J,2)(1),prime(J,3)(1),4.0/1.0);
    Y.set(prime(J,2)(1),prime(J,3)(2),4.0/2.0);
    Y.set(prime(J,2)(2),prime(J,3)(1),4.0/2.0);
    Y.set(prime(J,2)(2),prime(J,3)(2),4.0/4.0);

    EXPECT_TRUE( norm(X-Y) < eps );
    EXPECT_TRUE( norm(XX-Y) < eps );
}

// Solves Ax = B
// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dposv_ex.f.htm
TEST(LinearSystemCholesky2, Default_cotr) {
    double eps = 1.0e-08;
    Index K  = Index("k",5);
    Index Kp = prime(K);

    auto A = ITensor(K,Kp);
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

    auto B = ITensor(Kp);
    B.set(Kp(1),-7.29);
    B.set(Kp(2), 9.25);
    B.set(Kp(3), 5.99);
    B.set(Kp(4),-1.94);
    B.set(Kp(5),-8.30);

    CholeskySolver linsysSolver = CholeskySolver();
    ITensor X(K);
    linsystem(A,B,X,linsysSolver,{"dbg",true});

    // round to second digit
    auto round2 = [](Real r) { return std::round(r * 100.0) / 100.0; };
    X.apply(round2);

    // Solution
    auto Y = ITensor(K);
    Y.set(K(1),-6.02);
    Y.set(K(2),15.62);
    Y.set(K(3), 3.02);
    Y.set(K(4), 3.25);
    Y.set(K(5),-8.78);

    EXPECT_TRUE( norm(X-Y) < eps );
}

// Solves Ax = B
// https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/zposv_ex.f.htm
TEST(LinearSystemCholesky3, Default_cotr) {
    double eps = 1.0e-08;
    auto K  = Index("k",4);
    auto Kp = prime(K);

    auto A = ITensor(K,Kp);
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

    auto B = ITensor(K);
    B.set(K(1),-2.94+5.79_i);
    B.set(K(2), 8.12-9.12_i);
    B.set(K(3), 9.09-5.03_i);
    B.set(K(4), 7.36+6.77_i);

    CholeskySolver linsysSolver = CholeskySolver();
    ITensor X(Kp);
    linsystem(A,B,X,linsysSolver,{"dbg",true});

    // round to second digit
    auto round2 = [](Cplx c) { return 
        (std::round(c.real() * 100.0) / 100.0) + 
        (std::round(c.imag() * 100.0) / 100.0) * 1.0_i; };
    X.apply(round2);

    auto Y = ITensor(Kp);

    //   Solution
    Y.set(Kp(1), 0.80+1.62_i);
    Y.set(Kp(2), 1.26-1.78_i);
    Y.set(Kp(3), 3.38-0.29_i);
    Y.set(Kp(4), 3.46+2.92_i);

    EXPECT_TRUE( norm(X-Y) < eps );
}
