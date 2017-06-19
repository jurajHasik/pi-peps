#include <iostream>

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-result"
#include "itensor/all.h"
#pragma GCC diagnostic pop

using namespace itensor;

int main( int argc, char *argv[] ) {
    std::cout << "Hello" << std::endl;

    int auxD = 1;
    int physD = 2;
    Index lI("l", auxD);
    Index uI("u", auxD);
    Index rI("r", auxD);
    Index dI("d", auxD);
    
    // combiner indices
    Index llI("ll", auxD*auxD);
    Index uuI("uu", auxD*auxD);
    Index rrI("rr", auxD*auxD);
    Index ddI("dd", auxD*auxD);

    Index pI("S", physD, Site);

    /*
     *       u
     *       |
     *    l--A--r
     *       |
     *       d
     *
     *    A1--A2--A3
     *    |   |   |
     *    A4--A5--A6
     *    |   |   |
     *    A7--A8--A9
     *
     */

    // INITIALIZE
    ITensor A1(pI, rI, dI);
    randomize(A1);
    ITensor A2(pI, lI, rI, dI);
    randomize(A2);
    ITensor A3(pI, lI, dI);
    randomize(A3);

    ITensor A4(pI, uI, rI, dI);
    randomize(A4);
    ITensor A5(pI, lI, uI, rI, dI);
    randomize(A5);
    ITensor A6(pI, lI, uI, dI);
    randomize(A6);

    ITensor A7(pI, uI, rI);
    randomize(A7);
    ITensor A8(pI, lI, uI, rI);
    randomize(A8);
    ITensor A9(pI, lI, uI);
    randomize(A9);

    // CONTRACT
    auto ket = A1*delta(rI, prime(lI,1))*prime(A2);
    Print(ket);
    ket *= prime(A3,2)*delta(prime(rI,1), prime(lI,2));
    Print(ket);

    ket *= prime(A4,3)*delta(dI, prime(uI,3));
    Print(ket);
    ket *= (prime(A5,4)*delta(prime(dI,1), prime(uI,4)))*delta(prime(rI,3), prime(lI,4));
    Print(ket);
    ket *= (prime(A6,5)*delta(prime(dI,2), prime(uI,5)))*delta(prime(rI,4), prime(lI,5));
    Print(ket);

    ket *= prime(A7,6)*delta(prime(dI,3), prime(uI,6));
    Print(ket);
    ket *= (prime(A8,7)*delta(prime(dI,4), prime(uI,7)))*delta(prime(rI,6), prime(lI,7));
    Print(ket);
    ket *= (prime(A9,8)*delta(prime(dI,5), prime(uI,8)))*delta(prime(rI,7), prime(lI,8));
    Print(ket);

    auto llC = combiner(lI, prime(lI,1));
    auto uuC = combiner(uI, prime(uI,1));
    auto rrC = combiner(rI, prime(rI,1));
    auto ddC = combiner(dI, prime(dI,1));

    auto AA1 = (A1*conj(A1).prime(Link))*rrC*ddC;
    auto AA2 = (A2*conj(A2).prime(Link))*llC*rrC*ddC;
    auto AA3 = (A3*conj(A3).prime(Link))*llC*ddC;

    auto AA4 = (A4*conj(A4).prime(Link))*uuC*rrC*ddC;
    auto AA5 = (A5*conj(A5).prime(Link))*llC*uuC*rrC*ddC;
    auto AA6 = (A6*conj(A6).prime(Link))*llC*uuC*ddC;

    auto AA7 = (A7*conj(A7).prime(Link))*uuC*rrC;
    auto AA8 = (A8*conj(A8).prime(Link))*llC*uuC*rrC;
    auto AA9 = (A9*conj(A9).prime(Link))*llC*uuC;

    auto cIl = commonIndex(llC, AA2);
    auto cIu = commonIndex(uuC, AA4);
    auto cIr = commonIndex(rrC, AA1);
    auto cId = commonIndex(ddC, AA1);

    AA1 = AA1*delta(cIr, rrI)*delta(cId, ddI);
    AA2 = AA2*delta(cIl, llI)*delta(cIr, rrI)*delta(cId, ddI);
    AA3 = AA3*delta(cIl, llI)*delta(cId, ddI);

    AA4 = AA4*delta(cIr, rrI)*delta(cIu, uuI)*delta(cId, ddI);
    AA5 = AA5*delta(cIl, llI)*delta(cIu, uuI)*delta(cIr, rrI)*delta(cId, ddI);
    AA6 = AA6*delta(cIl, llI)*delta(cIu, uuI)*delta(cId, ddI);

    AA7 = AA7*delta(cIu, uuI)*delta(cIr, rrI);
    AA8 = AA8*delta(cIl, llI)*delta(cIu, uuI)*delta(cIr, rrI);
    AA9 = AA9*delta(cIl, llI)*delta(cIu, uuI);

    Print(AA1);
    Print(AA2);
    Print(AA3);
    Print(AA4);
    Print(AA5);
    Print(AA6);
    Print(AA7);
    Print(AA8);
    Print(AA9);

    auto normT = AA1*delta(rrI, prime(llI,1))*prime(AA2);
    Print(normT);
    normT *= prime(AA3,2)*delta(prime(rrI,1), prime(llI,2));
    Print(normT);

    normT *= prime(AA4,3)*delta(ddI, prime(uuI,3));
    Print(normT);
    normT *= (prime(AA5,4)*delta(prime(ddI,1), prime(uuI,4)))*delta(prime(rrI,3), prime(llI,4));
    Print(normT);
    normT *= (prime(AA6,5)*delta(prime(ddI,2), prime(uuI,5)))*delta(prime(rrI,4), prime(llI,5));
    Print(normT);

    normT *= prime(AA7,6)*delta(prime(ddI,3), prime(uuI,6));
    Print(normT);
    normT *= (prime(AA8,7)*delta(prime(ddI,4), prime(uuI,7)))*delta(prime(rrI,6), prime(llI,7));
    Print(normT);
    normT *= (prime(AA9,8)*delta(prime(ddI,5), prime(uuI,8)))*delta(prime(rrI,7), prime(llI,8));
    Print(normT);

    std::cout<< "Norm(ket): " << norm(ket)*norm(ket) << std::endl;

    //PrintData(ket);
}