#include <iostream>
#include <gtest/gtest.h>
#include "ctm-cluster-basic.h"
#include "ctm-cluster-env_v2.h"

using namespace itensor;

TEST(CtmEnv0, Default_cotr){
	Cluster cls;

	Index aIA, aIB, pIA, pIB, aIC, aID, pIC, pID;
	ITensor A, B, C, D;

    // ----- DEFINE BLANK CLUSTER ----------------------------------
    cls = Cluster_2x2_ABCD();
    cls.auxBondDim = 1;
    cls.physDim    = 1;

    cls.siteIds = std::vector< std::string >(4);
    cls.siteIds = { "A", "B", "C", "D" };
    cls.SI = { {"A",0}, {"B",1}, {"C",2}, {"D",3} };

    cls.cToS  = {
        {std::make_pair(0,0),"A"},
        {std::make_pair(1,0),"B"},
        {std::make_pair(0,1),"C"},
        {std::make_pair(1,1),"D"}
    };
    cls.vToId = { {{0,0},"A"}, {{1,0},"B"}, {{0,1},"C"}, {{1,1},"D"} };

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

    cls.aux  = {aIA, aIB, aIC, aID};
    cls.phys = {pIA, pIB, pIC, pID};

    cls.sites = {{"A", A}, {"B", B}, {"C",C}, {"D",D}};

    // Define siteToWeights
    cls.siteToWeights["A"] = {
        {{"A","B"},{2,0},"L1"},
        {{"A","B"},{0,2},"L2"},
        {{"A","C"},{1,3},"L3"},
        {{"A","C"},{3,1},"L4"}
    };
    cls.siteToWeights["B"] = {
        {{"B","A"},{2,0},"L2"},
        {{"B","A"},{0,2},"L1"},
        {{"B","D"},{1,3},"L5"},
        {{"B","D"},{3,1},"L6"}
    };
    cls.siteToWeights["C"] = {
        {{"C","D"},{2,0},"L7"},
        {{"C","D"},{0,2},"L8"},
        {{"C","A"},{1,3},"L4"},
        {{"C","A"},{3,1},"L3"}
    };
    cls.siteToWeights["D"] = {
        {{"D","B"},{3,1},"L5"},
        {{"D","B"},{1,3},"L6"},
        {{"D","C"},{2,0},"L8"},
        {{"D","C"},{0,2},"L7"}
    };

    setSites(cls, "NEEL", true);

    std::cout<<cls;
}