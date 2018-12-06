#include <iostream>
#include <gtest/gtest.h>
#include "ctm-cluster-basic.h"
#include "ctm-cluster-env_v2.h"

using namespace itensor;

TEST(CtmEnv_buildCorner0, Default_cotr){
	
    auto cls = Cluster_2x2_ABCD("AFM",2,3);
    std::cout<<cls;

    auto pSvdSolver = std::unique_ptr<SvdSolver>(new SvdSolver());
    CtmEnv ctmEnv("TEST_2x2_ABCD", 2, cls, *pSvdSolver,
        {"isoPseudoInvCutoff",1.0e-8,
         "SVD_METHOD","itensor",
         "dbg",true,
         "dbgLevel",3}
        );
    ctmEnv.init(CtmEnv::INIT_ENV_ctmrg, false, true);

    // 1|2
    // 4|3
    // site A = Vertex(0,0)
    // Check construction of the corners
    auto t = ctmEnv.build_corner_V2(1,cls,Vertex(0,0)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),6)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),7)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(2,cls,Vertex(0,0)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),4)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),7)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(3,cls,Vertex(0,0)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),5)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),4)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(4,cls,Vertex(0,0)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),5)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("A"),6)) );

    // site B = Vertex(1,0)
    // Check construction of the corners
    t = ctmEnv.build_corner_V2(1,cls,Vertex(1,0)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),6)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),7)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(2,cls,Vertex(1,0)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),4)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),7)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(3,cls,Vertex(1,0)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),5)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),4)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(4,cls,Vertex(1,0)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),5)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("B"),6)) );


    // site C = Vertex(0,1)
    // Check construction of the corners
    t = ctmEnv.build_corner_V2(1,cls,Vertex(0,1)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),6)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),7)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(2,cls,Vertex(0,1)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),4)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),7)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(3,cls,Vertex(0,1)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),5)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),4)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(4,cls,Vertex(0,1)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),5)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("C"),6)) );


    // site D = Vertex(1,1)
    // Check construction of the corners
    t = ctmEnv.build_corner_V2(1,cls,Vertex(1,1)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),6)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),7)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(2,cls,Vertex(1,1)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),4)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),7)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(3,cls,Vertex(1,1)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),5)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),4)) );

    // Check construction of the corners
    t = ctmEnv.build_corner_V2(4,cls,Vertex(1,1)); // upper-left
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),5)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.maux.at("D"),6)) );
}