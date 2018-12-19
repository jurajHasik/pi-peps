#include <iostream>
#include <gtest/gtest.h>
#include "ctm-cluster-basic.h"
#include "ctm-cluster-env_v2.h"

using namespace itensor;

TEST(CtmEnv_init0, Default_cotr){
    auto cls = Cluster_2x2_ABCD("AFM",3,2);
    std::cout<<cls;

    auto pSvdSolver = std::unique_ptr<SvdSolver>(new SvdSolver());
    CtmEnv ctmEnv("TEST_2x2_ABCD", 2, cls, *pSvdSolver,
        {"isoPseudoInvCutoff",1.0e-8,
         "SVD_METHOD","itensor",
         "dbg",true,
         "dbgLevel",3}
        );
    ctmEnv.init(CtmEnv::INIT_ENV_ctmrg, false, true);

    std::cout << ctmEnv; 
}

TEST(CtmEnv_buildCorner0, Default_cotr){
    typedef CtmEnv::CORNER CORNER;
    typedef CtmEnv::DIRECTION DIRECTION;
	
    auto cls = Cluster_2x2_ABCD("AFM",3,2);
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
    // site A = Vertex(0,0) ----------------------------------------------------
    auto t = ctmEnv.build_corner_V2(CORNER::LU,Vertex(0,0)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("A",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("A")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("A")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RU,Vertex(0,0)); // upper-right
    EXPECT_TRUE( hasindex(t,cls.AIc("A",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("A")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("A")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RD,Vertex(0,0)); // lower-right
    EXPECT_TRUE( hasindex(t,cls.AIc("A",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("A")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("A")[0]) );

    t = ctmEnv.build_corner_V2(CORNER::LD,Vertex(0,0)); // lower-left
    EXPECT_TRUE( hasindex(t,cls.AIc("A",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("A")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("A")[2]) );


    // site B = Vertex(1,0) ----------------------------------------------------
    t = ctmEnv.build_corner_V2(CORNER::LU,Vertex(1,0)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("B",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("B")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("B")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RU,Vertex(1,0)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("B",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("B")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("B")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RD,Vertex(1,0)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("B",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("B")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("B")[0]) );

    t = ctmEnv.build_corner_V2(CORNER::LD,Vertex(1,0)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("B",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("B")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("B")[2]) );


    // site C = Vertex(0,1) ----------------------------------------------------
    t = ctmEnv.build_corner_V2(CORNER::LU,Vertex(0,1)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("C",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("C")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("C")[3]) );
    
    t = ctmEnv.build_corner_V2(CORNER::RU,Vertex(0,1)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("C",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("C")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("C")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RD,Vertex(0,1)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("C",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("C")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("C")[0]) );

    t = ctmEnv.build_corner_V2(CORNER::LD,Vertex(0,1)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("C",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("C")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("C")[2]) );

    // site D = Vertex(1,1) ----------------------------------------------------
    t = ctmEnv.build_corner_V2(CORNER::LU,Vertex(1,1)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("D",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("D")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("D")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RU,Vertex(1,1)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("D",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("D")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("D")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RD,Vertex(1,1)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("D",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("D")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("D")[0]) );

    t = ctmEnv.build_corner_V2(CORNER::LD,Vertex(1,1)); // upper-left
    EXPECT_TRUE( hasindex(t,cls.AIc("D",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("D")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("D")[2]) );
}

TEST(CtmEnv_buildHalves0, Default_cotr){
    typedef CtmEnv::DIRECTION DIRECTION;

    // A B 
    // C D
    auto cls = Cluster_2x2_ABCD("AFM",3,2);
    std::cout<<cls;

    auto pSvdSolver = std::unique_ptr<SvdSolver>(new SvdSolver());
    CtmEnv ctmEnv("TEST_2x2_ABCD", 2, cls, *pSvdSolver,
        {"isoPseudoInvCutoff",1.0e-8,
         "SVD_METHOD","itensor",
         "dbg",true,
         "dbgLevel",3}
        );
    ctmEnv.init(CtmEnv::INIT_ENV_ctmrg, false, true);

    //              1 
    // directions 0   2  where 0: A B , 1: B A , 2: D C , 3: C D
    //              3             C D      D C      B A      A B
    // site A = Vertex(0,0) ----------------------------------------------------
    ITensor t, tt;
    ctmEnv.build_halves_V2(DIRECTION::LEFT,Vertex(0,0),t,tt); // direction left: t->upper half, tt->lower half
    EXPECT_TRUE( hasindex(t,cls.AIc("A",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("A")[3]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("B")[3]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("D",1)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("D",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("C",1)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("C",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::LEFT].at("C")[1]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::RIGHT].at("D")[1]) );
    
    ctmEnv.build_halves_V2(DIRECTION::UP,Vertex(0,0),t,tt); // direction up: t->right half, tt->left half
    EXPECT_TRUE( hasindex(t,cls.AIc("A",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("A")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("C")[0]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("B",2)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("B",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("D",2)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("D",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::UP].at("B")[2]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::DOWN].at("D")[2]) );

    ctmEnv.build_halves_V2(DIRECTION::RIGHT,Vertex(0,0),t,tt); // direction right: t->lower half, tt->upper half
    EXPECT_TRUE( hasindex(t,cls.AIc("A",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("A")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("B")[1]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("C",3)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("C",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("D",3)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("D",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::RIGHT].at("C")[3]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::LEFT].at("D")[3]) );

    ctmEnv.build_halves_V2(DIRECTION::DOWN,Vertex(0,0),t,tt); // direction down: t->left half, tt->right half
    EXPECT_TRUE( hasindex(t,cls.AIc("A",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("A")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("C")[2]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("B",0)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("B",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("D",0)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("D",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::DOWN].at("B")[0]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::UP].at("D")[0]) );


    // site B = Vertex(1,0) ----------------------------------------------------
    ctmEnv.build_halves_V2(DIRECTION::LEFT,Vertex(1,0),t,tt); // direction left: t->upper half, tt->lower half
    // B--A
    // 3  3
    // 1  1
    // D--C
    EXPECT_TRUE( hasindex(t,cls.AIc("B",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("B")[3]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("A")[3]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("D",1)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("D",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("C",1)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("C",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::LEFT].at("D")[1]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::RIGHT].at("C")[1]) );

    ctmEnv.build_halves_V2(DIRECTION::UP,Vertex(1,0),t,tt); // direction up: t->right half, tt->left half
    // A--2 0--B
    // |       |
    // C--2 0--D
    EXPECT_TRUE( hasindex(t,cls.AIc("B",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("B")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("D")[0]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("A",2)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("A",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("C",2)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("C",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::UP].at("A")[2]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::DOWN].at("C")[2]) );

    ctmEnv.build_halves_V2(DIRECTION::RIGHT,Vertex(1,0),t,tt); // direction right: t->lower half, tt->upper half
    // C--D
    // 3  3
    // 1  1
    // A--B
    EXPECT_TRUE( hasindex(t,cls.AIc("B",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("B")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("A")[1]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("C",3)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("C",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("D",3)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("D",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::RIGHT].at("D")[3]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::LEFT].at("C")[3]) );

    ctmEnv.build_halves_V2(DIRECTION::DOWN,Vertex(1,0),t,tt); // direction down: t->left half, tt->right half
    // D--2 0--C
    // |       |
    // B--2 0--A
    EXPECT_TRUE( hasindex(t,cls.AIc("B",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("B")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("D")[2]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("A",0)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("A",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("C",0)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("C",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::DOWN].at("A")[0]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::UP].at("C")[0]) );

    // site C = Vertex(0,1) ----------------------------------------------------
    ctmEnv.build_halves_V2(DIRECTION::LEFT,Vertex(0,1),t,tt); // direction left: t->upper half, tt->lower half
    // C--D
    // 3  3
    // 1  1
    // A--B
    EXPECT_TRUE( hasindex(t,cls.AIc("C",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("C")[3]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("D")[3]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("A",1)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("A",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("B",1)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("B",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::LEFT].at("A")[1]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::RIGHT].at("B")[1]) );

    ctmEnv.build_halves_V2(DIRECTION::UP,Vertex(0,1),t,tt); // direction up: t->right half, tt->left half
    // D--2 0--C
    // |       |
    // B--2 0--A
    EXPECT_TRUE( hasindex(t,cls.AIc("C",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("C")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("A")[0]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("D",2)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("D",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("B",2)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("B",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::UP].at("D")[2]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::DOWN].at("B")[2]) );

    ctmEnv.build_halves_V2(DIRECTION::RIGHT,Vertex(0,1),t,tt); // direction right: t->lower half, tt->upper half
    // B--A
    // 3  3
    // 1  1
    // D--C
    EXPECT_TRUE( hasindex(t,cls.AIc("C",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("C")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("D")[1]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("A",3)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("A",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("B",3)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("B",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::RIGHT].at("A")[3]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::LEFT].at("B")[3]) );

    ctmEnv.build_halves_V2(DIRECTION::DOWN,Vertex(0,1),t,tt); // direction down: t->left half, tt->right half
    // A--2 0--B
    // |       |
    // C--2 0--D
    EXPECT_TRUE( hasindex(t,cls.AIc("C",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("C")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("A")[2]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("D",0)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("D",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("B",0)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("B",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::DOWN].at("D")[0]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::UP].at("B")[0]) );


    // site D = Vertex(1,1) ----------------------------------------------------
    ctmEnv.build_halves_V2(DIRECTION::LEFT,Vertex(1,1),t,tt); // direction left: t->upper half, tt->lower half
    // D--C
    // 3  3
    // 1  1
    // B--A
    EXPECT_TRUE( hasindex(t,cls.AIc("D",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("D")[3]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("C")[3]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("A",1)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("A",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("B",1)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("B",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::LEFT].at("B")[1]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::RIGHT].at("A")[1]) );

    ctmEnv.build_halves_V2(DIRECTION::UP,Vertex(1,1),t,tt); // direction up: t->right half, tt->left half
    // C--2 0--D
    // |       |
    // A--2 0--B
    EXPECT_TRUE( hasindex(t,cls.AIc("D",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("D")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("B")[0]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("C",2)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("C",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("A",2)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("A",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::UP].at("C")[2]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::DOWN].at("A")[2]) );

    ctmEnv.build_halves_V2(DIRECTION::RIGHT,Vertex(1,1),t,tt); // direction right: t->lower half, tt->upper half
    // A--B
    // 3  3  
    // 1  1
    // C--D
    EXPECT_TRUE( hasindex(t,cls.AIc("D",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("D")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("C")[1]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("A",3)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("A",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("B",3)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("B",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::RIGHT].at("B")[3]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::LEFT].at("A")[3]) );

    ctmEnv.build_halves_V2(DIRECTION::DOWN,Vertex(1,1),t,tt); // direction down: t->left half, tt->right half
    // B--2 0--A
    // |       |
    // D--2 0--C
    EXPECT_TRUE( hasindex(t,cls.AIc("D",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("D")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("B")[2]) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("A",0)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("A",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,cls.AIc("C",0)) );
    EXPECT_TRUE( hasindex(tt,prime(cls.AIc("C",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::DOWN].at("C")[0]) );
    EXPECT_TRUE( hasindex(tt,ctmEnv.itaux[DIRECTION::UP].at("A")[0]) );
}