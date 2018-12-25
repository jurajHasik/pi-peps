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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("A")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("A")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RU,Vertex(0,0)); // upper-right
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("A")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("A")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RD,Vertex(0,0)); // lower-right
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("A")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("A")[0]) );

    t = ctmEnv.build_corner_V2(CORNER::LD,Vertex(0,0)); // lower-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("A",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("A",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("A")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("A")[2]) );


    // site B = Vertex(1,0) ----------------------------------------------------
    t = ctmEnv.build_corner_V2(CORNER::LU,Vertex(1,0)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("B")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("B")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RU,Vertex(1,0)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("B")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("B")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RD,Vertex(1,0)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("B")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("B")[0]) );

    t = ctmEnv.build_corner_V2(CORNER::LD,Vertex(1,0)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("B",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("B",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("B")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("B")[2]) );


    // site C = Vertex(0,1) ----------------------------------------------------
    t = ctmEnv.build_corner_V2(CORNER::LU,Vertex(0,1)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("C")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("C")[3]) );
    
    t = ctmEnv.build_corner_V2(CORNER::RU,Vertex(0,1)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("C")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("C")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RD,Vertex(0,1)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("C")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("C")[0]) );

    t = ctmEnv.build_corner_V2(CORNER::LD,Vertex(0,1)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("C",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("C",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("C")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("C")[2]) );

    // site D = Vertex(1,1) ----------------------------------------------------
    t = ctmEnv.build_corner_V2(CORNER::LU,Vertex(1,1)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",2)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",2), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("D")[2]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::LEFT].at("D")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RU,Vertex(1,1)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",3)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",3), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::UP].at("D")[0]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("D")[3]) );

    t = ctmEnv.build_corner_V2(CORNER::RD,Vertex(1,1)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",0)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",0), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,cls.AIc("D",1)) );
    EXPECT_TRUE( hasindex(t,prime(cls.AIc("D",1), cls.BRAKET_OFFSET)) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::RIGHT].at("D")[1]) );
    EXPECT_TRUE( hasindex(t,ctmEnv.itaux[DIRECTION::DOWN].at("D")[0]) );

    t = ctmEnv.build_corner_V2(CORNER::LD,Vertex(1,1)); // upper-left
    EXPECT_TRUE( t.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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
    EXPECT_TRUE( t.r() == 6 );
    EXPECT_TRUE( tt.r() == 6 );
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

// TODO verify indices on isometries
TEST(CtmEnv_compute_IsometriesT4, Default_cotr){
    using DIRECTION = CtmEnv::DIRECTION;

    auto cls = Cluster_2x2_ABCD("AFM",3,2);

    auto pSvdSolver = std::unique_ptr<SvdSolver>(new SvdSolver());
    CtmEnv ctmEnv("TEST_2x2_ABCD", 2, cls, *pSvdSolver,
        {"isoPseudoInvCutoff",1.0e-8,
         "SVD_METHOD","itensor",
         "dbg",true,
         "dbgLevel",3}
        );
    ctmEnv.init(CtmEnv::INIT_ENV_ctmrg, false, true);


    std::map<std::string, Index> ip; 
    std::map<std::string, Index> ipt;
    std::map<std::string, ITensor> P, Pt;
    std::vector<double> accT(8,0.0);

    auto verifyIndicesIso = [&ctmEnv,&P,&Pt](
        DIRECTION direction, std::string const& id) {

        Shift shift;
        switch (direction) {
            case DIRECTION::LEFT: { 
                shift    = Shift( 0, 1);
                break; 
            }
            case DIRECTION::UP: { 
                shift    = Shift(-1, 0);
                break; 
            }  
            case DIRECTION::RIGHT: {
                shift    = Shift( 0,-1);
                break; 
            } 
            case DIRECTION::DOWN: {
                shift = Shift( 1, 0); 
                break; 
            }
        }

        int dir0 = (direction + 3) % 4;
        int dir1 = (dir0 + 2) % 4;

        auto p  = P.at(id);
        auto pt = Pt.at(id);

        // 
        auto id_shifted = ctmEnv.p_cluster->vertexToId(
            ctmEnv.p_cluster->idToV.at(id) + shift);

        EXPECT_TRUE( p.r() == 4 );
        EXPECT_TRUE( hasindex(p,ctmEnv.p_cluster->AIc(id,dir0)) );
        EXPECT_TRUE( hasindex(p,prime(ctmEnv.p_cluster->AIc(id,dir0),
            ctmEnv.p_cluster->BRAKET_OFFSET)) );
        EXPECT_TRUE( hasindex(p,ctmEnv.itaux[direction].at(id)[dir0]) );

        EXPECT_TRUE( pt.r() == 4);
        EXPECT_TRUE( hasindex(pt,ctmEnv.p_cluster->AIc(id_shifted,dir1)) );
        EXPECT_TRUE( hasindex(pt,prime(ctmEnv.p_cluster->AIc(id_shifted,dir1),
            ctmEnv.p_cluster->BRAKET_OFFSET)) );
        EXPECT_TRUE( hasindex(pt,ctmEnv.itaux[direction].at(id_shifted)[dir1]) );

    };

    ctmEnv.compute_IsometriesT4(DIRECTION::LEFT, ip, ipt, P, Pt, accT);

    ctmEnv.compute_IsometriesT4(DIRECTION::UP, ip, ipt, P, Pt, accT);

    ctmEnv.compute_IsometriesT4(DIRECTION::RIGHT, ip, ipt, P, Pt, accT);

    ctmEnv.compute_IsometriesT4(DIRECTION::DOWN, ip, ipt, P, Pt, accT);
}

TEST(CtmEnv_move_singleDirection, Default_cotr){
    using DIRECTION = CtmEnv::DIRECTION;
    using ISOMETRY  = CtmEnv::ISOMETRY;

    auto cls = Cluster_2x2_ABCD("AFM",3,2);

    auto pSvdSolver = std::unique_ptr<SvdSolver>(new SvdSolver());
    CtmEnv ctmEnv("TEST_2x2_ABCD", 2, cls, *pSvdSolver,
        {"isoPseudoInvCutoff",1.0e-8,
         "SVD_METHOD","itensor",
         "dbg",true,
         "dbgLevel",3}
        );
    ctmEnv.init(CtmEnv::INIT_ENV_ctmrg, false, true);

    // consistency of indices of T_* tensors
    auto verifyIndicesT = [&ctmEnv](DIRECTION direction, std::string const& id) {

        std::map< std::string, ITensor > * ptr_T;
        switch (direction) {
            case DIRECTION::LEFT: { 
                ptr_T = &ctmEnv.T_L;
                break; 
            }
            case DIRECTION::UP: { 
                ptr_T = &ctmEnv.T_U;
                break; 
            }
            case DIRECTION::RIGHT: {
                ptr_T = &ctmEnv.T_R;
                break; }
            case DIRECTION::DOWN: { 
                ptr_T = &ctmEnv.T_D;
                break; }
        }
        std::map< std::string, ITensor > const& Taux  = *ptr_T;

        auto t = Taux.at(id);
        // get orthogonal directions
        int dir0 = (direction + 1) % 4;
        int dir1 = (dir0 + 2) % 4;

        EXPECT_TRUE( t.r() == 4 );
        EXPECT_TRUE( hasindex(t, ctmEnv.p_cluster->AIc(id,direction) ));
        EXPECT_TRUE( hasindex(t, prime(ctmEnv.p_cluster->AIc(id,direction),
            ctmEnv.p_cluster->BRAKET_OFFSET) ));
        EXPECT_TRUE( hasindex(t, ctmEnv.itaux[direction].at(id)[dir0]) );
        EXPECT_TRUE( hasindex(t, ctmEnv.itaux[direction].at(id)[dir1]) );
    };

    // consistency of indices of C_* tensors
    auto verifyIndicesC = [&ctmEnv](DIRECTION direction, std::string const& id) {

        std::map< std::string, ITensor > * ptr_C;
        std::map< std::string, ITensor > * ptr_Ct;
        switch (direction) {
            case DIRECTION::LEFT: { 
                ptr_C  = &ctmEnv.C_LU;
                ptr_Ct = &ctmEnv.C_LD;
                break; 
            }
            case DIRECTION::UP: { 
                ptr_C  = &ctmEnv.C_RU;
                ptr_Ct = &ctmEnv.C_LU;
                break; 
            }
            case DIRECTION::RIGHT: {
                ptr_C  = &ctmEnv.C_RD;
                ptr_Ct = &ctmEnv.C_RU;
                break; }
            case DIRECTION::DOWN: { 
                ptr_C  = &ctmEnv.C_LD;
                ptr_Ct = &ctmEnv.C_RD;
                break; }
        }
        std::map< std::string, ITensor > const& C = *ptr_C;
        std::map< std::string, ITensor > const& Ct = *ptr_Ct;

        auto c  = C.at(id);
        auto ct = Ct.at(id);

        // get directions for environment indices
        int dir0 = (direction + 1) % 4;
        int dir1 = (dir0 + 2) % 4;

        EXPECT_TRUE( c.r() == 2 );
        EXPECT_TRUE( hasindex(c,ctmEnv.itaux[dir0].at(id)[direction]) );
        EXPECT_TRUE( hasindex(c,ctmEnv.itaux[direction].at(id)[dir0]) );

        EXPECT_TRUE( ct.r() == 2 );
        EXPECT_TRUE( hasindex(ct,ctmEnv.itaux[dir1].at(id)[direction]) );
        EXPECT_TRUE( hasindex(ct,ctmEnv.itaux[direction].at(id)[dir1]) );
    };

    std::vector<double> accT(8,0.0);

    ctmEnv.move_singleDirection(DIRECTION::LEFT, ISOMETRY::ISOMETRY_T4, accT);
    for (auto const& id : cls.siteIds) verifyIndicesT(DIRECTION::LEFT, id);
    for (auto const& id : cls.siteIds) verifyIndicesC(DIRECTION::LEFT, id);

    ctmEnv.move_singleDirection(DIRECTION::UP, ISOMETRY::ISOMETRY_T4, accT);
    for (auto const& id : cls.siteIds) verifyIndicesT(DIRECTION::UP, id);
    for (auto const& id : cls.siteIds) verifyIndicesC(DIRECTION::UP, id);

    ctmEnv.move_singleDirection(DIRECTION::RIGHT, ISOMETRY::ISOMETRY_T4, accT);
    for (auto const& id : cls.siteIds) verifyIndicesT(DIRECTION::RIGHT, id);
    for (auto const& id : cls.siteIds) verifyIndicesC(DIRECTION::RIGHT, id);

    ctmEnv.move_singleDirection(DIRECTION::DOWN, ISOMETRY::ISOMETRY_T4, accT);
    for (auto const& id : cls.siteIds) verifyIndicesT(DIRECTION::DOWN, id);
    for (auto const& id : cls.siteIds) verifyIndicesC(DIRECTION::DOWN, id);
}