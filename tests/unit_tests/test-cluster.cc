#include "pi-peps/config.h"
#include <gtest/gtest.h>
#include "pi-peps/ctm-cluster-basic.h"
#include "pi-peps/ctm-cluster-io.h"
#include "pi-peps/ctm-cluster.h"
#include <iostream>
#include <string>

using namespace itensor;

TEST(Shift0, Default_cotr) {
  auto s1 = Shift(1, 1);
  auto s2 = Shift(2, 3);
  auto s3 = Shift(-2, 4);

  auto s4 = s1 + s2;
  EXPECT_TRUE(s4 == Shift(3, 4));

  auto s5 = s2 + s3;
  EXPECT_TRUE(s5 == Shift(0, 7));

  s2 += s1;
  EXPECT_TRUE(s2 == Shift(3, 4));

  s3 -= s1;
  EXPECT_TRUE(s3 == Shift(-3, 3));
}

TEST(ShiftVertex0, Default_cotr) {
  auto s1 = Shift(1, 1);
  auto s2 = Shift(2, 3);
  auto v1 = Vertex(-2, 4);
  auto v2 = Vertex(3, 2);

  auto v3 = v1 + s1;
  EXPECT_TRUE(v3 == Vertex(-1, 5));

  auto v4 = v1 - s2;
  EXPECT_TRUE(v4 == Vertex(-4, 1));

  v2 += s2;
  EXPECT_TRUE(v2 == Vertex(5, 5));

  v2 -= s1;
  EXPECT_TRUE(v2 == Vertex(4, 4));
}

TEST(ClusterBasic0_1x1_A, Default_cotr) {
  nlohmann::json jCls;
  jCls["type"] = "1X1_A";
  jCls["physDim"] = 2;
  jCls["auxBondDim"] = 3;
  jCls["initBy"] = "ZPRST";

  auto p_cls = Cluster_1x1_A::create(jCls);

  std::string out_file = "out.in";
  writeCluster(out_file, *p_cls);

  auto p_cls_ff = p_readCluster(out_file);

  EXPECT_TRUE(p_cls_ff->siteIds.size() == 1);

  for (auto const& siteId : p_cls_ff->siteIds) {
    EXPECT_TRUE(p_cls_ff->mphys.at(siteId).m() == jCls["physDim"]);

    for (auto const& ai : p_cls_ff->caux.at(siteId)) {
      EXPECT_TRUE(ai.m() == jCls["auxBondDim"]);
    }
  }
}

TEST(ClusterBasic0_2x2_ABBA, Default_cotr) {
  nlohmann::json jCls;
  jCls["type"] = "2X2_ABBA";
  jCls["physDim"] = 2;
  jCls["auxBondDim"] = 3;
  jCls["initBy"] = "ZPRST";

  auto p_cls = Cluster_2x2_ABBA::create(jCls);

  std::string out_file = "out.in";
  writeCluster(out_file, *p_cls);

  auto p_cls_ff = p_readCluster(out_file);

  EXPECT_TRUE(p_cls_ff->siteIds.size() == 2);

  for (auto const& siteId : p_cls_ff->siteIds) {
    EXPECT_TRUE(p_cls_ff->mphys.at(siteId).m() == jCls["physDim"]);

    for (auto const& ai : p_cls_ff->caux.at(siteId)) {
      EXPECT_TRUE(ai.m() == jCls["auxBondDim"]);
    }
  }
}

TEST(ClusterBasic0_2x2_ABCD, Default_cotr) {
  nlohmann::json jCls;
  jCls["type"] = "2X2_ABCD";
  jCls["physDim"] = 2;
  jCls["auxBondDim"] = 3;
  jCls["initBy"] = "ZPRST";

  auto p_cls = Cluster_2x2_ABCD::create(jCls);

  std::string out_file = "out.in";
  writeCluster(out_file, *p_cls);

  auto p_cls_ff = p_readCluster(out_file);

  EXPECT_TRUE(p_cls_ff->siteIds.size() == 4);

  for (auto const& siteId : p_cls_ff->siteIds) {
    EXPECT_TRUE(p_cls_ff->mphys.at(siteId).m() == jCls["physDim"]);

    for (auto const& ai : p_cls_ff->caux.at(siteId)) {
      EXPECT_TRUE(ai.m() == jCls["auxBondDim"]);
    }
  }
}

// TEST(ClusterIO1, Default_cotr) {
//   auto cluster = Cluster_2x2_ABCD(3, 2);
//   std::cout << cluster;
// }
