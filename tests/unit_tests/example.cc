#include <gtest/gtest.h>
// additional include needed for this test from this project
#include "p-ipeps/engine.h"

TEST(TrotterDecomposition, Default_cotr){
  ASSERT_NO_THROW(TrotterDecomposition<int>{});
  ASSERT_NO_THROW(TrotterDecomposition<double>{});
}
