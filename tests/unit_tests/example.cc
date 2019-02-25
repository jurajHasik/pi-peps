#include "pi-peps/config.h"
#include <gtest/gtest.h>
// additional include needed for this test from this project
#include "pi-peps/engine.h"

TEST(TrotterDecomposition, Default_cotr){
  ASSERT_NO_THROW(TrotterDecomposition<int>{});
  ASSERT_NO_THROW(TrotterDecomposition<double>{});
}
