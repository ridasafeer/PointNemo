#pragma once
#include <vector>
#include "params.hpp"
namespace pn {
class FxLMS {
public:
  explicit FxLMS(const FxParams&){};
  void load_S_hat(const std::vector<float>&){};
};
}
