// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <type_traits>

namespace cdfmm {

/** @brief Canonical magnitude and component-wise sign encoding of Tensor6. */
template <typename Scalar> struct CanonicalTensor6 {
  std::array<Scalar, 6> values{};
  std::uint8_t sign_mask{0};
};

/** @brief Canonicalises xx, xy, xz, yy, yz, zz independently. */
template <typename Scalar>
[[nodiscard]] inline CanonicalTensor6<Scalar>
canonicalise_tensor6(const std::array<Scalar, 6> &tensor) noexcept {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>);
  CanonicalTensor6<Scalar> result;
  for (int component = 0; component < 6; ++component) {
    const Scalar value = tensor[static_cast<std::size_t>(component)];
    if (value < Scalar{0}) {
      result.sign_mask |= static_cast<std::uint8_t>(1U << component);
      result.values[static_cast<std::size_t>(component)] = -value;
    } else {
      // Convert -0 to +0 before exact-bit dictionary lookup.
      result.values[static_cast<std::size_t>(component)] =
          value == Scalar{0} ? Scalar{0} : value;
    }
  }
  return result;
}

/** @brief Exact-bit key for a canonical Tensor6 at its execution precision. */
template <typename Scalar> struct Tensor6BitKey {
  using Bits = std::conditional_t<std::is_same_v<Scalar, float>, std::uint32_t,
                                  std::uint64_t>;
  std::array<Bits, 6> values{};
  [[nodiscard]] bool operator==(const Tensor6BitKey &) const noexcept = default;
};

template <typename Scalar>
[[nodiscard]] inline Tensor6BitKey<Scalar>
tensor6_bit_key(const std::array<Scalar, 6> &tensor) noexcept {
  Tensor6BitKey<Scalar> result;
  for (int component = 0; component < 6; ++component) {
    result.values[static_cast<std::size_t>(component)] = std::bit_cast<
        typename Tensor6BitKey<Scalar>::Bits>(tensor[static_cast<std::size_t>(component)]);
  }
  return result;
}

/** @brief Packs a 26-bit dictionary ID and six Tensor6 sign bits. */
[[nodiscard]] constexpr std::uint32_t pack_tensor6_token(
    const std::uint32_t tensor_id, const std::uint8_t sign_mask) noexcept {
  return (tensor_id << 6U) | (static_cast<std::uint32_t>(sign_mask) & 0x3FU);
}
[[nodiscard]] constexpr std::uint32_t tensor6_token_id(const std::uint32_t token) noexcept { return token >> 6U; }
[[nodiscard]] constexpr std::uint8_t tensor6_token_sign_mask(const std::uint32_t token) noexcept { return static_cast<std::uint8_t>(token & 0x3FU); }

} // namespace cdfmm
