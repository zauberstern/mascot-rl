#include "sabr_hagan.hpp"

#include <cmath>
#include <algorithm>

float sabr_hagan_iv_beta1(
    float forward,
    float strike,
    float tau,
    float sigma0,
    float nu,
    float rho
) {
    if (!(forward > 0.0f) || !(strike > 0.0f) || !(sigma0 > 0.0f) || tau <= 0.0f) {
        return sigma0;
    }
    const float logfk = std::log(forward / strike);
    if (std::fabs(logfk) < 1e-7f || std::fabs(nu) < 1e-12f) {
        // ATM / low-vol-of-vol limit
        const float corr =
            1.0f
            + (0.25f * rho * nu * sigma0 + (2.0f - 3.0f * rho * rho) * nu * nu / 24.0f)
                  * tau;
        return sigma0 * corr;
    }
    const float z = (nu / sigma0) * logfk;
    const float disc = std::sqrt(std::max(0.0f, 1.0f - 2.0f * rho * z + z * z));
    const float xz = std::log((disc - rho + z) / (1.0f - rho));
    float z_over_x = 1.0f;
    if (std::fabs(xz) > 1e-12f) {
        z_over_x = z / xz;
    }
    const float corr =
        1.0f
        + (0.25f * rho * nu * sigma0 + (2.0f - 3.0f * rho * rho) * nu * nu / 24.0f) * tau;
    return std::max(sigma0 * z_over_x * corr, 1e-4f);
}
