#pragma once

// Hagan (2002) implied vol for SABR with beta = 1 (Cao M2 parity).

float sabr_hagan_iv_beta1(
    float forward,
    float strike,
    float tau,
    float sigma0,
    float nu,
    float rho
);
