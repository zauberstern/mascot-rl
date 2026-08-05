#pragma once

// Heston characteristic function (Albrecher little-trap branch) and Lewis
// integral pricing via 128-node Gauss-Legendre on u ∈ (0, 200).

struct HestonParams {
    float v0 = 0.04f;
    float theta = 0.04f;
    float kappa = 2.0f;
    float xi = 0.30f;
    float rho = -0.70f;
};

float heston_call_price(
    float spot,
    float strike,
    float tau,
    float rate,
    float div_q,
    const HestonParams& p
);
