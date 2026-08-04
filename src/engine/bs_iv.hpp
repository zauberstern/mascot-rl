#pragma once

// Black-Scholes European call price and Brent implied-vol inversion.
// Used by multi-world generators (Heston CF, GJR-GARCH table, SABR) to fill
// the 5D surface tensor with implied vols on the Dupire strike/maturity grid.

float bs_call_price(
    float spot,
    float strike,
    float tau,
    float sigma,
    float rate,
    float div_q
);

// Invert BS call price to implied vol on [1e-4, 5.0]. Returns NaN on failure.
float brent_implied_vol(
    float price,
    float spot,
    float strike,
    float tau,
    float rate,
    float div_q,
    float tol = 1e-8f,
    int max_iter = 100
);
