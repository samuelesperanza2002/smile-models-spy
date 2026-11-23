"""
Vanna-Volga method (Castagna and Mercurio 2007), exact version, adapted to equity options.

The idea: start from the Black price with one flat volatility (the at-the-money one) and add a
correction. The correction comes from three liquid options, the "pillars", whose market prices
are known. The weights x1, x2, x3 of the pillars are chosen so that the three pillars together
have the same vega, vanna and volga as the option to price (Chapter 3 of the thesis).

    V_VV(K) = V_BS(K; sigma_ref) + sum_i x_i(K) * [V_BS(K_i; s_i) - V_BS(K_i; sigma_ref)]

The pillars have strikes K1 < K2 < K3 and implied volatilities s1, s2, s3. sigma_ref is the
volatility of the at-the-money pillar. Calls and puts get the same correction (put-call parity).
The implied volatility of the result is found by inverting Black's formula.
"""

import numpy as np
import pandas as pd
import black

PILLAR_SETS = {                    # name: target forward delta of each of the three pillars
    "P1": (("put", -0.25), ("atm", 0.50), ("call", 0.25)),      # FX standard
    "P2": (("put", -0.10), ("put", -0.25), ("atm", 0.50)),      # shifted towards the puts
    "P3": (("put", -0.10), ("atm", 0.50), ("call", 0.10)),      # wide
}


def select_pillars(smile, set_name):
    """Choose the three pillars among the out-of-the-money options of one smile.

    For each target delta, the pillar is the listed option with the closest forward delta.
    For the ATM pillar a put counts with its call-equivalent delta (1 + delta_put), and the option
    closest to 0.50 is used. Returns a table of 3 rows sorted by strike, or None when two targets
    fall on the same strike."""
    rows = []
    for kind, target in PILLAR_SETS[set_name]:
        if kind == "put":
            cand = smile[smile.right == "PUT"]
            if not len(cand):
                return None
            i = (cand.delta - target).abs().idxmin()
        elif kind == "call":
            cand = smile[smile.right == "CALL"]
            if not len(cand):
                return None
            i = (cand.delta - target).abs().idxmin()
        else:
            dcall = np.where(smile.right == "CALL", smile.delta, 1 + smile.delta)
            i = smile.index[int(np.argmin(np.abs(dcall - target)))]
        r = smile.loc[i]
        rows.append(dict(kind=kind, target=target, strike=float(r.strike), iv=float(r.iv), right=r.right,
                         delta=float(r.delta), delta_call=float(r.delta if r.right == "CALL" else 1 + r.delta),
                         idx=i))
    p = pd.DataFrame(rows)
    if p.strike.nunique() < 3:
        return None
    p = p.sort_values("strike").reset_index(drop=True)
    return p


class VannaVolga:
    """Exact Vanna-Volga pricer for one smile (one expiry at one point in time).

    K, sig: strikes and implied volatilities of the three pillars, sorted by strike.
    ref_index: position (0, 1 or 2) of the at-the-money pillar, whose volatility is sigma_ref."""

    def __init__(self, F, tau, D, K, sig, ref_index):
        self.F, self.tau, self.D = F, tau, D
        self.K = np.asarray(K, float)                 # K1 < K2 < K3
        self.sig = np.asarray(sig, float)             # s1, s2, s3
        self.ref_index = int(ref_index)
        self.sref = float(self.sig[self.ref_index])   # reference volatility = ATM pillar
        self.lK = np.log(self.K)
        self.vega_p = black.vega(F, self.K, tau, self.sref, D)
        # correction at each pillar: market price (Black with the pillar IV) minus Black price with sigma_ref
        self.corr_p = black.price(F, self.K, tau, self.sig, D, True) - black.price(F, self.K, tau, self.sref, D, True)

    def weights(self, K):
        """Weights x1, x2, x3 of the three pillars for the strikes K (closed form of Castagna and Mercurio)."""
        K = np.asarray(K, float)
        lK = np.log(K)
        K1, K2, K3 = self.lK
        v = black.vega(self.F, K, self.tau, self.sref, self.D)
        x1 = v / self.vega_p[0] * (K2 - lK) * (K3 - lK) / ((K2 - K1) * (K3 - K1))
        x2 = v / self.vega_p[1] * (lK - K1) * (K3 - lK) / ((K2 - K1) * (K3 - K2))
        x3 = v / self.vega_p[2] * (lK - K1) * (lK - K2) / ((K3 - K1) * (K3 - K2))
        return x1, x2, x3

    def price(self, K, is_call):
        """Vanna-Volga price: Black price at sigma_ref plus the weighted pillar corrections."""
        x1, x2, x3 = self.weights(K)
        base = black.price(self.F, np.asarray(K, float), self.tau, self.sref, self.D, is_call)
        return base + x1 * self.corr_p[0] + x2 * self.corr_p[1] + x3 * self.corr_p[2]

    def iv(self, K, is_call):
        """Implied volatility of the Vanna-Volga price. It is nan where the price breaks the
        no-arbitrage bounds: this counts as a failure of the method."""
        K = np.atleast_1d(np.asarray(K, float))
        is_call = np.broadcast_to(np.asarray(is_call, bool), K.shape)
        p = self.price(K, is_call)
        return np.array([black.implied_vol(float(pi), self.F, float(k), self.tau, self.D, bool(c))
                         for pi, k, c in zip(p, K, is_call)])

    def shifted(self, F_new, tau_new, D_new, delta_call):
        """New pricer for a later snapshot, without new market data ("sticky delta", Chapter 3).

        Each pillar keeps its implied volatility and its call-equivalent delta. Its new strike follows
        from the new forward and time to expiry: K = F exp(-q s sqrt(tau) + s^2 tau / 2), q = N^-1(delta).
        The "market" price of a moved pillar is the Black price with its stored implied volatility."""
        K_new = black.strike_from_delta(F_new, tau_new, self.sig, np.asarray(delta_call, float))
        order = np.argsort(K_new)
        ref_new = int(np.where(order == self.ref_index)[0][0])
        return VannaVolga(F_new, tau_new, D_new, K_new[order], self.sig[order], ref_new)


def ref_index(set_name):
    """Position of the ATM pillar after sorting the pillars by strike: P1 -> 1, P2 -> 2, P3 -> 1."""
    return {"P1": 1, "P2": 2, "P3": 1}[set_name]
