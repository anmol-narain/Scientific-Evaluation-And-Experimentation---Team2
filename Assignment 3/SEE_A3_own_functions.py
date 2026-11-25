# --- Team 2: end pose from the raw pen marks (team2_recalc.py, Assignment 3, data review) ---
L_PEN = 3.8   # cm
def pose_from_pen_marks(ax, ay, bx, by, dx=0.0, dy=0.0, L=L_PEN):
    """End pose (x, y, theta) of the axle centre from the two pen marks A (left) and B (right)."""
    theta = math.degrees(math.atan2(by - ay, bx - ax)) + 90.0       # heading, CCW from +x
    x = (ax + bx) / 2 + L * math.cos(math.radians(theta)) + dx
    y = (ay + by) / 2 + L * math.sin(math.radians(theta)) + dy
    return x, y, theta


# --- commanded motion for ideal motors (Assignment 2) ---
def computed_pose(m, b=AXLE_B):
    vl, vr = (p / 100 * MAX_SPEED * math.pi / 180 * WHEEL_D / 2 for p in SPEED[m]) # cm/s
    v, w = (vl + vr) / 2, (vr - vl) / b
    if abs(w) < 1e-12:
        return 0.0, v * T_RUN, 90.0, v, w
    phi = w * T_RUN
    R = v / w
    return -R * (1 - math.cos(phi)), R * math.sin(phi), 90 + math.degrees(phi), v, w


# --- chi-square goodness-of-fit test with equal-probability bins (Assignment 3) ---
def chi2_normal(x, k=None):
    x = np.asarray(x, float)
    n = len(x)
    k = k or max(5, min(10, n // 10))
    mu, sd = x.mean(), x.std(ddof=1)
    edges = stats.norm.ppf(np.linspace(0, 1, k + 1), mu, sd)
    obs = np.histogram(x, bins=edges)[0]
    exp = np.full(k, n / k)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    dof = k - 3
    return chi2, dof, float(stats.chi2.sf(chi2, dof)), k


# ---------- Chebyshev 2-sigma outlier removal (combined data, per motion) ----------
keep = df.valid.copy()
removed = []
cheb = {}
for m in MOTIONS:
    d = df[(df.motion == m) & df.valid]
    cheb[m] = {}
    for v, _ in VARS:
        mu, sd = d[v].mean(), d[v].std(ddof=1)
        out = (d[v] - mu).abs() > K_CHEB * sd
        cheb[m][v] = {"mean": mu, "sd": sd, "low": mu - K_CHEB * sd, "high": mu + K_CHEB * sd,
                      "n_out": int(out.sum())}
        for i in d.index[out]:
            removed.append({"team": df.at[i, "team"], "motion": m, "run": int(df.at[i, "run"]),
                            "variable": v, "value": float(df.at[i, v])})
        keep &= ~df.index.isin(d.index[out])
df["kept"] = keep
clean = df[keep].copy()
R["chebyshev"] = cheb
R["removed_list"] = removed
R["removed_counts"] = (df[df.valid & ~keep].groupby(["motion", "team"]).size().unstack(fill_value=0)
                       .reindex(index=MOTIONS, columns=TEAMS, fill_value=0).to_dict("index"))
R["n_before"] = {m: int(((df.motion == m) & df.valid).sum()) for m in MOTIONS}
R["n_after"] = {m: int((clean.motion == m).sum()) for m in MOTIONS}
