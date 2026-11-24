"""
SEE WS26 - Task 3, Assignment 6: statistical evaluation of the KUKA youBot experiment
Team 2 (Pen_sqr): Bhoumik Meher, Anmol Narain, Vishv Ravel

This script does everything that was used for the Assignment 6 report. It uses the combined
data of all five teams written by assignment5_preprocessing.py:
    output_a5/csv/combined_object_poses_optitrack.csv     (1135 final object poses)
    output_a5/csv/combined_endeffector_poses_youbot.csv   (1131 final end-effector poses)

What it computes (equation numbers as in the A6 report)
  - accuracy  = distance of the mean from the ground truth                 (Eq. 1)
  - precision = variance / standard deviation of the runs                  (Eq. 2)
  - Gaussian?   Shapiro-Wilk test, H0: data is Gaussian, alpha = 0.05
  - PCA on (x, y) (not on theta) before the variance tests                 (Eq. 3)
  - precision differences: Levene (and Bartlett) tests, also pair-wise     (Eq. 4)
  - accuracy differences: Kruskal-Wallis on the error of each run
  - which factor matters more: two-way ANOVA, effect size eta^2           (Eq. 5)
  - Table 5 of the manual (mean, variance, accuracy, test statistic, p value, H0)
  - the four figures of the report

How to run
  pip install numpy pandas scipy statsmodels matplotlib
  python assignment6_statistics.py
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# ============================================================== settings
A5_CSV = './output_a5/csv'          # output of assignment5_preprocessing.py
OUT_DIR = './output_a6'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
ALPHA = 0.05
SIZES = ['large', 'medium', 'small']
DIRS = ['left', 'right', 'straight']   # order of Table 5

# ground truth from the manual (Tables 1 and 2), world frame, metres / radians
GT_OBJ = {'straight': (0.150, -0.212, -1.65), 'left': (0.356, -0.283, -2.08), 'right': (-0.064, -0.352, -1.13)}
GT_EE = {'straight': (0.167, -0.317, -1.62), 'left': (0.331, -0.380, -2.11), 'right': (-0.01, -0.433, -1.11)}


# ============================================================== functions we wrote
def pca_xy(g):
    """PCA of the (x, y) points of one group (Eq. 3).
    Returns the PC scores (n x 2), the eigenvalues (variance along PC1, PC2) and the angle of PC1."""
    X = g[['x_mm', 'y_mm']].values
    Xc = X - X.mean(0)                        # shift to the group mean
    vals, vecs = np.linalg.eigh(np.cov(Xc.T))  # eigenvalues in ascending order
    vals, vecs = vals[::-1], vecs[:, ::-1]    # PC1 first
    scores = Xc @ vecs                        # PC1 and PC2 scores
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return scores, vals, (ang + 90) % 180 - 90   # axis direction in -90..90 deg


def table5_row(v, ground_truth):
    """Mean, variance, accuracy (Eq. 1, 2) and Shapiro-Wilk test of one variable."""
    W, p = stats.shapiro(v)
    return dict(n=len(v), mean=v.mean(), variance=v.var(ddof=1), sd=v.std(ddof=1),
                ground_truth=ground_truth, accuracy=abs(v.mean() - ground_truth),
                bias=v.mean() - ground_truth, W=W, p=p,
                h0='Reject' if p < ALPHA else 'Fail to Reject')


def var_tests(groups):
    """Bartlett and Levene (median-centred) tests for equal variances (Eq. 4)."""
    b = stats.bartlett(*groups)
    l = stats.levene(*groups, center='median')
    return b.statistic, b.pvalue, l.statistic, l.pvalue


def eta2(formula, data):
    """Two-way ANOVA and the share of the variation explained by each factor (Eq. 5)."""
    a = anova_lm(smf.ols(formula, data=data).fit(), typ=2)
    a['eta2'] = a['sum_sq'] / a['sum_sq'].sum()
    return a


# ============================================================== load the data
def load():
    o = pd.read_csv(os.path.join(A5_CSV, 'combined_object_poses_optitrack.csv'))
    o = o[['team', 'object', 'direction', 'run', 'x_m', 'y_m', 'theta_rad']].copy()
    o['x_mm'] = o.x_m * 1000
    o['y_mm'] = o.y_m * 1000
    o['th_deg'] = np.degrees(o.theta_rad)
    o['gx'] = o.direction.map(lambda d: GT_OBJ[d][0] * 1000)
    o['gy'] = o.direction.map(lambda d: GT_OBJ[d][1] * 1000)
    o['gth'] = o.direction.map(lambda d: np.degrees(GT_OBJ[d][2]))
    o['err_xy'] = np.hypot(o.x_mm - o.gx, o.y_mm - o.gy)   # position error of one run (mm)
    o['err_th'] = (o.th_deg - o.gth).abs()                  # orientation error of one run (deg)
    b = pd.read_csv(os.path.join(A5_CSV, 'combined_endeffector_poses_youbot.csv'))
    return o, b


# ============================================================== analysis
def analyse(o, b):
    R = {}

    # ---- PCA per object/direction combination; PC scores for every run
    o['pc1'] = np.nan
    o['pc2'] = np.nan
    rows = []
    for (s, d), g in o.groupby(['object', 'direction']):
        sc, vals, ang = pca_xy(g)
        o.loc[g.index, 'pc1'] = sc[:, 0]
        o.loc[g.index, 'pc2'] = sc[:, 1]
        rows.append(dict(object=s, direction=d, n=len(g), var_pc1=vals[0], var_pc2=vals[1],
                         sd_pc1=np.sqrt(vals[0]), sd_pc2=np.sqrt(vals[1]), angle_pc1_deg=ang,
                         explained_pc1=vals[0] / vals.sum()))
    R['pca'] = pd.DataFrame(rows)

    # ---- Table 5 of the manual
    rows = []
    for s in SIZES:
        for d in DIRS:
            g = o[(o.object == s) & (o.direction == d)]
            for var, col, gt in [('X', 'x_mm', 'gx'), ('Y', 'y_mm', 'gy'), ('Theta', 'th_deg', 'gth')]:
                rows.append(dict(size=s, direction=d, variable=var, **table5_row(g[col], g[gt].iloc[0])))
    R['table5'] = pd.DataFrame(rows)

    # ---- accuracy and precision per combination (short table in the report)
    rows = []
    for (s, d), g in o.groupby(['object', 'direction']):
        pr = R['pca'][(R['pca'].object == s) & (R['pca'].direction == d)].iloc[0]
        rows.append(dict(object=s, direction=d, n=len(g),
                         acc_xy_mm=np.hypot(g.x_mm.mean() - g.gx.iloc[0], g.y_mm.mean() - g.gy.iloc[0]),
                         acc_th_deg=abs(g.th_deg.mean() - g.gth.iloc[0]),
                         sd_pc1_mm=pr.sd_pc1, sd_pc2_mm=pr.sd_pc2, sd_th_deg=g.th_deg.std(ddof=1)))
    R['accuracy_precision'] = pd.DataFrame(rows)

    # ---- Gaussian: PC scores and single team sessions (for the discussion)
    pc_tests = [stats.shapiro(g[c]).pvalue for _, g in o.groupby(['object', 'direction']) for c in ['pc1', 'pc2']]
    sess_tests = [stats.shapiro(g[c]).pvalue for _, g in o.groupby(['team', 'object', 'direction'])
                  for c in ['x_mm', 'y_mm', 'th_deg']]

    # ---- precision: Levene / Bartlett, effect of the object (pose fixed) and of the pose (object fixed)
    rows = []
    for d in DIRS:
        for c, lab in [('pc1', 'PC1'), ('pc2', 'PC2'), ('th_deg', 'Theta')]:
            B, pB, L, pL = var_tests([o[(o.direction == d) & (o.object == s)][c].values for s in SIZES])
            rows.append(dict(factor='object', fixed=d, variable=lab, bartlett=B, p_bartlett=pB, levene=L, p_levene=pL))
    for s in SIZES:
        for c, lab in [('pc1', 'PC1'), ('pc2', 'PC2'), ('th_deg', 'Theta')]:
            B, pB, L, pL = var_tests([o[(o.object == s) & (o.direction == d)][c].values for d in DIRS])
            rows.append(dict(factor='direction', fixed=s, variable=lab, bartlett=B, p_bartlett=pB, levene=L, p_levene=pL))
    R['variance_tests'] = pd.DataFrame(rows)

    # ---- pair-wise Levene tests
    rows = []
    for d in DIRS:
        for a, c2 in itertools.combinations(SIZES, 2):
            r = dict(factor='object', fixed=d, pair=f'{a}-{c2}')
            for c, lab in [('pc1', 'PC1'), ('pc2', 'PC2'), ('th_deg', 'Theta')]:
                r[lab] = stats.levene(o[(o.direction == d) & (o.object == a)][c],
                                      o[(o.direction == d) & (o.object == c2)][c], center='median').pvalue
            rows.append(r)
    for s in SIZES:
        for a, c2 in itertools.combinations(DIRS, 2):
            r = dict(factor='direction', fixed=s, pair=f'{a}-{c2}')
            for c, lab in [('pc1', 'PC1'), ('pc2', 'PC2'), ('th_deg', 'Theta')]:
                r[lab] = stats.levene(o[(o.object == s) & (o.direction == a)][c],
                                      o[(o.object == s) & (o.direction == c2)][c], center='median').pvalue
            rows.append(r)
    R['pairwise_levene'] = pd.DataFrame(rows)

    # ---- accuracy: Kruskal-Wallis on the error of each run
    rows = []
    for d in DIRS:
        for c, lab in [('err_xy', 'position error'), ('err_th', 'orientation error')]:
            H, p = stats.kruskal(*[o[(o.direction == d) & (o.object == s)][c] for s in SIZES])
            rows.append(dict(factor='object', fixed=d, variable=lab, H=H, p=p))
    for s in SIZES:
        for c, lab in [('err_xy', 'position error'), ('err_th', 'orientation error')]:
            H, p = stats.kruskal(*[o[(o.object == s) & (o.direction == d)][c] for d in DIRS])
            rows.append(dict(factor='direction', fixed=s, variable=lab, H=H, p=p))
    R['accuracy_tests'] = pd.DataFrame(rows)

    # ---- which factor matters more: two-way ANOVA, eta^2
    # precision measure: absolute deviation of each run from its group median (what Levene compares)
    for c in ['pc1', 'pc2', 'th_deg']:
        o[f'absdev_{c}'] = (o[c] - o.groupby(['object', 'direction'])[c].transform('median')).abs()
    eff = {}
    for lab, col in [('accuracy: position error', 'err_xy'), ('accuracy: orientation error', 'err_th'),
                     ('precision: |dev| PC1', 'absdev_pc1'), ('precision: |dev| PC2', 'absdev_pc2'),
                     ('precision: |dev| theta', 'absdev_th_deg')]:
        a = eta2(f'{col} ~ C(object) * C(direction)', o)
        eff[lab] = {k: float(a.loc[k, 'eta2']) for k in a.index}
    eff_team = {}
    for lab, col in [('accuracy: position error', 'err_xy'), ('precision: |dev| PC1', 'absdev_pc1')]:
        a = eta2(f'{col} ~ C(object) + C(direction) + C(team)', o)
        eff_team[lab] = {k: float(a.loc[k, 'eta2']) for k in a.index}

    # ---- end effector (internal data)
    rows = []
    for d in DIRS:
        g = b[b.direction == d]
        rows.append(dict(direction=d, n=len(g), sd_x_mm=g.x_m.std() * 1000, sd_y_mm=g.y_m.std() * 1000,
                         sd_th_deg=np.degrees(g.theta_rad.std()),
                         acc_xy_mm=np.hypot(g.x_m.mean() - GT_EE[d][0], g.y_m.mean() - GT_EE[d][1]) * 1000,
                         acc_th_deg=abs(np.degrees(g.theta_rad.mean() - GT_EE[d][2]))))
    R['endeffector'] = pd.DataFrame(rows)

    R['summary'] = dict(
        n_runs=int(len(o)),
        table5_rejected=int((R['table5'].h0 == 'Reject').sum()), table5_rows=int(len(R['table5'])),
        pc_rejected=int(sum(p < ALPHA for p in pc_tests)), pc_tests=len(pc_tests),
        single_session_gaussian=int(sum(p >= ALPHA for p in sess_tests)), single_session_tests=len(sess_tests),
        within_session_sd=(o.groupby(['team', 'object', 'direction'])[['x_mm', 'y_mm', 'th_deg']].std().mean()).round(2).to_dict(),
        eta2=eff, eta2_with_team=eff_team)
    return R


# ============================================================== figures
def make_figures(o, R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    acc = R['accuracy_precision']
    C = {'large': '#2b6cb0', 'medium': '#d69e2e', 'small': '#b83280'}
    INK, MUTED, GRID = '#1f2933', '#616e7c', '#e4e7eb'
    NAVY, GREY = '#1F3A5F', '#9AA5B1'
    D3 = ['left', 'straight', 'right']
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
        'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.fontsize': 8, 'legend.frameon': False,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'axes.titleweight': 'bold'})

    # Figure 1: accuracy and precision
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, col, title in [(axs[0], 'acc_xy_mm', 'Accuracy (distance from ground truth)'),
                           (axs[1], 'sd_pc1_mm', 'Precision (spread along PC1)')]:
        w = 0.26
        for i, s in enumerate(SIZES):
            vals = [acc[(acc.object == s) & (acc.direction == d)][col].iloc[0] for d in D3]
            ax.bar(np.arange(3) + (i - 1) * w, vals, width=w - 0.03, color=C[s], label=s, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(D3); ax.grid(axis='x', visible=False)
        ax.set_title(title, loc='left'); ax.set_ylabel('mm')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.99, -0.04, 'lower is better', ha='right', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f'{FIG_DIR}/acc_prec.pdf'); plt.close(fig)

    # Figure 2: Gaussian or not (large object, left pose)
    g = o[(o.object == 'large') & (o.direction == 'left')]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, v, col, title in [(axs[0], g.y_mm.values, C['large'], 'All 5 teams together: two humps'),
                              (axs[1], g[g.team == 'Team 2'].y_mm.values, C['medium'], 'Only our team: one bell shape')]:
        ax.hist(v, bins=18 if len(v) < 40 else 30, density=True, color=col, alpha=0.85, ec='white', lw=0.5, zorder=3)
        xx = np.linspace(v.min() - 1, v.max() + 1, 200)
        ax.plot(xx, stats.norm.pdf(xx, v.mean(), v.std(ddof=1)), color=INK, lw=1.4)
        W, p = stats.shapiro(v)
        ax.text(0.97, 0.95, f'Shapiro-Wilk p {"< 0.001" if p < 0.001 else "= %.2f" % p}\n'
                f'{"not Gaussian" if p < ALPHA else "Gaussian"}', transform=ax.transAxes, ha='right', va='top', fontsize=8)
        ax.set_title(title, loc='left'); ax.set_xlabel('final y position (mm)'); ax.set_yticks([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    axs[0].set_ylabel('how often')
    fig.tight_layout(); fig.savefig(f'{FIG_DIR}/gaussian.pdf'); plt.close(fig)

    # Figure 3: PCA example (large object, left pose)
    X = g[['x_mm', 'y_mm']].values; Xc = X - X.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(Xc.T)); vals, vecs = vals[::-1], vecs[:, ::-1]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(Xc[:, 0], Xc[:, 1], s=9, color=C['large'], alpha=0.6, lw=0, zorder=2)
    for k, name, col in [(0, 'PC1 (largest spread)', NAVY), (1, 'PC2', '#d69e2e')]:
        vec = vecs[:, k] * 2 * np.sqrt(vals[k])
        ax.annotate('', xy=vec, xytext=-vec, arrowprops=dict(arrowstyle='<->', color=col, lw=2), zorder=4)
        end = vec if vec[0] > 0 else -vec
        pos = end * 1.1 if k == 0 else end * 1.25
        ax.text(pos[0] + 0.8, pos[1], name, color=col, fontsize=8, fontweight='bold', va='center')
    ax.set_xlim(-18, 18); ax.set_ylim(-18, 18); ax.set_aspect('equal')
    ax.set_xlabel('x − mean (mm)'); ax.set_ylabel('y − mean (mm)')
    ax.set_title('PCA example: large object, left', loc='left')
    fig.tight_layout(); fig.savefig(f'{FIG_DIR}/pca.pdf'); plt.close(fig)

    # Figure 4: which factor matters more
    eff = R['summary']['eta2']
    labs = [('accuracy: position error', 'Accuracy\n(position)'), ('accuracy: orientation error', 'Accuracy\n(angle)'),
            ('precision: |dev| PC1', 'Precision\n(PC1)'), ('precision: |dev| PC2', 'Precision\n(PC2)'),
            ('precision: |dev| theta', 'Precision\n(angle)')]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    w = 0.36
    for j, (key, col, name) in enumerate([('C(object)', NAVY, 'object'), ('C(direction)', GREY, 'placing pose')]):
        v = [eff[k][key] * 100 for k, _ in labs]
        xs = np.arange(len(labs)) + (j - 0.5) * w
        ax.bar(xs, v, width=w - 0.04, color=col, label=name, zorder=3)
        for xx, vv in zip(xs, v):
            ax.text(xx, vv + 0.8, f'{vv:.1f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([n for _, n in labs])
    ax.set_ylabel('effect size η² (%)'); ax.grid(axis='x', visible=False); ax.set_ylim(0, 55)
    ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(f'{FIG_DIR}/effects.pdf'); plt.close(fig)


# ============================================================== main
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    o, b = load()
    R = analyse(o, b)
    for name in ['table5', 'accuracy_precision', 'pca', 'variance_tests', 'pairwise_levene',
                 'accuracy_tests', 'endeffector']:
        R[name].to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    json.dump(R['summary'], open(os.path.join(OUT_DIR, 'summary.json'), 'w'), indent=1)
    make_figures(o, R)

    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)
    print('=== Table 5 (x, y in mm, theta in deg) ===')
    print(R['table5'][['size', 'direction', 'variable', 'mean', 'variance', 'accuracy', 'W', 'p', 'h0']].round(3).to_string(index=False))
    print('\n=== Accuracy and precision per combination ===')
    print(R['accuracy_precision'].round(1).to_string(index=False))
    print('\n=== Levene / Bartlett p values ===')
    print(R['variance_tests'][['factor', 'fixed', 'variable', 'p_bartlett', 'p_levene']].round(4).to_string(index=False))
    print('\n=== Kruskal-Wallis p values ===')
    print(R['accuracy_tests'][['factor', 'fixed', 'variable', 'p']].round(4).to_string(index=False))
    print('\n=== Summary ===')
    print(json.dumps(R['summary'], indent=1))
    print(f'\nDone. Tables in {OUT_DIR}, figures in {FIG_DIR}')
