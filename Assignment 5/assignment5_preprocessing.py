"""
SEE WS26 - Task 3, Assignment 5: KUKA youBot pick-and-place experiment
Team 2 (Pen_sqr): Bhoumik Meher, Anmol Narain, Vishv Ravel

This script does everything that was used for the Assignment 5 report:
  1. reads the OptiTrack (Motive CSV) takes and the youBot rosbags of all 5 teams
  2. finds the pick and final window of every take and removes outliers
  3. converts everything into the world frame of the ground truth
  4. writes the preprocessed CSV files (Table 3 layout + combined files)
  5. makes the figures of the report
  6. prints the numbers used in the report text and tables

How to run
  pip install numpy pandas matplotlib rosbags
  1. Unzip all team archives into one folder and set DATA_DIR below.
     Expected layout (the folder names of the archives we received):
        DATA_DIR/origin-small.csv                         (Team 1 origin take)
        DATA_DIR/notbags/<direction>/<size>/*.csv         (Team 1 OptiTrack)
        DATA_DIR/bags/<sizedirection>/*.bag               (Team 1 rosbags)
        DATA_DIR/Team2_task3/...                          (Team 2)
        DATA_DIR/Team_3_task3/...                         (Team 3)
        DATA_DIR/Team_4_task3/SEE FINALLLL/...            (Team 4)
        DATA_DIR/Team_5/Team_5/...                        (Team 5)
  2. python assignment5_preprocessing.py

Units: OptiTrack exports are in metres (header "Length Units, Meters"), rosbags in metres
and radians. All output CSVs are in metres and radians.
"""
import csv
import warnings
import glob
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean of frames where all markers are missing

# ============================================================== settings
DATA_DIR = './data'          # folder with all unzipped team archives
OUT_DIR = './output_a5'      # all results go here
FIG_DIR = os.path.join(OUT_DIR, 'figures')

WIN = 30                     # static window: 30 frames = 0.25 s at 120 Hz
STATIC_TOL = 1e-3            # object counts as still if x and y stay within 1 mm in the window
FROZEN_MIN = 10              # >= 10 identical rigid-body frames in a row = Motive lost the object

# one origin take per team (a single marker on the world origin)
ORIGIN_FILES = {
    'Team 1': 'origin-small.csv',
    'Team 2': 'Team2_task3/Team2_task3/Team_2optitrack/team2_Origin.csv',
    'Team 3': 'Team_3_task3/Team_3_CSV_file/Origin_Team_3.csv',
    'Team 4': 'Team_4_task3/SEE FINALLLL/Team_4 csv readings/origin-team4.csv',
    'Team 5': 'Team_5/Team_5/Origin_Team_5.csv',
}

# ground truth from the manual (Tables 1 and 2), world frame, metres / radians
GT_OBJ = {'pick': (0.143, -0.351, -1.65), 'straight': (0.150, -0.212, -1.65),
          'left': (0.356, -0.283, -2.08), 'right': (-0.064, -0.352, -1.13)}
GT_EE = {'pick': (0.152, -0.457, -1.65), 'straight': (0.167, -0.317, -1.62),
         'left': (0.331, -0.380, -2.11), 'right': (-0.01, -0.433, -1.11)}


# ============================================================== 1. finding the files
def norm_obj(s):
    """Folder name -> 'large' / 'medium' / 'small'."""
    s = s.lower()
    for o in ('large', 'medium', 'small'):
        if o in s:
            return o
    return {'l': 'large', 'm': 'medium', 's': 'small'}.get(s[0])


def norm_dir(s):
    """Folder name -> 'left' / 'right' / 'straight' (some folders have typos)."""
    s = s.lower()
    if 'left' in s:
        return 'left'
    if 'right' in s:
        return 'right'
    if 'straight' in s or 'straigth' in s or 'stright' in s:
        return 'straight'
    return None


def optitrack_files():
    """List of (team, object, direction, path) for every OptiTrack take."""
    B = DATA_DIR
    out = []
    for f in glob.glob(f'{B}/notbags/*/*/*.csv'):                                  # Team 1
        d, o = f.split('/')[-3:-1]
        out.append(('Team 1', norm_obj(o), norm_dir(d), f))
    for f in glob.glob(f'{B}/Team2_task3/Team2_task3/Team_2optitrack/*/*/*.csv'):  # Team 2
        o, d = f.split('/')[-3:-1]
        out.append(('Team 2', norm_obj(o), norm_dir(d), f))
    for f in glob.glob(f'{B}/Team_3_task3/Team_3_CSV_file/*/*/*.csv'):             # Team 3
        d, o = f.split('/')[-3:-1]
        out.append(('Team 3', norm_obj(o), norm_dir(d), f))
    for f in glob.glob(f'{B}/Team_4_task3/SEE FINALLLL/Team_4 csv readings/*/*/*.csv'):  # Team 4
        o, d = f.split('/')[-3:-1]
        out.append(('Team 4', norm_obj(o), norm_dir(d), f))
    t5 = {'ll': 'left', 'lr': 'right', 'ls': 'straight', 'ml': 'left', 'mr': 'right', 'ms': 'straight',
          'sl': 'left', 'sr': 'right', 'small straight': 'straight'}
    for f in glob.glob(f'{B}/Team_5/Team_5/*/*/*.csv'):                            # Team 5
        o, d = f.split('/')[-3:-1]
        out.append(('Team 5', norm_obj(o), t5[d], f))
    return sorted(out)


def bag_files():
    """List of (team, object, direction, path) for every rosbag. .bag.active files are skipped."""
    B = DATA_DIR
    out = []
    for f in glob.glob(f'{B}/bags/*/*.bag'):                                       # Team 1
        d = f.split('/')[-2]
        if d == 'group1_SS26':       # old data from a previous semester, not used
            continue
        out.append(('Team 1', norm_obj({'straightmed': 'medium', 'rightsmall': 'small'}.get(d, d)), norm_dir(d), f))
    for f in glob.glob(f'{B}/Team2_task3/Team2_task3/WS26team2/*/*/*.bag'):        # Team 2
        o, d = f.split('/')[-3:-1]
        out.append(('Team 2', norm_obj(o), norm_dir(d), f))
    for f in glob.glob(f'{B}/Team_3_task3/Team_3_bagfile/*/*/*.bag'):              # Team 3
        d, o = f.split('/')[-3:-1]
        out.append(('Team 3', norm_obj(o), norm_dir(d), f))
    for f in glob.glob(f'{B}/Team_4_task3/SEE FINALLLL/SEEWS26, TEAM-4 rosbagfiles/*/*/*.bag'):  # Team 4
        o, d = f.split('/')[-3:-1]
        out.append(('Team 4', norm_obj(o), norm_dir(d), f))
    for f in glob.glob(f'{B}/Team_5/Team_5/SEE_Team5WS26/*/*.bag'):                # Team 5
        d = f.split('/')[-2]
        out.append(('Team 5', norm_obj(d), norm_dir(d), f))
    return sorted(out)


# ============================================================== 2. reading Motive CSVs
def read_take(path):
    """Read a Motive CSV export. Returns time, rigid-body rotation/position and all markers."""
    rows = list(csv.reader(open(path, encoding='utf-8-sig')))
    meta = rows[0]
    md = {meta[i]: meta[i + 1] for i in range(0, len(meta) - 1, 2)}
    typ, name, kind, axis = rows[2], rows[3], rows[6], rows[7]
    data = rows[8:]

    def col(i):
        out = np.full(len(data), np.nan)
        for j, r in enumerate(data):
            if i < len(r) and r[i] != '':
                out[j] = float(r[i])
        return out

    res = {'meta': md, 't': col(1)}
    rb = [i for i, ty in enumerate(typ) if ty == 'Rigid Body']
    if rb:
        res['rb_name'] = name[rb[0]]
        idx = {(kind[i], axis[i]): i for i in rb}
        res['q'] = np.stack([col(idx[('Rotation', a)]) for a in 'XYZW'], 1)
        res['p'] = np.stack([col(idx[('Position', a)]) for a in 'XYZ'], 1)
    markers = {}
    for i in [i for i, ty in enumerate(typ) if ty == 'Marker']:
        markers.setdefault(name[i], {})[axis[i]] = i
    res['markers'] = {n: np.stack([col(d[a]) for a in 'XYZ'], 1) for n, d in markers.items()}
    return res


def yaw_from_quat(q):
    """Heading of the rigid body's local X axis in the table plane (world x = Motive X, world y = Motive Z)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx = 1 - 2 * (y * y + z * z)
    vz = 2 * (x * z - w * y)
    return np.arctan2(vz, vx)


def origin(team):
    """Mean position of the single marker in the team's origin take."""
    r = read_take(os.path.join(DATA_DIR, ORIGIN_FILES[team]))
    ms = [m for n, m in r['markers'].items() if not np.all(np.isnan(m))]
    return np.nanmean(ms[0], 0)


# ============================================================== 3. small helpers
def circmean(a):
    """Circular mean of angles (Eq. 5 of the A5 report)."""
    a = a[~np.isnan(a)]
    return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))) if len(a) else np.nan


def wrap(a):
    """Angle into [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def fold90(a):
    """Angle into [-45, 45) deg: the marker sets are 4-fold symmetric, so Motive can flip by 90 deg."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def blocks(mask):
    """Start/stop indices of every run of True values."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def robust_mean(xy):
    """Outlier rule (Eq. 3 of the A5 report): reject frames with
    d > max(median(d) + 3 * 1.4826 * MAD(d), 0.5 mm), then average the rest."""
    med = np.median(xy, 0)
    d = np.linalg.norm(xy - med, axis=1)
    thr = max(np.median(d) + 3 * 1.4826 * np.median(np.abs(d - np.median(d))), 5e-4)
    keep = d <= thr
    return xy[keep].mean(0), keep


# ============================================================== 4. one OptiTrack take
def process_take(f, O, h_ref=np.nan):
    """Pick and final pose of one take (positions in the world frame, Eq. 1 of the A5 report)."""
    r = read_take(f)
    q, n = r['q'], len(r['t'])
    lab = {k: m for k, m in r['markers'].items() if ':' in k}       # markers of the object
    unl = {k: m for k, m in r['markers'].items() if ':' not in k}   # unlabelled markers

    # frozen rigid body: Motive repeats the last pose when it loses the object
    rb_same = np.r_[False, np.all(np.diff(r['p'], axis=0) == 0, axis=1)]
    frozen = np.zeros(n, bool)
    for s, e in blocks(rb_same):
        if e - s >= FROZEN_MIN - 1:
            frozen[max(s - 1, 0):e] = True
    res = dict(n_frames=n, n_frozen=int(frozen.sum()))

    if lab:
        M = np.stack(list(lab.values()))                  # markers x frames x 3
        good = ~np.isnan(M).any((0, 2)) & ~frozen
        c = np.nanmean(M, 0)                              # marker centre
        heading = yaw_from_quat(q)
        res['source'] = 'labelled markers'
        res['n_markers'] = len(lab)
    else:
        # rigid body never solved (Team 1 medium object): use the unlabelled markers
        full = {k: m for k, m in unl.items() if (~np.isnan(m[:, 0])).mean() > 0.95}
        M = np.stack(list(full.values()))
        good = ~np.isnan(M).any((0, 2))
        c = M.mean(0)
        a, b = M[0], M[1]
        heading = np.arctan2(b[:, 2] - a[:, 2], b[:, 0] - a[:, 0])
        res['source'] = 'unlabelled markers'
        res['n_markers'] = len(full)
    res['n_bad'] = int((~good).sum())

    # Eq. 1: Motive -> world frame
    x = c[:, 0] - O[0]
    y = c[:, 2] - O[2]
    h = c[:, 1] - O[1]

    # Eq. 2: static frames = x and y range below 1 mm over 0.25 s
    xs, ys = np.nan_to_num(x), np.nan_to_num(y)
    ok = (np.ptp(swv(xs, WIN), 1) < STATIC_TOL) & (np.ptp(swv(ys, WIN), 1) < STATIC_TOL) & swv(good, WIN).all(1)
    static = np.zeros(n, bool)
    for i in np.where(ok)[0]:
        static[i:i + WIN] = True
    bl = [b for b in blocks(static) if b[1] - b[0] >= WIN]
    if not bl:
        res['status'] = 'no static segment'
        return res

    def pose(s, e):
        xy, keep = robust_mean(np.c_[x[s:e], y[s:e]])
        return xy, circmean(heading[s:e][keep]), int((~keep).sum()), e - s, float(h[s:e].mean())

    near_pick = lambda xy: np.hypot(xy[0] - 0.15, xy[1] + 0.35) < 0.05
    on_table = lambda p: np.isnan(h_ref) or abs(p[4] - h_ref) < 0.01

    # pick window: longest static block in the container, at resting height
    init = None
    for b in bl:
        p = pose(*b)
        if not near_pick(p[0]):
            break
        if on_table(p) and (init is None or p[3] > init[3]):
            init = p
    first = init if init is not None else pose(*bl[0])
    h_table = h_ref if not np.isnan(h_ref) else (first[4] if init is not None else np.nan)

    # final window: last static block on the table (same height as in the container)
    fin = None
    for b in reversed(bl):
        p = pose(*b)
        if near_pick(p[0]):
            continue
        if np.isnan(h_table) or abs(p[4] - h_table) < 0.01:
            fin = p
            break
    res['final_window'] = 'static block'
    if fin is None:
        # object still rocking at the end of the take: use the last 0.5 s
        idx = np.where(good)[0][-60:]
        p = pose(idx[0], idx[-1] + 1)
        if not near_pick(p[0]) and (np.isnan(h_table) or abs(p[4] - h_table) < 0.01):
            fin = p
            res['final_window'] = 'last 0.5 s (settling)'
    if fin is None:
        res['status'] = 'no final pose on the table'
        return res
    if init is not None:
        res.update(x0=init[0][0], y0=init[0][1], head0=init[1], out0=init[2], n0=init[3], h0=init[4])
    res.update(x1=fin[0][0], y1=fin[0][1], head1=fin[1], out1=fin[2], n1=fin[3], h1=fin[4],
               status='ok' if init is not None else 'final pose only')
    return res


# ============================================================== 5. one rosbag
def process_bag(f):
    """Final end-effector pose: average of the messages after the arm has stopped."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    vals = []
    with AnyReader([Path(f)], default_typestore=ts) as rd:
        cs = [c for c in rd.connections if c.topic == '/end_effector_pose']
        for c, t, raw in rd.messages(connections=cs):
            m = rd.deserialize(raw, c.msgtype)
            vals.append((m.x, m.y, m.theta))
    v = np.array(vals)
    if len(v) == 0:
        return dict(status='empty bag')
    last = v[-1]
    keep = (np.hypot(v[:, 0] - last[0], v[:, 1] - last[1]) <= 1e-3) & (np.abs(wrap(v[:, 2] - last[2])) <= 0.05)
    k = v[keep]
    return dict(status='ok', n_msgs=len(v), n_used=int(keep.sum()),
                x=k[:, 0].mean(), y=k[:, 1].mean(), theta=circmean(k[:, 2]))


# ============================================================== 6. pick-pose reference
def pick_reference(o):
    """Orientation (theta = -1.65 + delta_theta) and pick-pose correction (Eq. 4 of the A5 report):
    p_final = c_final + R(delta_theta) (p_pick_GT - c_pick)."""
    o = o.copy()
    key = ['team', 'object', 'direction', 'source']                  # one session
    pick = o.groupby(key)[['x0', 'y0']].transform('mean')
    o['dx_pick'] = pick.x0 - GT_OBJ['pick'][0]
    o['dy_pick'] = pick.y0 - GT_OBJ['pick'][1]

    # heading change between pick and final window; takes without a pick window use the
    # session mean (averaged on the 4x circle because of the 90 deg symmetry)
    def ref4(s):
        a = s.dropna().values
        return np.arctan2(np.sin(4 * a).mean(), np.cos(4 * a).mean()) / 4 if len(a) else np.nan
    head0 = o.head0.fillna(o.groupby(key).head0.transform(ref4))
    dth = fold90(wrap(o.head1 - head0))
    o['theta'] = GT_OBJ['pick'][2] + dth

    # alternative: session-mean pick reference (kept in the CSV for comparison)
    ox, oy = -o.dx_pick, -o.dy_pick
    o['x_session'] = o.x1 + np.cos(dth) * ox - np.sin(dth) * oy
    o['y_session'] = o.y1 + np.sin(dth) * ox + np.cos(dth) * oy

    # main result: every take referenced to its own pick pose
    px = GT_OBJ['pick'][0] - o.x0.fillna(pick.x0)
    py = GT_OBJ['pick'][1] - o.y0.fillna(pick.y0)
    o['x'] = o.x1 + np.cos(dth) * px - np.sin(dth) * py
    o['y'] = o.y1 + np.sin(dth) * px + np.cos(dth) * py
    o['pick_ref'] = np.where(o.x0.notna(), 'own pick pose', 'session mean pick pose')
    return o


# ============================================================== 7. time stamps (for the run numbers)
def take_time(f):
    h = next(csv.reader(open(f, encoding='utf-8-sig')))
    return datetime.strptime(h[h.index('Capture Start Time') + 1], '%Y-%m-%d %I.%M.%S.%f %p')


def bag_time(f):
    m = re.search(r'(\d{4})_(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d)', os.path.basename(f))
    return datetime(*map(int, m.groups()))


# ============================================================== 8. CSV output
def write_csvs(obj, b):
    obj_out = pd.DataFrame({
        'team': obj.team, 'object': obj.object, 'direction': obj.direction, 'run': obj.run,
        'x_m': obj.x.round(8), 'y_m': obj.y.round(8), 'theta_rad': obj.theta.round(8),
        'pick_reference': obj.pick_ref,
        'x_session_ref_m': obj.x_session.round(5), 'y_session_ref_m': obj.y_session.round(5),
        'x_origin_only_m': obj.x1.round(5), 'y_origin_only_m': obj.y1.round(5),
        'pick_x_m': obj.x0.round(5), 'pick_y_m': obj.y0.round(5),
        'pick_offset_x_m': obj.dx_pick.round(5), 'pick_offset_y_m': obj.dy_pick.round(5),
        'marker_source': obj.source, 'final_window': obj.final_window, 'status': obj.status,
        'frames_in_take': obj.n_frames, 'frozen_frames': obj.n_frozen,
        'final_window_frames': obj.n1, 'final_window_outliers': obj.out1,
        'pick_window_frames': obj.n0, 'pick_window_outliers': obj.out0,
        'capture_start': obj.t.dt.strftime('%Y-%m-%d %H:%M:%S'), 'file': obj.file})
    ee_out = pd.DataFrame({
        'team': b.team, 'object': b.object, 'direction': b.direction, 'run': b.run,
        'x_m': b.x.round(8), 'y_m': b.y.round(8), 'theta_rad': b.theta.round(8),
        'messages_in_bag': b.n_msgs, 'messages_averaged': b.n_used,
        'bag_start': b.t.dt.strftime('%Y-%m-%d %H:%M:%S'), 'file': b.file})
    D = os.path.join(OUT_DIR, 'csv')
    shutil.rmtree(D, ignore_errors=True)
    os.makedirs(f'{D}/table3_object_optitrack'); os.makedirs(f'{D}/table3_endeffector_youbot')
    obj_out.to_csv(f'{D}/combined_object_poses_optitrack.csv', index=False)
    ee_out.to_csv(f'{D}/combined_endeffector_poses_youbot.csv', index=False)
    names = {'run': 'Run', 'x_m': 'X axis (m)', 'y_m': 'Y axis (m)', 'theta_rad': 'Orientation (rad)'}
    for df, sub in [(obj_out, 'table3_object_optitrack'), (ee_out, 'table3_endeffector_youbot')]:
        for (t, ob, d), g in df.groupby(['team', 'object', 'direction']):
            fn = f"{t.replace(' ', '').lower()}_{ob}_{d}.csv"
            g[['run', 'x_m', 'y_m', 'theta_rad']].rename(columns=names).to_csv(f'{D}/{sub}/{fn}', index=False)


# ============================================================== 9. figures
def make_figures(obj, ee, origins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D
    os.makedirs(FIG_DIR, exist_ok=True)

    C = {'large': '#2b6cb0', 'medium': '#d69e2e', 'small': '#b83280'}
    MK = {'large': 'o', 'medium': 's', 'small': '^'}
    INK, MUTED, GRID = '#1f2933', '#616e7c', '#e4e7eb'
    DIRS = ['left', 'straight', 'right']; SIZES = ['large', 'medium', 'small']
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
        'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'legend.fontsize': 7.5, 'legend.frameon': False,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'axes.titleweight': 'bold'})

    def ellipse(ax, x, y, color, n_std=2.0):
        vals, vecs = np.linalg.eigh(np.cov(x, y))
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        w, h = 2 * n_std * np.sqrt(vals[::-1])
        ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang, fill=False, ec=color, lw=1.1))

    # Figure 3: every team separately
    TEAMS = ['Team 1', 'Team 2', 'Team 3', 'Team 4', 'Team 5']
    TCOL = dict(zip(TEAMS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))
    TM = {'Team 1': 'o', 'Team 2': 's', 'Team 3': '^', 'Team 4': 'D', 'Team 5': 'v'}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.1))
    for ax, d in zip(axs, DIRS):
        gd = obj[obj.direction == d]
        means = {}
        for tm in TEAMS:
            g = gd[gd.team == tm]
            ax.scatter(g.x * 100, g.y * 100, s=4, color=TCOL[tm], alpha=0.25, lw=0, zorder=2)
        for tm in TEAMS:
            g = gd[gd.team == tm]
            mx, my = g.x.mean() * 100, g.y.mean() * 100; means[tm] = (mx, my)
            ax.errorbar(mx, my, xerr=g.x.std() * 100, yerr=g.y.std() * 100, fmt='none',
                        ecolor=TCOL[tm], elinewidth=1.0, capsize=2.5, zorder=3)
            ax.scatter(mx, my, s=58, marker=TM[tm], color=TCOL[tm], ec='white', lw=0.9, zorder=4)
        gx, gy, _ = GT_OBJ[d]
        ax.scatter(gx * 100, gy * 100, marker='+', s=120, color=INK, lw=1.5, zorder=5)
        xs = [m[0] for m in means.values()] + [gx * 100]; ys = [m[1] for m in means.values()] + [gy * 100]
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 0.35
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_aspect('equal')
        ax.set_title(d, loc='left'); ax.set_xlabel('x (cm)')
    axs[0].set_ylabel('y (cm)')
    h = [Line2D([], [], marker=TM[t], ls='', color=TCOL[t], mec='white', ms=7, label=t) for t in TEAMS]
    h.append(Line2D([], [], marker='+', ls='', color=INK, ms=9, mew=1.5, label='ground truth'))
    fig.legend(handles=h, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.03), handletextpad=0.2, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f'{FIG_DIR}/teams.pdf'); plt.close(fig)

    # Figure 4: filtering example (one of our takes) + outliers per take
    ex = obj[(obj.team == 'Team 2') & (obj.object == 'medium') & (obj.direction == 'left')].iloc[3]
    r = read_take(os.path.join(DATA_DIR, ex.file))
    O = origins['Team 2']
    M = np.stack([m for k, m in r['markers'].items() if ':' in k]); c = M.mean(0)
    x = c[:, 0] - O[0]; y = c[:, 2] - O[2]; hh = c[:, 1] - O[1]; t = r['t']
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axs[0]
    ax.plot(t, x * 100, color=C['large'], lw=1.2, label='x')
    ax.plot(t, y * 100, color=C['medium'], lw=1.2, label='y')
    ax.plot(t, hh * 100, color=C['small'], lw=1.2, label='height')
    ok = (np.ptp(swv(x, WIN), 1) < STATIC_TOL) & (np.ptp(swv(y, WIN), 1) < STATIC_TOL)
    st = np.zeros(len(x), bool)
    for i in np.where(ok)[0]:
        st[i:i + WIN] = True
    bl = [b for b in blocks(st) if b[1] - b[0] >= WIN]
    pick_bl = max([b for b in bl if abs(hh[b[0]:b[1]].mean() - hh[0]) < 0.01 and abs(x[b[0]] - x[0]) < 0.01],
                  key=lambda b: b[1] - b[0])
    for (s, e), lab in [(pick_bl, 'pick window'), (bl[-1], 'final window')]:
        ax.axvspan(t[s], t[e - 1], color='#e4e7eb', alpha=0.9, lw=0)
        ax.text((t[s] + t[e - 1]) / 2, 37, lab, ha='center', va='bottom', fontsize=7, color=MUTED)
    ax.set_xlabel('time in take (s)'); ax.set_ylabel('position (cm)')
    ax.set_ylim(-40, 45); ax.legend(loc='lower right', ncol=3)
    ax.set_title('One take (Team 2, medium object, left)', loc='left')
    ax = axs[1]
    ax.hist(obj.out1.clip(upper=60), bins=np.arange(0, 62, 2), color=C['large'], ec='white', lw=0.6)
    ax.set_xlabel('frames rejected in final window'); ax.set_ylabel('takes')
    ax.set_title('Outliers per take (all teams)', loc='left')
    ax.text(0.97, 0.95, f"median {obj.out1.median():.0f} of {obj.n1.median():.0f} frames\nlast bin = 60 or more",
            transform=ax.transAxes, ha='right', va='top', fontsize=7, color=MUTED)
    fig.tight_layout(); fig.savefig(f'{FIG_DIR}/filtering.pdf'); plt.close(fig)

    # Figure 5: overview of all final poses
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for s in SIZES:
        g = obj[obj.object == s]
        ax.scatter(g.x * 100, g.y * 100, s=7, marker=MK[s], color=C[s], ec='none', alpha=0.7,
                   label=f'object, {s} (OptiTrack)', zorder=3)
    ax.scatter(ee.x * 100, ee.y * 100, s=55, marker='X', color=INK, ec='white', lw=0.6,
               label='end effector (youBot encoders)', zorder=5)
    for d in DIRS + ['pick']:
        gx, gy, _ = GT_OBJ[d]; ex_, ey_, _ = GT_EE[d]
        ax.scatter(gx * 100, gy * 100, marker='+', s=90, color=INK, lw=1.2, zorder=6,
                   label='object ground truth' if d == 'left' else None)
        ax.scatter(ex_ * 100, ey_ * 100, marker='D', s=26, facecolor='none', ec=INK, lw=0.9, zorder=6,
                   label='end-effector ground truth' if d == 'left' else None)
        ax.annotate(d, (gx * 100, gy * 100), xytext=(0, 12) if d != 'left' else (0, -16),
                    textcoords='offset points', ha='center', fontsize=7.5, color=INK)
    for d in DIRS:
        g = obj[obj.direction == d]; th = np.arctan2(np.sin(g.theta).mean(), np.cos(g.theta).mean())
        ax.annotate('', xy=(g.x.mean() * 100 + 4 * np.cos(th), g.y.mean() * 100 + 4 * np.sin(th)),
                    xytext=(g.x.mean() * 100, g.y.mean() * 100),
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.9), zorder=4)
    ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)'); ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), handletextpad=0.4)
    ax.set_title('All final poses in the world frame (5 teams combined)', loc='left')
    fig.tight_layout(); fig.savefig(f'{FIG_DIR}/overview.pdf'); plt.close(fig)

    # Figure 6: object positions per placing pose with 2-sigma ellipses
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9))
    for ax, d in zip(axs, DIRS):
        for s in SIZES:
            g = obj[(obj.direction == d) & (obj.object == s)]
            ax.scatter(g.x * 100, g.y * 100, s=9, marker=MK[s], color=C[s], ec='white', lw=0.25, alpha=0.85, label=s, zorder=3)
            ellipse(ax, g.x.values * 100, g.y.values * 100, C[s])
        gx, gy, _ = GT_OBJ[d]
        ax.scatter(gx * 100, gy * 100, marker='+', s=90, color=INK, lw=1.3, zorder=5, label='ground truth')
        ax.set_title(d, loc='left'); ax.set_xlabel('x (cm)')
        ax.set_aspect('equal', adjustable='datalim')
    axs[0].set_ylabel('y (cm)')
    axs[0].legend(loc='lower left', ncol=2, columnspacing=0.6, handletextpad=0.2)
    fig.tight_layout(); fig.savefig(f'{FIG_DIR}/object_by_direction.pdf'); plt.close(fig)

    # Figure 7: final object orientation
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, d in zip(axs, DIRS):
        rng = np.random.default_rng(1)
        for i, s in enumerate(SIZES):
            g = obj[(obj.direction == d) & (obj.object == s)].theta
            ax.scatter(np.full(len(g), i) + rng.uniform(-0.22, 0.22, len(g)), np.degrees(g), s=6,
                       marker=MK[s], color=C[s], lw=0, alpha=0.7)
            ax.plot([i - 0.32, i + 0.32], [np.degrees(g.mean())] * 2, color=INK, lw=1.4)
        ax.axhline(np.degrees(GT_OBJ[d][2]), color=INK, lw=0.9, ls='--')
        ax.set_xticks(range(3)); ax.set_xticklabels(SIZES); ax.set_xlim(-0.6, 2.6)
        ax.set_title(d, loc='left')
    axs[0].set_ylabel('final orientation θ (deg)')
    h = [Line2D([], [], color=INK, lw=1.4, label='mean of the runs'),
         Line2D([], [], color=INK, ls='--', lw=0.9, label='ground truth (manual Table 1)')]
    fig.legend(handles=h, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f'{FIG_DIR}/orientation.pdf'); plt.close(fig)


# ============================================================== 10. numbers for the report
def report_numbers(obj, b):
    k = ['team', 'object', 'direction']
    print('\n=== Table 2: data per team ===')
    t = pd.DataFrame({
        'takes': obj.groupby('team').size(), 'bags': b.groupby('team').size(),
        'SD_x_mm': (obj.groupby(k).x.std().groupby('team').mean() * 1000).round(1),
        'SD_y_mm': (obj.groupby(k).y.std().groupby('team').mean() * 1000).round(1),
        'outliers_per_take': obj.groupby('team').out1.mean().round(1)})
    print(t.to_string())

    print('\n=== Table 3: combined data (cm, mm, deg) ===')
    rows = []
    for d in ['left', 'straight', 'right']:
        for s in ['large', 'medium', 'small']:
            g = obj[(obj.direction == d) & (obj.object == s)]; e = b[(b.direction == d) & (b.object == s)]
            rows.append(dict(direction=d, object=s, n=len(g), x_cm=g.x.mean() * 100, y_cm=g.y.mean() * 100,
                             theta_deg=np.degrees(g.theta.mean()), SDx_mm=g.x.std() * 1000, SDy_mm=g.y.std() * 1000,
                             SDtheta_deg=np.degrees(g.theta.std()), ee_theta_deg=np.degrees(e.theta.mean()),
                             ee_SDtheta_deg=np.degrees(e.theta.std())))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print('\n=== Table 4: end effector, ground truth vs robot reading ===')
    for d in ['left', 'straight', 'right']:
        e = b[b.direction == d]
        print(f"{d:9s} reading ({e.x.mean():.3f}, {e.y.mean():.3f}, {e.theta.mean():.2f})  GT {GT_EE[d]}  "
              f"pos diff {np.hypot(e.x.mean() - GT_EE[d][0], e.y.mean() - GT_EE[d][1]) * 1000:.0f} mm  "
              f"angle diff {abs(np.degrees(e.theta.mean() - GT_EE[d][2])):.0f} deg")

    print('\n=== Filtering ===')
    print(f"takes: {len(obj)}, bags: {len(b)}")
    print(f"final window: median {obj.n1.median():.0f} frames, outliers median {obj.out1.median():.0f}, "
          f"mean {obj.out1.mean():.1f}, max {obj.out1.max():.0f}, takes without outliers {(obj.out1 == 0).mean() * 100:.0f}%")
    print(f"pick window: median {obj.n0.median():.0f} frames, outliers median {obj.out0.median():.0f}")
    print(f"takes with frozen frames: {(obj.n_frozen > 0).sum()}, fully frozen: {(obj.n_frozen == obj.n_frames).sum()}")
    print(f"final pose from the last 0.5 s (still rocking): {(obj.final_window != 'static block').sum()}")
    print(f"takes without pick window: {(obj.status != 'ok').sum()}")
    print(f"rosbags: median {b.n_msgs.median():.0f} messages, {b.n_used.median():.0f} averaged")
    print('within-session SD (mm):', (obj.groupby(k)[['x', 'y']].std().mean() * 1000).round(2).to_dict())

    print('\n=== Ground-truth check ===')
    for d in ['straight', 'left', 'right']:
        g = obj[obj.direction == d]
        print(f"{d:9s} rotation pick->place {np.degrees((g.theta - GT_OBJ['pick'][2]).mean()):6.1f} deg "
              f"(manual {np.degrees(GT_OBJ[d][2] - GT_OBJ['pick'][2]):6.1f}), "
              f"position error of the mean {np.hypot(g.x.mean() - GT_OBJ[d][0], g.y.mean() - GT_OBJ[d][1]) * 1000:.1f} mm")


# ============================================================== main
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    # origins of the five teams
    origins = {t: origin(t) for t in ORIGIN_FILES}
    json.dump({t: list(map(float, o)) for t, o in origins.items()},
              open(os.path.join(OUT_DIR, 'origins.json'), 'w'), indent=1)

    # OptiTrack: pass 1 finds the resting height of each object in the container,
    # pass 2 uses it to pick the right windows
    files = optitrack_files()
    print(f'{len(files)} OptiTrack takes')
    h_ref = {}
    for rnd in (1, 2):
        rows = []
        for team, ob, d, f in files:
            r = process_take(f, origins[team], h_ref.get((team, ob), np.nan))
            r.update(team=team, object=ob, direction=d, file=os.path.relpath(f, DATA_DIR), t=take_time(f))
            rows.append(r)
        st = pd.DataFrame(rows)
        h_ref = st.groupby(['team', 'object']).h0.median().to_dict()

    # rosbags
    bags = bag_files()
    print(f'{len(bags)} rosbags')
    rows = []
    for team, ob, d, f in bags:
        r = process_bag(f)
        r.update(team=team, object=ob, direction=d, file=os.path.relpath(f, DATA_DIR), t=bag_time(f))
        rows.append(r)
    b = pd.DataFrame(rows)

    # world frame + pick reference, run numbers in recording order
    obj = pick_reference(st)
    obj = obj.sort_values(['team', 'object', 'direction', 't'])
    obj['run'] = obj.groupby(['team', 'object', 'direction']).cumcount() + 1
    b = b.sort_values(['team', 'object', 'direction', 't'])
    b['run'] = b.groupby(['team', 'object', 'direction']).cumcount() + 1

    write_csvs(obj, b)
    make_figures(obj, b, origins)
    report_numbers(obj, b)
    print(f'\nDone. CSV files in {OUT_DIR}/csv, figures in {FIG_DIR}')
