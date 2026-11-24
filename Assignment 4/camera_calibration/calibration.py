import os
import numpy as np
import cv2
import glob

CHECKERBOARD = (7, 9)    # inner corners
square_size = 0.020      # metres, measure your printout!
USE_FIXED_MODEL = True   # False = free 5-parameter model

# Create a folder to save the images with rainbow corners drawn
os.makedirs('detected_corners', exist_ok=True)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints, imgpoints, used = [], [], []
images = sorted(glob.glob('images/*.jpg') + glob.glob('images/*.png'))
print(f"-> Found {len(images)} images.")

size = None
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if size is None:
        size = gray.shape[::-1]
    elif gray.shape[::-1] != size:
        print(f"[SKIP] {fname}: size {gray.shape[::-1]} != {size}")
        continue

    ok, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        print(f"[FAILED] {fname}")
        continue

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners)
    used.append(fname)

    # --- DRAW RAINBOW CORNERS AND SAVE ---
    vis_img = img.copy()
    cv2.drawChessboardCorners(vis_img, CHECKERBOARD, corners, ok)
    out_path = os.path.join('detected_corners', os.path.basename(fname))
    cv2.imwrite(out_path, vis_img)

print(f"-> Using {len(objpoints)} images, size {size}")
print("-> Saved all rainbow corner images in 'detected_corners/' folder.")
if len(objpoints) < 8:
    raise SystemExit("Too few valid images.")

flags = (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST) if USE_FIXED_MODEL else 0
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, size, None, None, flags=flags)

print(f"\nRMS reprojection error: {ret:.4f} px")
print("K:\n", mtx)
print("dist [k1 k2 p1 p2 k3]:\n", dist)

print("\nPer-image RMS:")
for i, f in enumerate(used):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    d = imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2)
    print(f"  {f}: {np.sqrt(np.mean(np.sum(d**2, axis=1))):.3f} px")

np.savez('calib_result.npz', mtx=mtx, dist=dist, rms=ret)

# undistortion for the report (raw vs undistorted)
img = cv2.imread(used[0])
und = cv2.undistort(img, mtx, dist)
cv2.imwrite('before_after.png', np.hstack([img, und]))

# Also save a side-by-side of Raw vs Rainbow Corners for the first image
first_rainbow = img.copy()
cv2.drawChessboardCorners(first_rainbow, CHECKERBOARD, imgpoints[0], True)
cv2.imwrite('before_after_corners.png', np.hstack([img, first_rainbow]))

print("\nSaved calib_result.npz, before_after.png, and before_after_corners.png")