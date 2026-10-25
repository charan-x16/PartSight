"""Optional LoFTR feature matching alignment backend.

Install ``torch`` and ``kornia`` to enable this experiment. The matcher only
estimates image correspondence; component presence and position are checked
separately by the inspection pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .detector import AlignmentResult


def align_loftr_ransac(
    photo_rgb: np.ndarray, cad_rgb: np.ndarray, *, max_side: int = 1280
) -> tuple[AlignmentResult, np.ndarray | None]:
    """Estimate photo-to-CAD homography from pretrained LoFTR matches + RANSAC.

    LoFTR dependencies are loaded lazily so the traditional pipeline remains
    usable without PyTorch. This function never treats a raw match set as a
    valid transform: the same inlier, reprojection, spread, and canvas checks
    used by the baseline are applied here.
    """
    import cv2

    diag: dict[str, Any] = {
        "method": "LoFTR",
        "keypoints_photo": 0,
        "keypoints_cad": 0,
        "correspondences": 0,
        "descriptors_photo_available": False,
        "descriptors_cad_available": False,
        "raw_matches": 0,
        "knn_candidates": 0,
        "filtered_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error": None,
        "homography_valid": False,
        "inlier_spread_ratio": None,
        "projected_area_ratio": None,
        "overlap_ratio": None,
        "match_points": (),
        "correlation_score": None,
    }

    def fail(reason: str) -> tuple[AlignmentResult, None]:
        result = AlignmentResult(False, None, reason, **diag)
        return result, None

    try:
        import torch
        import kornia.feature as KF
    except ImportError as exc:
        return fail(
            "LoFTR requires optional dependencies torch and kornia; install the "
            "runtime with `uv pip install torch kornia` to enable it."
        )

    photo = np.asarray(photo_rgb)
    cad = np.asarray(cad_rgb)
    if photo.ndim != 3 or photo.shape[2] not in {3, 4} or cad.ndim != 3 or cad.shape[2] not in {3, 4}:
        return fail(f"LoFTR expects RGB/RGBA images; got {photo.shape} and {cad.shape}.")
    photo_rgb = photo[..., :3]
    cad_rgb = cad[..., :3]
    photo_h, photo_w = photo_rgb.shape[:2]
    cad_h, cad_w = cad_rgb.shape[:2]

    def prepare(image: np.ndarray) -> tuple[np.ndarray, float, float]:
        height, width = image.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        out_w = max(8, int(round(width * scale / 8)) * 8)
        out_h = max(8, int(round(height * scale / 8)) * 8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return resized, out_w / width, out_h / height

    photo_gray, photo_sx, photo_sy = prepare(photo_rgb)
    cad_gray, cad_sx, cad_sy = prepare(cad_rgb)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        matcher = KF.LoFTR(pretrained="outdoor").eval().to(device)
        image0 = torch.from_numpy(photo_gray).to(device=device, dtype=torch.float32)[None, None] / 255.0
        image1 = torch.from_numpy(cad_gray).to(device=device, dtype=torch.float32)[None, None] / 255.0
        with torch.inference_mode():
            prediction = matcher({"image0": image0, "image1": image1})
        points_photo = prediction["keypoints0"].detach().cpu().numpy().astype(np.float32)
        points_cad = prediction["keypoints1"].detach().cpu().numpy().astype(np.float32)
    except Exception as exc:  # model weights/device can be unavailable offline
        return fail(f"LoFTR model could not run (weights or runtime unavailable): {exc}")

    # LoFTR does not emit detector keypoints or reusable descriptors; these
    # are matched coordinate pairs produced directly by the network.
    diag["keypoints_photo"] = 0
    diag["keypoints_cad"] = 0
    diag["correspondences"] = int(len(points_photo))
    diag["raw_matches"] = diag["filtered_matches"] = diag["good_matches"] = int(len(points_photo))
    diag["match_points"] = tuple(
        (
            float(points_photo[i, 0] / photo_sx),
            float(points_photo[i, 1] / photo_sy),
            float(points_cad[i, 0] / cad_sx),
            float(points_cad[i, 1] / cad_sy),
            False,
        )
        for i in np.linspace(0, len(points_photo) - 1, min(300, len(points_photo)), dtype=int)
    ) if len(points_photo) else ()
    if len(points_photo) < 12 or points_photo.shape != points_cad.shape:
        return fail(f"LoFTR produced only {len(points_photo)} correspondences; at least 12 are required.")

    try:
        h_small, inlier_mask = cv2.findHomography(
            points_photo, points_cad, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.995
        )
    except cv2.error as exc:
        return fail(f"LoFTR RANSAC failed: {exc}")
    if h_small is None or inlier_mask is None or h_small.shape != (3, 3) or not np.isfinite(h_small).all():
        return fail("LoFTR RANSAC returned no valid homography.")
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    ratio = inlier_count / len(points_photo)
    diag.update(inliers=inlier_count, inlier_ratio=ratio)
    if inlier_count < 8 or ratio < 0.20:
        return fail(f"Weak LoFTR RANSAC fit: {inlier_count}/{len(points_photo)} inliers ({ratio:.0%}).")
    projected = cv2.perspectiveTransform(points_photo[inliers, None, :], h_small)[:, 0, :]
    errors = np.linalg.norm(projected - points_cad[inliers], axis=1) if inlier_count else np.array([])
    median_error = float(np.median(errors)) if errors.size else None
    hull = cv2.convexHull(points_cad[inliers]) if inlier_count >= 3 else None
    spread = float(cv2.contourArea(hull)) / max(1.0, cad_gray.shape[0] * cad_gray.shape[1]) if hull is not None else 0.0
    diag.update(median_reprojection_error=median_error, inlier_spread_ratio=spread)
    if median_error is None or median_error > 5.0 or spread < 0.001:
        return fail(f"LoFTR inliers have weak spread or high reprojection error ({median_error}px; spread={spread:.4%}).")
    # Hull area does not detect a long thin strip of inliers. A homography is
    # unconstrained perpendicular to near-collinear points, so parts away from
    # the strip would be placed by extrapolation.
    principal_std = np.sqrt(np.clip(np.linalg.eigvalsh(np.cov(points_cad[inliers].T)), 0.0, None))
    elongation = float(principal_std[0] / principal_std[1]) if principal_std[1] > 0 else 0.0
    if elongation < 0.25:
        return fail(
            f"LoFTR inliers are nearly collinear (minor/major spread {elongation:.2f}); "
            "the homography is unconstrained away from the inlier strip."
        )

    photo_scale = np.diag([photo_sx, photo_sy, 1.0])
    cad_scale = np.diag([cad_sx, cad_sy, 1.0])
    matrix = np.linalg.inv(cad_scale) @ h_small @ photo_scale
    corners = np.float32([[[0, 0], [photo_w - 1, 0], [photo_w - 1, photo_h - 1], [0, photo_h - 1]]])
    projected_corners = cv2.perspectiveTransform(corners, matrix)[0]
    area_ratio = abs(float(cv2.contourArea(projected_corners))) / float(cad_w * cad_h)
    cad_rect = np.float32([[[0, 0], [cad_w - 1, 0], [cad_w - 1, cad_h - 1], [0, cad_h - 1]]])
    try:
        overlap, _ = cv2.intersectConvexConvex(projected_corners.astype(np.float32), cad_rect)
    except cv2.error:
        overlap = 0.0
    overlap_ratio = float(overlap) / float(cad_w * cad_h)
    diag.update(projected_area_ratio=area_ratio, overlap_ratio=overlap_ratio)
    if not (0.08 <= area_ratio <= 5.0) or overlap_ratio < 0.20:
        return fail(f"LoFTR projection is implausible (area={area_ratio:.3f}x, overlap={overlap_ratio:.1%}).")
    diag["homography_valid"] = True

    inlier_cad = points_cad[inliers]
    cad_points = tuple((float(x / cad_sx), float(y / cad_sy)) for x, y in inlier_cad)
    diag["match_points"] = tuple(
        (
            float(points_photo[i, 0] / photo_sx), float(points_photo[i, 1] / photo_sy),
            float(points_cad[i, 0] / cad_sx), float(points_cad[i, 1] / cad_sy), bool(inliers[i]),
        )
        for i in np.linspace(0, len(points_photo) - 1, min(300, len(points_photo)), dtype=int)
    )
    aligned = cv2.warpPerspective(photo_rgb, matrix, (cad_w, cad_h))
    return AlignmentResult(True, matrix, "LoFTR/RANSAC passed alignment quality checks.", cad_inlier_points=cad_points, **diag), aligned
