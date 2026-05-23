# Image Stitching + Super-Resolution

Pipeline for stitching multiple overlapping images into a panorama and restoring output resolution using a deep learning super-resolution model. Built for the Inter-IIT Tech Meet 2023 — ISRO problem statement.

---

## Problem

Satellite/aerial imagery is often captured as multiple overlapping frames. Stitching these into a single coherent image introduces resolution loss. The goal: produce a stitched panorama at resolution comparable to the original input images.

---

## Pipeline

```
Input images (3x)
      │
      ▼
 Resize to 40%          ← input images are 3000×1200; scaled before stitching
      │                    to fit memory and screen constraints
      ▼
OpenCV Stitcher         ← automatic feature detection, homography estimation,
      │                    warping and blending
      ▼
Stitched panorama       ← lower effective resolution due to warping
      │
      ▼
EDSR ×4 Upscale         ← Enhanced Deep Super Resolution CNN
      │                    (EDSR_x4.pb — 4× upscale factor)
      ▼
High-resolution output
```

---

## Methods

**Stitching — OpenCV `Stitcher`:**
Handles feature detection (ORB/SIFT), feature matching, homography estimation, image warping, and multi-band blending automatically. Input images are downscaled to 40% before stitching to reduce memory overhead.

**Super-resolution — EDSR (Enhanced Deep Super Resolution):**
A residual CNN trained for 4× upscaling. Outperforms bicubic interpolation on natural images by learning perceptually meaningful high-frequency details rather than interpolating them. Applied via OpenCV's `dnn_superres` module.

The output is compared side-by-side against standard bicubic upscaling (OpenCV `resize` with fx=4, fy=4) to show the quality difference.

**Alternative:** MDSR (Multi-Scale Deep Super Resolution) can be used in place of EDSR for multi-scale upscaling. Weights available in the same model repo.

---

## Setup

**Model weights** — download `EDSR_x4.pb` from the [EDSR_Tensorflow repo](https://github.com/Saafke/EDSR_Tensorflow/tree/master/models) and place it in the project root.

```bash
pip install opencv-python opencv-contrib-python matplotlib
jupyter notebook "stitch 2.ipynb"
```

> `opencv-contrib-python` is required for `cv2.dnn_superres`.

---

## Stack

`Python` `OpenCV` `OpenCV DNN Super Resolution` `EDSR` `Matplotlib`

---

## Context

Inter-IIT Tech Meet 2023 · ISRO problem statement · IIT (BHU) Varanasi
