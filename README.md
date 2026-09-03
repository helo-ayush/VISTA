# VISTA Face Detection

A lightning-fast, client-side face detection web application built with **React**, **Vite**, and **ONNX Runtime Web (WASM)**, powered by the OpenCV **YuNet** neural network model.

All machine learning inference runs directly inside the user's browser via WebAssembly—**100% locally and privately**, with zero backend server dependencies.

---

## Features

- ⚡ **Zero Server Latency**: Runs entirely in-browser using WebAssembly (`onnxruntime-web`).
- 🎯 **Accurate Detection**: Powered by OpenCV's lightweight **YuNet** model (`face_detection_yunet_2023mar.onnx`).
- 📐 **Full Face Geometry**:
  - High-precision bounding box coordinates (`x`, `y`, `width`, `height`).
  - 5 facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner).
- 🔍 **Multi-Scale Processing**: Custom JavaScript post-processor decoding multi-scale anchors across Strides 8, 16, and 32 with Non-Maximum Suppression (NMS) to eliminate duplicate detections.
- ⏱️ **Real-Time Performance Badge**: Measures and displays exact inference and processing duration per frame in milliseconds using `performance.now()`.
- 🖼️ **Interactive Canvas**: Automatically overlays green bounding boxes on top of the original uploaded image with responsive scaling.

---

## Technical Architecture & Pipeline

```
[ User Image ] 
      │
      ▼
[ Offscreen Canvas (640 × 640) ]
      │
      ▼
[ Image Preprocessing ]
  • Convert RGBA interleaved (0–255) to Planar BGR Float32
  • Reshape to Tensor shape: [1, 3, 640, 640]
      │
      ▼
[ ONNX Runtime Web (WASM Inference) ]
      │
      ▼
[ 12 Raw Output Tensors ]
  • Classification: cls_8, cls_16, cls_32
  • Objectness:     obj_8, obj_16, obj_32
  • Bounding Boxes: bbox_8, bbox_16, bbox_32
  • Landmarks:      kps_8, kps_16, kps_32
      │
      ▼
[ Post-Processing & NMS ]
  • Combined Score = sqrt(cls * obj) >= 0.5
  • Anchor center & exp(bbox) decoding
  • Non-Maximum Suppression (IoU threshold: 0.3)
  • Coordinate re-scaling back to original image aspect ratio
      │
      ▼
[ Visible Canvas Rendering ]
  • Paint bounding boxes & landmarks
  • Update React State (`faceData`, `timeTaken`)
```

---

## Tech Stack

- **Framework**: [React 19](https://react.dev/) + [Vite](https://vitejs.dev/)
- **Machine Learning Runtime**: [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (`onnxruntime-web`)
- **Model**: OpenCV YuNet Face Detector (`face_detection_yunet_2023mar.onnx`)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **Graphics API**: HTML5 2D Canvas API

---

## Getting Started

### Prerequisites

Make sure you have [Node.js](https://nodejs.org/) installed (v18 or newer recommended).

### 1. Clone the Repository

```bash
git clone https://github.com/helo-ayush/VISTA.git
cd VISTA
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Start Development Server

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser to test.

### 4. Build for Production

```bash
npm run build
```

---

## Project Structure

```
├── public/
│   └── models/
│       └── face_detection_yunet_2023mar.onnx   # ONNX model binary (~232 KB)
├── src/
│   ├── App.jsx                                 # Main component + YuNet decoder + Canvas
│   ├── main.jsx                                # React entrypoint
│   ├── index.css                               # Tailwind setup
│   └── assets/                                 # Project assets
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

---

## License

This project is open-source and available under the [MIT License](LICENSE).
YuNet model is licensed by OpenCV under the Apache 2.0 license.
