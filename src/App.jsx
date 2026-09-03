import React, { useState, useRef } from 'react';
import * as ort from 'onnxruntime-web';

// ==========================================
// 1. CONFIGURATION & MODEL INITIALIZATION
// ==========================================

// Point ONNX Runtime Web to the CDN where WASM binaries are hosted
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@dev/dist/';

const MODEL_PATH = '/models/face_detection_yunet_2023mar.onnx';
const INPUT_SIZE = 640;           // YuNet standard input resolution (640x640)
const SCORE_THRESHOLD = 0.5;      // Minimum confidence to keep a face candidate
const NMS_THRESHOLD = 0.3;        // IoU threshold to eliminate overlapping duplicates

// Cache the model session in memory to avoid re-allocating WASM memory on every run
let cachedSession = null;
export async function loadFaceDetector() {
  if (!cachedSession) {
    cachedSession = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
    });
  }
  return cachedSession;
}

// ==========================================
// 2. YUNET DECODER & POST-PROCESSING
// ==========================================

/**
 * Decodes the 12 raw tensor outputs from YuNet (strides 8, 16, 32) into
 * actual pixel coordinates scaled back to the original image dimensions.
 *
 * @param {object} response - Raw ONNX model output containing cls, obj, bbox, kps tensors
 * @param {number} origW - Natural width of original image
 * @param {number} origH - Natural height of original image
 * @returns {Array} List of detected faces with bounding box and landmarks
 */

function decodeYuNetOutput(response, origW, origH) {
  const strides = [8, 16, 32];
  const candidates = [];

  for (const stride of strides) {
    const cols = Math.floor(INPUT_SIZE / stride);
    const rows = Math.floor(INPUT_SIZE / stride);

    const cls = response[`cls_${stride}`]?.data;
    const obj = response[`obj_${stride}`]?.data;
    const bbox = response[`bbox_${stride}`]?.data;
    const kps = response[`kps_${stride}`]?.data;

    if (!cls || !obj || !bbox) continue;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;

        // Combined face confidence = sqrt(classification * objectness)
        const clsScore = Math.min(Math.max(cls[idx], 0), 1);
        const objScore = Math.min(Math.max(obj[idx], 0), 1);
        const score = Math.sqrt(clsScore * objScore);

        if (score < SCORE_THRESHOLD) continue;

        // Decode bounding box offsets on the 640x640 space
        const cx = (c + bbox[idx * 4 + 0]) * stride;
        const cy = (r + bbox[idx * 4 + 1]) * stride;
        const w = Math.exp(bbox[idx * 4 + 2]) * stride;
        const h = Math.exp(bbox[idx * 4 + 3]) * stride;

        const x1 = cx - w / 2;
        const y1 = cy - h / 2;

        // Scale factors to map 640x640 coordinates back to original image size
        const scaleX = origW / INPUT_SIZE;
        const scaleY = origH / INPUT_SIZE;

        // Decode 5 facial landmarks (right eye, left eye, nose, right mouth, left mouth)
        const landmarks = [];
        if (kps) {
          for (let n = 0; n < 5; n++) {
            landmarks.push({
              x: (c + kps[idx * 10 + 2 * n]) * stride * scaleX,
              y: (r + kps[idx * 10 + 2 * n + 1]) * stride * scaleY,
            });
          }
        }

        candidates.push({
          x: x1 * scaleX,
          y: y1 * scaleY,
          width: w * scaleX,
          height: h * scaleY,
          score,
          landmarks,
        });
      }
    }
  }

  // Non-Maximum Suppression (NMS): filter out overlapping candidate boxes
  candidates.sort((a, b) => b.score - a.score);
  const finalFaces = [];

  for (const box of candidates) {
    let keep = true;
    for (const chosen of finalFaces) {
      const x1 = Math.max(box.x, chosen.x);
      const y1 = Math.max(box.y, chosen.y);
      const x2 = Math.min(box.x + box.width, chosen.x + chosen.width);
      const y2 = Math.min(box.y + box.height, chosen.y + chosen.height);

      const interW = Math.max(0, x2 - x1);
      const interH = Math.max(0, y2 - y1);
      const interArea = interW * interH;
      const unionArea = box.width * box.height + chosen.width * chosen.height - interArea;
      const iou = unionArea > 0 ? interArea / unionArea : 0;

      if (iou > NMS_THRESHOLD) {
        keep = false;
        break;
      }
    }
    if (keep) finalFaces.push(box);
  }

  return finalFaces;
}

// ==========================================
// 3. MAIN REACT COMPONENT
// ==========================================

const App = () => {
  // Application State
  const [image, setImage] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [faceData, setFaceData] = useState([]);       // Stores latest detected face objects
  const [timeTaken, setTimeTaken] = useState(null);   // Total execution time in ms

  // DOM & Memory References
  const canvasRef = useRef(null);
  const imageElementRef = useRef(null);

  /**
   * Helper to decode model response and synchronize it with the top-level faceData state.
  */
  const updateFaceData = (rawResponse) => {
    const img = imageElementRef.current;
    if (!img || !rawResponse) return [];

    const faces = decodeYuNetOutput(rawResponse, img.naturalWidth, img.naturalHeight);
    setFaceData(faces);
    console.log('=== LATEST FACE DATA (ARRAY) ===', faces);
    return faces;
  };

  /**
   * Step 1: User selects an image file.
   * Loads it into memory, sizes the canvas, and displays the clean preview.
   */
  const handleImageInput = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setTimeTaken(null);
    setFaceData([]);

    // Create temporary URL to read file into an HTML Image Element
    const imgUrl = URL.createObjectURL(file);
    const img = new Image();
    img.src = imgUrl;

    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');

      // Match canvas internal resolution to the real image resolution
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;

      // Draw initial clean image onto the canvas
      ctx.drawImage(img, 0, 0);

      // Cache HTMLImageElement in ref for instant reuse during detection
      imageElementRef.current = img;

      // Revoke temporary blob URL to free up browser memory
      URL.revokeObjectURL(imgUrl);
    };
  };

  /**
   * Step 2: Runs AI Face Detection.
   * Preprocesses image to planar BGR Float32, executes model, decodes boxes,
   * and draws bounding boxes on the canvas.
   */
  const detectFace = async () => {
    const img = imageElementRef.current;
    if (!img) return;

    setGenerating(true);
    const startTime = performance.now(); // ⏱️ Start timer

    const model = await loadFaceDetector();

    // 1. Offscreen Canvas: Resize image to 640x640 expected by YuNet
    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE); // RGBA pixel array

    // 2. Preprocessing: Convert RGBA interleaved (0-255) to Planar BGR Float32
    // Shape required by model: [1, 3, 640, 640] -> Blue channel, Green channel, Red channel
    const totalPixels = INPUT_SIZE * INPUT_SIZE;
    const float32Data = new Float32Array(3 * totalPixels);
    const gOffset = totalPixels;
    const rOffset = 2 * totalPixels;

    for (let i = 0, j = 0; i < totalPixels; i++, j += 4) {
      float32Data[i] = data[j + 2];          // B (Blue)
      float32Data[gOffset + i] = data[j + 1]; // G (Green)
      float32Data[rOffset + i] = data[j];     // R (Red)
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    // 3. Run model inference
    const inputNames = model.inputNames[0];
    const response = await model.run({ [inputNames]: inputTensor });

    // 4. Decode raw tensors and update state
    const faces = updateFaceData(response);

    // 5. Render detection bounding boxes on the visible canvas
    const visibleCanvas = canvasRef.current;
    if (visibleCanvas) {
      const visibleCtx = visibleCanvas.getContext('2d');

      // Clear previous boxes by repainting the clean image
      visibleCtx.drawImage(img, 0, 0);

      // Draw bounding box for each detected face
      visibleCtx.strokeStyle = '#00FF00'; // Bright Green
      visibleCtx.lineWidth = 4;

      faces.forEach((face) => {
        visibleCtx.strokeRect(face.x, face.y, face.width, face.height);
      });
    }

    // 6. Record total time taken
    const endTime = performance.now(); // ⏱️ End timer
    setTimeTaken(Math.round(endTime - startTime));
    setGenerating(false);
  };

  return (
    <div className='flex flex-col items-center m-10 gap-3'>
      <div className='text-xl font-bold'>Face Detection with ONNX Runtime Web (YuNet)</div>

      {/* File Selector */}
      <input
        className="bg-slate-200 w-fit p-2 cursor-pointer rounded-4xl"
        accept="image/*"
        onChange={handleImageInput}
        type="file"
      />

      {/* Detection Trigger Button */}
      <button
        onClick={detectFace}
        disabled={generating || !image}
        className='cursor-pointer bg-slate-950 px-5 text-white py-3 rounded-4xl disabled:opacity-50'
      >
        {!generating ? 'Generate Face Points' : 'Generating...'}
      </button>

      {/* Performance & Results Badge */}
      {timeTaken !== null && (
        <div className='flex items-center gap-2 bg-slate-100 text-slate-700 px-4 py-1.5 rounded-full text-sm font-medium border border-slate-300 shadow-sm'>
          <span>⚡ Time taken: <strong className='text-slate-900'>{timeTaken} ms</strong></span>
          <span>•</span>
          <span>Faces detected: <strong className='text-slate-900'>{faceData.length}</strong></span>
        </div>
      )}

      {/* Responsive Canvas Display */}
      <div className='mt-4 max-w-full overflow-auto'>
        <canvas
          ref={canvasRef}
          className='max-w-160 w-full h-auto border border-gray-300 rounded-lg shadow'
        />
      </div>
    </div>
  );
};

export default App;
