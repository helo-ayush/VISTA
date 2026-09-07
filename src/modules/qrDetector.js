import jsQR from 'jsqr';

/**
 * Detects QR codes and 2D matrix barcodes on an image element.
 * Uses native BarcodeDetector API when available in modern browsers,
 * with multi-scale jsQR fallback, and visual high-frequency matrix finder.
 * 
 * @param {HTMLImageElement|HTMLCanvasElement} imgElement
 * @returns {Promise<Array<{ x: number, y: number, width: number, height: number, tag: string }>>}
 */
export async function detectQRCodes(imgElement) {
  if (!imgElement) return [];
  const qrBoxes = [];

  const w = imgElement.naturalWidth || imgElement.width || 0;
  const h = imgElement.naturalHeight || imgElement.height || 0;
  if (w <= 0 || h <= 0) return [];

  // 1. Primary: Native browser BarcodeDetector API (fast, hardware-accelerated in Chrome/Edge/Android)
  if (typeof window !== 'undefined' && 'BarcodeDetector' in window) {
    try {
      const barcodeDetector = new window.BarcodeDetector({ formats: ['qr_code', 'data_matrix', 'aztec'] });
      const barcodes = await barcodeDetector.detect(imgElement);
      if (barcodes && Array.isArray(barcodes) && barcodes.length > 0) {
        for (const b of barcodes) {
          const bb = b.boundingBox;
          if (bb && bb.width > 20 && bb.height > 20) {
            qrBoxes.push({
              x: Math.max(0, Math.round(bb.x)),
              y: Math.max(0, Math.round(bb.y)),
              width: Math.round(bb.width),
              height: Math.round(bb.height),
              tag: 'QR'
            });
          }
        }
        if (qrBoxes.length > 0) {
          return qrBoxes;
        }
      }
    } catch (nativeErr) {
      console.warn('[QRDetector] Native BarcodeDetector error, falling back to jsQR:', nativeErr);
    }
  }

  // 2. Secondary: jsQR via offscreen Canvas
  try {
    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    const ctx = offscreen.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(imgElement, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);

    // Pass A: Full frame
    let code = jsQR(imageData.data, imageData.width, imageData.height, {
      inversionAttempts: 'attemptBoth'
    });

    // Pass B: If not found, scan bottom-center sub-region with 2x upscale (typical for mark sheets/certificates)
    if (!code && h > 200) {
      const cropX = Math.round(w * 0.25);
      const cropY = Math.round(h * 0.60);
      const cropW = Math.round(w * 0.50);
      const cropH = h - cropY;

      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = cropW * 2;
      cropCanvas.height = cropH * 2;
      const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
      cropCtx.imageSmoothingEnabled = true;
      cropCtx.drawImage(offscreen, cropX, cropY, cropW, cropH, 0, 0, cropW * 2, cropH * 2);

      const cropData = cropCtx.getImageData(0, 0, cropCanvas.width, cropCanvas.height);
      const subCode = jsQR(cropData.data, cropData.width, cropData.height, {
        inversionAttempts: 'attemptBoth'
      });

      if (subCode && subCode.location) {
        const loc = subCode.location;
        const minX = Math.min(loc.topLeftCorner.x, loc.bottomLeftCorner.x) / 2 + cropX;
        const maxX = Math.max(loc.topRightCorner.x, loc.bottomRightCorner.x) / 2 + cropX;
        const minY = Math.min(loc.topLeftCorner.y, loc.topRightCorner.y) / 2 + cropY;
        const maxY = Math.max(loc.bottomLeftCorner.y, loc.bottomRightCorner.y) / 2 + cropY;

        code = {
          location: {
            topLeftCorner: { x: minX, y: minY },
            topRightCorner: { x: maxX, y: minY },
            bottomLeftCorner: { x: minX, y: maxY },
            bottomRightCorner: { x: maxX, y: maxY }
          }
        };
      }
    }

    if (code && code.location) {
      const loc = code.location;
      const minX = Math.min(loc.topLeftCorner.x, loc.bottomLeftCorner.x);
      const maxX = Math.max(loc.topRightCorner.x, loc.bottomRightCorner.x);
      const minY = Math.min(loc.topLeftCorner.y, loc.topRightCorner.y);
      const maxY = Math.max(loc.bottomLeftCorner.y, loc.bottomRightCorner.y);
      const boxW = Math.round(maxX - minX);
      const boxH = Math.round(maxY - minY);

      if (boxW > 20 && boxH > 20) {
        qrBoxes.push({
          x: Math.max(0, Math.round(minX)),
          y: Math.max(0, Math.round(minY)),
          width: boxW,
          height: boxH,
          tag: 'QR'
        });
      }
    }

    // 3. Fallback: Visual 2D Matrix / QR Finder Heuristic (when barcode payload is un-decodable)
    if (qrBoxes.length === 0 && w >= 300 && h >= 300) {
      // Downscale to ~320px width for 2-3ms scan
      const scale = 320 / w;
      const smallW = 320;
      const smallH = Math.round(h * scale);

      const smallCanvas = document.createElement('canvas');
      smallCanvas.width = smallW;
      smallCanvas.height = smallH;
      const sCtx = smallCanvas.getContext('2d', { willReadFrequently: true });
      sCtx.drawImage(offscreen, 0, 0, smallW, smallH);
      const smallData = sCtx.getImageData(0, 0, smallW, smallH).data;

      // Compute luminance
      const lum = new Uint8Array(smallW * smallH);
      for (let i = 0, j = 0; i < smallData.length; i += 4, j++) {
        lum[j] = Math.round(smallData[i] * 0.299 + smallData[i + 1] * 0.587 + smallData[i + 2] * 0.114);
      }

      // Check candidate square windows (typical QR sizes 25-60px in downscaled image)
      let bestScore = 0;
      let bestBox = null;

      for (const win of [28, 38, 50]) {
        // Focus search on bottom half and right half where document QRs live
        const startY = Math.round(smallH * 0.35);
        for (let y = startY; y < smallH - win; y += 4) {
          for (let x = 10; x < smallW - win - 10; x += 4) {
            let transitions = 0;
            let totalPairs = 0;

            for (let r = y; r < y + win; r += 2) {
              for (let c = x; c < x + win - 1; c += 2) {
                const idx = r * smallW + c;
                if (Math.abs(lum[idx] - lum[idx + 1]) > 28) transitions++;
                totalPairs++;
              }
            }

            const density = transitions / Math.max(1, totalPairs);
            // QR codes have a dense checkerboard distribution with density ~0.30 - 0.45
            if (density > 0.32 && density > bestScore) {
              bestScore = density;
              bestBox = {
                x: Math.round(x / scale),
                y: Math.round(y / scale),
                width: Math.round(win / scale),
                height: Math.round(win / scale),
                tag: 'QR'
              };
            }
          }
        }
      }

      if (bestBox && bestScore > 0.32) {
        qrBoxes.push(bestBox);
      }
    }
  } catch (jsQrErr) {
    console.warn('[QRDetector] jsQR scanning notice:', jsQrErr);
  }

  return qrBoxes;
}
