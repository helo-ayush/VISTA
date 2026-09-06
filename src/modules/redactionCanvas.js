import { REGEX_RULES, UI_STOPWORDS, COMPANY_INDICATORS } from './piiDetector.js';

const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/**
 * Computes a surgical sub-bounding box for an in-line substring within an OCR text line box.
 * Allows pinpoint in-line redaction (e.g. redacting just "Rajesh Kumar" in a full sentence).
 */
function getCharWeight(ch) {
  if ('il1!|:;\',. -`'.includes(ch)) return 0.52;
  if ('frtI()[]{}'.includes(ch)) return 0.68;
  if ('mwWM@%#&'.includes(ch)) return 1.40;
  if (ch >= 'A' && ch <= 'Z') return 1.18;
  if (ch >= '0' && ch <= '9') return 1.12;
  return 0.95;
}

function computeSubBox(box, fullText, startIdx, endIdx) {
  const b = box || { x: 0, y: 0, width: 100, height: 20 };
  const str = fullText || '';
  let totalW = 0;
  const prefixWeights = [0];
  for (let i = 0; i < str.length; i++) {
    totalW += getCharWeight(str[i]);
    prefixWeights.push(totalW);
  }
  totalW = Math.max(0.1, totalW);

  const startRatio = prefixWeights[Math.min(startIdx, str.length)] / totalW;
  const endRatio = prefixWeights[Math.min(endIdx, str.length)] / totalW;

  const boxW = b.width || 100;
  const rawX = (b.x || 0) + boxW * startRatio;
  const rawEnd = (b.x || 0) + boxW * endRatio;

  // Modest pixel padding (8px) so characters (like trailing digits "90" or leading "HDFC") are never clipped
  const padPx = 8;
  const subX = Math.max(b.x || 0, Math.round(rawX - (startIdx > 0 ? padPx : 0)));
  const subEnd = Math.min((b.x || 0) + boxW, Math.round(rawEnd + (endIdx < str.length ? padPx : 0)));
  const subW = Math.max(16, subEnd - subX);

  return {
    x: subX,
    y: b.y ?? 0,
    width: subW,
    height: b.height ?? 20
  };
}

/**
 * Matches detected OCR visual bounding boxes against extracted PII entities.
 * Eliminates false positives on icons, UI buttons, breadcrumbs, and random text fragments.
 * Computes surgical in-line sub-boxes so only the sensitive words are masked, not the whole line.
 *
 * @param {Array} ocrBoxes - Detected OCR boxes with { text, box, confidence }
 * @param {Array} redactedEntities - Extracted PII entities with { originalValue, tag }
 * @returns {Array} List of boxes to redact with assigned tags
 */
export function findBoxesToRedact(ocrBoxes, redactedEntities) {
  const boxesToRedact = [];

  for (const item of ocrBoxes) {
    const itemText = (item.text || '').trim();
    if (!itemText) continue;

    // Reject standalone UI stopwords or 1-2 character OCR noise
    const cleanLower = itemText.toLowerCase().replace(/^[^\w]+|[^\w]+$/g, '');
    if (cleanLower.length <= 2 && !/^\d{2}$/.test(cleanLower)) {
      continue; // Skip single letters, punctuation, icon noise like "v", "[PI", "o"
    }

    if (UI_STOPWORDS.has(cleanLower) || UI_STOPWORDS.has(itemText.toLowerCase())) {
      continue; // Skip generic buttons/navigation: "Home", "Download Invoice", "Cart", etc.
    }

    // Skip company, corporate, or organization names (e.g. Seller: CultX, Flipkart, Inc, Ltd, etc.)
    if (COMPANY_INDICATORS.test(itemText) || COMPANY_INDICATORS.test(cleanLower)) {
      continue;
    }

    // Reject standalone code identifiers, repo slugs, file paths, or URLs (e.g. "helo-ayush/VISTA_PII")
    // Do NOT reject lines containing dates (12/30), parentage (S/o), or bank account labels (A/c)
    if (/^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+$/.test(itemText) && !/\d{1,2}\/\d{1,2}/.test(itemText)) {
      continue;
    }
    if (/^(?:https?:\/\/|www\.)\S+$/i.test(itemText)) {
      continue;
    }
    if (/^[a-zA-Z0-9]+_[a-zA-Z0-9_]+$/.test(cleanLower)) {
      continue;
    }

    // Skip UI category lists or multi-word navigation where >= 50% of words are UI stopwords
    const words = cleanLower.split(/[\s,&/+-]+/).filter(w => w.length >= 2);
    if (words.length > 0) {
      const stopwordCount = words.filter(w => UI_STOPWORDS.has(w) || COMPANY_INDICATORS.test(w)).length;
      if (stopwordCount / words.length >= 0.50) {
        continue;
      }
    }

    const itemLower = itemText.toLowerCase();
    const matchedSubSpans = [];

    // 1. Match against extracted PII entities
    for (const ent of redactedEntities) {
      const val = (ent.originalValue || '').trim();
      if (!val || val.length < 3) continue;

      const valLower = val.toLowerCase();
      const cleanVal = val.replace(/^[^\w]+|[^\w]+$/g, '').toLowerCase();
      const cleanItem = itemText.replace(/^[^\w]+|[^\w]+$/g, '').toLowerCase();

      // Case A: The OCR line contains the sensitive PII (either full entity or substring)
      let searchStart = 0;
      while (searchStart < itemLower.length) {
        const idx = itemLower.indexOf(valLower, searchStart);
        if (idx === -1) break;
        matchedSubSpans.push({
          start: idx,
          end: idx + val.length,
          tag: ent.tag,
          value: val
        });
        searchStart = idx + val.length;
      }

      // Case B: For person names, match individual name tokens (length >= 3)
      if (ent.tag === 'NAME' && val.split(/\s+/).length > 1) {
        const nameParts = val.split(/\s+/).filter(p => p.length >= 3 && !UI_STOPWORDS.has(p.toLowerCase()) && !COMPANY_INDICATORS.test(p));
        for (const part of nameParts) {
          const partLower = part.toLowerCase();
          let pStart = 0;
          while (pStart < itemLower.length) {
            const pIdx = itemLower.indexOf(partLower, pStart);
            if (pIdx === -1) break;
            const isWordStart = pIdx === 0 || !/\w/.test(itemLower[pIdx - 1]);
            const isWordEnd = (pIdx + part.length >= itemLower.length) || !/\w/.test(itemLower[pIdx + part.length]);
            if (isWordStart && isWordEnd) {
              matchedSubSpans.push({
                start: pIdx,
                end: pIdx + part.length,
                tag: 'NAME',
                value: part
              });
            }
            pStart = pIdx + part.length;
          }
        }
      }

      // Case C: The PII entity contains this OCR box (e.g. truncated address "Home Plot-42...")
      if (cleanItem.length >= 4 && !UI_STOPWORDS.has(cleanLower)) {
        if (cleanVal.includes(cleanItem)) {
          matchedSubSpans.push({
            start: 0,
            end: itemText.length,
            tag: ent.tag,
            value: itemText
          });
        }
      }

      // Case D: Handles & Usernames (@username)
      if (ent.tag === 'HANDLE') {
        const cleanHandle = valLower.replace(/^[@#\s]+/, '');
        if (cleanHandle.length >= 3) {
          let hStart = 0;
          while (hStart < itemLower.length) {
            const hIdx = itemLower.indexOf(cleanHandle, hStart);
            if (hIdx === -1) break;
            const actualStart = (hIdx > 0 && (itemLower[hIdx - 1] === '@' || itemLower[hIdx - 1] === '#')) ? hIdx - 1 : hIdx;
            matchedSubSpans.push({
              start: actualStart,
              end: hIdx + cleanHandle.length,
              tag: 'HANDLE',
              value: cleanHandle
            });
            hStart = hIdx + cleanHandle.length;
          }
        }
      }
    }

    // 2. High-precision direct regex rules on this line (PHONE, CARD, AADHAAR, PAN, EMAIL, UPI, ACCOUNT_NUMBER, CVV, etc.)
    for (const rule of REGEX_RULES) {
      if (rule.tag === 'ADDRESS') continue;
      const pat = new RegExp(rule.pattern.source, rule.pattern.flags);
      let m;
      while ((m = pat.exec(itemText)) !== null) {
        if (m[1]) {
          const valStart = m.index + m[0].lastIndexOf(m[1]);
          matchedSubSpans.push({
            start: valStart,
            end: valStart + m[1].length,
            tag: rule.tag,
            value: m[1]
          });
        } else {
          matchedSubSpans.push({
            start: m.index,
            end: m.index + m[0].length,
            tag: rule.tag,
            value: m[0]
          });
        }
      }
    }

    // 3. Direct handle detection: any OCR box starting with @ or containing @username
    const handleRegex = /(?<=\s|^|[([:;,])@\s*[a-zA-Z0-9_.-]{2,32}\b/gi;
    let hm;
    while ((hm = handleRegex.exec(itemText)) !== null) {
      matchedSubSpans.push({
        start: hm.index,
        end: hm.index + hm[0].length,
        tag: 'HANDLE',
        value: hm[0]
      });
    }

    if (matchedSubSpans.length > 0) {
      // Sort and merge overlapping sub-spans within this line
      matchedSubSpans.sort((a, b) => a.start - b.start);
      const mergedLineSpans = [];
      for (const span of matchedSubSpans) {
        if (mergedLineSpans.length === 0) {
          mergedLineSpans.push({ ...span });
        } else {
          const last = mergedLineSpans[mergedLineSpans.length - 1];
          if (span.start <= last.end) {
            last.end = Math.max(last.end, span.end);
            if (span.tag === 'NAME' || last.tag === 'NAME') last.tag = 'NAME';
          } else {
            mergedLineSpans.push({ ...span });
          }
        }
      }

      // Generate surgical boxes for each matched entity
      for (const span of mergedLineSpans) {
        const itemBox = item.box || { x: item.x || 0, y: item.y || 0, width: item.width || 100, height: item.height || 20 };
        const spanRatio = (span.end - span.start) / itemText.length;
        if (spanRatio >= 0.85) {
          // Entity occupies the whole line -> redact full box
          boxesToRedact.push({
            ...itemBox,
            text: itemText,
            tag: span.tag,
            confidence: item.confidence
          });
        } else {
          // Entity is an in-line substring -> compute pinpoint sub-box!
          const sub = computeSubBox(itemBox, itemText, span.start, span.end);
          boxesToRedact.push({
            ...sub,
            text: itemText.substring(span.start, span.end),
            tag: span.tag,
            confidence: item.confidence
          });
        }
      }
    }
  }

  return boxesToRedact;
}

/**
 * Applies mathematical pixelation blur to an arbitrary bounding box on the canvas.
 */
function applyPixelatedBlur(ctx, x, y, width, height, pixelSize = 12) {
  const scaledW = Math.max(1, Math.floor(width / pixelSize));
  const scaledH = Math.max(1, Math.floor(height / pixelSize));

  const offscreen = document.createElement('canvas');
  offscreen.width = scaledW;
  offscreen.height = scaledH;

  const offCtx = offscreen.getContext('2d');
  offCtx.drawImage(ctx.canvas, x, y, width, height, 0, 0, scaledW, scaledH);

  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, 0, 0, scaledW, scaledH, x, y, width, height);
  ctx.restore();
}

/**
 * Draws rounded rectangle helper.
 */
function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

/**
 * Main visual rendering engine on HTML5 Canvas.
 * Renders document image, blurred faces with <FACE_HIDDEN>, and blurred/guided PII text boxes (<NAME_HIDDEN>, etc.).
 *
 * @param {HTMLCanvasElement} canvas - Target display canvas
 * @param {HTMLImageElement} img - Source image
 * @param {Array} boxesToRedact - Bounding boxes of detected PII text
 * @param {Array} allOcrBoxes - All detected OCR boxes (for inspection mode)
 * @param {Array} faces - Detected faces from YuNet
 * @param {string} viewMode - 'guided' | 'blackout' | 'boxes' | 'original'
 * @returns {number} Time taken to render in milliseconds
 */
export function renderCanvasOverlay(canvas, img, boxesToRedact = [], allOcrBoxes = [], faces = [], viewMode = 'guided') {
  if (!canvas || !img) return 0;
  const startTime = performance.now();
  const ctx = canvas.getContext('2d');

  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;

  // 1. Draw pristine original image
  ctx.drawImage(img, 0, 0);

  if (viewMode === 'original') {
    return Math.round(performance.now() - startTime);
  }

  // --- MODE: FACE BOXES ---
  if (viewMode === 'faces') {
    faces.forEach((face, idx) => {
      ctx.lineWidth = Math.max(3, Math.round(canvas.width / 400));
      ctx.strokeStyle = '#a855f7'; // Purple-500
      ctx.strokeRect(face.x, face.y, face.width, face.height);

      // Subtle fill
      ctx.fillStyle = 'rgba(168, 85, 247, 0.15)';
      ctx.fillRect(face.x, face.y, face.width, face.height);

      const fontSize = Math.max(12, Math.round(canvas.width / 70));
      ctx.font = `bold ${fontSize}px monospace`;
      const faceLabel = `Face #${idx + 1} (${Math.round(face.score * 100)}%)`;
      const metrics = ctx.measureText(faceLabel);

      ctx.fillStyle = 'rgba(147, 51, 234, 0.95)';
      ctx.fillRect(face.x, Math.max(0, face.y - fontSize - 6), metrics.width + 10, fontSize + 6);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(faceLabel, face.x + 5, Math.max(fontSize, face.y - 4));

      // Facial landmarks
      if (face.landmarks && Array.isArray(face.landmarks)) {
        ctx.fillStyle = '#fbbf24';
        for (const pt of face.landmarks) {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, Math.max(3, Math.round(canvas.width / 350)), 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    });
    return Math.round(performance.now() - startTime);
  }

  // --- MODE: OCR BOXES ---
  if (viewMode === 'ocr') {
    for (const item of allOcrBoxes) {
      const isRedacted = boxesToRedact.some(b => b.x === item.box.x && b.y === item.box.y);
      ctx.lineWidth = Math.max(1.5, Math.round(canvas.width / 600));
      ctx.strokeStyle = isRedacted ? '#ef4444' : '#10b981';
      ctx.strokeRect(item.box.x, item.box.y, item.box.width, item.box.height);

      ctx.fillStyle = isRedacted ? 'rgba(239, 68, 68, 0.12)' : 'rgba(16, 185, 129, 0.08)';
      ctx.fillRect(item.box.x, item.box.y, item.box.width, item.box.height);

      const fontSize = Math.max(10, Math.round(canvas.width / 90));
      ctx.font = `bold ${fontSize}px monospace`;
      // Display the full scanned text without artificial 15-character truncation
      const label = isRedacted ? `[${item.tag || 'PII'}]` : (item.text || `${Math.round(item.confidence * 100)}%`);
      const textMetrics = ctx.measureText(label);

      ctx.fillStyle = isRedacted ? 'rgba(239, 68, 68, 0.9)' : 'rgba(16, 185, 129, 0.85)';
      ctx.fillRect(item.box.x, Math.max(0, item.box.y - fontSize - 4), textMetrics.width + 6, fontSize + 4);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, item.box.x + 3, Math.max(fontSize, item.box.y - 3));
    }
    return Math.round(performance.now() - startTime);
  }

  // --- MODE 1: GUIDED SEMANTIC PRIVACY MASKING (DEFAULT & RECOMMENDED) ---
  // A. Blur & Mask Faces with <FACE_HIDDEN>
  for (const face of faces) {
    const fx = Math.max(0, face.x);
    const fy = Math.max(0, face.y);
    const fw = Math.min(canvas.width - fx, face.width);
    const fh = Math.min(canvas.height - fy, face.height);

    // Apply privacy pixelation blur
    const pixelSize = Math.max(8, Math.round(fw / 12));
    applyPixelatedBlur(ctx, fx, fy, fw, fh, pixelSize);

    // Frosted dark privacy veil over face
    ctx.fillStyle = 'rgba(15, 23, 42, 0.45)';
    ctx.fillRect(fx, fy, fw, fh);

    // Face border outline
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.7)';
    ctx.lineWidth = Math.max(2, Math.round(canvas.width / 500));
    ctx.strokeRect(fx, fy, fw, fh);

    // Centered Guided Label Badge: <FACE_HIDDEN>
    const badgeText = '<FACE_HIDDEN>';
    const fontSize = Math.max(12, Math.min(22, Math.round(fw / 6)));
    ctx.font = `bold ${fontSize}px ui-monospace, SFMono-Regular, monospace`;
    const metrics = ctx.measureText(badgeText);
    const badgeW = metrics.width + 16;
    const badgeH = fontSize + 10;
    const badgeX = fx + (fw - badgeW) / 2;
    const badgeY = fy + (fh - badgeH) / 2;

    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    drawRoundedRect(ctx, badgeX, badgeY, badgeW, badgeH, 6);
    ctx.fill();
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(badgeText, badgeX + 8, badgeY + badgeH / 2);
    ctx.restore();
  }

  // B. Blur & Fill Sensitive PII Text with Guided Semantic Badges (<NAME_HIDDEN>, etc.)
  for (const box of boxesToRedact) {
    const padX = Math.max(3, Math.round(box.width * 0.02));
    const padY = Math.max(3, Math.round(box.height * 0.08));
    const rx = Math.max(0, box.x - padX);
    const ry = Math.max(0, box.y - padY);
    const rw = Math.min(canvas.width - rx, box.width + padX * 2);
    const rh = Math.min(canvas.height - ry, box.height + padY * 2);

    // Apply privacy pixelation blur over the underlying text
    applyPixelatedBlur(ctx, rx, ry, rw, rh, Math.max(4, Math.round(rh / 3)));

    // Deep semantic background fill
    ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
    drawRoundedRect(ctx, rx, ry, rw, rh, 4);
    ctx.fill();

    // Guided semantic token: <TAG_HIDDEN> or compact [TAG] if box is narrow
    const tag = (box.tag || 'PII').toUpperCase();
    const token = rw >= 110 ? `<${tag}_HIDDEN>` : (rw >= 50 ? `[${tag}]` : `*`);

    const maxFontSize = Math.max(10, Math.min(18, Math.round(rh * 0.65)));
    ctx.font = `bold ${maxFontSize}px ui-monospace, SFMono-Regular, monospace`;
    let metrics = ctx.measureText(token);

    // Auto-fit font size if badge is wider than the box
    let fittedFontSize = maxFontSize;
    if (metrics.width > rw - 6 && rw > 25) {
      fittedFontSize = Math.max(8, Math.floor(maxFontSize * (rw - 6) / metrics.width));
      ctx.font = `bold ${fittedFontSize}px ui-monospace, SFMono-Regular, monospace`;
      metrics = ctx.measureText(token);
    }

    ctx.save();
    ctx.fillStyle = '#38bdf8'; // Sky-400 for high-contrast machine-readable AI recognition
    ctx.textBaseline = 'middle';
    const textX = rx + Math.max(3, (rw - metrics.width) / 2);
    const textY = ry + rh / 2;
    ctx.fillText(token, textX, textY);
    ctx.restore();
  }

  return Math.round(performance.now() - startTime);
}
