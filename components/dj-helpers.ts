// ═══════════════════════════════════════════════════════════════
// DJ TRAINER — Audio Analysis & DSP Helpers
// ═══════════════════════════════════════════════════════════════

export function buildPeaks(buffer: AudioBuffer, buckets = 1100): Float32Array {
  const ch0 = buffer.getChannelData(0);
  const ch1 = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : null;
  const len = ch0.length;
  const step = Math.floor(len / buckets) || 1;
  const peaks = new Float32Array(buckets);
  for (let b = 0; b < buckets; b++) {
    let max = 0;
    const off = b * step;
    for (let i = 0; i < step && off + i < len; i++) {
      let v = Math.abs(ch0[off + i]);
      if (ch1) v = Math.max(v, Math.abs(ch1[off + i]));
      if (v > max) max = v;
    }
    peaks[b] = max;
  }
  return peaks;
}

export function detectBPM(buffer: AudioBuffer): number | null {
  const sr = buffer.sampleRate;
  const data = buffer.getChannelData(0);
  const samples = Math.min(data.length, Math.floor(sr * 45));
  const dec = Math.round(sr / 200);
  const dLen = Math.floor(samples / dec);
  if (dLen < 400) return null;
  const bass = new Float32Array(dLen);
  for (let i = 0; i < dLen; i++) {
    let sum = 0;
    const off = i * dec;
    for (let j = 0; j < dec; j++) { const s = data[off + j]; sum += s * s; }
    bass[i] = sum / dec;
  }
  const onset = new Float32Array(dLen - 1);
  let maxO = 0;
  for (let i = 0; i < dLen - 1; i++) {
    onset[i] = Math.max(0, bass[i + 1] - bass[i]);
    if (onset[i] > maxO) maxO = onset[i];
  }
  if (maxO > 0) for (let i = 0; i < onset.length; i++) onset[i] /= maxO;
  const effRate = sr / dec;
  const minLag = Math.round(effRate * 60 / 200);
  const maxLag = Math.round(effRate * 60 / 60);
  const corr = new Float32Array(maxLag + 1);
  let bestLag = minLag, bestCorr = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let c = 0;
    const n = onset.length - lag;
    for (let i = 0; i < n; i++) c += onset[i] * onset[i + lag];
    c /= n; corr[lag] = c;
    if (c > bestCorr) { bestCorr = c; bestLag = lag; }
  }
  if (bestCorr <= 0) return null;
  let refined = bestLag;
  if (bestLag > minLag && bestLag < maxLag) {
    const a = corr[bestLag - 1], b = corr[bestLag], c2 = corr[bestLag + 1];
    const denom = a - 2 * b + c2;
    if (Math.abs(denom) > 1e-12) refined = bestLag + 0.5 * (a - c2) / denom;
  }
  let bpm = 60 * effRate / refined;
  const halfLag = Math.round(refined * 2);
  if (halfLag <= maxLag && bpm > 140 && (corr[halfLag] || 0) > bestCorr * 0.8) bpm /= 2;
  const dblLag = Math.round(refined / 2);
  if (dblLag >= minLag && bpm < 80 && (corr[dblLag] || 0) > bestCorr * 0.8) bpm *= 2;
  if (bpm < 60 || bpm > 200) return null;
  return Math.round(bpm * 10) / 10;
}

export function computeRMS(buffer: AudioBuffer): number {
  const data = buffer.getChannelData(0);
  const blockLen = Math.min(data.length, buffer.sampleRate * 10);
  // Find loudest 2-second window
  const windowLen = Math.min(data.length, buffer.sampleRate * 2);
  let maxRms = 0;
  const step = Math.floor(windowLen / 2);
  for (let off = 0; off < blockLen - windowLen; off += step) {
    let sum = 0;
    for (let i = 0; i < windowLen; i++) { const s = data[off + i]; sum += s * s; }
    const rms = Math.sqrt(sum / windowLen);
    if (rms > maxRms) maxRms = rms;
  }
  return maxRms;
}

// Target RMS for normalization (~-14 dBFS, good club level)
const TARGET_RMS = 0.2;

export function normalizeGain(rms: number): number {
  if (rms <= 0.001) return 1;
  return Math.min(2.0, TARGET_RMS / rms); // cap at 2x to prevent distortion
}

export function detectKey(buffer: AudioBuffer): string | null {
  const data = buffer.getChannelData(0);
  const sr = buffer.sampleRate;
  const N = 4096;
  const start = Math.floor(Math.min(data.length * 0.2, sr * 10));
  if (start + N * 3 > data.length) return null;

  const chroma = new Float32Array(12);
  const baseFreqs = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88];

  for (let s = 0; s < 3; s++) {
    const off = start + s * N;
    for (let note = 0; note < 12; note++) {
      for (const oct of [0.5, 1, 2]) {
        const freq = baseFreqs[note] * oct;
        let re = 0, im = 0;
        const w = 2 * Math.PI * freq / sr;
        for (let i = 0; i < N; i++) {
          re += data[off + i] * Math.cos(w * i);
          im += data[off + i] * Math.sin(w * i);
        }
        chroma[note] += Math.sqrt(re * re + im * im);
      }
    }
  }
  const maxC = Math.max(...chroma);
  if (maxC <= 0) return null;
  for (let i = 0; i < 12; i++) chroma[i] /= maxC;

  const maj = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const min = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  const names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];

  let bestKey = 'C', bestCorr = -Infinity;
  for (let shift = 0; shift < 12; shift++) {
    let cMaj = 0, cMin = 0;
    for (let i = 0; i < 12; i++) {
      cMaj += chroma[(i + shift) % 12] * maj[i];
      cMin += chroma[(i + shift) % 12] * min[i];
    }
    if (cMaj > bestCorr) { bestCorr = cMaj; bestKey = `${names[shift]}`; }
    if (cMin > bestCorr) { bestCorr = cMin; bestKey = `${names[shift]}m`; }
  }
  return bestKey;
}

export function createReverbIR(ctx: AudioContext, duration = 2.5, decay = 2.5): AudioBuffer {
  const length = Math.floor(ctx.sampleRate * duration);
  const ir = ctx.createBuffer(2, length, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const data = ir.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
    }
  }
  return ir;
}

export function fmtTime(s: number): string {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const f = Math.floor((s % 1) * 100);
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(f).padStart(2, '0')}`;
}

// Snap position to nearest beat boundary
export function quantizePos(pos: number, bpm: number): number {
  if (!bpm || bpm <= 0) return pos;
  const beatSec = 60 / bpm;
  return Math.round(pos / beatSec) * beatSec;
}

// Camelot key notation for harmonic mixing
const CAMELOT: Record<string, string> = {
  'Ab': '1B', 'Abm': '1A', 'Eb': '2B', 'Ebm': '2A', 'Bb': '3B', 'Bbm': '3A',
  'F': '4B', 'Fm': '4A', 'C': '5B', 'Cm': '5A', 'G': '6B', 'Gm': '6A',
  'D': '7B', 'Dm': '7A', 'A': '8B', 'Am': '8A', 'E': '9B', 'Em': '9A',
  'B': '10B', 'Bm': '10A', 'Gb': '11B', 'Gbm': '11A', 'Db': '12B', 'Dbm': '12A',
};

export function toCamelot(key: string | null): string | null {
  if (!key) return null;
  return CAMELOT[key] || null;
}

export function keysCompatible(keyA: string | null, keyB: string | null): boolean {
  const a = toCamelot(keyA), b = toCamelot(keyB);
  if (!a || !b) return false;
  const numA = parseInt(a), numB = parseInt(b);
  const letterA = a.slice(-1), letterB = b.slice(-1);
  if (a === b) return true; // same key
  if (letterA === letterB && (Math.abs(numA - numB) === 1 || Math.abs(numA - numB) === 11)) return true; // adjacent
  if (numA === numB && letterA !== letterB) return true; // relative major/minor
  return false;
}
