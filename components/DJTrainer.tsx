'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';

// ═══════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════

type DeckId = 'A' | 'B';

interface DeckState {
  name: string | null;
  loaded: boolean;
  playing: boolean;
  pos: number;
  dur: number;
  tempo: number;
  range: number;
  bpm: string;
  masterTempo: boolean;
  peaks: Float32Array | null;
  cue: number;
  hotCues: (number | null)[];
  loopIn: number | null;
  loopOut: number | null;
  loopActive: boolean;
}

interface ChannelState {
  trim: number;
  hi: number;
  mid: number;
  low: number;
  fader: number;
  color: number;
}

interface MixerState {
  xfader: number;
  chA: ChannelState;
  chB: ChannelState;
  master: number;
}

interface AudioDeck {
  trim: GainNode;
  low: BiquadFilterNode;
  mid: BiquadFilterNode;
  hi: BiquadFilterNode;
  color: BiquadFilterNode;
  chGain: GainNode;
  analyser: AnalyserNode;
  xf: GainNode;
  src: AudioBufferSourceNode | null;
  buffer: AudioBuffer | null;
  startTime: number;
  startOffset: number;
  rate: number;
  effRate: number;
  playing: boolean;
  stopping: boolean;
  braking: boolean;
  cue: number;
  dur: number;
  hotCues: (number | null)[];
  loopIn: number | null;
  loopOut: number | null;
  loopActive: boolean;
  cuePreview: boolean;
}

interface AudioEngine {
  ctx: AudioContext;
  master: GainNode;
  masterAnalyser: AnalyserNode;
  decks: Record<DeckId, AudioDeck>;
}

// ═══════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════

const C = {
  bg: '#08080a', panel: '#141418', panelHi: '#1c1c22', edge: '#2a2a32',
  cyan: '#24c1e8', cyanDim: '#0c5a6e', orange: '#ff7a1a', green: '#3ce08a',
  text: '#e7e7ec', dim: '#7a7a86', red: '#ff4d4d', yellow: '#ffd23d', purple: '#c97aff',
};
const HC_COLORS = [C.red, C.yellow, C.green, C.purple];
const HC_LABELS = ['A', 'B', 'C', 'D'];
const GUIDE = [
  'Load a track on each deck \u2014 drag an MP3 onto the jog wheel, or tap LOAD.',
  'BPM is auto-detected on load. Check the BoothMatch bar \u2014 it turns green when both BPMs are locked.',
  'Press PLAY on Deck A. Push channel 1 fader up and crossfader left.',
  'Press CUE on Deck B to set a start point. Move the TEMPO fader until BPMs match.',
  'Nudge the JOG WHEEL on Deck B to slide its beats into alignment with Deck A.',
  'Slowly sweep the CROSSFADER right, kill Deck A\u2019s LOW as Deck B\u2019s LOW comes in. You\u2019re mixing.',
];

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════

function buildPeaks(buffer: AudioBuffer, buckets = 1100): Float32Array {
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

function detectBPM(buffer: AudioBuffer): number | null {
  const sr = buffer.sampleRate;
  const data = buffer.getChannelData(0);
  const samples = Math.min(data.length, Math.floor(sr * 45));

  // Decimate to ~200Hz to isolate bass/kick energy
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

  // Onset = half-wave rectified difference
  const onset = new Float32Array(dLen - 1);
  let maxO = 0;
  for (let i = 0; i < dLen - 1; i++) {
    onset[i] = Math.max(0, bass[i + 1] - bass[i]);
    if (onset[i] > maxO) maxO = onset[i];
  }
  if (maxO > 0) for (let i = 0; i < onset.length; i++) onset[i] /= maxO;

  // Autocorrelation over 60-200 BPM range
  const effRate = sr / dec;
  const minLag = Math.round(effRate * 60 / 200);
  const maxLag = Math.round(effRate * 60 / 60);
  const corr = new Float32Array(maxLag + 1);
  let bestLag = minLag, bestCorr = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let c = 0;
    const n = onset.length - lag;
    for (let i = 0; i < n; i++) c += onset[i] * onset[i + lag];
    c /= n;
    corr[lag] = c;
    if (c > bestCorr) { bestCorr = c; bestLag = lag; }
  }
  if (bestCorr <= 0) return null;

  // Parabolic interpolation for sub-sample accuracy
  let refined = bestLag;
  if (bestLag > minLag && bestLag < maxLag) {
    const a = corr[bestLag - 1], b = corr[bestLag], c2 = corr[bestLag + 1];
    const denom = a - 2 * b + c2;
    if (Math.abs(denom) > 1e-12) refined = bestLag + 0.5 * (a - c2) / denom;
  }
  let bpm = 60 * effRate / refined;

  // Octave correction
  const halfLag = Math.round(refined * 2);
  if (halfLag <= maxLag && bpm > 140 && (corr[halfLag] || 0) > bestCorr * 0.8) bpm /= 2;
  const dblLag = Math.round(refined / 2);
  if (dblLag >= minLag && bpm < 80 && (corr[dblLag] || 0) > bestCorr * 0.8) bpm *= 2;

  if (bpm < 60 || bpm > 200) return null;
  return Math.round(bpm * 10) / 10;
}

function fmtTime(s: number): string {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const f = Math.floor((s % 1) * 100);
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}.${String(f).padStart(2, '0')}`;
}

function mkDeckState(): DeckState {
  return {
    name: null, loaded: false, playing: false, pos: 0, dur: 0,
    tempo: 0.5, range: 8, bpm: '', masterTempo: false,
    peaks: null, cue: 0, hotCues: [null, null, null, null],
    loopIn: null, loopOut: null, loopActive: false,
  };
}

// ═══════════════════════════════════════════════════════════════
// WAVEFORM RENDERERS
// ═══════════════════════════════════════════════════════════════

function drawOverview(canvas: HTMLCanvasElement | null, st: DeckState): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0c0c10'; ctx.fillRect(0, 0, W, H);
  if (!st.peaks) {
    ctx.fillStyle = C.dim; ctx.font = "11px 'IBM Plex Mono'";
    ctx.fillText('DRAG TRACK OR TAP LOAD', W * 0.1, H / 2 + 4);
    return;
  }
  const prog = st.dur ? st.pos / st.dur : 0;
  const n = st.peaks.length;
  for (let i = 0; i < n; i++) {
    const x = (i / n) * W, h = st.peaks[i] * (H * 0.44);
    ctx.fillStyle = (i / n < prog) ? C.cyan : C.cyanDim;
    ctx.fillRect(x, H / 2 - h, Math.max(1, W / n), h * 2);
  }
  if (st.loopIn !== null && st.loopOut !== null && st.dur > 0) {
    const x1 = (st.loopIn / st.dur) * W, x2 = (st.loopOut / st.dur) * W;
    ctx.fillStyle = st.loopActive ? 'rgba(60,224,138,0.12)' : 'rgba(60,224,138,0.05)';
    ctx.fillRect(x1, 0, x2 - x1, H);
    ctx.fillStyle = C.green; ctx.fillRect(x1 - 1, 0, 2, H); ctx.fillRect(x2 - 1, 0, 2, H);
  }
  st.hotCues.forEach((hc, i) => {
    if (hc !== null && st.dur > 0) {
      ctx.fillStyle = HC_COLORS[i]; ctx.fillRect((hc / st.dur) * W - 1, 0, 2, H);
    }
  });
  if (st.dur > 0) { ctx.fillStyle = C.orange; ctx.fillRect((st.cue / st.dur) * W - 1, 0, 2, H); }
  ctx.fillStyle = '#fff'; ctx.fillRect(prog * W - 1, 0, 2, H);
}

function drawZoom(canvas: HTMLCanvasElement | null, buffer: AudioBuffer | null, pos: number, dur: number, st: DeckState): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0a0e'; ctx.fillRect(0, 0, W, H);
  if (!buffer || dur <= 0) return;

  const windowSec = 6;
  const startSec = pos - windowSec / 2;
  const sr = buffer.sampleRate;
  const ch0 = buffer.getChannelData(0);
  const samplesPerPx = Math.max(1, Math.round((windowSec * sr) / W));

  for (let px = 0; px < W; px++) {
    const t = startSec + (px / W) * windowSec;
    const si = Math.floor(t * sr);
    if (si < 0 || si + samplesPerPx >= ch0.length) continue;
    let max = 0;
    for (let j = 0; j < samplesPerPx; j++) max = Math.max(max, Math.abs(ch0[si + j]));
    const h = max * H * 0.42;
    ctx.fillStyle = t < pos ? C.cyan : C.cyanDim;
    ctx.fillRect(px, H / 2 - h, 1, h * 2);
  }

  // Loop region
  if (st.loopIn !== null && st.loopOut !== null) {
    const lx1 = ((st.loopIn - startSec) / windowSec) * W;
    const lx2 = ((st.loopOut - startSec) / windowSec) * W;
    if (lx2 > 0 && lx1 < W) {
      ctx.fillStyle = st.loopActive ? 'rgba(60,224,138,0.1)' : 'rgba(60,224,138,0.04)';
      ctx.fillRect(Math.max(0, lx1), 0, Math.min(W, lx2) - Math.max(0, lx1), H);
    }
  }
  // Hot cue + cue markers
  st.hotCues.forEach((hc, i) => {
    if (hc !== null) {
      const hx = ((hc - startSec) / windowSec) * W;
      if (hx >= -2 && hx <= W + 2) { ctx.fillStyle = HC_COLORS[i]; ctx.fillRect(hx - 1, 0, 2, H); }
    }
  });
  const cueX = ((st.cue - startSec) / windowSec) * W;
  if (cueX >= -2 && cueX <= W + 2) { ctx.fillStyle = C.orange; ctx.fillRect(cueX - 1, 0, 2, H); }
  // Playhead (center)
  ctx.fillStyle = '#ffffff'; ctx.fillRect(W / 2 - 1, 0, 2, H);
  // Beat grid ticks
  const bpm = parseFloat(st.bpm);
  if (bpm > 0) {
    const beatSec = 60 / bpm;
    const firstBeat = Math.ceil(startSec / beatSec) * beatSec;
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    for (let t = firstBeat; t < startSec + windowSec; t += beatSec) {
      const bx = ((t - startSec) / windowSec) * W;
      ctx.fillRect(bx, 0, 1, H);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// CONTROLS — Touch-perfect with setPointerCapture
// ═══════════════════════════════════════════════════════════════

interface KnobProps {
  value: number; onChange: (v: number) => void; label: string;
  color?: string; size?: number; hint?: string; onHint?: (h: string) => void;
}

function Knob({ value, onChange, label, color = C.cyan, size = 48, hint, onHint }: KnobProps) {
  const drag = useRef<{ y: number; v: number; id: number } | null>(null);
  const angle = -135 + value * 270;
  return (
    <div className="flex flex-col items-center select-none" style={{ width: size + 8 }}
      onMouseEnter={() => hint && onHint?.(hint)}>
      <div className="rounded-full relative cursor-ns-resize"
        style={{
          width: size, height: size, touchAction: 'none',
          background: 'radial-gradient(circle at 35% 30%, #3a3a44, #141418 70%)',
          border: `1px solid ${C.edge}`,
          boxShadow: 'inset 0 2px 4px rgba(0,0,0,.6), 0 1px 2px rgba(0,0,0,.5)',
        }}
        onPointerDown={e => {
          e.preventDefault(); e.stopPropagation();
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
          drag.current = { y: e.clientY, v: value, id: e.pointerId };
        }}
        onPointerMove={e => {
          if (!drag.current || e.pointerId !== drag.current.id) return;
          onChange(Math.max(0, Math.min(1, drag.current.v + (drag.current.y - e.clientY) / 140)));
        }}
        onPointerUp={e => { if (drag.current?.id === e.pointerId) drag.current = null; }}
        onPointerCancel={e => { if (drag.current?.id === e.pointerId) drag.current = null; }}
        onDoubleClick={() => onChange(0.5)}
        title={label}>
        <div className="absolute left-1/2 top-1/2" style={{
          width: 2, height: size * 0.4, background: color, borderRadius: 2,
          transform: `translate(-50%,-100%) rotate(${angle}deg)`,
          transformOrigin: 'bottom center', boxShadow: `0 0 4px ${color}`,
        }} />
      </div>
      <span style={{ color: C.dim, fontSize: 9, marginTop: 3, letterSpacing: 1, fontFamily: 'Oxanium' }}>{label}</span>
    </div>
  );
}

interface FaderProps {
  value: number; onChange: (v: number) => void; label?: string;
  color?: string; height?: number; center?: boolean; hint?: string; onHint?: (h: string) => void;
}

function Fader({ value, onChange, label, color = C.cyan, height = 130, center, hint, onHint }: FaderProps) {
  const drag = useRef<{ y: number; v: number; id: number } | null>(null);
  return (
    <div className="flex flex-col items-center select-none" onMouseEnter={() => hint && onHint?.(hint)}>
      <div className="relative" style={{ height, width: 36, touchAction: 'none' }}
        onPointerDown={e => {
          e.preventDefault(); e.stopPropagation();
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
          drag.current = { y: e.clientY, v: value, id: e.pointerId };
        }}
        onPointerMove={e => {
          if (!drag.current || e.pointerId !== drag.current.id) return;
          onChange(Math.max(0, Math.min(1, drag.current.v + (drag.current.y - e.clientY) / height)));
        }}
        onPointerUp={e => { if (drag.current?.id === e.pointerId) drag.current = null; }}
        onPointerCancel={e => { if (drag.current?.id === e.pointerId) drag.current = null; }}>
        <div className="absolute left-1/2 top-0 bottom-0" style={{
          width: 4, transform: 'translateX(-50%)', background: '#0a0a0c', borderRadius: 3, border: `1px solid ${C.edge}`,
        }} />
        {center && <div className="absolute left-1/2 top-1/2" style={{ width: 14, height: 1, background: C.dim, transform: 'translate(-50%,-50%)' }} />}
        <div className="absolute left-1/2" style={{
          width: 34, height: 22, transform: 'translateX(-50%)',
          bottom: `calc(${value * 100}% - 11px)`,
          background: 'linear-gradient(180deg,#48484f,#1a1a1e)', borderRadius: 4,
          border: `1px solid ${C.edge}`, boxShadow: '0 0 5px rgba(0,0,0,.6)',
        }}>
          <div style={{ height: 2, background: color, margin: '9px 4px', borderRadius: 2, boxShadow: `0 0 4px ${color}` }} />
        </div>
      </div>
      {label && <span style={{ color: C.dim, fontSize: 9, marginTop: 4, letterSpacing: 1, fontFamily: 'Oxanium' }}>{label}</span>}
    </div>
  );
}

function VUMeter({ level }: { level: number }) {
  const segs = 14, lit = Math.round(level * segs);
  return (
    <div className="flex flex-col-reverse gap-0.5">
      {Array.from({ length: segs }).map((_, i) => {
        const on = i < lit;
        const col = i > segs * 0.85 ? C.red : i > segs * 0.65 ? C.yellow : C.green;
        return <div key={i} style={{ width: 7, height: 5, borderRadius: 1, background: on ? col : '#1d1d22', boxShadow: on ? `0 0 3px ${col}` : 'none' }} />;
      })}
    </div>
  );
}

function CrossfaderH({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const trackRef = useRef<HTMLDivElement>(null);
  const dragging = useRef<number | null>(null);
  return (
    <div ref={trackRef} className="relative cursor-ew-resize" style={{ height: 36, touchAction: 'none' }}
      onPointerDown={e => {
        e.preventDefault();
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        dragging.current = e.pointerId;
        if (trackRef.current) {
          const r = trackRef.current.getBoundingClientRect();
          onChange(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
        }
      }}
      onPointerMove={e => {
        if (dragging.current !== e.pointerId || !trackRef.current) return;
        const r = trackRef.current.getBoundingClientRect();
        onChange(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
      }}
      onPointerUp={e => { if (dragging.current === e.pointerId) dragging.current = null; }}
      onPointerCancel={e => { if (dragging.current === e.pointerId) dragging.current = null; }}>
      <div className="absolute top-1/2 left-0 right-0" style={{ height: 6, transform: 'translateY(-50%)', background: '#0a0a0c', borderRadius: 3, border: `1px solid ${C.edge}` }} />
      <div className="absolute top-1/2" style={{
        width: 28, height: 32, transform: 'translate(-50%,-50%)', left: `${value * 100}%`,
        background: 'linear-gradient(180deg,#48484f,#1a1a1e)', borderRadius: 4,
        border: `1px solid ${C.edge}`, boxShadow: '0 0 5px rgba(0,0,0,.6)',
      }}>
        <div style={{ width: 2, height: 22, background: C.cyan, margin: '5px auto', borderRadius: 2, boxShadow: `0 0 4px ${C.cyan}` }} />
      </div>
    </div>
  );
}

function BeatIndicator({ pos, bpm }: { pos: number; bpm: number }) {
  if (!bpm || bpm <= 0) return null;
  const beatInBar = ((pos * bpm / 60) % 4);
  const current = Math.floor(beatInBar);
  return (
    <div className="flex gap-1">
      {[0, 1, 2, 3].map(i => (
        <div key={i} style={{
          width: 12, height: 12, borderRadius: 2,
          background: i === current ? (i === 0 ? C.orange : C.cyan) : '#1a1a1e',
          border: `1px solid ${i === current ? (i === 0 ? C.orange : C.cyan) : C.edge}`,
          boxShadow: i === current ? `0 0 6px ${i === 0 ? C.orange : C.cyan}55` : 'none',
          transition: 'background 0.05s',
        }} />
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════

export default function DJTrainer() {
  // ─── refs ───
  const audio = useRef<AudioEngine | null>(null);
  const cvsA = useRef<HTMLCanvasElement>(null);
  const cvsB = useRef<HTMLCanvasElement>(null);
  const zoomA = useRef<HTMLCanvasElement>(null);
  const zoomB = useRef<HTMLCanvasElement>(null);
  const fileRefA = useRef<HTMLInputElement>(null);
  const fileRefB = useRef<HTMLInputElement>(null);
  const platterRefA = useRef<HTMLDivElement>(null);
  const platterRefB = useRef<HTMLDivElement>(null);
  const platterAngle = useRef<Record<DeckId, number>>({ A: 0, B: 0 });
  const deckARef = useRef<DeckState>(mkDeckState());
  const deckBRef = useRef<DeckState>(mkDeckState());

  // ─── state ───
  const [initialized, setInitialized] = useState(false);
  const [learn, setLearn] = useState(true);
  const [hint, setHint] = useState<string | null>(null);
  const [guideStep, setGuideStep] = useState(0);
  const [showGuide, setShowGuide] = useState(false);
  const [showKeys, setShowKeys] = useState(false);
  const [deckA, setDeckA] = useState<DeckState>(mkDeckState());
  const [deckB, setDeckB] = useState<DeckState>(mkDeckState());
  const [mix, setMix] = useState<MixerState>({
    xfader: 0.5,
    chA: { trim: 0.7, hi: 0.5, mid: 0.5, low: 0.5, fader: 0.85, color: 0.5 },
    chB: { trim: 0.7, hi: 0.5, mid: 0.5, low: 0.5, fader: 0.85, color: 0.5 },
    master: 0.8,
  });
  const [vu, setVu] = useState({ A: 0, B: 0, M: 0 });
  const [mobileTab, setMobileTab] = useState<'A' | 'mix' | 'B'>('A');
  const [screenW, setScreenW] = useState(1200);

  const setters: Record<DeckId, React.Dispatch<React.SetStateAction<DeckState>>> = { A: setDeckA, B: setDeckB };
  const isMobile = screenW < 768;
  const isCompact = screenW < 1100;

  // Keep refs in sync with state for rAF access
  useEffect(() => { deckARef.current = deckA; }, [deckA]);
  useEffect(() => { deckBRef.current = deckB; }, [deckB]);

  // Track screen size
  useEffect(() => {
    const up = () => setScreenW(window.innerWidth);
    up();
    window.addEventListener('resize', up);
    return () => window.removeEventListener('resize', up);
  }, []);

  // Lock body scroll when booth is active
  useEffect(() => {
    if (initialized) document.body.classList.add('booth-active');
    return () => { document.body.classList.remove('booth-active'); };
  }, [initialized]);

  // ─── audio init ───
  const initAudio = useCallback(() => {
    if (audio.current) return;
    const ctx = new AudioContext();
    const master = ctx.createGain(); master.gain.value = 0.8;
    const masterAnalyser = ctx.createAnalyser(); masterAnalyser.fftSize = 256;
    master.connect(masterAnalyser); masterAnalyser.connect(ctx.destination);
    const mkAudioDeck = (): AudioDeck => {
      const trim = ctx.createGain(); trim.gain.value = 0.7;
      const low = ctx.createBiquadFilter(); low.type = 'lowshelf'; low.frequency.value = 200;
      const mid = ctx.createBiquadFilter(); mid.type = 'peaking'; mid.frequency.value = 1000; mid.Q.value = 0.9;
      const hi = ctx.createBiquadFilter(); hi.type = 'highshelf'; hi.frequency.value = 3500;
      const color = ctx.createBiquadFilter(); color.type = 'lowpass'; color.frequency.value = 22000;
      const chGain = ctx.createGain(); chGain.gain.value = 0.85;
      const analyser = ctx.createAnalyser(); analyser.fftSize = 256;
      const xf = ctx.createGain(); xf.gain.value = 0.5;
      trim.connect(low); low.connect(mid); mid.connect(hi); hi.connect(color);
      color.connect(chGain); chGain.connect(analyser); analyser.connect(xf); xf.connect(master);
      return {
        trim, low, mid, hi, color, chGain, analyser, xf,
        src: null, buffer: null, startTime: 0, startOffset: 0,
        rate: 1, effRate: 1, playing: false, stopping: false, braking: false,
        cue: 0, dur: 0, hotCues: [null, null, null, null],
        loopIn: null, loopOut: null, loopActive: false, cuePreview: false,
      };
    };
    audio.current = { ctx, master, masterAnalyser, decks: { A: mkAudioDeck(), B: mkAudioDeck() } };
    setInitialized(true);
  }, []);

  const ensure = useCallback((): AudioEngine => {
    if (!audio.current) throw new Error('Audio not initialized');
    audio.current.ctx.resume();
    return audio.current;
  }, []);

  // ─── position helpers ───
  const posOf = (d: AudioDeck): number =>
    d.playing ? d.startOffset + (audio.current!.ctx.currentTime - d.startTime) * d.effRate : d.startOffset;
  const rebase = (d: AudioDeck): void => { d.startOffset = posOf(d); d.startTime = audio.current!.ctx.currentTime; };

  // ─── source management ───
  const startSrc = (a: AudioEngine, d: AudioDeck, id: DeckId): void => {
    const src = a.ctx.createBufferSource();
    src.buffer = d.buffer; src.playbackRate.value = d.effRate; src.connect(d.trim);
    src.onended = () => { if (d.stopping) { d.stopping = false; return; } d.playing = false; setters[id](s => ({ ...s, playing: false })); };
    src.start(0, Math.max(0, Math.min(d.dur, d.startOffset)));
    d.src = src; d.startTime = a.ctx.currentTime; d.playing = true;
  };

  // ─── file loading with auto-BPM ───
  const loadFile = async (id: DeckId, file: File): Promise<void> => {
    const a = ensure();
    const arr = await file.arrayBuffer();
    const buf = await a.ctx.decodeAudioData(arr);
    const d = a.decks[id];
    if (d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } d.playing = false; }
    d.buffer = buf; d.dur = buf.duration; d.startOffset = 0; d.cue = 0;
    d.hotCues = [null, null, null, null]; d.loopIn = null; d.loopOut = null; d.loopActive = false; d.cuePreview = false; d.braking = false;
    const peaks = buildPeaks(buf);
    const bpm = detectBPM(buf);
    setters[id](() => ({
      ...mkDeckState(), name: file.name.replace(/\.[^.]+$/, ''),
      loaded: true, dur: buf.duration, peaks, bpm: bpm ? bpm.toFixed(1) : '',
    }));
  };

  // ─── transport ───
  const playDeck = (id: DeckId): void => {
    const a = ensure(); const d = a.decks[id];
    if (!d.buffer) return;
    if (d.cuePreview) { d.cuePreview = false; return; }
    if (d.playing) return;
    if (d.braking) { d.braking = false; } // cancel brake
    if (d.startOffset >= d.dur) d.startOffset = 0;
    d.effRate = d.rate; // restore rate after any brake
    startSrc(a, d, id);
    setters[id](s => ({ ...s, playing: true }));
  };

  const pauseDeck = (id: DeckId): void => {
    const a = ensure(); const d = a.decks[id];
    if (!d.playing || d.braking) return;
    // Vinyl brake: ramp rate to 0
    rebase(d);
    d.braking = true;
    if (d.src) {
      d.src.playbackRate.setValueAtTime(d.effRate, a.ctx.currentTime);
      d.src.playbackRate.linearRampToValueAtTime(0.001, a.ctx.currentTime + 0.25);
    }
    setTimeout(() => {
      if (!d.braking) return; // was cancelled by play
      const p = d.startOffset + d.effRate * 0.125; // approx midpoint of decel
      d.stopping = true;
      try { d.src?.stop(); } catch { /* ok */ }
      d.startOffset = Math.max(0, Math.min(d.dur, p));
      d.playing = false; d.cuePreview = false; d.braking = false;
      d.effRate = d.rate;
      setters[id](s => ({ ...s, playing: false, pos: d.startOffset }));
    }, 260);
  };

  const seekDeck = (id: DeckId, p: number): void => {
    const a = ensure(); const d = a.decks[id];
    p = Math.max(0, Math.min(d.dur, p));
    if (d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } d.startOffset = p; startSrc(a, d, id); }
    else { d.startOffset = p; }
    setters[id](s => ({ ...s, pos: p }));
  };

  const cuePress = (id: DeckId): void => {
    const a = ensure(); const d = a.decks[id]; if (!d.buffer) return;
    if (d.playing) {
      d.stopping = true; try { d.src?.stop(); } catch { /* ok */ }
      d.startOffset = d.cue; d.playing = false; d.cuePreview = false; d.braking = false;
      d.effRate = d.rate;
      setters[id](s => ({ ...s, playing: false, pos: d.cue }));
    } else {
      d.cue = d.startOffset; d.cuePreview = true;
      d.effRate = d.rate;
      startSrc(a, d, id);
      setters[id](s => ({ ...s, cue: d.cue, playing: true }));
    }
  };

  const cueRelease = (id: DeckId): void => {
    if (!audio.current) return;
    const d = audio.current.decks[id];
    if (d.cuePreview && d.playing) {
      d.stopping = true; try { d.src?.stop(); } catch { /* ok */ }
      d.startOffset = d.cue; d.playing = false; d.cuePreview = false;
      setters[id](s => ({ ...s, playing: false, pos: d.cue }));
    }
  };

  // ─── tempo (CDJ convention: fader DOWN = faster, UP = slower) ───
  const tempoPercent = (v: number, range: number) => (0.5 - v) * 2 * range;
  const setTempo = (id: DeckId, v: number): void => {
    if (!audio.current) return;
    const d = audio.current.decks[id];
    const range = id === 'A' ? deckA.range : deckB.range;
    const rate = 1 + tempoPercent(v, range) / 100;
    if (d.playing) rebase(d);
    d.rate = rate; d.effRate = rate;
    if (d.src && !d.braking) d.src.playbackRate.value = rate;
    setters[id](s => ({ ...s, tempo: v }));
  };

  // ─── jog wheel (touch-perfect with setPointerCapture) ───
  const jogDown = (id: DeckId, e: React.PointerEvent): void => {
    const a = ensure(); const d = a.decks[id]; if (!d.buffer) return;
    const el = e.currentTarget as HTMLElement;
    el.setPointerCapture(e.pointerId);
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
    let lastAng = Math.atan2(e.clientY - cy, e.clientX - cx);
    const pid = e.pointerId;

    const mv = (ev: PointerEvent) => {
      if (ev.pointerId !== pid) return;
      const na = Math.atan2(ev.clientY - cy, ev.clientX - cx);
      let dA = na - lastAng;
      if (dA > Math.PI) dA -= 2 * Math.PI;
      if (dA < -Math.PI) dA += 2 * Math.PI;
      lastAng = na;
      if (d.playing) {
        rebase(d);
        d.effRate = d.rate * (1 + Math.max(-0.5, Math.min(0.5, dA * 2.2)));
        if (d.src && !d.braking) d.src.playbackRate.value = d.effRate;
      } else {
        d.startOffset = Math.max(0, Math.min(d.dur, d.startOffset + (dA / (2 * Math.PI)) * 4));
        setters[id](s => ({ ...s, pos: d.startOffset }));
      }
    };
    const up = (ev: PointerEvent) => {
      if (ev.pointerId !== pid) return;
      if (d.playing) { rebase(d); d.effRate = d.rate; if (d.src && !d.braking) d.src.playbackRate.value = d.rate; }
      el.removeEventListener('pointermove', mv);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
    };
    el.addEventListener('pointermove', mv);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
  };

  // ─── hot cues ───
  const hotCuePress = (id: DeckId, idx: number): void => {
    const a = ensure(); const d = a.decks[id]; if (!d.buffer) return;
    if (d.hotCues[idx] === null) {
      const p = posOf(d); d.hotCues[idx] = p;
      setters[id](s => { const hc = [...s.hotCues]; hc[idx] = p; return { ...s, hotCues: hc }; });
    } else {
      const target = d.hotCues[idx]!;
      if (d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } }
      d.startOffset = target; d.cuePreview = false; d.braking = false; d.effRate = d.rate;
      startSrc(a, d, id);
      setters[id](s => ({ ...s, playing: true, pos: target }));
    }
  };
  const hotCueClear = (id: DeckId, idx: number): void => {
    if (!audio.current) return;
    audio.current.decks[id].hotCues[idx] = null;
    setters[id](s => { const hc = [...s.hotCues]; hc[idx] = null; return { ...s, hotCues: hc }; });
  };

  // ─── loops ───
  const setLoopInPoint = (id: DeckId): void => {
    if (!audio.current) return; const d = audio.current.decks[id]; if (!d.buffer) return;
    const p = posOf(d); d.loopIn = p; setters[id](s => ({ ...s, loopIn: p }));
  };
  const setLoopOutPoint = (id: DeckId): void => {
    if (!audio.current) return; const d = audio.current.decks[id]; if (!d.buffer) return;
    const p = posOf(d); d.loopOut = p; d.loopActive = true;
    setters[id](s => ({ ...s, loopOut: p, loopActive: true }));
  };
  const toggleLoop = (id: DeckId): void => {
    if (!audio.current) return; const d = audio.current.decks[id];
    if (d.loopIn === null || d.loopOut === null) return;
    d.loopActive = !d.loopActive;
    setters[id](s => ({ ...s, loopActive: !s.loopActive }));
  };

  // ─── beat jump ───
  const beatJump = (id: DeckId, beats: number): void => {
    if (!audio.current) return;
    const d = audio.current.decks[id];
    const st = id === 'A' ? deckARef.current : deckBRef.current;
    const bpm = parseFloat(st.bpm);
    if (!bpm || !d.buffer) return;
    seekDeck(id, posOf(d) + beats * (60 / bpm));
  };

  // ─── bpm calc (matches inverted CDJ tempo fader) ───
  const adjBpm = (st: DeckState): number | null => {
    const b = parseFloat(st.bpm);
    if (!b || isNaN(b)) return null;
    return b * (1 + tempoPercent(st.tempo, st.range) / 100);
  };

  // ─── effect: overview waveforms ───
  useEffect(() => { drawOverview(cvsA.current, deckA); },
    [deckA.peaks, deckA.pos, deckA.dur, deckA.cue, deckA.hotCues, deckA.loopIn, deckA.loopOut, deckA.loopActive]);
  useEffect(() => { drawOverview(cvsB.current, deckB); },
    [deckB.peaks, deckB.pos, deckB.dur, deckB.cue, deckB.hotCues, deckB.loopIn, deckB.loopOut, deckB.loopActive]);

  // ─── effect: mixer ───
  useEffect(() => {
    if (!audio.current) return;
    const a = audio.current;
    const eq = (val: number) => (val - 0.5) * 2 * 26 * (val < 0.5 ? 1.6 : 0.23); // kill=-41dB, boost=+6dB (DJM-900 spec)
    const apply = (id: DeckId, ch: ChannelState) => {
      const d = a.decks[id];
      d.trim.gain.value = ch.trim * 1.4;
      d.low.gain.value = eq(ch.low); d.mid.gain.value = eq(ch.mid); d.hi.gain.value = eq(ch.hi);
      if (Math.abs(ch.color - 0.5) < 0.04) { d.color.type = 'lowpass'; d.color.frequency.value = 22000; }
      else if (ch.color < 0.5) { d.color.type = 'lowpass'; d.color.frequency.value = 200 + Math.pow(ch.color / 0.5, 2) * 12000; }
      else { d.color.type = 'highpass'; d.color.frequency.value = 30 + Math.pow((ch.color - 0.5) / 0.5, 2) * 7000; }
      d.chGain.gain.value = ch.fader;
    };
    apply('A', mix.chA); apply('B', mix.chB);
    a.decks.A.xf.gain.value = Math.cos(mix.xfader * Math.PI / 2);
    a.decks.B.xf.gain.value = Math.cos((1 - mix.xfader) * Math.PI / 2);
    a.master.gain.value = mix.master;
  }, [mix, initialized]);

  // ─── effect: animation frame ───
  useEffect(() => {
    if (!initialized) return;
    let raf: number;
    const buf = new Uint8Array(128);
    const rms = (an: AnalyserNode): number => {
      an.getByteTimeDomainData(buf); let s = 0;
      for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; s += v * v; }
      return Math.min(1, Math.sqrt(s / buf.length) * 2.2);
    };
    const tick = () => {
      const a = audio.current;
      if (a) {
        setVu({ A: rms(a.decks.A.analyser), B: rms(a.decks.B.analyser), M: rms(a.masterAnalyser) });
        (['A', 'B'] as DeckId[]).forEach(id => {
          const d = a.decks[id];
          const el = id === 'A' ? platterRefA.current : platterRefB.current;
          if (el) {
            if (d.playing && !d.braking) platterAngle.current[id] += d.effRate * 3.33;
            el.style.transform = `rotate(${platterAngle.current[id]}deg)`;
          }
          if (d.playing) {
            const p = posOf(d);
            if (d.loopActive && d.loopIn !== null && d.loopOut !== null && p >= d.loopOut) {
              d.stopping = true; try { d.src?.stop(); } catch { /* ok */ }
              d.startOffset = d.loopIn; startSrc(a, d, id);
            }
            setters[id](s => (Math.abs(s.pos - p) > 0.02 ? { ...s, pos: p } : s));
          }
          // Draw zoom waveform from buffer
          const zEl = id === 'A' ? zoomA.current : zoomB.current;
          const st = id === 'A' ? deckARef.current : deckBRef.current;
          if (zEl && d.buffer) drawZoom(zEl, d.buffer, d.playing ? posOf(d) : d.startOffset, d.dur, st);
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialized]);

  // ─── effect: keyboard ───
  useEffect(() => {
    if (!initialized) return;
    const onDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      switch (e.key) {
        case 'q': e.preventDefault(); cuePress('A'); break;
        case 'w': e.preventDefault(); if (!e.repeat) { audio.current?.decks.A.playing ? pauseDeck('A') : playDeck('A'); } break;
        case '1': hotCuePress('A', 0); break; case '2': hotCuePress('A', 1); break;
        case '3': hotCuePress('A', 2); break; case '4': hotCuePress('A', 3); break;
        case 'a': setLoopInPoint('A'); break;
        case 's': e.preventDefault(); setLoopOutPoint('A'); break;
        case 'd': e.preventDefault(); toggleLoop('A'); break;
        case 'z': e.preventDefault(); beatJump('A', -4); break;
        case 'x': e.preventDefault(); beatJump('A', -1); break;
        case 'c': e.preventDefault(); beatJump('A', 1); break;
        case 'v': e.preventDefault(); beatJump('A', 4); break;
        case '[': e.preventDefault(); cuePress('B'); break;
        case ']': e.preventDefault(); if (!e.repeat) { audio.current?.decks.B.playing ? pauseDeck('B') : playDeck('B'); } break;
        case '7': hotCuePress('B', 0); break; case '8': hotCuePress('B', 1); break;
        case '9': hotCuePress('B', 2); break; case '0': hotCuePress('B', 3); break;
        case 'l': setLoopInPoint('B'); break;
        case ';': setLoopOutPoint('B'); break;
        case "'": toggleLoop('B'); break;
        case ',': beatJump('B', -4); break; case '.': beatJump('B', -1); break;
        case '/': e.preventDefault(); beatJump('B', 1); break;
      }
    };
    const onUp = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === 'q') cueRelease('A');
      if (e.key === '[') cueRelease('B');
    };
    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    return () => { window.removeEventListener('keydown', onDown); window.removeEventListener('keyup', onUp); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialized]);

  // ═══════════════════════════════════════════════════════════
  // INIT + PORTRAIT BLOCK
  // ═══════════════════════════════════════════════════════════

  if (!initialized) {
    return (
      <div className="flex flex-col items-center justify-center select-none"
        style={{ background: C.bg, minHeight: '100dvh', fontFamily: 'Oxanium', color: C.text, padding: 24 }}>
        <div style={{ fontWeight: 700, fontSize: 'clamp(28px, 6vw, 42px)', letterSpacing: 6, textAlign: 'center' }}>
          VIRTUAL <span style={{ color: C.cyan }}>BOOTH</span>
        </div>
        <div style={{ color: C.dim, fontSize: 'clamp(10px, 2vw, 14px)', fontFamily: "'IBM Plex Mono'", marginTop: 8, letterSpacing: 2, textAlign: 'center' }}>
          CDJ-2000NXS &times;2 &middot; DJM-900NXS
        </div>
        <div style={{ marginTop: 16, color: C.dim, fontSize: 12, fontFamily: "'IBM Plex Mono'", textAlign: 'center', lineHeight: 1.6 }}>
          Auto-BPM detection &middot; Zoomed waveform &middot; Beat counter<br />
          Vinyl brake &middot; Beat jump &middot; Works on phone, iPad &amp; desktop
        </div>
        <button onClick={initAudio} className="touch-target" style={{
          marginTop: 40, padding: '16px 48px', borderRadius: 8,
          background: 'transparent', border: `2px solid ${C.cyan}`, color: C.cyan,
          fontFamily: 'Oxanium', fontWeight: 700, fontSize: 16, letterSpacing: 4, cursor: 'pointer',
        }}>START SESSION</button>
        <div style={{ marginTop: 32, color: C.dim, fontSize: 11, fontFamily: "'IBM Plex Mono'", textAlign: 'center', maxWidth: 440, lineHeight: 1.7 }}>
          Load your own tracks (MP3 / WAV / FLAC / OGG).<br />
          On mobile, rotate to landscape for the full rig.
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════

  const showHint = (t: string) => { if (learn) setHint(t); };
  const bpmA = adjBpm(deckA), bpmB = adjBpm(deckB);
  const matched = bpmA !== null && bpmB !== null && Math.abs(bpmA - bpmB) < 0.15;
  const jogSize = isMobile ? 130 : 160;
  const knobSize = isMobile ? 38 : 46;
  const faderH = isMobile ? 90 : 115;

  // ─── deck render ───
  const renderDeck = (id: DeckId) => {
    const st = id === 'A' ? deckA : deckB;
    const cvsRef = id === 'A' ? cvsA : cvsB;
    const zoomRef = id === 'A' ? zoomA : zoomB;
    const fileRef = id === 'A' ? fileRefA : fileRefB;
    const plRef = id === 'A' ? platterRefA : platterRefB;
    const remaining = (st.dur || 0) - st.pos;
    const endWarn = st.loaded && st.playing && remaining < 30;
    const endCrit = endWarn && remaining < 10;
    const deckColor = id === 'A' ? C.cyan : C.orange;
    const bpmNum = parseFloat(st.bpm) || 0;

    return (
      <div className="rounded-lg p-2 flex flex-col gap-1.5" style={{ background: C.panel, border: `1px solid ${C.edge}`, flex: 1, minWidth: isMobile ? 0 : 280 }}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span style={{ fontFamily: 'Oxanium', fontWeight: 700, color: deckColor, letterSpacing: 2, fontSize: 12 }}>DECK {id}</span>
            <BeatIndicator pos={st.pos} bpm={bpmNum} />
          </div>
          <span style={{ fontFamily: "'IBM Plex Mono'", color: endCrit ? C.red : endWarn ? C.orange : C.dim, fontSize: 10 }}
            className={endCrit ? 'warning-flash' : ''}>
            {st.loaded ? fmtTime(st.pos) : '--:--'} / -{fmtTime(remaining)}
          </span>
        </div>

        {/* Zoomed waveform (main display) */}
        <div className="rounded overflow-hidden" style={{ background: '#0a0a0e', border: `1px solid ${C.edge}` }}>
          <canvas ref={zoomRef} width={400} height={60}
            style={{ width: '100%', height: isMobile ? 44 : 56, display: 'block' }} />
        </div>

        {/* Overview waveform (mini) */}
        <div className="rounded overflow-hidden" style={{ background: '#0c0c10', border: `1px solid ${C.edge}`, cursor: 'pointer' }}
          onMouseEnter={() => showHint('Overview waveform. Tap anywhere to jump to that position.')}>
          <canvas ref={cvsRef} width={400} height={32}
            style={{ width: '100%', height: isMobile ? 22 : 28, display: 'block' }}
            onClick={e => {
              const rect = e.currentTarget.getBoundingClientRect();
              seekDeck(id, ((e.clientX - rect.left) / rect.width) * st.dur);
            }}
            onPointerDown={e => {
              // Touch seek
              if (e.pointerType === 'touch') {
                const rect = e.currentTarget.getBoundingClientRect();
                seekDeck(id, ((e.clientX - rect.left) / rect.width) * st.dur);
              }
            }} />
          <div className="flex items-center justify-center px-1" style={{ fontFamily: "'IBM Plex Mono'", fontSize: 9, color: C.dim, height: 14 }}>
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{st.name || 'No track loaded'}</span>
          </div>
        </div>

        {/* Jog wheel */}
        <div className="flex items-center justify-center">
          <div
            onPointerDown={e => { e.preventDefault(); jogDown(id, e); }}
            onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadFile(id, f); }}
            onDragOver={e => e.preventDefault()}
            onMouseEnter={() => showHint('JOG WHEEL: Playing \u2192 nudge to pitch-bend. Paused \u2192 spin to scrub. Drag a file here to load.')}
            className="rounded-full relative cursor-grab active:cursor-grabbing select-none flex items-center justify-center"
            style={{
              width: jogSize, height: jogSize, touchAction: 'none',
              background: 'radial-gradient(circle at 50% 45%, #2c2c34, #16161a 55%, #0c0c0e)',
              border: '3px solid #1a1a1e',
              boxShadow: '0 4px 16px rgba(0,0,0,.5), inset 0 0 24px rgba(0,0,0,.6)',
            }}>
            <div className="absolute rounded-full" style={{ inset: 6, border: '1px dashed #2a2a32' }} />
            <div ref={plRef} className="rounded-full relative flex items-center justify-center"
              style={{
                width: jogSize * 0.64, height: jogSize * 0.64,
                background: 'radial-gradient(circle, #3a3a44, #1c1c22)',
                border: `1px solid ${C.edge}`,
              }}>
              <div className="absolute" style={{ width: jogSize * 0.53, height: 2, background: st.playing ? deckColor : C.dim, opacity: 0.5 }} />
              <div className="rounded-full" style={{
                width: jogSize * 0.2, height: jogSize * 0.2, background: '#0c0c10',
                border: `2px solid ${st.playing ? deckColor : C.edge}`,
                boxShadow: st.playing ? `0 0 8px ${deckColor}` : 'none',
              }} />
            </div>
          </div>
        </div>

        {/* Transport */}
        <div className="flex items-center gap-2 justify-center">
          <button onPointerDown={() => cuePress(id)} onPointerUp={() => cueRelease(id)} onPointerLeave={() => cueRelease(id)}
            onMouseEnter={() => showHint('CUE: Paused\u2192set cue + preview (hold). Playing\u2192snap to cue.')}
            className="rounded-full flex items-center justify-center touch-target"
            style={{ width: 48, height: 48,
              background: !st.playing ? `${C.orange}18` : '#1a1a1e',
              border: `2px solid ${C.orange}`, color: C.orange,
              fontFamily: 'Oxanium', fontWeight: 700, fontSize: 11,
              boxShadow: !st.playing && st.loaded ? `0 0 12px ${C.orange}66` : `0 0 4px ${C.orange}22`,
            }}>CUE</button>
          <button onPointerDown={() => st.playing ? pauseDeck(id) : playDeck(id)}
            onMouseEnter={() => showHint('PLAY/PAUSE with vinyl brake effect.')}
            className="rounded-full flex items-center justify-center touch-target"
            style={{ width: 48, height: 48,
              background: st.playing ? `${C.green}18` : '#1a1a1e',
              border: `2px solid ${C.green}`, color: C.green,
              fontFamily: 'Oxanium', fontWeight: 700, fontSize: 17,
              boxShadow: st.playing ? `0 0 12px ${C.green}66` : `0 0 4px ${C.green}22`,
            }}>{st.playing ? '\u275A\u275A' : '\u25B6'}</button>
          <button onClick={() => fileRef.current?.click()}
            onMouseEnter={() => showHint('LOAD a track. BPM is auto-detected.')}
            className="rounded flex items-center justify-center touch-target"
            style={{ width: 48, height: 48, background: '#1a1a1e', border: `1px solid ${C.edge}`, color: C.dim, fontFamily: 'Oxanium', fontSize: 9 }}>
            <span style={{ fontSize: 14 }}>{'\u2913'}</span>
          </button>
          <input ref={fileRef} type="file" accept="audio/*" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) loadFile(id, f); }} />
        </div>

        {/* Hot cues + loop + beat jump */}
        <div className="flex items-center gap-1 justify-center flex-wrap">
          {st.hotCues.map((hc, i) => (
            <button key={i} onClick={() => hotCuePress(id, i)}
              onContextMenu={e => { e.preventDefault(); hotCueClear(id, i); }}
              className="rounded flex items-center justify-center touch-target"
              style={{
                width: 30, height: 28, fontSize: 9, fontFamily: 'Oxanium', fontWeight: 700,
                background: hc !== null ? `${HC_COLORS[i]}18` : '#1a1a1e',
                border: `2px solid ${hc !== null ? HC_COLORS[i] : C.edge}`,
                color: hc !== null ? HC_COLORS[i] : C.dim,
              }}>{HC_LABELS[i]}</button>
          ))}
          <div style={{ width: 1, height: 20, background: C.edge }} />
          <button onClick={() => setLoopInPoint(id)} className="rounded flex items-center justify-center"
            style={{ width: 28, height: 28, fontSize: 7, fontFamily: 'Oxanium', fontWeight: 600,
              background: st.loopIn !== null ? `${C.green}18` : '#1a1a1e',
              border: `1px solid ${st.loopIn !== null ? C.green : C.edge}`, color: st.loopIn !== null ? C.green : C.dim }}>IN</button>
          <button onClick={() => setLoopOutPoint(id)} className="rounded flex items-center justify-center"
            style={{ width: 28, height: 28, fontSize: 7, fontFamily: 'Oxanium', fontWeight: 600,
              background: st.loopOut !== null ? `${C.green}18` : '#1a1a1e',
              border: `1px solid ${st.loopOut !== null ? C.green : C.edge}`, color: st.loopOut !== null ? C.green : C.dim }}>OUT</button>
          <button onClick={() => toggleLoop(id)} className="rounded flex items-center justify-center"
            style={{ width: 32, height: 28, fontSize: 7, fontFamily: 'Oxanium', fontWeight: 600,
              background: st.loopActive ? `${C.green}22` : '#1a1a1e',
              border: `1px solid ${st.loopActive ? C.green : C.edge}`, color: st.loopActive ? C.green : C.dim }}>
            {st.loopActive ? '\u27F2ON' : 'LOOP'}</button>
          <div style={{ width: 1, height: 20, background: C.edge }} />
          {/* Beat jump */}
          {[{ b: -4, l: '\u00AB4' }, { b: -1, l: '\u2039' }, { b: 1, l: '\u203A' }, { b: 4, l: '4\u00BB' }].map(({ b, l }) => (
            <button key={b} onClick={() => beatJump(id, b)} className="rounded flex items-center justify-center"
              onMouseEnter={() => showHint(`Beat jump: skip ${Math.abs(b)} beat${Math.abs(b) > 1 ? 's' : ''} ${b > 0 ? 'forward' : 'back'}. Needs BPM.`)}
              style={{ width: 24, height: 28, fontSize: 9, fontFamily: 'Oxanium',
                background: '#1a1a1e', border: `1px solid ${C.edge}`, color: bpmNum > 0 ? C.text : C.dim }}>{l}</button>
          ))}
        </div>

        {/* Tempo + BPM */}
        <div className="flex items-stretch gap-2 justify-between">
          <div className="flex flex-col items-center" style={{ flex: 1 }}>
            <div className="flex items-center gap-0.5 mb-1">
              {([6, 10, 16, 100] as const).map(r => (
                <button key={r} onClick={() => setters[id](s => ({ ...s, range: r }))}
                  style={{ fontFamily: 'Oxanium', fontSize: 7, padding: '1px 3px', borderRadius: 3,
                    background: st.range === r ? C.cyanDim : '#1a1a1e', color: st.range === r ? C.cyan : C.dim, border: `1px solid ${C.edge}` }}>
                  {r === 100 ? 'W' : `\u00B1${r}`}</button>
              ))}
            </div>
            <Fader value={st.tempo} onChange={v => setTempo(id, v)} label="TEMPO" color={deckColor} height={faderH} center
              onHint={showHint} hint="TEMPO: pull DOWN to speed up, push UP to slow down (matches real CDJ). Center=0%. Pros beatmatch on \u00B16." />
          </div>
          <div className="flex flex-col items-center justify-end gap-1" style={{ width: isMobile ? 80 : 90 }}>
            <div className="w-full rounded text-center" style={{ background: '#0c0c10', border: `1px solid ${C.edge}`, padding: 3 }}>
              <div style={{ fontFamily: "'IBM Plex Mono'", fontSize: 14, color: deckColor }}>
                {tempoPercent(st.tempo, st.range) >= 0 ? '+' : ''}{tempoPercent(st.tempo, st.range).toFixed(1)}%
              </div>
            </div>
            <input value={st.bpm} onChange={e => setters[id](s => ({ ...s, bpm: e.target.value.replace(/[^\d.]/g, '') }))}
              placeholder="BPM" inputMode="decimal"
              onMouseEnter={() => showHint('BPM: auto-detected on load. Edit to correct.')}
              style={{ width: '100%', background: '#0c0c10', border: `1px solid ${C.edge}`, color: C.text,
                fontFamily: "'IBM Plex Mono'", fontSize: 12, padding: '3px 4px', borderRadius: 4, textAlign: 'center', outline: 'none' }} />
            <div className="w-full text-center" style={{ fontFamily: "'IBM Plex Mono'", fontSize: 11, color: adjBpm(st) ? C.green : C.dim }}>
              {adjBpm(st) ? `\u2192 ${adjBpm(st)!.toFixed(1)}` : '\u2014'}
            </div>
          </div>
        </div>
      </div>
    );
  };

  // ─── channel strip ───
  const renderChannelStrip = (id: DeckId) => {
    const ch = id === 'A' ? mix.chA : mix.chB;
    const setCh = (p: Partial<ChannelState>) => setMix(m => id === 'A' ? { ...m, chA: { ...m.chA, ...p } } : { ...m, chB: { ...m.chB, ...p } });
    const sc = id === 'A' ? C.cyan : C.orange;
    return (
      <div className="flex flex-col items-center gap-1.5 rounded-lg p-1.5" style={{ background: C.panelHi, border: `1px solid ${C.edge}` }}>
        <span style={{ fontFamily: 'Oxanium', fontWeight: 700, color: sc, fontSize: 11 }}>CH {id === 'A' ? '1' : '2'}</span>
        <Knob value={ch.trim} onChange={v => setCh({ trim: v })} label="TRIM" size={knobSize} onHint={showHint}
          hint="TRIM: input gain. Peak into orange, never red." />
        <Knob value={ch.hi} onChange={v => setCh({ hi: v })} label="HI" size={knobSize} onHint={showHint}
          hint="HI EQ: treble. Full left = kill." />
        <Knob value={ch.mid} onChange={v => setCh({ mid: v })} label="MID" size={knobSize} onHint={showHint}
          hint="MID EQ: vocals/snares. Kill during blends." />
        <Knob value={ch.low} onChange={v => setCh({ low: v })} label="LOW" color={C.orange} size={knobSize} onHint={showHint}
          hint="LOW EQ: bass/kick. THE mixing control. Swap lows between tracks." />
        <div className="flex items-end gap-1.5">
          <Fader value={ch.fader} onChange={v => setCh({ fader: v })} color={sc} height={faderH}
            onHint={showHint} hint="CHANNEL FADER: volume for this deck." />
          <VUMeter level={id === 'A' ? vu.A : vu.B} />
        </div>
        <Knob value={ch.color} onChange={v => setCh({ color: v })} label="COLOR" size={knobSize - 8} onHint={showHint}
          hint="COLOR FILTER: Left=low-pass, Right=high-pass. Sweep on buildups." />
      </div>
    );
  };

  // ─── mixer render ───
  const renderMixer = () => (
    <div className="rounded-lg p-2 flex flex-col gap-2" style={{ background: C.panel, border: `1px solid ${C.edge}`, minWidth: isMobile ? 0 : 200 }}>
      <div className="flex items-center justify-between">
        <span style={{ fontFamily: 'Oxanium', fontWeight: 700, color: C.cyan, letterSpacing: 2, fontSize: 12 }}>MIXER</span>
        <span style={{ fontFamily: 'Oxanium', color: C.dim, fontSize: 9 }}>DJM-900NXS</span>
      </div>
      <div className="flex gap-2 justify-center">
        {renderChannelStrip('A')}
        {renderChannelStrip('B')}
      </div>
      <div className="flex items-center gap-2">
        <Knob value={mix.master} onChange={v => setMix(m => ({ ...m, master: v }))} label="MASTER" size={knobSize} onHint={showHint}
          hint="MASTER: overall output. Set once, leave it." />
        <VUMeter level={vu.M} />
      </div>
      <div>
        <div style={{ color: C.dim, fontSize: 8, letterSpacing: 1, textAlign: 'center', marginBottom: 3, fontFamily: 'Oxanium' }}>CROSSFADER</div>
        <div className="px-1" onMouseEnter={() => showHint('CROSSFADER: A (left) to B (right). For smooth mixes, many DJs leave it centered.')}>
          <CrossfaderH value={mix.xfader} onChange={v => setMix(m => ({ ...m, xfader: v }))} />
        </div>
        <div className="flex justify-between px-1" style={{ fontFamily: 'Oxanium', fontSize: 9, color: C.dim }}>
          <span style={{ color: C.cyan }}>A</span><span style={{ color: C.orange }}>B</span>
        </div>
      </div>
    </div>
  );

  // ═══════════════════════════════════════════════════════════
  // MAIN LAYOUT
  // ═══════════════════════════════════════════════════════════

  return (
    <div className="select-none" style={{ background: C.bg, minHeight: '100dvh', fontFamily: 'Oxanium', color: C.text }}>
      {/* Portrait rotation prompt (phones only) */}
      <div className="portrait-block flex-col items-center justify-center gap-4"
        style={{ position: 'fixed', inset: 0, background: C.bg, zIndex: 100, fontFamily: 'Oxanium', color: C.text, padding: 32 }}>
        <div style={{ fontSize: 48 }}>{'\u21BB'}</div>
        <div style={{ fontWeight: 700, fontSize: 20, letterSpacing: 3 }}>ROTATE TO LANDSCAPE</div>
        <div style={{ color: C.dim, fontSize: 12, fontFamily: "'IBM Plex Mono'", textAlign: 'center', lineHeight: 1.6 }}>
          The DJ booth needs horizontal space.<br />Turn your phone sideways to start mixing.
        </div>
      </div>

      <div className="landscape-content" style={{ padding: isMobile ? 4 : 8 }}>
        {/* Top bar */}
        <div className="flex items-center justify-between mb-1.5 flex-wrap gap-1">
          <div className="flex items-center gap-2">
            <span style={{ fontWeight: 700, fontSize: isMobile ? 14 : 18, letterSpacing: 3 }}>
              VIRTUAL <span style={{ color: C.cyan }}>BOOTH</span>
            </span>
            {!isMobile && <span style={{ color: C.dim, fontSize: 10, fontFamily: "'IBM Plex Mono'" }}>CDJ-2000NXS &times;2 &middot; DJM-900NXS</span>}
          </div>
          <div className="flex items-center gap-1">
            {!isMobile && (
              <button onClick={() => setShowKeys(v => !v)}
                style={{ background: C.panel, border: `1px solid ${showKeys ? C.cyan : C.edge}`, color: showKeys ? C.cyan : C.dim, padding: '4px 8px', borderRadius: 4, fontSize: 10 }}>
                Keys</button>
            )}
            <button onClick={() => setShowGuide(v => !v)}
              style={{ background: C.panel, border: `1px solid ${C.edge}`, color: C.text, padding: '4px 8px', borderRadius: 4, fontSize: 10 }}>
              Guide</button>
            <button onClick={() => setLearn(v => !v)}
              style={{ background: learn ? C.cyanDim : C.panel, border: `1px solid ${learn ? C.cyan : C.edge}`, color: learn ? C.cyan : C.dim, padding: '4px 8px', borderRadius: 4, fontSize: 10 }}>
              Learn</button>
          </div>
        </div>

        {/* BoothMatch — always visible */}
        <div className="rounded mb-1.5 px-2 py-1 flex items-center justify-center gap-3 flex-wrap"
          style={{ background: C.panel, border: `1px solid ${matched ? C.green : C.edge}`, transition: 'border-color 0.3s' }}>
          <span style={{ fontSize: 9, color: C.dim, letterSpacing: 2 }}>BOOTHMATCH</span>
          <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 13, color: C.cyan }}>A {bpmA ? bpmA.toFixed(1) : '--'}</span>
          <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 11, color: matched ? C.green : C.dim }}>
            {bpmA && bpmB ? `\u0394${(bpmA - bpmB).toFixed(2)}` : '\u00B7'}
          </span>
          <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 13, color: C.orange }}>B {bpmB ? bpmB.toFixed(1) : '--'}</span>
          {matched && <span style={{ color: C.green, fontWeight: 700, fontSize: 10 }}>{'\u25CF'} LOCKED</span>}
        </div>

        {/* Mobile tabs */}
        {isMobile && (
          <div className="flex gap-1 mb-1.5">
            {(['A', 'mix', 'B'] as const).map(tab => (
              <button key={tab} onClick={() => setMobileTab(tab)}
                style={{
                  flex: 1, padding: '6px 0', borderRadius: 4, fontSize: 11, fontWeight: 700, letterSpacing: 2,
                  background: mobileTab === tab ? (tab === 'A' ? C.cyanDim : tab === 'B' ? `${C.orange}22` : C.panelHi) : C.panel,
                  border: `1px solid ${mobileTab === tab ? (tab === 'A' ? C.cyan : tab === 'B' ? C.orange : C.edge) : C.edge}`,
                  color: mobileTab === tab ? (tab === 'A' ? C.cyan : tab === 'B' ? C.orange : C.text) : C.dim,
                }}>
                {tab === 'mix' ? 'MIXER' : `DECK ${tab}`}
              </button>
            ))}
          </div>
        )}

        {/* Main rig */}
        {isMobile ? (
          <div>
            {mobileTab === 'A' && renderDeck('A')}
            {mobileTab === 'mix' && renderMixer()}
            {mobileTab === 'B' && renderDeck('B')}
          </div>
        ) : (
          <div className="flex gap-2 items-start justify-center" style={{ flexWrap: isCompact ? 'wrap' : 'nowrap' }}>
            {renderDeck('A')}
            {renderMixer()}
            {renderDeck('B')}
          </div>
        )}

        {/* Learn hint */}
        {learn && (
          <div className="rounded mt-1.5 px-3 py-2" style={{ background: C.panelHi, border: `1px solid ${C.cyanDim}`, minHeight: 36 }}>
            <span style={{ color: C.cyan, fontWeight: 700, fontSize: 10, letterSpacing: 2 }}>LEARN {'\u25B8'} </span>
            <span style={{ color: hint ? C.text : C.dim, fontSize: isMobile ? 11 : 12, fontFamily: "'IBM Plex Mono'", lineHeight: 1.5 }}>
              {hint || 'Hover (or tap) any control to learn what it does.'}
            </span>
          </div>
        )}

        {/* Guide */}
        {showGuide && (
          <div className="rounded mt-1.5 p-3" style={{ background: C.panel, border: `1px solid ${C.edge}` }}>
            <div className="flex items-center justify-between mb-1">
              <span style={{ fontWeight: 700, color: C.cyan, letterSpacing: 2, fontSize: 11 }}>FIRST MIX &mdash; 6 STEPS</span>
              <span style={{ fontFamily: "'IBM Plex Mono'", color: C.dim, fontSize: 10 }}>{guideStep + 1}/6</span>
            </div>
            <div style={{ fontSize: 12, fontFamily: "'IBM Plex Mono'", color: C.text, lineHeight: 1.6 }}>
              <span style={{ color: C.orange, fontWeight: 600 }}>Step {guideStep + 1}. </span>{GUIDE[guideStep]}
            </div>
            <div className="flex gap-2 mt-2 items-center">
              <button onClick={() => setGuideStep(s => Math.max(0, s - 1))}
                style={{ background: C.panelHi, border: `1px solid ${C.edge}`, color: C.text, padding: '4px 10px', borderRadius: 4, fontSize: 11 }}>{'\u2190'}</button>
              <button onClick={() => setGuideStep(s => Math.min(5, s + 1))}
                style={{ background: C.cyanDim, border: `1px solid ${C.cyan}`, color: C.cyan, padding: '4px 10px', borderRadius: 4, fontSize: 11 }}>{'\u2192'}</button>
              <div className="flex gap-1 ml-1">
                {GUIDE.map((_, i) => (
                  <div key={i} onClick={() => setGuideStep(i)} className="cursor-pointer rounded-full"
                    style={{ width: 7, height: 7, background: i === guideStep ? C.cyan : '#2a2a32' }} />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Keyboard shortcuts (desktop only) */}
        {showKeys && !isMobile && (
          <div className="rounded mt-1.5 p-3" style={{ background: C.panel, border: `1px solid ${C.edge}` }}>
            <div style={{ fontWeight: 700, color: C.cyan, letterSpacing: 2, marginBottom: 4, fontSize: 10 }}>KEYBOARD</div>
            <div className="grid grid-cols-2 gap-3" style={{ fontFamily: "'IBM Plex Mono'", fontSize: 10, color: C.dim, lineHeight: 1.8 }}>
              <div>
                <div style={{ color: C.cyan, fontWeight: 600 }}>DECK A</div>
                <div><b>Q</b> CUE (hold) &middot; <b>W</b> PLAY &middot; <b>1-4</b> Hot Cues</div>
                <div><b>A/S/D</b> Loop In/Out/Reloop &middot; <b>Z/X/C/V</b> Beat Jump \u00B11/\u00B14</div>
              </div>
              <div>
                <div style={{ color: C.orange, fontWeight: 600 }}>DECK B</div>
                <div><b>[</b> CUE (hold) &middot; <b>]</b> PLAY &middot; <b>7-0</b> Hot Cues</div>
                <div><b>L/;/&apos;</b> Loop &middot; <b>,/./</b> Beat Jump</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
