'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { buildPeaks, detectBPM, computeRMS, normalizeGain, detectKey, toCamelot, keysCompatible, createReverbIR, fmtTime, quantizePos } from './dj-helpers';

// ═══════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════

type DeckId = 'A' | 'B';
type XfCurve = 'thru' | 'smooth' | 'sharp';
type FxType = 'off' | 'echo' | 'reverb';

interface DeckState {
  name: string | null; loaded: boolean; playing: boolean; pos: number; dur: number;
  tempo: number; range: number; bpm: string; masterTempo: boolean;
  peaks: Float32Array | null; cue: number; hotCues: (number | null)[];
  loopIn: number | null; loopOut: number | null; loopActive: boolean;
  quantize: boolean; key: string | null;
}

interface ChannelState {
  trim: number; hi: number; mid: number; low: number; fader: number; color: number;
}

interface MixerState {
  xfader: number; chA: ChannelState; chB: ChannelState; master: number;
  xfCurve: XfCurve; fxType: FxType; fxWet: number;
  splitCue: boolean; cueA: boolean; cueB: boolean;
}

interface AudioDeck {
  trim: GainNode; low: BiquadFilterNode; mid: BiquadFilterNode; hi: BiquadFilterNode;
  color: BiquadFilterNode; chGain: GainNode; analyser: AnalyserNode; xf: GainNode;
  cueGain: GainNode;
  src: AudioBufferSourceNode | null; buffer: AudioBuffer | null;
  startTime: number; startOffset: number; rate: number; effRate: number;
  playing: boolean; stopping: boolean; braking: boolean;
  cue: number; dur: number; hotCues: (number | null)[]; loopIn: number | null;
  loopOut: number | null; loopActive: boolean; cuePreview: boolean;
}

interface AudioEngine {
  ctx: AudioContext; master: GainNode; masterAnalyser: AnalyserNode;
  masterPanner: StereoPannerNode;
  masterDry: GainNode; masterMix: GainNode;
  echoDelay: DelayNode; echoFeedback: GainNode; echoWet: GainNode;
  reverbConv: ConvolverNode; reverbWet: GainNode;
  cueBus: GainNode; cuePanner: StereoPannerNode;
  recordDest: MediaStreamAudioDestinationNode;
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
  'Load a track on each deck \u2014 tap LOAD or the waveform area.',
  'BPM + KEY auto-detected. Check BoothMatch \u2014 green = BPMs locked, Camelot keys shown.',
  'Press PLAY on Deck A. Push channel fader up, crossfader left.',
  'Press CUE on Deck B. Move TEMPO fader (down=faster) until BPMs match.',
  'Watch the PHASE METER \u2014 nudge the JOG WHEEL until the needle is green (within 15ms).',
  'Sweep CROSSFADER right. Kill Deck A\u2019s LOW as B\u2019s LOW comes in. Try the ECHO FX on the transition.',
];

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
  if (!st.peaks) { ctx.fillStyle = C.dim; ctx.font = "11px 'IBM Plex Mono'"; ctx.fillText('TAP TO LOAD TRACK', W * 0.1, H / 2 + 4); return; }
  const prog = st.dur ? st.pos / st.dur : 0;
  const n = st.peaks.length;
  for (let i = 0; i < n; i++) {
    const x = (i / n) * W, h = st.peaks[i] * (H * 0.44);
    ctx.fillStyle = (i / n < prog) ? C.cyan : C.cyanDim;
    ctx.fillRect(x, H / 2 - h, Math.max(1, W / n), h * 2);
  }
  // Phrase markers (every 16 bars = 64 beats)
  const bpm = parseFloat(st.bpm);
  if (bpm > 0 && st.dur > 0) {
    const phraseSec = 64 * 60 / bpm;
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    for (let t = 0; t < st.dur; t += phraseSec) { ctx.fillRect((t / st.dur) * W, 0, 1, H); }
  }
  if (st.loopIn !== null && st.loopOut !== null && st.dur > 0) {
    const x1 = (st.loopIn / st.dur) * W, x2 = (st.loopOut / st.dur) * W;
    ctx.fillStyle = st.loopActive ? 'rgba(60,224,138,0.12)' : 'rgba(60,224,138,0.05)';
    ctx.fillRect(x1, 0, x2 - x1, H); ctx.fillStyle = C.green; ctx.fillRect(x1 - 1, 0, 2, H); ctx.fillRect(x2 - 1, 0, 2, H);
  }
  st.hotCues.forEach((hc, i) => { if (hc !== null && st.dur > 0) { ctx.fillStyle = HC_COLORS[i]; ctx.fillRect((hc / st.dur) * W - 1, 0, 2, H); } });
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
  if (st.loopIn !== null && st.loopOut !== null) {
    const lx1 = ((st.loopIn - startSec) / windowSec) * W, lx2 = ((st.loopOut - startSec) / windowSec) * W;
    if (lx2 > 0 && lx1 < W) { ctx.fillStyle = st.loopActive ? 'rgba(60,224,138,0.1)' : 'rgba(60,224,138,0.04)'; ctx.fillRect(Math.max(0, lx1), 0, Math.min(W, lx2) - Math.max(0, lx1), H); }
  }
  st.hotCues.forEach((hc, i) => { if (hc !== null) { const hx = ((hc - startSec) / windowSec) * W; if (hx >= -2 && hx <= W + 2) { ctx.fillStyle = HC_COLORS[i]; ctx.fillRect(hx - 1, 0, 2, H); } } });
  const cueX = ((st.cue - startSec) / windowSec) * W;
  if (cueX >= -2 && cueX <= W + 2) { ctx.fillStyle = C.orange; ctx.fillRect(cueX - 1, 0, 2, H); }
  ctx.fillStyle = '#ffffff'; ctx.fillRect(W / 2 - 1, 0, 2, H);
  // Beat grid + phrase ticks
  const bpm = parseFloat(st.bpm);
  if (bpm > 0) {
    const beatSec = 60 / bpm;
    const firstBeat = Math.ceil(startSec / beatSec) * beatSec;
    for (let t = firstBeat; t < startSec + windowSec; t += beatSec) {
      const bx = ((t - startSec) / windowSec) * W;
      const beatNum = Math.round(t / beatSec);
      const isPhrase = beatNum % 16 === 0;
      const isBar = beatNum % 4 === 0;
      ctx.fillStyle = isPhrase ? 'rgba(255,255,255,0.2)' : isBar ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.04)';
      ctx.fillRect(bx, 0, 1, H);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// CONTROLS
// ═══════════════════════════════════════════════════════════════

function Knob({ value, onChange, label, color = C.cyan, size = 48, hint, onHint }: {
  value: number; onChange: (v: number) => void; label: string;
  color?: string; size?: number; hint?: string; onHint?: (h: string) => void;
}) {
  const drag = useRef<{ cx: number; cy: number; lastAng: number; id: number } | null>(null);
  const valRef = useRef(value); valRef.current = value;
  const angDeg = -135 + value * 270;
  // Arc path for value indicator
  const r = size / 2 - 3;
  const cx = size / 2, cy = size / 2;
  const toRad = (d: number) => (d - 90) * Math.PI / 180;
  const sa = toRad(-135), ea = toRad(angDeg);
  const x1 = cx + r * Math.cos(sa), y1 = cy + r * Math.sin(sa);
  const x2 = cx + r * Math.cos(ea), y2 = cy + r * Math.sin(ea);
  const arc = value > 0.001 ? `M ${x1} ${y1} A ${r} ${r} 0 ${value > 0.5 ? 1 : 0} 1 ${x2} ${y2}` : '';

  return (
    <div className="flex flex-col items-center select-none" style={{ width: size + 8 }} onMouseEnter={() => hint && onHint?.(hint)}>
      <div className="rounded-full relative cursor-grab active:cursor-grabbing" style={{ width: size, height: size, touchAction: 'none',
        background: 'radial-gradient(circle at 35% 30%, #3a3a44, #141418 70%)', border: `1px solid ${C.edge}`,
        boxShadow: 'inset 0 2px 4px rgba(0,0,0,.6), 0 1px 2px rgba(0,0,0,.5)', transition: 'box-shadow 0.15s' }}
        onPointerDown={e => {
          e.preventDefault(); e.stopPropagation();
          const el = e.currentTarget as HTMLElement;
          el.setPointerCapture(e.pointerId);
          el.style.boxShadow = `inset 0 2px 4px rgba(0,0,0,.6), 0 0 8px ${color}33`;
          const rect = el.getBoundingClientRect();
          drag.current = { cx: rect.left + rect.width / 2, cy: rect.top + rect.height / 2, lastAng: Math.atan2(e.clientY - (rect.top + rect.height / 2), e.clientX - (rect.left + rect.width / 2)), id: e.pointerId };
        }}
        onPointerMove={e => {
          if (!drag.current || e.pointerId !== drag.current.id) return;
          const ang = Math.atan2(e.clientY - drag.current.cy, e.clientX - drag.current.cx);
          let dA = ang - drag.current.lastAng;
          if (dA > Math.PI) dA -= 2 * Math.PI; if (dA < -Math.PI) dA += 2 * Math.PI;
          drag.current.lastAng = ang;
          let nv = Math.max(0, Math.min(1, valRef.current + dA / (Math.PI * 1.5)));
          // Center detent with haptic
          if (Math.abs(nv - 0.5) < 0.018 && Math.abs(valRef.current - 0.5) >= 0.018) {
            nv = 0.5; try { navigator.vibrate?.(4); } catch { /* ok */ }
          } else if (Math.abs(nv - 0.5) < 0.018) nv = 0.5;
          onChange(nv);
        }}
        onPointerUp={e => { if (drag.current?.id === e.pointerId) { drag.current = null; (e.currentTarget as HTMLElement).style.boxShadow = 'inset 0 2px 4px rgba(0,0,0,.6), 0 1px 2px rgba(0,0,0,.5)'; } }}
        onPointerCancel={e => { if (drag.current?.id === e.pointerId) { drag.current = null; (e.currentTarget as HTMLElement).style.boxShadow = 'inset 0 2px 4px rgba(0,0,0,.6), 0 1px 2px rgba(0,0,0,.5)'; } }}
        onDoubleClick={() => onChange(0.5)} title={label}>
        {/* Value arc */}
        <svg className="absolute inset-0" width={size} height={size} style={{ pointerEvents: 'none' }}>
          {arc && <path d={arc} stroke={color} strokeWidth={2.5} fill="none" opacity={0.5} strokeLinecap="round" />}
        </svg>
        {/* Indicator line */}
        <div className="absolute left-1/2 top-1/2" style={{ width: 2, height: size * 0.35, background: color, borderRadius: 2,
          transform: `translate(-50%,-100%) rotate(${angDeg}deg)`, transformOrigin: 'bottom center', boxShadow: `0 0 5px ${color}` }} />
      </div>
      <span style={{ color: C.dim, fontSize: 9, marginTop: 3, letterSpacing: 1, fontFamily: 'Oxanium' }}>{label}</span>
    </div>
  );
}

function Fader({ value, onChange, label, color = C.cyan, height = 130, center, hint, onHint }: {
  value: number; onChange: (v: number) => void; label?: string;
  color?: string; height?: number; center?: boolean; hint?: string; onHint?: (h: string) => void;
}) {
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
          let nv = Math.max(0, Math.min(1, drag.current.v + (drag.current.y - e.clientY) / height));
          // Detent at center for tempo faders
          if (center && Math.abs(nv - 0.5) < 0.012 && Math.abs(drag.current.v + (drag.current.y - e.clientY) / height - 0.5) >= 0.012) {
            nv = 0.5; try { navigator.vibrate?.(4); } catch { /* ok */ }
          } else if (center && Math.abs(nv - 0.5) < 0.012) nv = 0.5;
          onChange(nv);
        }}
        onPointerUp={e => { if (drag.current?.id === e.pointerId) drag.current = null; }}
        onPointerCancel={e => { if (drag.current?.id === e.pointerId) drag.current = null; }}>
        {/* Track */}
        <div className="absolute left-1/2 top-0 bottom-0" style={{ width: 4, transform: 'translateX(-50%)', background: '#0a0a0c', borderRadius: 3, border: `1px solid ${C.edge}` }} />
        {/* Track fill */}
        <div className="absolute left-1/2 bottom-0" style={{ width: 4, transform: 'translateX(-50%)', height: `${value * 100}%`, background: color, opacity: 0.25, borderRadius: 3 }} />
        {center && <div className="absolute left-1/2 top-1/2" style={{ width: 14, height: 1, background: C.dim, transform: 'translate(-50%,-50%)' }} />}
        {/* Handle */}
        <div className="absolute left-1/2" style={{ width: 34, height: 22, transform: 'translateX(-50%)', bottom: `calc(${value * 100}% - 11px)`,
          background: 'linear-gradient(180deg,#48484f,#1a1a1e)', borderRadius: 4, border: `1px solid ${C.edge}`, boxShadow: '0 0 5px rgba(0,0,0,.6)',
          transition: 'box-shadow 0.1s' }}>
          <div style={{ height: 2, background: color, margin: '9px 4px', borderRadius: 2, boxShadow: `0 0 4px ${color}` }} />
        </div>
      </div>
      {label && <span style={{ color: C.dim, fontSize: 9, marginTop: 4, letterSpacing: 1, fontFamily: 'Oxanium' }}>{label}</span>}
    </div>
  );
}

function VUMeter({ level }: { level: number }) {
  const segs = 14, lit = Math.round(level * segs);
  return (<div className="flex flex-col-reverse gap-0.5">{Array.from({ length: segs }).map((_, i) => {
    const on = i < lit; const col = i > segs * 0.85 ? C.red : i > segs * 0.65 ? C.yellow : C.green;
    return <div key={i} style={{ width: 7, height: 5, borderRadius: 1, background: on ? col : '#1d1d22', boxShadow: on ? `0 0 3px ${col}` : 'none' }} />;
  })}</div>);
}

function CrossfaderH({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const trackRef = useRef<HTMLDivElement>(null);
  const dragging = useRef<number | null>(null);
  return (
    <div ref={trackRef} className="relative cursor-ew-resize" style={{ height: 36, touchAction: 'none' }}
      onPointerDown={e => { e.preventDefault(); (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); dragging.current = e.pointerId;
        if (trackRef.current) { const r = trackRef.current.getBoundingClientRect(); onChange(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))); } }}
      onPointerMove={e => { if (dragging.current !== e.pointerId || !trackRef.current) return; const r = trackRef.current.getBoundingClientRect(); onChange(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))); }}
      onPointerUp={e => { if (dragging.current === e.pointerId) dragging.current = null; }}
      onPointerCancel={e => { if (dragging.current === e.pointerId) dragging.current = null; }}>
      <div className="absolute top-1/2 left-0 right-0" style={{ height: 6, transform: 'translateY(-50%)', background: '#0a0a0c', borderRadius: 3, border: `1px solid ${C.edge}` }} />
      <div className="absolute top-1/2" style={{ width: 28, height: 32, transform: 'translate(-50%,-50%)', left: `${value * 100}%`,
        background: 'linear-gradient(180deg,#48484f,#1a1a1e)', borderRadius: 4, border: `1px solid ${C.edge}`, boxShadow: '0 0 5px rgba(0,0,0,.6)' }}>
        <div style={{ width: 2, height: 22, background: C.cyan, margin: '5px auto', borderRadius: 2, boxShadow: `0 0 4px ${C.cyan}` }} />
      </div>
    </div>
  );
}

function BeatIndicator({ pos, bpm }: { pos: number; bpm: number }) {
  if (!bpm || bpm <= 0) return null;
  const current = Math.floor((pos * bpm / 60) % 4);
  return (<div className="flex gap-1">{[0, 1, 2, 3].map(i => (
    <div key={i} style={{ width: 10, height: 10, borderRadius: 2, background: i === current ? (i === 0 ? C.orange : C.cyan) : '#1a1a1e',
      border: `1px solid ${i === current ? (i === 0 ? C.orange : C.cyan) : C.edge}`,
      boxShadow: i === current ? `0 0 6px ${i === 0 ? C.orange : C.cyan}55` : 'none' }} />
  ))}</div>);
}

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════

function mkDeck(): DeckState {
  return { name: null, loaded: false, playing: false, pos: 0, dur: 0, tempo: 0.5, range: 8,
    bpm: '', masterTempo: false, peaks: null, cue: 0, hotCues: [null, null, null, null],
    loopIn: null, loopOut: null, loopActive: false, quantize: false, key: null };
}

export default function DJTrainer() {
  const audio = useRef<AudioEngine | null>(null);
  const cvsA = useRef<HTMLCanvasElement>(null); const cvsB = useRef<HTMLCanvasElement>(null);
  const zoomA = useRef<HTMLCanvasElement>(null); const zoomB = useRef<HTMLCanvasElement>(null);
  const platterRefA = useRef<HTMLDivElement>(null); const platterRefB = useRef<HTMLDivElement>(null);
  const platterAngle = useRef<Record<DeckId, number>>({ A: 0, B: 0 });
  const deckARef = useRef<DeckState>(mkDeck()); const deckBRef = useRef<DeckState>(mkDeck());
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recChunks = useRef<Blob[]>([]);

  const [initialized, setInitialized] = useState(false);
  const [learn, setLearn] = useState(true);
  const [hint, setHint] = useState<string | null>(null);
  const [guideStep, setGuideStep] = useState(0);
  const [showGuide, setShowGuide] = useState(false);
  const [showKeys, setShowKeys] = useState(false);
  const [deckA, setDeckA] = useState<DeckState>(mkDeck());
  const [deckB, setDeckB] = useState<DeckState>(mkDeck());
  const [mix, setMix] = useState<MixerState>({
    xfader: 0.5, chA: { trim: 0.7, hi: 0.5, mid: 0.5, low: 0.5, fader: 0.85, color: 0.5 },
    chB: { trim: 0.7, hi: 0.5, mid: 0.5, low: 0.5, fader: 0.85, color: 0.5 },
    master: 0.8, xfCurve: 'thru', fxType: 'off', fxWet: 0.3, splitCue: false, cueA: false, cueB: false,
  });
  const [vu, setVu] = useState({ A: 0, B: 0, M: 0 });
  const [mobileTab, setMobileTab] = useState<'A' | 'mix' | 'B'>('A');
  const [screenW, setScreenW] = useState(1200);
  const [loading, setLoading] = useState<Record<DeckId, boolean>>({ A: false, B: false });
  const [syncFlash, setSyncFlash] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [drillActive, setDrillActive] = useState(false);
  const [drillStart, setDrillStart] = useState(0);
  const [drillBest, setDrillBest] = useState<number | null>(null);

  const setters: Record<DeckId, React.Dispatch<React.SetStateAction<DeckState>>> = { A: setDeckA, B: setDeckB };
  const isMobile = screenW < 768;
  const isCompact = screenW < 1100;

  useEffect(() => { deckARef.current = deckA; }, [deckA]);
  useEffect(() => { deckBRef.current = deckB; }, [deckB]);
  useEffect(() => { const up = () => setScreenW(window.innerWidth); up(); window.addEventListener('resize', up); return () => window.removeEventListener('resize', up); }, []);
  useEffect(() => { if (initialized) document.body.classList.add('booth-active'); return () => { document.body.classList.remove('booth-active'); }; }, [initialized]);

  // ─── audio init (FX + split cue + recording) ───
  const initAudio = useCallback(() => {
    if (audio.current) return;
    const ctx = new AudioContext();
    const master = ctx.createGain(); master.gain.value = 0.8;
    // FX chain: master → dry/echo/reverb → mix → analyser → panner → dest
    const masterDry = ctx.createGain(); masterDry.gain.value = 1;
    const masterMix = ctx.createGain(); masterMix.gain.value = 1;
    const echoDelay = ctx.createDelay(2); echoDelay.delayTime.value = 0.375;
    const echoFeedback = ctx.createGain(); echoFeedback.gain.value = 0.35;
    const echoWet = ctx.createGain(); echoWet.gain.value = 0;
    const reverbConv = ctx.createConvolver(); reverbConv.buffer = createReverbIR(ctx);
    const reverbWet = ctx.createGain(); reverbWet.gain.value = 0;
    master.connect(masterDry); masterDry.connect(masterMix);
    master.connect(echoDelay); echoDelay.connect(echoFeedback); echoFeedback.connect(echoDelay);
    echoDelay.connect(echoWet); echoWet.connect(masterMix);
    master.connect(reverbConv); reverbConv.connect(reverbWet); reverbWet.connect(masterMix);
    const masterAnalyser = ctx.createAnalyser(); masterAnalyser.fftSize = 256;
    masterMix.connect(masterAnalyser);
    const masterPanner = ctx.createStereoPanner(); masterPanner.pan.value = 0;
    masterAnalyser.connect(masterPanner); masterPanner.connect(ctx.destination);
    // Headphone cue bus
    const cueBus = ctx.createGain(); cueBus.gain.value = 1;
    const cuePanner = ctx.createStereoPanner(); cuePanner.pan.value = -1;
    cueBus.connect(cuePanner); cuePanner.connect(ctx.destination);
    // Recording destination
    const recordDest = ctx.createMediaStreamDestination();
    masterMix.connect(recordDest);

    const mkAudioDeck = (): AudioDeck => {
      const trim = ctx.createGain(); trim.gain.value = 0.7;
      const low = ctx.createBiquadFilter(); low.type = 'lowshelf'; low.frequency.value = 200;
      const mid = ctx.createBiquadFilter(); mid.type = 'peaking'; mid.frequency.value = 1000; mid.Q.value = 0.9;
      const hi = ctx.createBiquadFilter(); hi.type = 'highshelf'; hi.frequency.value = 3500;
      const color = ctx.createBiquadFilter(); color.type = 'lowpass'; color.frequency.value = 22000;
      const chGain = ctx.createGain(); chGain.gain.value = 0.85;
      const analyser = ctx.createAnalyser(); analyser.fftSize = 256;
      const xf = ctx.createGain(); xf.gain.value = 0.5;
      const cueGain = ctx.createGain(); cueGain.gain.value = 0;
      trim.connect(low); low.connect(mid); mid.connect(hi); hi.connect(color);
      color.connect(chGain); chGain.connect(analyser); analyser.connect(xf); xf.connect(master);
      analyser.connect(cueGain); cueGain.connect(cueBus);
      return { trim, low, mid, hi, color, chGain, analyser, xf, cueGain,
        src: null, buffer: null, startTime: 0, startOffset: 0, rate: 1, effRate: 1,
        playing: false, stopping: false, braking: false, cue: 0, dur: 0,
        hotCues: [null, null, null, null], loopIn: null, loopOut: null, loopActive: false, cuePreview: false };
    };
    audio.current = { ctx, master, masterAnalyser, masterPanner, masterDry, masterMix,
      echoDelay, echoFeedback, echoWet, reverbConv, reverbWet, cueBus, cuePanner, recordDest,
      decks: { A: mkAudioDeck(), B: mkAudioDeck() } };
    setInitialized(true);
  }, []);

  const ensure = useCallback((): AudioEngine => { if (!audio.current) throw new Error('Not init'); audio.current.ctx.resume(); return audio.current; }, []);
  const posOf = (d: AudioDeck): number => d.playing ? d.startOffset + (audio.current!.ctx.currentTime - d.startTime) * d.effRate : d.startOffset;
  const rebase = (d: AudioDeck): void => { d.startOffset = posOf(d); d.startTime = audio.current!.ctx.currentTime; };
  const startSrc = (a: AudioEngine, d: AudioDeck, id: DeckId): void => {
    const src = a.ctx.createBufferSource(); src.buffer = d.buffer; src.playbackRate.value = d.effRate; src.connect(d.trim);
    src.onended = () => { if (d.stopping) { d.stopping = false; return; } d.playing = false; setters[id](s => ({ ...s, playing: false })); };
    src.start(0, Math.max(0, Math.min(d.dur, d.startOffset))); d.src = src; d.startTime = a.ctx.currentTime; d.playing = true;
  };

  // ─── file loading (normalize + key detect + BPM) ───
  const loadFile = async (id: DeckId, file: File): Promise<void> => {
    setLoading(l => ({ ...l, [id]: true }));
    const a = ensure();
    const buf = await a.ctx.decodeAudioData(await file.arrayBuffer());
    const d = a.decks[id];
    if (d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } d.playing = false; }
    d.buffer = buf; d.dur = buf.duration; d.startOffset = 0; d.cue = 0;
    d.hotCues = [null, null, null, null]; d.loopIn = null; d.loopOut = null; d.loopActive = false; d.cuePreview = false; d.braking = false;
    const peaks = buildPeaks(buf);
    const bpm = detectBPM(buf);
    const key = detectKey(buf);
    // Auto-normalize: adjust trim to target loudness
    const rms = computeRMS(buf);
    const normGain = normalizeGain(rms);
    d.trim.gain.value = normGain;
    setters[id](() => ({ ...mkDeck(), name: file.name.replace(/\.[^.]+$/, ''), loaded: true, dur: buf.duration, peaks, bpm: bpm ? bpm.toFixed(1) : '', key }));
    setMix(m => id === 'A' ? { ...m, chA: { ...m.chA, trim: Math.min(1, normGain / 1.4) } } : { ...m, chB: { ...m.chB, trim: Math.min(1, normGain / 1.4) } });
    setLoading(l => ({ ...l, [id]: false }));
  };

  // ─── transport ───
  const tempoPercent = (v: number, range: number) => (0.5 - v) * 2 * range;
  const playDeck = (id: DeckId): void => { const a = ensure(); const d = a.decks[id]; if (!d.buffer) return; if (d.cuePreview) { d.cuePreview = false; return; } if (d.playing) return; if (d.braking) d.braking = false; if (d.startOffset >= d.dur) d.startOffset = 0; d.effRate = d.rate; startSrc(a, d, id); setters[id](s => ({ ...s, playing: true })); };
  const pauseDeck = (id: DeckId): void => { const a = ensure(); const d = a.decks[id]; if (!d.playing || d.braking) return; rebase(d); d.braking = true; if (d.src) { d.src.playbackRate.setValueAtTime(d.effRate, a.ctx.currentTime); d.src.playbackRate.linearRampToValueAtTime(0.001, a.ctx.currentTime + 0.25); } setTimeout(() => { if (!d.braking) return; const p = d.startOffset + d.effRate * 0.125; d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } d.startOffset = Math.max(0, Math.min(d.dur, p)); d.playing = false; d.cuePreview = false; d.braking = false; d.effRate = d.rate; setters[id](s => ({ ...s, playing: false, pos: d.startOffset })); }, 260); };
  const seekDeck = (id: DeckId, p: number): void => { const a = ensure(); const d = a.decks[id]; p = Math.max(0, Math.min(d.dur, p)); if (d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } d.startOffset = p; startSrc(a, d, id); } else d.startOffset = p; setters[id](s => ({ ...s, pos: p })); };
  const cuePress = (id: DeckId): void => { const a = ensure(); const d = a.decks[id]; if (!d.buffer) return; if (d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } d.startOffset = d.cue; d.playing = false; d.cuePreview = false; d.braking = false; d.effRate = d.rate; setters[id](s => ({ ...s, playing: false, pos: d.cue })); } else { d.cue = d.startOffset; d.cuePreview = true; d.effRate = d.rate; startSrc(a, d, id); setters[id](s => ({ ...s, cue: d.cue, playing: true })); } };
  const cueRelease = (id: DeckId): void => { if (!audio.current) return; const d = audio.current.decks[id]; if (d.cuePreview && d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } d.startOffset = d.cue; d.playing = false; d.cuePreview = false; setters[id](s => ({ ...s, playing: false, pos: d.cue })); } };
  const setTempo = (id: DeckId, v: number): void => { if (!audio.current) return; const d = audio.current.decks[id]; const range = id === 'A' ? deckA.range : deckB.range; const rate = 1 + tempoPercent(v, range) / 100; if (d.playing) rebase(d); d.rate = rate; d.effRate = rate; if (d.src && !d.braking) d.src.playbackRate.value = rate; setters[id](s => ({ ...s, tempo: v })); };

  // ─── jog wheel ───
  const jogDown = (id: DeckId, e: React.PointerEvent): void => {
    const a = ensure(); const d = a.decks[id]; if (!d.buffer) return;
    const el = e.currentTarget as HTMLElement; el.setPointerCapture(e.pointerId);
    const rect = el.getBoundingClientRect(); const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
    let lastAng = Math.atan2(e.clientY - cy, e.clientX - cx); const pid = e.pointerId;
    const mv = (ev: PointerEvent) => { if (ev.pointerId !== pid) return; const na = Math.atan2(ev.clientY - cy, ev.clientX - cx); let dA = na - lastAng; if (dA > Math.PI) dA -= 2 * Math.PI; if (dA < -Math.PI) dA += 2 * Math.PI; lastAng = na;
      if (d.playing) { rebase(d); d.effRate = d.rate * (1 + Math.max(-0.5, Math.min(0.5, dA * 2.2))); if (d.src && !d.braking) d.src.playbackRate.value = d.effRate; }
      else { d.startOffset = Math.max(0, Math.min(d.dur, d.startOffset + (dA / (2 * Math.PI)) * 4)); setters[id](s => ({ ...s, pos: d.startOffset })); } };
    const up = (ev: PointerEvent) => { if (ev.pointerId !== pid) return; if (d.playing) { rebase(d); d.effRate = d.rate; if (d.src && !d.braking) d.src.playbackRate.value = d.rate; } el.removeEventListener('pointermove', mv); el.removeEventListener('pointerup', up); el.removeEventListener('pointercancel', up); };
    el.addEventListener('pointermove', mv); el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
  };

  // ─── hot cues (double-tap to clear) ───
  const lastHcTap = useRef<Record<string, number>>({});
  const hotCueTap = (id: DeckId, idx: number): void => {
    const a = ensure(); const d = a.decks[id]; if (!d.buffer) return;
    const st = id === 'A' ? deckARef.current : deckBRef.current;
    const key = `${id}${idx}`; const now = Date.now(); const last = lastHcTap.current[key] || 0;
    if (now - last < 350 && d.hotCues[idx] !== null) { d.hotCues[idx] = null; setters[id](s => { const hc = [...s.hotCues]; hc[idx] = null; return { ...s, hotCues: hc }; }); lastHcTap.current[key] = 0; return; }
    lastHcTap.current[key] = now;
    if (d.hotCues[idx] === null) {
      let p = posOf(d); if (st.quantize) p = quantizePos(p, parseFloat(st.bpm));
      d.hotCues[idx] = p; setters[id](s => { const hc = [...s.hotCues]; hc[idx] = p; return { ...s, hotCues: hc }; });
    } else {
      const target = d.hotCues[idx]!; if (d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } }
      d.startOffset = target; d.cuePreview = false; d.braking = false; d.effRate = d.rate; startSrc(a, d, id); setters[id](s => ({ ...s, playing: true, pos: target }));
    }
  };

  // ─── loops ───
  const setLoopInPoint = (id: DeckId): void => { if (!audio.current) return; const d = audio.current.decks[id]; if (!d.buffer) return; const st = id === 'A' ? deckARef.current : deckBRef.current; let p = posOf(d); if (st.quantize) p = quantizePos(p, parseFloat(st.bpm)); d.loopIn = p; setters[id](s => ({ ...s, loopIn: p })); };
  const setLoopOutPoint = (id: DeckId): void => { if (!audio.current) return; const d = audio.current.decks[id]; if (!d.buffer) return; const st = id === 'A' ? deckARef.current : deckBRef.current; let p = posOf(d); if (st.quantize) p = quantizePos(p, parseFloat(st.bpm)); d.loopOut = p; d.loopActive = true; setters[id](s => ({ ...s, loopOut: p, loopActive: true })); };
  const toggleLoop = (id: DeckId): void => { if (!audio.current) return; const d = audio.current.decks[id]; if (d.loopIn === null || d.loopOut === null) return; d.loopActive = !d.loopActive; setters[id](s => ({ ...s, loopActive: !s.loopActive })); };
  const clearLoop = (id: DeckId): void => { if (!audio.current) return; const d = audio.current.decks[id]; d.loopIn = null; d.loopOut = null; d.loopActive = false; setters[id](s => ({ ...s, loopIn: null, loopOut: null, loopActive: false })); };
  const beatLoop = (id: DeckId, beats: number): void => { if (!audio.current) return; const d = audio.current.decks[id]; const st = id === 'A' ? deckARef.current : deckBRef.current; const bpm = parseFloat(st.bpm); if (!bpm || !d.buffer) return; const p = posOf(d); const dur = beats * 60 / bpm; d.loopIn = p; d.loopOut = p + dur; d.loopActive = true; setters[id](s => ({ ...s, loopIn: p, loopOut: p + dur, loopActive: true })); };
  const halveLoop = (id: DeckId): void => { if (!audio.current) return; const d = audio.current.decks[id]; if (d.loopIn === null || d.loopOut === null) return; d.loopOut = d.loopIn + (d.loopOut - d.loopIn) / 2; setters[id](s => s.loopIn !== null && s.loopOut !== null ? { ...s, loopOut: s.loopIn + (s.loopOut - s.loopIn) / 2 } : s); };
  const doubleLoop = (id: DeckId): void => { if (!audio.current) return; const d = audio.current.decks[id]; if (d.loopIn === null || d.loopOut === null) return; d.loopOut = Math.min(d.dur, d.loopIn + (d.loopOut - d.loopIn) * 2); setters[id](s => s.loopIn !== null && s.loopOut !== null ? { ...s, loopOut: Math.min(s.dur, s.loopIn + (s.loopOut - s.loopIn) * 2) } : s); };
  const beatJump = (id: DeckId, beats: number): void => { if (!audio.current) return; const d = audio.current.decks[id]; const st = id === 'A' ? deckARef.current : deckBRef.current; const bpm = parseFloat(st.bpm); if (!bpm || !d.buffer) return; seekDeck(id, posOf(d) + beats * (60 / bpm)); };

  // ─── sync ───
  const adjBpm = (st: DeckState): number | null => { const b = parseFloat(st.bpm); if (!b || isNaN(b)) return null; return b * (1 + tempoPercent(st.tempo, st.range) / 100); };
  const syncDeck = (id: DeckId): void => {
    const other = id === 'A' ? deckB : deckA; const self = id === 'A' ? deckA : deckB;
    const otherAdj = adjBpm(other); const selfBpm = parseFloat(self.bpm);
    if (!otherAdj || !selfBpm) { setSyncFlash('Load tracks with BPM on both decks'); setTimeout(() => setSyncFlash(null), 2000); return; }
    const v = 0.5 - ((otherAdj / selfBpm - 1) * 100) / (2 * self.range);
    if (v < 0 || v > 1) { setSyncFlash(`Range too narrow \u2014 widen to \u00B1${Math.ceil(Math.abs((otherAdj / selfBpm - 1) * 100))}% or more`); setTimeout(() => setSyncFlash(null), 3000); return; }
    setTempo(id, v); setSyncFlash(`Synced to ${otherAdj.toFixed(1)}`); setTimeout(() => setSyncFlash(null), 1500);
  };
  const ejectDeck = (id: DeckId): void => { if (!audio.current) return; const d = audio.current.decks[id]; if (d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } d.playing = false; } d.buffer = null; d.dur = 0; d.startOffset = 0; d.cue = 0; d.hotCues = [null, null, null, null]; d.loopIn = null; d.loopOut = null; d.loopActive = false; d.cuePreview = false; d.braking = false; setters[id](() => mkDeck()); };

  // ─── session recording ───
  const toggleRecording = (): void => {
    if (!audio.current) return;
    if (recording) { recorderRef.current?.stop(); setRecording(false); return; }
    recChunks.current = [];
    const mr = new MediaRecorder(audio.current.recordDest.stream, { mimeType: 'audio/webm;codecs=opus' });
    mr.ondataavailable = (e) => { if (e.data.size > 0) recChunks.current.push(e.data); };
    mr.onstop = () => { const blob = new Blob(recChunks.current, { type: 'audio/webm' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `mix-${new Date().toISOString().slice(0, 16)}.webm`; a.click(); URL.revokeObjectURL(url); };
    mr.start(1000); recorderRef.current = mr; setRecording(true);
  };

  // ─── effect: waveforms ───
  useEffect(() => { drawOverview(cvsA.current, deckA); }, [deckA.peaks, deckA.pos, deckA.dur, deckA.cue, deckA.hotCues, deckA.loopIn, deckA.loopOut, deckA.loopActive, deckA.bpm]);
  useEffect(() => { drawOverview(cvsB.current, deckB); }, [deckB.peaks, deckB.pos, deckB.dur, deckB.cue, deckB.hotCues, deckB.loopIn, deckB.loopOut, deckB.loopActive, deckB.bpm]);

  // ─── effect: mixer + FX + crossfader curve + cue routing ───
  useEffect(() => {
    if (!audio.current) return;
    const a = audio.current;
    const eq = (val: number): number => val >= 0.5 ? (val - 0.5) * 12 : -32 * Math.pow((0.5 - val) * 2, 2);
    const apply = (id: DeckId, ch: ChannelState) => {
      const d = a.decks[id];
      d.trim.gain.value = ch.trim * 1.4;
      d.low.gain.value = eq(ch.low); d.mid.gain.value = eq(ch.mid); d.hi.gain.value = eq(ch.hi);
      if (Math.abs(ch.color - 0.5) < 0.04) { d.color.type = 'lowpass'; d.color.frequency.value = 22000; }
      else if (ch.color < 0.5) { d.color.type = 'lowpass'; d.color.frequency.value = 150 + Math.pow(ch.color / 0.5, 2) * 18000; }
      else { d.color.type = 'highpass'; d.color.frequency.value = 20 + Math.pow((ch.color - 0.5) / 0.5, 2) * 8000; }
      d.chGain.gain.value = ch.fader * ch.fader;
    };
    apply('A', mix.chA); apply('B', mix.chB);
    // Crossfader curve
    const x = mix.xfader;
    if (mix.xfCurve === 'thru') { a.decks.A.xf.gain.value = x < 0.5 ? 1 : Math.max(0, 2 * (1 - x)); a.decks.B.xf.gain.value = x > 0.5 ? 1 : Math.max(0, 2 * x); }
    else if (mix.xfCurve === 'smooth') { a.decks.A.xf.gain.value = Math.cos(x * Math.PI / 2) * 1.414; a.decks.B.xf.gain.value = Math.cos((1 - x) * Math.PI / 2) * 1.414; }
    else { a.decks.A.xf.gain.value = x < 0.1 ? 1 : x > 0.9 ? 0 : 1 - (x - 0.1) / 0.8; a.decks.B.xf.gain.value = x > 0.9 ? 1 : x < 0.1 ? 0 : (x - 0.1) / 0.8; }
    a.master.gain.value = mix.master * mix.master;
    // FX
    a.echoWet.gain.value = mix.fxType === 'echo' ? mix.fxWet : 0;
    a.reverbWet.gain.value = mix.fxType === 'reverb' ? mix.fxWet * 0.6 : 0;
    // Split cue
    a.masterPanner.pan.value = mix.splitCue ? 1 : 0;
    a.cuePanner.pan.value = -1;
    a.decks.A.cueGain.gain.value = mix.cueA ? 1 : 0;
    a.decks.B.cueGain.gain.value = mix.cueB ? 1 : 0;
  }, [mix, initialized]);

  // ─── rAF ───
  useEffect(() => {
    if (!initialized) return;
    let raf: number;
    const buf = new Uint8Array(128);
    const rms = (an: AnalyserNode): number => { an.getByteTimeDomainData(buf); let s = 0; for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; s += v * v; } return Math.min(1, Math.sqrt(s / buf.length) * 2.2); };
    const tick = () => {
      const a = audio.current;
      if (a) {
        setVu({ A: rms(a.decks.A.analyser), B: rms(a.decks.B.analyser), M: rms(a.masterAnalyser) });
        (['A', 'B'] as DeckId[]).forEach(id => {
          const d = a.decks[id];
          const el = id === 'A' ? platterRefA.current : platterRefB.current;
          if (el) { if (d.playing && !d.braking) platterAngle.current[id] += d.effRate * 3.33; el.style.transform = `rotate(${platterAngle.current[id]}deg)`; }
          if (d.playing) {
            const p = posOf(d);
            if (d.loopActive && d.loopIn !== null && d.loopOut !== null && p >= d.loopOut) { d.stopping = true; try { d.src?.stop(); } catch { /* ok */ } d.startOffset = d.loopIn; startSrc(a, d, id); }
            setters[id](s => (Math.abs(s.pos - p) > 0.02 ? { ...s, pos: p } : s));
          }
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

  // ─── keyboard ───
  useEffect(() => {
    if (!initialized) return;
    const onDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      switch (e.key) {
        case 'q': e.preventDefault(); cuePress('A'); break;
        case 'w': e.preventDefault(); if (!e.repeat) { audio.current?.decks.A.playing ? pauseDeck('A') : playDeck('A'); } break;
        case '1': hotCueTap('A', 0); break; case '2': hotCueTap('A', 1); break; case '3': hotCueTap('A', 2); break; case '4': hotCueTap('A', 3); break;
        case 'a': setLoopInPoint('A'); break; case 's': e.preventDefault(); setLoopOutPoint('A'); break; case 'd': e.preventDefault(); toggleLoop('A'); break;
        case 'z': e.preventDefault(); beatJump('A', -4); break; case 'x': e.preventDefault(); beatJump('A', -1); break;
        case 'c': e.preventDefault(); beatJump('A', 1); break; case 'v': e.preventDefault(); beatJump('A', 4); break;
        case '[': e.preventDefault(); cuePress('B'); break;
        case ']': e.preventDefault(); if (!e.repeat) { audio.current?.decks.B.playing ? pauseDeck('B') : playDeck('B'); } break;
        case '7': hotCueTap('B', 0); break; case '8': hotCueTap('B', 1); break; case '9': hotCueTap('B', 2); break; case '0': hotCueTap('B', 3); break;
        case 'l': setLoopInPoint('B'); break; case ';': setLoopOutPoint('B'); break; case "'": toggleLoop('B'); break;
        case ',': beatJump('B', -4); break; case '.': beatJump('B', -1); break; case '/': e.preventDefault(); beatJump('B', 1); break;
      }
    };
    const onUp = (e: KeyboardEvent) => { if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return; if (e.key === 'q') cueRelease('A'); if (e.key === '[') cueRelease('B'); };
    window.addEventListener('keydown', onDown); window.addEventListener('keyup', onUp);
    return () => { window.removeEventListener('keydown', onDown); window.removeEventListener('keyup', onUp); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialized]);

  // ─── drill: track phase accuracy when both playing ───
  useEffect(() => {
    if (!drillActive || !deckA.playing || !deckB.playing) return;
    const bA = adjBpm(deckA), bB = adjBpm(deckB);
    if (!bA || !bB) return;
    const phA = (deckA.pos * bA / 60) % 1, phB = (deckB.pos * bB / 60) % 1;
    let diff = phA - phB; if (diff > 0.5) diff -= 1; if (diff < -0.5) diff += 1;
    const ms = Math.abs(diff * 60000 / ((bA + bB) / 2));
    if (Math.abs(bA - bB) < 0.2 && ms < 15) {
      const elapsed = (Date.now() - drillStart) / 1000;
      if (!drillBest || elapsed < drillBest) setDrillBest(elapsed);
      setDrillActive(false);
      setSyncFlash(`DRILL COMPLETE \u2014 ${elapsed.toFixed(1)}s${drillBest ? ` (best: ${Math.min(elapsed, drillBest).toFixed(1)}s)` : ''}`);
      setTimeout(() => setSyncFlash(null), 4000);
    }
  }, [deckA.pos, deckB.pos, drillActive, drillStart, drillBest, deckA, deckB]);

  // ═══════════════════════════════════════════════════════════
  // INIT
  // ═══════════════════════════════════════════════════════════
  if (!initialized) {
    return (
      <div className="flex flex-col items-center justify-center select-none" style={{ background: C.bg, minHeight: '100dvh', fontFamily: 'Oxanium', color: C.text, padding: 24 }}>
        <div style={{ fontWeight: 700, fontSize: 'clamp(28px, 6vw, 42px)', letterSpacing: 6, textAlign: 'center' }}>VIRTUAL <span style={{ color: C.cyan }}>BOOTH</span></div>
        <div style={{ color: C.dim, fontSize: 'clamp(10px, 2vw, 14px)', fontFamily: "'IBM Plex Mono'", marginTop: 8, letterSpacing: 2, textAlign: 'center' }}>CDJ-2000NXS &times;2 &middot; DJM-900NXS</div>
        <div style={{ marginTop: 16, color: C.dim, fontSize: 12, fontFamily: "'IBM Plex Mono'", textAlign: 'center', lineHeight: 1.6 }}>
          Beat FX &middot; Split headphone cue &middot; Practice drills &middot; Session recording<br />
          Auto BPM + Key detection &middot; Track normalization &middot; Phrase markers
        </div>
        <button onClick={initAudio} className="touch-target" style={{ marginTop: 40, padding: '16px 48px', borderRadius: 8, background: 'transparent', border: `2px solid ${C.cyan}`, color: C.cyan, fontFamily: 'Oxanium', fontWeight: 700, fontSize: 16, letterSpacing: 4, cursor: 'pointer' }}>START SESSION</button>
        <div style={{ marginTop: 32, color: C.dim, fontSize: 11, fontFamily: "'IBM Plex Mono'", textAlign: 'center', maxWidth: 440, lineHeight: 1.7 }}>
          Load your own tracks. On mobile, rotate to landscape.<br />Split cue: left ear = cue, right ear = master.
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
  const phaseMs = (() => { if (!bpmA || !bpmB) return null; const phA = (deckA.pos * bpmA / 60) % 1, phB = (deckB.pos * bpmB / 60) % 1; let d = phA - phB; if (d > 0.5) d -= 1; if (d < -0.5) d += 1; return d * 60000 / ((bpmA + bpmB) / 2); })();
  const phaseOk = phaseMs !== null && Math.abs(phaseMs) < 15;
  const phaseTight = phaseMs !== null && Math.abs(phaseMs) < 30;
  const jogSize = isMobile ? 120 : 150;
  const knobSize = isMobile ? 36 : 44;
  const faderH = isMobile ? 85 : 110;
  const keyA = toCamelot(deckA.key), keyB = toCamelot(deckB.key);
  const keysMatch = keysCompatible(deckA.key, deckB.key);

  const renderDeck = (id: DeckId) => {
    const st = id === 'A' ? deckA : deckB;
    const cvsRef = id === 'A' ? cvsA : cvsB;
    const zoomRef = id === 'A' ? zoomA : zoomB;
    const plRef = id === 'A' ? platterRefA : platterRefB;
    const remaining = (st.dur || 0) - st.pos;
    const endWarn = st.loaded && st.playing && remaining < 30;
    const endCrit = endWarn && remaining < 10;
    const dc = id === 'A' ? C.cyan : C.orange;
    const bp = parseFloat(st.bpm) || 0;
    return (
      <div className="rounded-lg p-2 flex flex-col gap-1 relative" style={{ background: C.panel, border: `1px solid ${C.edge}`, flex: 1, minWidth: isMobile ? 0 : 260 }}>
        {loading[id] && <div className="absolute inset-0 flex flex-col items-center justify-center rounded-lg" style={{ background: 'rgba(8,8,10,0.92)', zIndex: 20 }}><div style={{ color: C.cyan, fontSize: 13, fontWeight: 700, letterSpacing: 3 }}>ANALYZING</div><div style={{ color: C.dim, fontSize: 10, marginTop: 4 }}>BPM + Key + Normalization...</div></div>}
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span style={{ fontWeight: 700, color: dc, letterSpacing: 2, fontSize: 11 }}>DECK {id}</span>
            <BeatIndicator pos={st.pos} bpm={bp} />
            {st.key && <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 9, color: C.purple, background: `${C.purple}18`, padding: '1px 4px', borderRadius: 3 }}>{toCamelot(st.key) || st.key}</span>}
            {st.quantize && <span style={{ fontSize: 8, color: C.green, background: `${C.green}18`, padding: '1px 3px', borderRadius: 2 }}>Q</span>}
          </div>
          <span style={{ fontFamily: "'IBM Plex Mono'", color: endCrit ? C.red : endWarn ? C.orange : C.dim, fontSize: 9 }} className={endCrit ? 'warning-flash' : ''}>{st.loaded ? fmtTime(st.pos) : '--:--'} / -{fmtTime(remaining)}</span>
        </div>
        {/* Zoomed waveform */}
        <div className="rounded overflow-hidden relative" style={{ background: '#0a0a0e', border: `1px solid ${C.edge}` }}>
          <canvas ref={zoomRef} width={400} height={56} style={{ width: '100%', height: isMobile ? 40 : 50, display: 'block' }} />
          {!st.loaded && <label htmlFor={`file-${id}`} className="absolute inset-0 flex items-center justify-center cursor-pointer" style={{ background: 'rgba(10,10,14,0.85)' }}><span style={{ color: C.dim, fontSize: 11 }}>TAP TO LOAD TRACK</span></label>}
        </div>
        {/* Overview */}
        <div className="rounded overflow-hidden" style={{ background: '#0c0c10', border: `1px solid ${C.edge}`, cursor: 'pointer' }}>
          <canvas ref={cvsRef} width={400} height={28} style={{ width: '100%', height: isMobile ? 18 : 24, display: 'block' }}
            onClick={e => { const r = e.currentTarget.getBoundingClientRect(); seekDeck(id, ((e.clientX - r.left) / r.width) * st.dur); }}
            onPointerDown={e => { if (e.pointerType === 'touch') { const r = e.currentTarget.getBoundingClientRect(); seekDeck(id, ((e.clientX - r.left) / r.width) * st.dur); } }} />
          <div className="flex items-center justify-center px-1" style={{ fontFamily: "'IBM Plex Mono'", fontSize: 8, color: C.dim, height: 12 }}><span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{st.name || ''}</span></div>
        </div>
        {/* Jog */}
        <div className="flex items-center justify-center">
          <div onPointerDown={e => { e.preventDefault(); jogDown(id, e); }} onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadFile(id, f); }} onDragOver={e => e.preventDefault()}
            onMouseEnter={() => showHint('JOG: Playing=pitch-bend, Paused=scrub.')}
            className="rounded-full relative cursor-grab active:cursor-grabbing select-none flex items-center justify-center"
            style={{ width: jogSize, height: jogSize, touchAction: 'none', background: 'radial-gradient(circle at 50% 45%, #2c2c34, #16161a 55%, #0c0c0e)', border: '3px solid #1a1a1e', boxShadow: '0 4px 16px rgba(0,0,0,.5), inset 0 0 24px rgba(0,0,0,.6)' }}>
            <div className="absolute rounded-full" style={{ inset: 6, border: '1px dashed #2a2a32' }} />
            <div ref={plRef} className="rounded-full relative flex items-center justify-center" style={{ width: jogSize * 0.64, height: jogSize * 0.64, background: 'radial-gradient(circle, #3a3a44, #1c1c22)', border: `1px solid ${C.edge}` }}>
              <div className="absolute" style={{ width: jogSize * 0.53, height: 2, background: st.playing ? dc : C.dim, opacity: 0.5 }} />
              <div className="rounded-full" style={{ width: jogSize * 0.2, height: jogSize * 0.2, background: '#0c0c10', border: `2px solid ${st.playing ? dc : C.edge}`, boxShadow: st.playing ? `0 0 8px ${dc}` : 'none' }} />
            </div>
          </div>
        </div>
        {/* Transport */}
        <div className="flex items-center gap-1.5 justify-center flex-wrap">
          <button onPointerDown={() => cuePress(id)} onPointerUp={() => cueRelease(id)} onPointerLeave={() => cueRelease(id)}
            className="rounded-full flex items-center justify-center touch-target"
            style={{ width: 44, height: 44, background: !st.playing ? `${C.orange}18` : '#1a1a1e', border: `2px solid ${C.orange}`, color: C.orange, fontFamily: 'Oxanium', fontWeight: 700, fontSize: 10, boxShadow: !st.playing && st.loaded ? `0 0 10px ${C.orange}55` : 'none' }}>CUE</button>
          <button onPointerDown={() => st.playing ? pauseDeck(id) : playDeck(id)}
            className="rounded-full flex items-center justify-center touch-target"
            style={{ width: 44, height: 44, background: st.playing ? `${C.green}18` : '#1a1a1e', border: `2px solid ${C.green}`, color: C.green, fontFamily: 'Oxanium', fontWeight: 700, fontSize: 16, boxShadow: st.playing ? `0 0 10px ${C.green}55` : 'none' }}>{st.playing ? '\u275A\u275A' : '\u25B6'}</button>
          <button onClick={() => syncDeck(id)} className="rounded flex items-center justify-center" style={{ height: 32, padding: '0 6px', background: '#1a1a1e', border: `1px solid ${C.cyan}`, color: C.cyan, fontFamily: 'Oxanium', fontWeight: 700, fontSize: 8 }}>SYNC</button>
          <button onClick={() => setters[id](s => ({ ...s, quantize: !s.quantize }))}
            onMouseEnter={() => showHint('QUANTIZE: snap hot cues and loops to the nearest beat.')}
            className="rounded flex items-center justify-center" style={{ height: 32, padding: '0 5px', background: st.quantize ? `${C.green}18` : '#1a1a1e', border: `1px solid ${st.quantize ? C.green : C.edge}`, color: st.quantize ? C.green : C.dim, fontFamily: 'Oxanium', fontWeight: 600, fontSize: 8 }}>QTZ</button>
          <label htmlFor={`file-${id}`} className="rounded flex items-center justify-center cursor-pointer" style={{ height: 32, padding: '0 6px', background: '#1a1a1e', border: `1px solid ${C.edge}`, color: C.dim, fontFamily: 'Oxanium', fontSize: 8 }}>
            LOAD<input id={`file-${id}`} type="file" accept="audio/*,.mp3,.wav,.m4a,.aac,.ogg,.flac,.mp4" style={{ position: 'absolute', width: 1, height: 1, opacity: 0, overflow: 'hidden', pointerEvents: 'none' }} onChange={e => { const f = e.target.files?.[0]; if (f) { loadFile(id, f); e.target.value = ''; } }} /></label>
          <button onClick={() => ejectDeck(id)} className="rounded flex items-center justify-center" style={{ height: 32, padding: '0 4px', background: '#1a1a1e', border: `1px solid ${C.edge}`, color: C.dim, fontSize: 10 }}>{'\u23CF'}</button>
        </div>
        {/* Hot cues */}
        <div className="flex items-center gap-1 justify-center">
          {st.hotCues.map((hc, i) => (<button key={i} onClick={() => hotCueTap(id, i)}
            onMouseEnter={() => showHint(`HOT CUE ${HC_LABELS[i]}: ${hc !== null ? `${fmtTime(hc)}. Tap=jump. 2x tap=clear.` : 'Tap=save position.'}`)}
            className="rounded flex items-center justify-center relative" style={{ width: 32, height: 28, fontSize: 9, fontFamily: 'Oxanium', fontWeight: 700,
              background: hc !== null ? `${HC_COLORS[i]}18` : '#1a1a1e', border: `2px solid ${hc !== null ? HC_COLORS[i] : C.edge}`, color: hc !== null ? HC_COLORS[i] : C.dim,
              boxShadow: hc !== null ? `0 0 5px ${HC_COLORS[i]}33` : 'none' }}>{HC_LABELS[i]}{hc !== null && <span className="absolute" style={{ top: -2, right: -2, width: 6, height: 6, borderRadius: '50%', background: HC_COLORS[i] }} />}</button>))}
        </div>
        {/* Loops */}
        <div className="flex items-center gap-0.5 justify-center flex-wrap">
          <button onClick={() => setLoopInPoint(id)} className="rounded flex items-center justify-center" style={{ height: 24, padding: '0 4px', fontSize: 7, fontFamily: 'Oxanium', fontWeight: 600, background: st.loopIn !== null ? `${C.green}18` : '#1a1a1e', border: `1px solid ${st.loopIn !== null ? C.green : C.edge}`, color: st.loopIn !== null ? C.green : C.dim }}>IN</button>
          <button onClick={() => setLoopOutPoint(id)} className="rounded flex items-center justify-center" style={{ height: 24, padding: '0 4px', fontSize: 7, fontFamily: 'Oxanium', fontWeight: 600, background: st.loopOut !== null ? `${C.green}18` : '#1a1a1e', border: `1px solid ${st.loopOut !== null ? C.green : C.edge}`, color: st.loopOut !== null ? C.green : C.dim }}>OUT</button>
          {[0.5, 1, 2, 4, 8].map(b => (<button key={b} onClick={() => beatLoop(id, b)} className="rounded flex items-center justify-center" style={{ height: 24, padding: '0 3px', fontSize: 7, fontFamily: "'IBM Plex Mono'", background: '#1a1a1e', border: `1px solid ${C.edge}`, color: bp > 0 ? C.text : C.dim }}>{b < 1 ? '\u00BD' : b}</button>))}
          <button onClick={() => halveLoop(id)} className="rounded flex items-center justify-center" style={{ height: 24, padding: '0 3px', fontSize: 8, fontFamily: 'Oxanium', background: '#1a1a1e', border: `1px solid ${st.loopActive ? C.green : C.edge}`, color: st.loopActive ? C.green : C.dim }}>/2</button>
          <button onClick={() => doubleLoop(id)} className="rounded flex items-center justify-center" style={{ height: 24, padding: '0 3px', fontSize: 8, fontFamily: 'Oxanium', background: '#1a1a1e', border: `1px solid ${st.loopActive ? C.green : C.edge}`, color: st.loopActive ? C.green : C.dim }}>x2</button>
          <button onClick={() => toggleLoop(id)} className="rounded flex items-center justify-center" style={{ height: 24, padding: '0 4px', fontSize: 7, fontFamily: 'Oxanium', fontWeight: 700, background: st.loopActive ? `${C.green}22` : '#1a1a1e', border: `1px solid ${st.loopActive ? C.green : C.edge}`, color: st.loopActive ? C.green : C.dim, boxShadow: st.loopActive ? `0 0 5px ${C.green}33` : 'none' }}>{st.loopActive ? '\u27F2' : 'LP'}</button>
          {(st.loopIn !== null) && <button onClick={() => clearLoop(id)} className="rounded flex items-center justify-center" style={{ height: 24, padding: '0 3px', fontSize: 7, fontFamily: 'Oxanium', background: '#1a1a1e', border: `1px solid ${C.red}44`, color: C.red }}>X</button>}
        </div>
        {/* Beat jump */}
        <div className="flex items-center gap-0.5 justify-center">
          <span style={{ fontSize: 7, color: C.dim, letterSpacing: 1 }}>JUMP</span>
          {[{ b: -8, l: '\u00AB8' }, { b: -4, l: '\u00AB4' }, { b: -1, l: '\u2039' }, { b: 1, l: '\u203A' }, { b: 4, l: '4\u00BB' }, { b: 8, l: '8\u00BB' }].map(({ b, l }) => (
            <button key={b} onClick={() => beatJump(id, b)} className="rounded flex items-center justify-center" style={{ width: 24, height: 22, fontSize: 8, fontFamily: 'Oxanium', background: '#1a1a1e', border: `1px solid ${C.edge}`, color: bp > 0 ? C.text : C.dim }}>{l}</button>))}
        </div>
        {/* Tempo */}
        <div className="flex items-stretch gap-2 justify-between">
          <div className="flex flex-col items-center" style={{ flex: 1 }}>
            <div className="flex items-center gap-0.5 mb-0.5">
              {([6, 10, 16, 100] as const).map(r => (<button key={r} onClick={() => setters[id](s => ({ ...s, range: r }))} style={{ fontFamily: 'Oxanium', fontSize: 7, padding: '1px 3px', borderRadius: 3, background: st.range === r ? C.cyanDim : '#1a1a1e', color: st.range === r ? C.cyan : C.dim, border: `1px solid ${C.edge}` }}>{r === 100 ? 'WIDE' : `\u00B1${r}%`}</button>))}
            </div>
            <Fader value={st.tempo} onChange={v => setTempo(id, v)} label="TEMPO" color={dc} height={faderH} center
              onHint={showHint} hint="TEMPO: pull DOWN=faster, push UP=slower (matches real CDJ)." />
          </div>
          <div className="flex flex-col items-center justify-end gap-1" style={{ width: isMobile ? 75 : 85 }}>
            <div className="w-full rounded text-center" style={{ background: '#0c0c10', border: `1px solid ${C.edge}`, padding: 2 }}>
              <div style={{ fontFamily: "'IBM Plex Mono'", fontSize: 13, color: dc }}>{tempoPercent(st.tempo, st.range) >= 0 ? '+' : ''}{tempoPercent(st.tempo, st.range).toFixed(1)}%</div>
            </div>
            <input value={st.bpm} onChange={e => setters[id](s => ({ ...s, bpm: e.target.value.replace(/[^\d.]/g, '') }))} placeholder="BPM" inputMode="decimal"
              style={{ width: '100%', background: '#0c0c10', border: `1px solid ${C.edge}`, color: C.text, fontFamily: "'IBM Plex Mono'", fontSize: 11, padding: '2px 4px', borderRadius: 4, textAlign: 'center', outline: 'none' }} />
            <div className="w-full text-center" style={{ fontFamily: "'IBM Plex Mono'", fontSize: 10, color: adjBpm(st) ? C.green : C.dim }}>{adjBpm(st) ? `\u2192 ${adjBpm(st)!.toFixed(1)}` : '\u2014'}</div>
          </div>
        </div>
      </div>
    );
  };

  const renderChannelStrip = (id: DeckId) => {
    const ch = id === 'A' ? mix.chA : mix.chB;
    const setCh = (p: Partial<ChannelState>) => setMix(m => id === 'A' ? { ...m, chA: { ...m.chA, ...p } } : { ...m, chB: { ...m.chB, ...p } });
    const sc = id === 'A' ? C.cyan : C.orange;
    const cueOn = id === 'A' ? mix.cueA : mix.cueB;
    return (
      <div className="flex flex-col items-center gap-1 rounded-lg p-1.5" style={{ background: C.panelHi, border: `1px solid ${C.edge}` }}>
        <span style={{ fontWeight: 700, color: sc, fontSize: 10 }}>CH {id === 'A' ? '1' : '2'}</span>
        <Knob value={ch.trim} onChange={v => setCh({ trim: v })} label="TRIM" size={knobSize} onHint={showHint} hint="TRIM: auto-set on load. Adjust if needed." />
        <Knob value={ch.hi} onChange={v => setCh({ hi: v })} label="HI" size={knobSize} onHint={showHint} hint="HI EQ: treble. Full left=kill." />
        <Knob value={ch.mid} onChange={v => setCh({ mid: v })} label="MID" size={knobSize} onHint={showHint} hint="MID EQ: vocals/snares." />
        <Knob value={ch.low} onChange={v => setCh({ low: v })} label="LOW" color={C.orange} size={knobSize} onHint={showHint} hint="LOW EQ: bass/kick. Swap lows between tracks." />
        <div className="flex items-end gap-1">
          <Fader value={ch.fader} onChange={v => setCh({ fader: v })} color={sc} height={faderH} onHint={showHint} hint="CHANNEL FADER" />
          <VUMeter level={id === 'A' ? vu.A : vu.B} />
        </div>
        <Knob value={ch.color} onChange={v => setCh({ color: v })} label="COLOR" size={knobSize - 8} onHint={showHint} hint="COLOR FILTER: L=LP, R=HP." />
        {/* Headphone CUE button */}
        <button onClick={() => setMix(m => id === 'A' ? { ...m, cueA: !m.cueA } : { ...m, cueB: !m.cueB })}
          onMouseEnter={() => showHint('CUE: route this channel to headphone (left ear when split cue is ON).')}
          className="rounded flex items-center justify-center" style={{ width: '100%', height: 22, fontSize: 8, fontFamily: 'Oxanium', fontWeight: 700,
            background: cueOn ? `${sc}22` : '#1a1a1e', border: `1px solid ${cueOn ? sc : C.edge}`, color: cueOn ? sc : C.dim }}>
          {'\uD83C\uDFA7'} CUE
        </button>
      </div>
    );
  };

  const renderMixer = () => (
    <div className="rounded-lg p-2 flex flex-col gap-1.5" style={{ background: C.panel, border: `1px solid ${C.edge}`, minWidth: isMobile ? 0 : 190 }}>
      <div className="flex items-center justify-between">
        <span style={{ fontWeight: 700, color: C.cyan, letterSpacing: 2, fontSize: 11 }}>MIXER</span>
        <span style={{ color: C.dim, fontSize: 8 }}>DJM-900NXS</span>
      </div>
      <div className="flex gap-1.5 justify-center">{renderChannelStrip('A')}{renderChannelStrip('B')}</div>
      {/* FX Section */}
      <div className="rounded p-1.5" style={{ background: '#0c0c10', border: `1px solid ${C.edge}` }}>
        <div className="flex items-center gap-1 justify-center mb-1">
          <span style={{ fontSize: 8, color: C.dim, letterSpacing: 1 }}>BEAT FX</span>
          {(['off', 'echo', 'reverb'] as FxType[]).map(t => (
            <button key={t} onClick={() => setMix(m => ({ ...m, fxType: t }))}
              style={{ fontSize: 7, padding: '2px 5px', borderRadius: 3, fontFamily: 'Oxanium', fontWeight: 600, textTransform: 'uppercase',
                background: mix.fxType === t ? `${C.purple}22` : '#1a1a1e', border: `1px solid ${mix.fxType === t ? C.purple : C.edge}`, color: mix.fxType === t ? C.purple : C.dim }}>{t}</button>
          ))}
        </div>
        {mix.fxType !== 'off' && (
          <Knob value={mix.fxWet} onChange={v => setMix(m => ({ ...m, fxWet: v }))} label="WET/DRY" color={C.purple} size={knobSize - 6}
            onHint={showHint} hint={`${mix.fxType.toUpperCase()} wet/dry mix. Turn up to add more effect.`} />
        )}
      </div>
      {/* Master + Crossfader */}
      <div className="flex items-center gap-1.5">
        <Knob value={mix.master} onChange={v => setMix(m => ({ ...m, master: v }))} label="MASTER" size={knobSize} onHint={showHint} hint="MASTER: overall output." />
        <VUMeter level={vu.M} />
        <button onClick={() => setMix(m => ({ ...m, splitCue: !m.splitCue }))}
          onMouseEnter={() => showHint('SPLIT CUE: left ear=cued channel, right ear=master. Use headphones.')}
          className="rounded flex items-center justify-center" style={{ height: 24, padding: '0 5px', fontSize: 7, fontFamily: 'Oxanium', fontWeight: 600,
            background: mix.splitCue ? `${C.cyan}22` : '#1a1a1e', border: `1px solid ${mix.splitCue ? C.cyan : C.edge}`, color: mix.splitCue ? C.cyan : C.dim }}>
          SPLIT{'\uD83C\uDFA7'}
        </button>
      </div>
      <div>
        <div className="flex items-center justify-between px-1 mb-0.5">
          <span style={{ fontSize: 7, color: C.dim, letterSpacing: 1 }}>CROSSFADER</span>
          <div className="flex gap-0.5">
            {(['thru', 'smooth', 'sharp'] as XfCurve[]).map(c => (
              <button key={c} onClick={() => setMix(m => ({ ...m, xfCurve: c }))}
                style={{ fontSize: 6, padding: '1px 3px', borderRadius: 2, fontFamily: 'Oxanium',
                  background: mix.xfCurve === c ? C.cyanDim : '#1a1a1e', color: mix.xfCurve === c ? C.cyan : C.dim, border: `1px solid ${C.edge}` }}>{c}</button>
            ))}
          </div>
        </div>
        <div className="px-1"><CrossfaderH value={mix.xfader} onChange={v => setMix(m => ({ ...m, xfader: v }))} /></div>
        <div className="flex justify-between px-1" style={{ fontSize: 9, color: C.dim }}><span style={{ color: C.cyan }}>A</span><span style={{ color: C.orange }}>B</span></div>
      </div>
      {/* Recording */}
      <button onClick={toggleRecording}
        onMouseEnter={() => showHint(recording ? 'Stop recording and download your mix.' : 'Record your mix session. Downloads as WebM when stopped.')}
        className="rounded flex items-center justify-center gap-1" style={{ height: 26, fontSize: 8, fontFamily: 'Oxanium', fontWeight: 600,
          background: recording ? `${C.red}22` : '#1a1a1e', border: `1px solid ${recording ? C.red : C.edge}`, color: recording ? C.red : C.dim }}>
        {recording ? '\u25CF REC STOP' : '\u25CF REC'}
      </button>
    </div>
  );

  // ═══════════════════════════════════════════════════════════
  // MAIN LAYOUT
  // ═══════════════════════════════════════════════════════════
  return (
    <div className="select-none" style={{ background: C.bg, minHeight: '100dvh', fontFamily: 'Oxanium', color: C.text }}>
      <div className="portrait-block flex-col items-center justify-center gap-4" style={{ position: 'fixed', inset: 0, background: C.bg, zIndex: 100, padding: 32 }}>
        <div style={{ fontSize: 48 }}>{'\u21BB'}</div>
        <div style={{ fontWeight: 700, fontSize: 20, letterSpacing: 3 }}>ROTATE TO LANDSCAPE</div>
        <div style={{ color: C.dim, fontSize: 12, fontFamily: "'IBM Plex Mono'", textAlign: 'center', lineHeight: 1.6 }}>Turn your phone sideways to mix.</div>
      </div>

      <div className="landscape-content" style={{ padding: isMobile ? 3 : 6 }}>
        {/* Top bar */}
        <div className="flex items-center justify-between mb-1 flex-wrap gap-1">
          <div className="flex items-center gap-2">
            <span style={{ fontWeight: 700, fontSize: isMobile ? 13 : 16, letterSpacing: 3 }}>VIRTUAL <span style={{ color: C.cyan }}>BOOTH</span></span>
          </div>
          <div className="flex items-center gap-1">
            {/* Practice drill */}
            <button onClick={() => { if (drillActive) { setDrillActive(false); } else { setDrillActive(true); setDrillStart(Date.now()); setSyncFlash('DRILL: Match BPM + Phase. Go!'); setTimeout(() => setSyncFlash(null), 2000); } }}
              style={{ background: drillActive ? `${C.orange}22` : C.panel, border: `1px solid ${drillActive ? C.orange : C.edge}`, color: drillActive ? C.orange : C.dim, padding: '3px 7px', borderRadius: 4, fontSize: 9, fontWeight: 700 }}>
              {drillActive ? `DRILL ${((Date.now() - drillStart) / 1000).toFixed(0)}s` : 'DRILL'}{drillBest && !drillActive ? ` (${drillBest.toFixed(1)}s)` : ''}
            </button>
            {!isMobile && <button onClick={() => setShowKeys(v => !v)} style={{ background: C.panel, border: `1px solid ${showKeys ? C.cyan : C.edge}`, color: showKeys ? C.cyan : C.dim, padding: '3px 7px', borderRadius: 4, fontSize: 9 }}>Keys</button>}
            <button onClick={() => setShowGuide(v => !v)} style={{ background: C.panel, border: `1px solid ${C.edge}`, color: C.text, padding: '3px 7px', borderRadius: 4, fontSize: 9 }}>Guide</button>
            <button onClick={() => setLearn(v => !v)} style={{ background: learn ? C.cyanDim : C.panel, border: `1px solid ${learn ? C.cyan : C.edge}`, color: learn ? C.cyan : C.dim, padding: '3px 7px', borderRadius: 4, fontSize: 9 }}>Learn</button>
          </div>
        </div>

        {/* BoothMatch + Phase + Key */}
        <div className="rounded mb-1 px-2 py-1 flex items-center justify-center gap-2 flex-wrap"
          style={{ background: C.panel, border: `1px solid ${matched && phaseOk ? C.green : matched ? C.yellow : C.edge}`, transition: 'border-color 0.3s' }}>
          <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 11, color: C.cyan }}>A {bpmA ? bpmA.toFixed(1) : '--'}</span>
          <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 9, color: matched ? C.green : C.dim }}>{bpmA && bpmB ? `\u0394${Math.abs(bpmA - bpmB).toFixed(2)}` : '\u00B7'}</span>
          <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 11, color: C.orange }}>B {bpmB ? bpmB.toFixed(1) : '--'}</span>
          {matched && <span style={{ color: C.green, fontWeight: 700, fontSize: 8 }}>{'\u25CF'}</span>}
          {/* Phase */}
          {deckA.playing && deckB.playing && phaseMs !== null && (<>
            <div style={{ width: 1, height: 14, background: C.edge }} />
            <div style={{ width: 50, height: 6, background: '#1a1a1e', borderRadius: 3, position: 'relative', border: `1px solid ${C.edge}` }}>
              <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: '#333' }} />
              <div style={{ position: 'absolute', top: -1, left: `${50 + Math.max(-50, Math.min(50, (phaseMs / 200) * 50))}%`, width: 5, height: 8, borderRadius: 3,
                background: phaseOk ? C.green : phaseTight ? C.yellow : C.red, transform: 'translateX(-50%)', boxShadow: `0 0 3px ${phaseOk ? C.green : phaseTight ? C.yellow : C.red}` }} />
            </div>
            <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 9, minWidth: 40, textAlign: 'right', color: phaseOk ? C.green : phaseTight ? C.yellow : C.red }}>{phaseMs > 0 ? '+' : ''}{phaseMs.toFixed(0)}ms</span>
          </>)}
          {/* Key compatibility */}
          {keyA && keyB && (<>
            <div style={{ width: 1, height: 14, background: C.edge }} />
            <span style={{ fontSize: 8, color: C.purple }}>{keyA}</span>
            <span style={{ fontSize: 8, color: keysMatch ? C.green : C.red }}>{keysMatch ? '\u2713' : '\u2717'}</span>
            <span style={{ fontSize: 8, color: C.purple }}>{keyB}</span>
          </>)}
        </div>

        {syncFlash && <div className="rounded mb-1 px-2 py-1 text-center" style={{ background: `${C.cyan}12`, border: `1px solid ${C.cyan}33`, fontSize: 10, color: C.cyan }}>{syncFlash}</div>}

        {isMobile && (
          <div className="flex gap-1 mb-1">{(['A', 'mix', 'B'] as const).map(tab => (
            <button key={tab} onClick={() => setMobileTab(tab)} style={{ flex: 1, padding: '5px 0', borderRadius: 4, fontSize: 10, fontWeight: 700, letterSpacing: 2,
              background: mobileTab === tab ? (tab === 'A' ? C.cyanDim : tab === 'B' ? `${C.orange}22` : C.panelHi) : C.panel,
              border: `1px solid ${mobileTab === tab ? (tab === 'A' ? C.cyan : tab === 'B' ? C.orange : C.edge) : C.edge}`,
              color: mobileTab === tab ? (tab === 'A' ? C.cyan : tab === 'B' ? C.orange : C.text) : C.dim }}>{tab === 'mix' ? 'MIXER' : `DECK ${tab}`}</button>
          ))}</div>
        )}

        {isMobile ? (
          <div>{mobileTab === 'A' && renderDeck('A')}{mobileTab === 'mix' && renderMixer()}{mobileTab === 'B' && renderDeck('B')}</div>
        ) : (
          <div className="flex gap-2 items-start justify-center" style={{ flexWrap: isCompact ? 'wrap' : 'nowrap' }}>{renderDeck('A')}{renderMixer()}{renderDeck('B')}</div>
        )}

        {/* Guide overlay */}
        {showGuide && (
          <div style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.7)' }} onClick={() => setShowGuide(false)}>
            <div className="rounded-lg p-4" style={{ background: C.panel, border: `1px solid ${C.edge}`, maxWidth: 500, width: '90%' }} onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-2"><span style={{ fontWeight: 700, color: C.cyan, letterSpacing: 2, fontSize: 12 }}>FIRST MIX</span><button onClick={() => setShowGuide(false)} style={{ color: C.dim, fontSize: 18, background: 'none', border: 'none', cursor: 'pointer' }}>&times;</button></div>
              <div style={{ fontSize: 13, fontFamily: "'IBM Plex Mono'", color: C.text, lineHeight: 1.7 }}><span style={{ color: C.orange, fontWeight: 600 }}>Step {guideStep + 1}. </span>{GUIDE[guideStep]}</div>
              <div className="flex gap-2 mt-3 items-center">
                <button onClick={() => setGuideStep(s => Math.max(0, s - 1))} style={{ background: C.panelHi, border: `1px solid ${C.edge}`, color: C.text, padding: '6px 14px', borderRadius: 4, fontSize: 12 }}>{'\u2190'}</button>
                <button onClick={() => setGuideStep(s => Math.min(5, s + 1))} style={{ background: C.cyanDim, border: `1px solid ${C.cyan}`, color: C.cyan, padding: '6px 14px', borderRadius: 4, fontSize: 12 }}>{'\u2192'}</button>
                <div className="flex gap-1 ml-2">{GUIDE.map((_, i) => (<div key={i} onClick={() => setGuideStep(i)} className="cursor-pointer rounded-full" style={{ width: 8, height: 8, background: i === guideStep ? C.cyan : '#2a2a32' }} />))}</div>
              </div>
            </div>
          </div>
        )}
        {showKeys && !isMobile && (
          <div style={{ position: 'fixed', inset: 0, zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.7)' }} onClick={() => setShowKeys(false)}>
            <div className="rounded-lg p-4" style={{ background: C.panel, border: `1px solid ${C.edge}`, maxWidth: 500, width: '90%' }} onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-2"><span style={{ fontWeight: 700, color: C.cyan, letterSpacing: 2, fontSize: 12 }}>KEYBOARD</span><button onClick={() => setShowKeys(false)} style={{ color: C.dim, fontSize: 18, background: 'none', border: 'none', cursor: 'pointer' }}>&times;</button></div>
              <div className="grid grid-cols-2 gap-4" style={{ fontFamily: "'IBM Plex Mono'", fontSize: 10, color: C.dim, lineHeight: 2 }}>
                <div><div style={{ color: C.cyan, fontWeight: 600 }}>DECK A</div><div><b>Q</b> CUE &middot; <b>W</b> PLAY &middot; <b>1-4</b> Cues</div><div><b>A/S/D</b> Loop &middot; <b>Z/X/C/V</b> Jump</div></div>
                <div><div style={{ color: C.orange, fontWeight: 600 }}>DECK B</div><div><b>[</b> CUE &middot; <b>]</b> PLAY &middot; <b>7-0</b> Cues</div><div><b>L/;/&apos;</b> Loop &middot; <b>,./</b> Jump</div></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {learn && (
        <div style={{ position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 40, background: `${C.panelHi}ee`, borderTop: `1px solid ${C.cyanDim}`, padding: '5px 10px', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}>
          <span style={{ color: C.cyan, fontWeight: 700, fontSize: 8, letterSpacing: 2 }}>LEARN {'\u25B8'} </span>
          <span style={{ color: hint ? C.text : C.dim, fontSize: isMobile ? 9 : 10, fontFamily: "'IBM Plex Mono'", lineHeight: 1.4 }}>{hint || 'Tap any control to learn what it does.'}</span>
        </div>
      )}
    </div>
  );
}
