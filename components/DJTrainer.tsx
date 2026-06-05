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
  'Type each track\u2019s BPM into the BPM box (the BoothMatch readout turns green when they\u2019re locked).',
  'Press PLAY on Deck A. Push channel 1 fader up and the crossfader to the left.',
  'Press CUE on Deck B to set a start point on a beat. Move the TEMPO fader until Deck B\u2019s BPM matches Deck A.',
  'Nudge the JOG WHEEL on Deck B to slide its beats into the pocket with Deck A.',
  'Slowly sweep the CROSSFADER to the right, ride the EQs (kill Deck A\u2019s LOW as B\u2019s LOW comes in). You\u2019re mixing.',
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
    const start = b * step;
    for (let i = 0; i < step; i++) {
      const idx = start + i;
      if (idx >= len) break;
      let v = Math.abs(ch0[idx]);
      if (ch1) v = Math.max(v, Math.abs(ch1[idx]));
      if (v > max) max = v;
    }
    peaks[b] = max;
  }
  return peaks;
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
// WAVEFORM RENDERER
// ═══════════════════════════════════════════════════════════════

function drawWaveform(canvas: HTMLCanvasElement | null, st: DeckState): void {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0c0c10';
  ctx.fillRect(0, 0, W, H);

  if (!st.peaks) {
    ctx.fillStyle = C.dim;
    ctx.font = "11px 'IBM Plex Mono'";
    ctx.fillText('DRAG TRACK HERE OR TAP LOAD', 10, H / 2 + 4);
    return;
  }

  const prog = st.dur ? st.pos / st.dur : 0;
  const n = st.peaks.length;

  // Waveform bars
  for (let i = 0; i < n; i++) {
    const x = (i / n) * W;
    const h = st.peaks[i] * (H * 0.44);
    ctx.fillStyle = (i / n < prog) ? C.cyan : C.cyanDim;
    ctx.fillRect(x, H / 2 - h, Math.max(1, W / n), h * 2);
  }

  // Loop region
  if (st.loopIn !== null && st.loopOut !== null && st.dur > 0) {
    const x1 = (st.loopIn / st.dur) * W;
    const x2 = (st.loopOut / st.dur) * W;
    ctx.fillStyle = st.loopActive ? 'rgba(60,224,138,0.12)' : 'rgba(60,224,138,0.05)';
    ctx.fillRect(x1, 0, x2 - x1, H);
    ctx.fillStyle = C.green;
    ctx.fillRect(x1 - 1, 0, 2, H);
    ctx.fillRect(x2 - 1, 0, 2, H);
  }

  // Hot cue markers
  st.hotCues.forEach((hc, i) => {
    if (hc !== null && st.dur > 0) {
      const hx = (hc / st.dur) * W;
      ctx.fillStyle = HC_COLORS[i];
      ctx.fillRect(hx - 1, 0, 2, H);
      ctx.font = 'bold 8px Oxanium';
      ctx.fillText(HC_LABELS[i], hx + 3, 10);
    }
  });

  // Cue marker
  if (st.dur > 0) {
    const cx = (st.cue / st.dur) * W;
    ctx.fillStyle = C.orange;
    ctx.fillRect(cx - 1, 0, 2, H);
  }

  // Playhead
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(prog * W - 1, 0, 2, H);
}

// ═══════════════════════════════════════════════════════════════
// CONTROLS
// ═══════════════════════════════════════════════════════════════

interface KnobProps {
  value: number;
  onChange: (v: number) => void;
  label: string;
  color?: string;
  size?: number;
  hint?: string;
  onHint?: (h: string) => void;
}

function Knob({ value, onChange, label, color = C.cyan, size = 48, hint, onHint }: KnobProps) {
  const drag = useRef<{ y: number; v: number } | null>(null);
  const angle = -135 + value * 270;
  const down = (e: React.PointerEvent) => {
    e.preventDefault();
    drag.current = { y: e.clientY, v: value };
    const move = (ev: PointerEvent) => {
      if (!drag.current) return;
      const nv = Math.max(0, Math.min(1, drag.current.v + (drag.current.y - ev.clientY) / 160));
      onChange(nv);
    };
    const up = () => {
      drag.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };
  return (
    <div className="flex flex-col items-center select-none" style={{ width: size + 8 }}
      onMouseEnter={() => hint && onHint?.(hint)}>
      <div onPointerDown={down} onDoubleClick={() => onChange(0.5)}
        className="rounded-full relative cursor-ns-resize"
        style={{
          width: size, height: size,
          background: 'radial-gradient(circle at 35% 30%, #3a3a44, #141418 70%)',
          border: `1px solid ${C.edge}`,
          boxShadow: 'inset 0 2px 4px rgba(0,0,0,.6), 0 1px 2px rgba(0,0,0,.5)',
        }} title={label}>
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
  value: number;
  onChange: (v: number) => void;
  label?: string;
  color?: string;
  height?: number;
  center?: boolean;
  hint?: string;
  onHint?: (h: string) => void;
}

function Fader({ value, onChange, label, color = C.cyan, height = 130, center, hint, onHint }: FaderProps) {
  const drag = useRef<{ y: number; v: number } | null>(null);
  const down = (e: React.PointerEvent) => {
    e.preventDefault();
    drag.current = { y: e.clientY, v: value };
    const move = (ev: PointerEvent) => {
      if (!drag.current) return;
      const nv = Math.max(0, Math.min(1, drag.current.v + (drag.current.y - ev.clientY) / height));
      onChange(nv);
    };
    const up = () => {
      drag.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };
  return (
    <div className="flex flex-col items-center select-none" onMouseEnter={() => hint && onHint?.(hint)}>
      <div className="relative" style={{ height, width: 30 }}>
        <div className="absolute left-1/2 top-0 bottom-0" style={{
          width: 4, transform: 'translateX(-50%)', background: '#0a0a0c',
          borderRadius: 3, border: `1px solid ${C.edge}`,
        }} />
        {center && <div className="absolute left-1/2 top-1/2" style={{ width: 12, height: 1, background: C.dim, transform: 'translate(-50%,-50%)' }} />}
        <div onPointerDown={down} className="absolute left-1/2 cursor-ns-resize" style={{
          width: 26, height: 16, transform: 'translateX(-50%)',
          bottom: `calc(${value * 100}% - 8px)`,
          background: 'linear-gradient(180deg,#48484f,#1a1a1e)',
          borderRadius: 3, border: `1px solid ${C.edge}`,
          boxShadow: '0 0 5px rgba(0,0,0,.6)',
        }}>
          <div style={{ height: 2, background: color, margin: '6px 3px', borderRadius: 2, boxShadow: `0 0 4px ${color}` }} />
        </div>
      </div>
      {label && <span style={{ color: C.dim, fontSize: 9, marginTop: 4, letterSpacing: 1, fontFamily: 'Oxanium' }}>{label}</span>}
    </div>
  );
}

function VUMeter({ level }: { level: number }) {
  const segs = 14;
  const lit = Math.round(level * segs);
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
  const dragging = useRef(false);
  const down = (e: React.PointerEvent) => {
    e.preventDefault();
    dragging.current = true;
    move(e.nativeEvent);
    const move2 = (ev: PointerEvent) => {
      if (!dragging.current || !trackRef.current) return;
      const r = trackRef.current.getBoundingClientRect();
      onChange(Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)));
    };
    const up = () => {
      dragging.current = false;
      window.removeEventListener('pointermove', move2);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move2);
    window.addEventListener('pointerup', up);
  };
  const move = (ev: PointerEvent | React.PointerEvent<HTMLDivElement>['nativeEvent']) => {
    if (!trackRef.current) return;
    const r = trackRef.current.getBoundingClientRect();
    onChange(Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)));
  };
  return (
    <div ref={trackRef} onPointerDown={down} className="relative cursor-ew-resize" style={{ height: 30 }}>
      <div className="absolute top-1/2 left-0 right-0" style={{ height: 5, transform: 'translateY(-50%)', background: '#0a0a0c', borderRadius: 3, border: `1px solid ${C.edge}` }} />
      <div className="absolute top-1/2" style={{
        width: 18, height: 26, transform: 'translate(-50%,-50%)', left: `${value * 100}%`,
        background: 'linear-gradient(180deg,#48484f,#1a1a1e)', borderRadius: 3,
        border: `1px solid ${C.edge}`, boxShadow: '0 0 5px rgba(0,0,0,.6)',
      }}>
        <div style={{ width: 2, height: 18, background: C.cyan, margin: '4px auto', borderRadius: 2, boxShadow: `0 0 4px ${C.cyan}` }} />
      </div>
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
  const fileRefA = useRef<HTMLInputElement>(null);
  const fileRefB = useRef<HTMLInputElement>(null);
  const platterRefA = useRef<HTMLDivElement>(null);
  const platterRefB = useRef<HTMLDivElement>(null);
  const platterAngle = useRef<Record<DeckId, number>>({ A: 0, B: 0 });

  // ─── state ───
  const [initialized, setInitialized] = useState(false);
  const [learn, setLearn] = useState(true);
  const [hint, setHint] = useState<string | null>(null);
  const [guideStep, setGuideStep] = useState(0);
  const [showGuide, setShowGuide] = useState(true);
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

  const setters: Record<DeckId, React.Dispatch<React.SetStateAction<DeckState>>> = {
    A: setDeckA, B: setDeckB,
  };

  // ─── audio init ───
  const initAudio = useCallback(() => {
    if (audio.current) return;
    const ctx = new AudioContext();
    const master = ctx.createGain();
    master.gain.value = 0.8;
    const masterAnalyser = ctx.createAnalyser();
    masterAnalyser.fftSize = 256;
    master.connect(masterAnalyser);
    masterAnalyser.connect(ctx.destination);

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
        rate: 1, effRate: 1, playing: false, stopping: false,
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

  const rebase = (d: AudioDeck): void => {
    d.startOffset = posOf(d);
    d.startTime = audio.current!.ctx.currentTime;
  };

  // ─── source management ───
  const startSrc = (a: AudioEngine, d: AudioDeck, id: DeckId): void => {
    const src = a.ctx.createBufferSource();
    src.buffer = d.buffer;
    src.playbackRate.value = d.effRate;
    src.connect(d.trim);
    src.onended = () => {
      if (d.stopping) { d.stopping = false; return; }
      d.playing = false;
      setters[id](s => ({ ...s, playing: false }));
    };
    src.start(0, Math.max(0, Math.min(d.dur, d.startOffset)));
    d.src = src;
    d.startTime = a.ctx.currentTime;
    d.playing = true;
  };

  // ─── file loading ───
  const loadFile = async (id: DeckId, file: File): Promise<void> => {
    const a = ensure();
    const arr = await file.arrayBuffer();
    const buf = await a.ctx.decodeAudioData(arr);
    const d = a.decks[id];
    if (d.playing) { d.stopping = true; try { d.src?.stop(); } catch { /* source already stopped */ } d.playing = false; }
    d.buffer = buf; d.dur = buf.duration; d.startOffset = 0; d.cue = 0;
    d.hotCues = [null, null, null, null];
    d.loopIn = null; d.loopOut = null; d.loopActive = false; d.cuePreview = false;
    const peaks = buildPeaks(buf);
    setters[id](() => ({
      ...mkDeckState(),
      name: file.name.replace(/\.[^.]+$/, ''),
      loaded: true, dur: buf.duration, peaks,
    }));
  };

  // ─── transport: play ───
  const playDeck = (id: DeckId): void => {
    const a = ensure();
    const d = a.decks[id];
    if (!d.buffer) return;
    // If CUE preview is active, promote to normal play
    if (d.cuePreview) { d.cuePreview = false; return; }
    if (d.playing) return;
    if (d.startOffset >= d.dur) d.startOffset = 0;
    startSrc(a, d, id);
    setters[id](s => ({ ...s, playing: true }));
  };

  // ─── transport: pause ───
  const pauseDeck = (id: DeckId): void => {
    const a = ensure();
    const d = a.decks[id];
    if (!d.playing) return;
    const p = posOf(d);
    d.stopping = true;
    try { d.src?.stop(); } catch { /* ok */ }
    d.startOffset = p; d.playing = false; d.cuePreview = false;
    setters[id](s => ({ ...s, playing: false, pos: p }));
  };

  // ─── transport: seek ───
  const seekDeck = (id: DeckId, p: number): void => {
    const a = ensure();
    const d = a.decks[id];
    p = Math.max(0, Math.min(d.dur, p));
    if (d.playing) {
      d.stopping = true;
      try { d.src?.stop(); } catch { /* ok */ }
      d.startOffset = p;
      startSrc(a, d, id);
    } else {
      d.startOffset = p;
    }
    setters[id](s => ({ ...s, pos: p }));
  };

  // ─── transport: CUE (authentic CDJ behavior) ───
  const cuePress = (id: DeckId): void => {
    const a = ensure();
    const d = a.decks[id];
    if (!d.buffer) return;

    if (d.playing) {
      // Playing (normal or preview) → snap to cue, pause
      d.stopping = true;
      try { d.src?.stop(); } catch { /* ok */ }
      d.startOffset = d.cue;
      d.playing = false;
      d.cuePreview = false;
      setters[id](s => ({ ...s, playing: false, pos: d.cue }));
    } else {
      // Paused → set cue at current position, start preview
      d.cue = d.startOffset;
      d.cuePreview = true;
      startSrc(a, d, id);
      setters[id](s => ({ ...s, cue: d.cue, playing: true }));
    }
  };

  const cueRelease = (id: DeckId): void => {
    if (!audio.current) return;
    const d = audio.current.decks[id];
    // If we're in CUE preview, snap back to cue and pause
    if (d.cuePreview && d.playing) {
      d.stopping = true;
      try { d.src?.stop(); } catch { /* ok */ }
      d.startOffset = d.cue;
      d.playing = false;
      d.cuePreview = false;
      setters[id](s => ({ ...s, playing: false, pos: d.cue }));
    }
  };

  // ─── tempo ───
  const setTempo = (id: DeckId, v: number): void => {
    if (!audio.current) return;
    const d = audio.current.decks[id];
    const range = id === 'A' ? deckA.range : deckB.range;
    const pct = (v - 0.5) * 2 * range;
    const rate = 1 + pct / 100;
    if (d.playing) rebase(d);
    d.rate = rate; d.effRate = rate;
    if (d.src) d.src.playbackRate.value = rate;
    setters[id](s => ({ ...s, tempo: v }));
  };

  // ─── jog wheel ───
  const jogDown = (id: DeckId, e: React.PointerEvent): void => {
    const a = ensure();
    const d = a.decks[id];
    if (!d.buffer) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    let lastAng = Math.atan2(e.clientY - cy, e.clientX - cx);

    const mv = (ev: PointerEvent) => {
      const na = Math.atan2(ev.clientY - cy, ev.clientX - cx);
      let dA = na - lastAng;
      if (dA > Math.PI) dA -= 2 * Math.PI;
      if (dA < -Math.PI) dA += 2 * Math.PI;
      lastAng = na;

      if (d.playing) {
        rebase(d);
        const bend = Math.max(-0.5, Math.min(0.5, dA * 2.2));
        d.effRate = d.rate * (1 + bend);
        if (d.src) d.src.playbackRate.value = d.effRate;
      } else {
        d.startOffset = Math.max(0, Math.min(d.dur, d.startOffset + (dA / (2 * Math.PI)) * 4));
        setters[id](s => ({ ...s, pos: d.startOffset }));
      }
    };
    const up = () => {
      if (d.playing) {
        rebase(d);
        d.effRate = d.rate;
        if (d.src) d.src.playbackRate.value = d.rate;
      }
      window.removeEventListener('pointermove', mv);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
  };

  // ─── hot cues ───
  const hotCuePress = (id: DeckId, idx: number): void => {
    const a = ensure();
    const d = a.decks[id];
    if (!d.buffer) return;

    if (d.hotCues[idx] === null) {
      const p = posOf(d);
      d.hotCues[idx] = p;
      setters[id](s => {
        const hc = [...s.hotCues];
        hc[idx] = p;
        return { ...s, hotCues: hc };
      });
    } else {
      const target = d.hotCues[idx]!;
      if (d.playing) {
        d.stopping = true;
        try { d.src?.stop(); } catch { /* ok */ }
      }
      d.startOffset = target;
      d.cuePreview = false;
      startSrc(a, d, id);
      setters[id](s => ({ ...s, playing: true, pos: target }));
    }
  };

  const hotCueClear = (id: DeckId, idx: number): void => {
    if (!audio.current) return;
    audio.current.decks[id].hotCues[idx] = null;
    setters[id](s => {
      const hc = [...s.hotCues];
      hc[idx] = null;
      return { ...s, hotCues: hc };
    });
  };

  // ─── loops ───
  const setLoopInPoint = (id: DeckId): void => {
    if (!audio.current) return;
    const d = audio.current.decks[id];
    if (!d.buffer) return;
    const p = posOf(d);
    d.loopIn = p;
    setters[id](s => ({ ...s, loopIn: p }));
  };

  const setLoopOutPoint = (id: DeckId): void => {
    if (!audio.current) return;
    const d = audio.current.decks[id];
    if (!d.buffer) return;
    const p = posOf(d);
    d.loopOut = p;
    d.loopActive = true;
    setters[id](s => ({ ...s, loopOut: p, loopActive: true }));
  };

  const toggleLoop = (id: DeckId): void => {
    if (!audio.current) return;
    const d = audio.current.decks[id];
    if (d.loopIn === null || d.loopOut === null) return;
    d.loopActive = !d.loopActive;
    setters[id](s => ({ ...s, loopActive: !s.loopActive }));
  };

  // ─── bpm calculation ───
  const adjBpm = (st: DeckState): number | null => {
    const b = parseFloat(st.bpm);
    if (!b || isNaN(b)) return null;
    return b * (1 + ((st.tempo - 0.5) * 2 * st.range) / 100);
  };

  // ─── effect: draw waveforms ───
  useEffect(() => { drawWaveform(cvsA.current, deckA); },
    [deckA.peaks, deckA.pos, deckA.dur, deckA.cue, deckA.hotCues, deckA.loopIn, deckA.loopOut, deckA.loopActive]);
  useEffect(() => { drawWaveform(cvsB.current, deckB); },
    [deckB.peaks, deckB.pos, deckB.dur, deckB.cue, deckB.hotCues, deckB.loopIn, deckB.loopOut, deckB.loopActive]);

  // ─── effect: mixer parameter application ───
  useEffect(() => {
    if (!audio.current) return;
    const a = audio.current;
    const eqGain = (val: number) => (val - 0.5) * 2 * 26 * (val < 0.5 ? 1.6 : 0.5);
    const apply = (id: DeckId, ch: ChannelState) => {
      const d = a.decks[id];
      d.trim.gain.value = ch.trim * 1.4;
      d.low.gain.value = eqGain(ch.low);
      d.mid.gain.value = eqGain(ch.mid);
      d.hi.gain.value = eqGain(ch.hi);
      if (Math.abs(ch.color - 0.5) < 0.04) {
        d.color.type = 'lowpass'; d.color.frequency.value = 22000;
      } else if (ch.color < 0.5) {
        d.color.type = 'lowpass'; d.color.frequency.value = 200 + Math.pow(ch.color / 0.5, 2) * 12000;
      } else {
        d.color.type = 'highpass'; d.color.frequency.value = 30 + Math.pow((ch.color - 0.5) / 0.5, 2) * 7000;
      }
      d.chGain.gain.value = ch.fader;
    };
    apply('A', mix.chA); apply('B', mix.chB);
    a.decks.A.xf.gain.value = Math.cos(mix.xfader * Math.PI / 2);
    a.decks.B.xf.gain.value = Math.cos((1 - mix.xfader) * Math.PI / 2);
    a.master.gain.value = mix.master;
  }, [mix, initialized]);

  // ─── effect: animation frame loop ───
  useEffect(() => {
    if (!initialized) return;
    let raf: number;
    const buf = new Uint8Array(128);
    const rms = (an: AnalyserNode): number => {
      an.getByteTimeDomainData(buf);
      let s = 0;
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
            if (d.playing) platterAngle.current[id] += d.effRate * 3.33;
            el.style.transform = `rotate(${platterAngle.current[id]}deg)`;
          }
          if (d.playing) {
            const p = posOf(d);
            // Loop enforcement
            if (d.loopActive && d.loopIn !== null && d.loopOut !== null && p >= d.loopOut) {
              d.stopping = true;
              try { d.src?.stop(); } catch { /* ok */ }
              d.startOffset = d.loopIn;
              startSrc(a, d, id);
            }
            setters[id](s => (Math.abs(s.pos - p) > 0.02 ? { ...s, pos: p } : s));
          }
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialized]);

  // ─── effect: keyboard shortcuts ───
  useEffect(() => {
    if (!initialized) return;
    const onDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const k = e.key;
      switch (k) {
        case 'q': e.preventDefault(); cuePress('A'); break;
        case 'w': e.preventDefault(); if (!e.repeat) { audio.current?.decks.A.playing ? pauseDeck('A') : playDeck('A'); } break;
        case '1': hotCuePress('A', 0); break;
        case '2': hotCuePress('A', 1); break;
        case '3': hotCuePress('A', 2); break;
        case '4': hotCuePress('A', 3); break;
        case 'a': setLoopInPoint('A'); break;
        case 's': e.preventDefault(); setLoopOutPoint('A'); break;
        case 'd': e.preventDefault(); toggleLoop('A'); break;
        case '[': e.preventDefault(); cuePress('B'); break;
        case ']': e.preventDefault(); if (!e.repeat) { audio.current?.decks.B.playing ? pauseDeck('B') : playDeck('B'); } break;
        case '7': hotCuePress('B', 0); break;
        case '8': hotCuePress('B', 1); break;
        case '9': hotCuePress('B', 2); break;
        case '0': hotCuePress('B', 3); break;
        case 'l': setLoopInPoint('B'); break;
        case ';': setLoopOutPoint('B'); break;
        case "'": toggleLoop('B'); break;
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
  // INIT OVERLAY
  // ═══════════════════════════════════════════════════════════

  if (!initialized) {
    return (
      <div className="flex flex-col items-center justify-center select-none"
        style={{ background: C.bg, minHeight: '100vh', fontFamily: 'Oxanium', color: C.text }}>
        <div style={{ fontWeight: 700, fontSize: 42, letterSpacing: 6 }}>
          VIRTUAL <span style={{ color: C.cyan }}>BOOTH</span>
        </div>
        <div style={{ color: C.dim, fontSize: 14, fontFamily: "'IBM Plex Mono'", marginTop: 8, letterSpacing: 2 }}>
          CDJ-2000NXS &times;2 &middot; DJM-900NXS
        </div>
        <div style={{ marginTop: 16, color: C.dim, fontSize: 12, fontFamily: "'IBM Plex Mono'", textAlign: 'center', lineHeight: 1.6 }}>
          Real Web Audio engine &middot; 3-band EQ + color filter &middot; Authentic CUE behavior<br />
          Jog-wheel pitch-bend &middot; Hot cues &middot; Loops &middot; Keyboard shortcuts
        </div>
        <button onClick={initAudio} style={{
          marginTop: 40, padding: '16px 56px', borderRadius: 8,
          background: 'transparent', border: `2px solid ${C.cyan}`, color: C.cyan,
          fontFamily: 'Oxanium', fontWeight: 700, fontSize: 16, letterSpacing: 4,
          cursor: 'pointer', transition: 'all 0.3s',
        }}
          onMouseEnter={e => { e.currentTarget.style.background = C.cyanDim; e.currentTarget.style.boxShadow = `0 0 30px ${C.cyan}44`; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.boxShadow = 'none'; }}>
          START SESSION
        </button>
        <div style={{ marginTop: 32, color: C.dim, fontSize: 11, fontFamily: "'IBM Plex Mono'", textAlign: 'center', maxWidth: 440, lineHeight: 1.7 }}>
          Load your own tracks (MP3 / WAV / FLAC / OGG). Headphones recommended.<br />
          Browser audio requires a user gesture to start &mdash; that&apos;s what this button does.
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════
  // RENDER HELPERS
  // ═══════════════════════════════════════════════════════════

  const showHint = (t: string) => { if (learn) setHint(t); };
  const bpmA = adjBpm(deckA);
  const bpmB = adjBpm(deckB);
  const matched = bpmA !== null && bpmB !== null && Math.abs(bpmA - bpmB) < 0.15;

  const renderDeck = (id: DeckId) => {
    const st = id === 'A' ? deckA : deckB;
    const cvsRef = id === 'A' ? cvsA : cvsB;
    const fileRef = id === 'A' ? fileRefA : fileRefB;
    const plRef = id === 'A' ? platterRefA : platterRefB;
    const remaining = (st.dur || 0) - st.pos;
    const endWarn = st.loaded && st.playing && remaining < 30;
    const endCrit = endWarn && remaining < 10;
    const hcShortcuts = id === 'A' ? ['1', '2', '3', '4'] : ['7', '8', '9', '0'];
    const deckColor = id === 'A' ? C.cyan : C.orange;

    return (
      <div className="rounded-lg p-3 flex flex-col gap-2" style={{ background: C.panel, border: `1px solid ${C.edge}`, minWidth: 300, flex: 1, maxWidth: 380 }}>
        {/* Header */}
        <div className="flex items-center justify-between">
          <span style={{ fontFamily: 'Oxanium', fontWeight: 700, color: deckColor, letterSpacing: 2, fontSize: 13 }}>DECK {id}</span>
          <span style={{ fontFamily: 'Oxanium', color: C.dim, fontSize: 10 }}>CDJ-2000NXS</span>
        </div>

        {/* Waveform display */}
        <div className="rounded" style={{ background: '#0c0c10', border: `1px solid ${C.edge}`, padding: 6 }}
          onMouseEnter={() => showHint('CDJ display: waveform overview. White line = playhead, orange = CUE point, colored markers = hot cues. Green region = active loop. Click anywhere on the waveform to jump to that position.')}>
          <canvas ref={cvsRef} width={276} height={70}
            style={{ width: '100%', height: 70, display: 'block', cursor: 'pointer' }}
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              seekDeck(id, ((e.clientX - rect.left) / rect.width) * st.dur);
            }} />
          <div className="flex items-center justify-between mt-1" style={{ fontFamily: "'IBM Plex Mono'", fontSize: 11 }}>
            <span style={{ color: C.cyan }}>{fmtTime(st.pos)}</span>
            <span style={{ color: C.text, fontSize: 10, maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{st.name || '\u2014'}</span>
            <span style={{ color: endCrit ? C.red : endWarn ? C.orange : C.orange }} className={endCrit ? 'warning-flash' : ''}>
              -{fmtTime(remaining)}
            </span>
          </div>
        </div>

        {/* Jog wheel */}
        <div className="flex items-center justify-center my-1">
          <div
            onPointerDown={(e) => jogDown(id, e)}
            onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) loadFile(id, f); }}
            onDragOver={(e) => e.preventDefault()}
            onMouseEnter={() => showHint('JOG WHEEL: the heart of the CDJ. While PLAYING, nudge forward to speed up, pull back to slow down \u2014 sliding beats into alignment. While PAUSED, spin to scrub and find your cue point. Drag a track file here to load it.')}
            className="rounded-full relative cursor-grab active:cursor-grabbing select-none flex items-center justify-center"
            style={{
              width: 168, height: 168, touchAction: 'none',
              background: 'radial-gradient(circle at 50% 45%, #2c2c34 0%, #16161a 55%, #0c0c0e 100%)',
              border: '3px solid #1a1a1e',
              boxShadow: '0 6px 20px rgba(0,0,0,.6), inset 0 0 30px rgba(0,0,0,.7)',
            }}>
            <div className="absolute inset-2 rounded-full" style={{ border: '1px dashed #33333c' }} />
            {/* Spinning platter — animated via ref in rAF */}
            <div ref={plRef} className="rounded-full relative flex items-center justify-center"
              style={{
                width: 108, height: 108,
                background: 'radial-gradient(circle at 50% 50%, #3a3a44, #1c1c22)',
                border: `1px solid ${C.edge}`,
              }}>
              <div className="absolute" style={{ width: 90, height: 2, background: st.playing ? deckColor : C.dim, opacity: 0.5 }} />
              <div className="rounded-full" style={{
                width: 34, height: 34, background: '#0c0c10',
                border: `2px solid ${st.playing ? deckColor : C.edge}`,
                boxShadow: st.playing ? `0 0 10px ${deckColor}` : 'none',
                transition: 'border-color 0.3s, box-shadow 0.3s',
              }} />
            </div>
          </div>
        </div>

        {/* Transport buttons */}
        <div className="flex items-center gap-3 justify-center">
          <button
            onPointerDown={() => cuePress(id)}
            onPointerUp={() => cueRelease(id)}
            onPointerLeave={() => cueRelease(id)}
            onMouseEnter={() => showHint('CUE: sets and returns to your start point. Paused \u2192 press to drop a CUE (hold to preview, release to snap back). Playing \u2192 press to snap back to CUE and pause. Hold CUE + press PLAY to commit and keep playing.')}
            className="rounded-full flex flex-col items-center justify-center"
            style={{ width: 54, height: 54, background: '#1a1a1e', border: `2px solid ${C.orange}`, color: C.orange, fontFamily: 'Oxanium', fontWeight: 700, fontSize: 12, boxShadow: `0 0 8px ${C.orange}55` }}>
            CUE
            <span style={{ fontSize: 7, opacity: 0.5, marginTop: -2 }}>{id === 'A' ? 'Q' : '['}</span>
          </button>
          <button
            onPointerDown={() => st.playing ? pauseDeck(id) : playDeck(id)}
            onMouseEnter={() => showHint('PLAY / PAUSE: starts or pauses the deck. If held during CUE preview, converts to normal play.')}
            className="rounded-full flex flex-col items-center justify-center"
            style={{ width: 54, height: 54, background: '#1a1a1e', border: `2px solid ${C.green}`, color: C.green, fontFamily: 'Oxanium', fontWeight: 700, fontSize: 18, boxShadow: `0 0 10px ${C.green}55` }}>
            {st.playing ? '\u275A\u275A' : '\u25B6'}
            <span style={{ fontSize: 7, opacity: 0.5, marginTop: -2 }}>{id === 'A' ? 'W' : ']'}</span>
          </button>
          <button onClick={() => fileRef.current?.click()}
            onMouseEnter={() => showHint('LOAD: choose an audio file from your computer. On the real rig this is your USB stick loaded from rekordbox.')}
            className="rounded flex flex-col items-center justify-center"
            style={{ width: 54, height: 54, background: '#1a1a1e', border: `1px solid ${C.edge}`, color: C.dim, fontFamily: 'Oxanium', fontSize: 10 }}>
            <span style={{ fontSize: 16 }}>{'\u2913'}</span>LOAD
          </button>
          <input ref={fileRef} type="file" accept="audio/*" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) loadFile(id, f); }} />
        </div>

        {/* Hot cues + Loop controls */}
        <div className="flex items-center gap-1.5 justify-center flex-wrap">
          {/* Hot cue pads */}
          {st.hotCues.map((hc, i) => (
            <button key={i}
              onClick={() => hotCuePress(id, i)}
              onContextMenu={(e) => { e.preventDefault(); hotCueClear(id, i); }}
              onMouseEnter={() => showHint(`HOT CUE ${HC_LABELS[i]}: ${hc !== null ? 'Jump to this saved point and play. Right-click to clear.' : 'Tap to save the current position as a hot cue. Jump back to it instantly anytime.'}`)}
              className="relative rounded flex items-center justify-center"
              style={{
                width: 34, height: 30, fontSize: 10, fontFamily: 'Oxanium', fontWeight: 700,
                background: hc !== null ? `${HC_COLORS[i]}18` : '#1a1a1e',
                border: `2px solid ${hc !== null ? HC_COLORS[i] : C.edge}`,
                color: hc !== null ? HC_COLORS[i] : C.dim,
                boxShadow: hc !== null ? `0 0 6px ${HC_COLORS[i]}33` : 'none',
              }}>
              {HC_LABELS[i]}
              <span className="absolute bottom-0 right-0.5" style={{ fontSize: 6, color: C.dim, fontWeight: 400 }}>{hcShortcuts[i]}</span>
            </button>
          ))}
          <div style={{ width: 1, height: 24, background: C.edge, margin: '0 2px' }} />
          {/* Loop controls */}
          <button onClick={() => setLoopInPoint(id)}
            onMouseEnter={() => showHint('LOOP IN: sets the start point of a loop. Press LOOP OUT to close the loop.')}
            className="rounded flex items-center justify-center"
            style={{ width: 34, height: 30, fontSize: 8, fontFamily: 'Oxanium', fontWeight: 600,
              background: st.loopIn !== null ? `${C.green}18` : '#1a1a1e',
              border: `1px solid ${st.loopIn !== null ? C.green : C.edge}`,
              color: st.loopIn !== null ? C.green : C.dim }}>
            IN
          </button>
          <button onClick={() => setLoopOutPoint(id)}
            onMouseEnter={() => showHint('LOOP OUT: sets the end point and activates the loop. The track will cycle between IN and OUT.')}
            className="rounded flex items-center justify-center"
            style={{ width: 34, height: 30, fontSize: 8, fontFamily: 'Oxanium', fontWeight: 600,
              background: st.loopOut !== null ? `${C.green}18` : '#1a1a1e',
              border: `1px solid ${st.loopOut !== null ? C.green : C.edge}`,
              color: st.loopOut !== null ? C.green : C.dim }}>
            OUT
          </button>
          <button onClick={() => toggleLoop(id)}
            onMouseEnter={() => showHint('RELOOP: toggle the loop on/off. When green and active, the track cycles between your IN and OUT points.')}
            className="rounded flex items-center justify-center"
            style={{ width: 42, height: 30, fontSize: 8, fontFamily: 'Oxanium', fontWeight: 600,
              background: st.loopActive ? `${C.green}22` : '#1a1a1e',
              border: `1px solid ${st.loopActive ? C.green : C.edge}`,
              color: st.loopActive ? C.green : C.dim,
              boxShadow: st.loopActive ? `0 0 8px ${C.green}33` : 'none' }}>
            {st.loopActive ? '\u27F2 ON' : 'LOOP'}
          </button>
        </div>

        {/* Tempo + BPM section */}
        <div className="flex items-stretch gap-3 mt-1 justify-between">
          <div className="flex flex-col items-center" style={{ flex: 1 }}>
            <div className="flex items-center gap-1 mb-1">
              {([6, 10, 16, 100] as const).map(r => (
                <button key={r} onClick={() => setters[id](s => ({ ...s, range: r }))}
                  style={{ fontFamily: 'Oxanium', fontSize: 8, padding: '2px 4px', borderRadius: 3,
                    background: st.range === r ? C.cyanDim : '#1a1a1e',
                    color: st.range === r ? C.cyan : C.dim,
                    border: `1px solid ${C.edge}` }}>
                  {r === 100 ? 'WIDE' : `\u00B1${r}`}
                </button>
              ))}
            </div>
            <Fader value={st.tempo} onChange={(v) => setTempo(id, v)} label="TEMPO" color={deckColor} height={120} center
              onHint={showHint} hint="TEMPO FADER: speeds up or slows down the track to match BPMs. Center = 0%. Range buttons above set precision \u2014 pros beatmatch on \u00B16 for fine control." />
          </div>
          <div className="flex flex-col items-center justify-end gap-2" style={{ width: 96 }}>
            <button onClick={() => setters[id](s => ({ ...s, masterTempo: !s.masterTempo }))}
              onMouseEnter={() => showHint('MASTER TEMPO (key lock): on the real CDJ this keeps pitch constant while speed changes. This trainer changes pitch with speed like vinyl mode \u2014 the beatmatching workflow is identical.')}
              style={{ fontFamily: 'Oxanium', fontSize: 9, padding: '4px 6px', borderRadius: 4, width: '100%',
                background: st.masterTempo ? C.cyanDim : '#1a1a1e',
                color: st.masterTempo ? C.cyan : C.dim,
                border: `1px solid ${C.edge}` }}>
              MASTER TEMPO
            </button>
            <div className="w-full rounded text-center" style={{ background: '#0c0c10', border: `1px solid ${C.edge}`, padding: 4 }}>
              <div style={{ fontFamily: "'IBM Plex Mono'", fontSize: 16, color: deckColor }}>
                {((st.tempo - 0.5) * 2 * st.range) >= 0 ? '+' : ''}{((st.tempo - 0.5) * 2 * st.range).toFixed(1)}%
              </div>
              <div style={{ fontFamily: 'Oxanium', fontSize: 8, color: C.dim }}>TEMPO</div>
            </div>
            <input value={st.bpm} onChange={(e) => setters[id](s => ({ ...s, bpm: e.target.value.replace(/[^\d.]/g, '') }))}
              placeholder="BPM" inputMode="decimal"
              onMouseEnter={() => showHint('Type the track\u2019s original BPM here (from rekordbox / CDJ display). The trainer shows your live adjusted BPM so you can match decks by the numbers while training your ears.')}
              style={{ width: '100%', background: '#0c0c10', border: `1px solid ${C.edge}`, color: C.text,
                fontFamily: "'IBM Plex Mono'", fontSize: 12, padding: '4px 6px', borderRadius: 4, textAlign: 'center',
                outline: 'none' }} />
            <div className="w-full text-center" style={{ fontFamily: "'IBM Plex Mono'", fontSize: 11, color: adjBpm(st) ? C.green : C.dim }}>
              {adjBpm(st) ? `\u2192 ${adjBpm(st)!.toFixed(1)}` : '\u2014'}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderChannelStrip = (id: DeckId) => {
    const ch = id === 'A' ? mix.chA : mix.chB;
    const setCh = (patch: Partial<ChannelState>) => {
      setMix(m => id === 'A'
        ? { ...m, chA: { ...m.chA, ...patch } }
        : { ...m, chB: { ...m.chB, ...patch } });
    };
    const stripColor = id === 'A' ? C.cyan : C.orange;
    return (
      <div className="flex flex-col items-center gap-2 rounded-lg p-2" style={{ background: C.panelHi, border: `1px solid ${C.edge}` }}>
        <span style={{ fontFamily: 'Oxanium', fontWeight: 700, color: stripColor, fontSize: 12 }}>CH {id === 'A' ? '1' : '2'}</span>
        <Knob value={ch.trim} onChange={(v) => setCh({ trim: v })} label="TRIM" onHint={showHint}
          hint="TRIM (gain): sets the channel\u2019s input level. Aim for loud parts peaking into orange on the meter \u2014 never red." />
        <Knob value={ch.hi} onChange={(v) => setCh({ hi: v })} label="HI" onHint={showHint}
          hint="HI EQ: treble (hi-hats, vocal air). Turn fully left to KILL the highs." />
        <Knob value={ch.mid} onChange={(v) => setCh({ mid: v })} label="MID" onHint={showHint}
          hint="MID EQ: vocals, snares, melody body. Killing mids during a blend stops two tracks fighting." />
        <Knob value={ch.low} onChange={(v) => setCh({ low: v })} label="LOW" color={C.orange} onHint={showHint}
          hint="LOW EQ: bass / kick. THE most important mixing control \u2014 never two kicks at once. Swap lows: kill outgoing track\u2019s LOW, raise the incoming one\u2019s." />
        <div className="flex items-end gap-2 mt-1">
          <Fader value={ch.fader} onChange={(v) => setCh({ fader: v })} color={stripColor} height={120}
            onHint={showHint} hint="CHANNEL FADER: volume for this deck. Push up to bring the deck into the mix." />
          <VUMeter level={id === 'A' ? vu.A : vu.B} />
        </div>
        <Knob value={ch.color} onChange={(v) => setCh({ color: v })} label="COLOR" size={40} onHint={showHint}
          hint="COLOR / FILTER: center = off. Left = low-pass (muffles, builds tension). Right = high-pass (thins, removes bass). Sweep it on a buildup \u2014 the club\u2019s favourite move." />
      </div>
    );
  };

  // ═══════════════════════════════════════════════════════════
  // MAIN RENDER
  // ═══════════════════════════════════════════════════════════

  return (
    <div className="p-3 select-none" style={{ background: C.bg, minHeight: '100vh', fontFamily: 'Oxanium', color: C.text }}>
      {/* Top bar */}
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div>
          <div style={{ fontWeight: 700, fontSize: 20, letterSpacing: 3 }}>
            VIRTUAL <span style={{ color: C.cyan }}>BOOTH</span>
          </div>
          <div style={{ color: C.dim, fontSize: 11, fontFamily: "'IBM Plex Mono'" }}>CDJ-2000NXS &times;2 &middot; DJM-900NXS &middot; real audio trainer</div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button onClick={() => setShowKeys(v => !v)}
            style={{ background: C.panel, border: `1px solid ${showKeys ? C.cyan : C.edge}`, color: showKeys ? C.cyan : C.dim, padding: '8px 12px', borderRadius: 6, fontSize: 12 }}>
            {showKeys ? 'Hide' : 'Show'} Keys
          </button>
          <button onClick={() => setShowGuide(v => !v)}
            style={{ background: C.panel, border: `1px solid ${C.edge}`, color: C.text, padding: '8px 12px', borderRadius: 6, fontSize: 12 }}>
            {showGuide ? 'Hide' : 'Show'} Guide
          </button>
          <button onClick={() => setLearn(v => !v)}
            style={{ background: learn ? C.cyanDim : C.panel, border: `1px solid ${learn ? C.cyan : C.edge}`, color: learn ? C.cyan : C.dim, padding: '8px 12px', borderRadius: 6, fontSize: 12 }}>
            Learn {learn ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      {/* BoothMatch readout */}
      <div className="rounded-lg mb-3 px-3 py-2 flex items-center justify-center gap-4 flex-wrap"
        style={{ background: C.panel, border: `1px solid ${matched ? C.green : C.edge}`, transition: 'border-color 0.3s' }}>
        <span style={{ fontFamily: 'Oxanium', fontSize: 11, color: C.dim, letterSpacing: 2 }}>BOOTHMATCH</span>
        <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 15, color: C.cyan }}>A {bpmA ? bpmA.toFixed(1) : '\u2013\u2013'}</span>
        <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 13, color: matched ? C.green : C.dim }}>
          {bpmA && bpmB ? `\u0394 ${(bpmA - bpmB).toFixed(2)}` : '\u00B7'}
        </span>
        <span style={{ fontFamily: "'IBM Plex Mono'", fontSize: 15, color: C.orange }}>B {bpmB ? bpmB.toFixed(1) : '\u2013\u2013'}</span>
        {matched && <span style={{ color: C.green, fontFamily: 'Oxanium', fontWeight: 700, fontSize: 12 }}>{'\u25CF'} LOCKED</span>}
      </div>

      {/* Main rig: Deck A | Mixer | Deck B */}
      <div className="flex gap-3 items-start justify-center flex-wrap xl:flex-nowrap">
        {renderDeck('A')}

        {/* MIXER */}
        <div className="rounded-lg p-3 flex flex-col gap-3" style={{ background: C.panel, border: `1px solid ${C.edge}`, minWidth: 230 }}>
          <div className="flex items-center justify-between">
            <span style={{ fontFamily: 'Oxanium', fontWeight: 700, color: C.cyan, letterSpacing: 2, fontSize: 13 }}>MIXER</span>
            <span style={{ fontFamily: 'Oxanium', color: C.dim, fontSize: 10 }}>DJM-900NXS</span>
          </div>
          <div className="flex gap-2 justify-center">
            {renderChannelStrip('A')}
            {renderChannelStrip('B')}
          </div>
          <div className="flex items-center justify-between gap-3 mt-1">
            <div className="flex items-center gap-2">
              <Knob value={mix.master} onChange={(v) => setMix(m => ({ ...m, master: v }))} label="MASTER" size={44}
                onHint={showHint} hint="MASTER LEVEL: overall output to speakers. Set once so the master meter peaks in the orange on the loudest track, then leave it." />
              <VUMeter level={vu.M} />
            </div>
          </div>
          <div className="mt-1">
            <div style={{ color: C.dim, fontSize: 9, letterSpacing: 1, textAlign: 'center', marginBottom: 4 }}>CROSSFADER</div>
            <div className="px-2" onMouseEnter={() => showHint('CROSSFADER: blends between Deck A (left) and Deck B (right). For smooth house/techno blends many DJs leave it centered and mix with channel faders + EQ instead.')}>
              <CrossfaderH value={mix.xfader} onChange={(v) => setMix(m => ({ ...m, xfader: v }))} />
            </div>
            <div className="flex justify-between px-2" style={{ fontFamily: 'Oxanium', fontSize: 10, color: C.dim }}>
              <span style={{ color: C.cyan }}>A</span><span style={{ color: C.orange }}>B</span>
            </div>
          </div>
        </div>

        {renderDeck('B')}
      </div>

      {/* Hint bar */}
      {learn && (
        <div className="rounded-lg mt-3 px-4 py-3" style={{ background: C.panelHi, border: `1px solid ${C.cyanDim}`, minHeight: 56 }}>
          <span style={{ color: C.cyan, fontFamily: 'Oxanium', fontWeight: 700, fontSize: 11, letterSpacing: 2 }}>LEARN {'\u25B8'} </span>
          <span style={{ color: hint ? C.text : C.dim, fontSize: 13, fontFamily: "'IBM Plex Mono'", lineHeight: 1.5 }}>
            {hint || 'Hover any control to learn what it does on the real CDJ-2000NXS / DJM-900NXS.'}
          </span>
        </div>
      )}

      {/* First-mix guide */}
      {showGuide && (
        <div className="rounded-lg mt-3 p-4" style={{ background: C.panel, border: `1px solid ${C.edge}` }}>
          <div className="flex items-center justify-between mb-2">
            <span style={{ fontFamily: 'Oxanium', fontWeight: 700, color: C.cyan, letterSpacing: 2 }}>YOUR FIRST MIX &mdash; 6 STEPS</span>
            <span style={{ fontFamily: "'IBM Plex Mono'", color: C.dim, fontSize: 12 }}>{guideStep + 1}/6</span>
          </div>
          <div style={{ fontSize: 14, fontFamily: "'IBM Plex Mono'", color: C.text, lineHeight: 1.6, minHeight: 44 }}>
            <span style={{ color: C.orange, fontWeight: 600 }}>Step {guideStep + 1}. </span>{GUIDE[guideStep]}
          </div>
          <div className="flex gap-2 mt-3 items-center">
            <button onClick={() => setGuideStep(s => Math.max(0, s - 1))}
              style={{ background: C.panelHi, border: `1px solid ${C.edge}`, color: C.text, padding: '6px 14px', borderRadius: 6, fontSize: 13 }}>{'\u2190'} Back</button>
            <button onClick={() => setGuideStep(s => Math.min(5, s + 1))}
              style={{ background: C.cyanDim, border: `1px solid ${C.cyan}`, color: C.cyan, padding: '6px 14px', borderRadius: 6, fontSize: 13 }}>Next {'\u2192'}</button>
            <div className="flex items-center gap-1 ml-2">
              {GUIDE.map((_, i) => (
                <div key={i} onClick={() => setGuideStep(i)} className="cursor-pointer rounded-full"
                  style={{ width: 9, height: 9, background: i === guideStep ? C.cyan : '#2a2a32' }} />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Keyboard shortcuts */}
      {showKeys && (
        <div className="rounded-lg mt-3 p-4" style={{ background: C.panel, border: `1px solid ${C.edge}` }}>
          <div style={{ fontFamily: 'Oxanium', fontWeight: 700, color: C.cyan, letterSpacing: 2, marginBottom: 8, fontSize: 12 }}>KEYBOARD SHORTCUTS</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4" style={{ fontFamily: "'IBM Plex Mono'", fontSize: 12, color: C.dim, lineHeight: 1.8 }}>
            <div>
              <div style={{ color: C.cyan, fontWeight: 600, marginBottom: 4 }}>DECK A (Left Hand)</div>
              <div><kbd style={kbd}>Q</kbd> CUE (hold to preview, release snaps back)</div>
              <div><kbd style={kbd}>W</kbd> PLAY / PAUSE</div>
              <div><kbd style={kbd}>1</kbd> <kbd style={kbd}>2</kbd> <kbd style={kbd}>3</kbd> <kbd style={kbd}>4</kbd> Hot Cues A&ndash;D</div>
              <div><kbd style={kbd}>A</kbd> Loop In &middot; <kbd style={kbd}>S</kbd> Loop Out &middot; <kbd style={kbd}>D</kbd> Reloop</div>
            </div>
            <div>
              <div style={{ color: C.orange, fontWeight: 600, marginBottom: 4 }}>DECK B (Right Hand)</div>
              <div><kbd style={kbd}>[</kbd> CUE (hold to preview, release snaps back)</div>
              <div><kbd style={kbd}>]</kbd> PLAY / PAUSE</div>
              <div><kbd style={kbd}>7</kbd> <kbd style={kbd}>8</kbd> <kbd style={kbd}>9</kbd> <kbd style={kbd}>0</kbd> Hot Cues A&ndash;D</div>
              <div><kbd style={kbd}>L</kbd> Loop In &middot; <kbd style={kbd}>;</kbd> Loop Out &middot; <kbd style={kbd}>&apos;</kbd> Reloop</div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{ color: C.dim, fontSize: 11, fontFamily: "'IBM Plex Mono'", textAlign: 'center', marginTop: 16, lineHeight: 1.7 }}>
        Skills that transfer 1:1 to the real booth: cueing on the beat, reading the tempo fader, EQ swaps on the LOW, and jog-nudge beatmatching.
        <br />
        <span style={{ fontSize: 10, opacity: 0.5 }}>Browser limit: no headphone pre-cue (single output). Pitch changes with speed (vinyl mode).</span>
      </div>
    </div>
  );
}

// Kbd style for keyboard shortcuts display
const kbd: React.CSSProperties = {
  display: 'inline-block',
  padding: '1px 6px',
  borderRadius: 3,
  background: '#1a1a1e',
  border: `1px solid ${C.edge}`,
  fontFamily: "'IBM Plex Mono'",
  fontSize: 11,
  color: C.text,
  marginRight: 4,
};
