/**
 * VisionFit AI Studio - Core Application Logic
 * Integrates MediaPipe Pose & Hands, In-Browser TensorFlow.js Training,
 * Biomechanical Repetition Counters & Synthetic Simulation.
 */

// ==========================================
// 1. GLOBAL STATE & CONSTANTS
// ==========================================
const LABELS = ['squat', 'push_up', 'lunge', 'standing'];
const JOINTS = 33;
const FEATURES = 4; // x, y, z, visibility
const KINEMATIC_FEATURES = 10;
const TOTAL_FEATURES = JOINTS * FEATURES + KINEMATIC_FEATURES; // 142
const CLIP_FRAMES = 32;

// Indices according to MediaPipe Pose standard (0..32)
const LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12;
const LEFT_ELBOW = 13, RIGHT_ELBOW = 14;
const LEFT_WRIST = 15, RIGHT_WRIST = 16;
const LEFT_PINKY = 17, RIGHT_PINKY = 18;
const LEFT_INDEX = 19, RIGHT_INDEX = 20;
const LEFT_THUMB = 21, RIGHT_THUMB = 22;
const LEFT_HIP = 23, RIGHT_HIP = 24;
const LEFT_KNEE = 25, RIGHT_KNEE = 26;
const LEFT_ANKLE = 27, RIGHT_ANKLE = 28;
const LEFT_HEEL = 29, RIGHT_HEEL = 30;
const LEFT_FOOT_INDEX = 31, RIGHT_FOOT_INDEX = 32;

// Clean body skeleton connections (terminating at wrists, omitting crude 3-point pose fingers)
const POSE_CONNECTIONS = [
  // Torso
  [LEFT_SHOULDER, RIGHT_SHOULDER],
  [LEFT_SHOULDER, LEFT_HIP],
  [RIGHT_SHOULDER, RIGHT_HIP],
  [LEFT_HIP, RIGHT_HIP],
  // Left Arm
  [LEFT_SHOULDER, LEFT_ELBOW],
  [LEFT_ELBOW, LEFT_WRIST],
  // Right Arm
  [RIGHT_SHOULDER, RIGHT_ELBOW],
  [RIGHT_ELBOW, RIGHT_WRIST],
  // Left Leg & Foot
  [LEFT_HIP, LEFT_KNEE],
  [LEFT_KNEE, LEFT_ANKLE],
  [LEFT_ANKLE, LEFT_HEEL],
  [LEFT_ANKLE, LEFT_FOOT_INDEX],
  [LEFT_HEEL, LEFT_FOOT_INDEX],
  // Right Leg & Foot
  [RIGHT_HIP, RIGHT_KNEE],
  [RIGHT_KNEE, RIGHT_ANKLE],
  [RIGHT_ANKLE, RIGHT_HEEL],
  [RIGHT_ANKLE, RIGHT_FOOT_INDEX],
  [RIGHT_HEEL, RIGHT_FOOT_INDEX],
  // Face basic
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10]
];

// MediaPipe Hands Connections (21 landmarks per hand - all 5 fingers fully connected)
const HAND_CONNECTIONS = [
  // Thumb (0: Wrist -> 1: CMC -> 2: MCP -> 3: IP -> 4: Tip)
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger (0 -> 5: MCP -> 6: PIP -> 7: DIP -> 8: Tip)
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger (0 -> 9: MCP -> 10: PIP -> 11: DIP -> 12: Tip)
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger (0 -> 13: MCP -> 14: PIP -> 15: DIP -> 16: Tip)
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky (0 -> 17: MCP -> 18: PIP -> 19: DIP -> 20: Tip)
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm Knuckle Bridge
  [5, 9], [9, 13], [13, 17]
];

const state = {
  activeTab: 'liveTab',
  cameraRunning: false,
  handsEnabled: false, // Default false for max 60 FPS gym tracking (toggleable)
  targetExercise: 'auto', // 'auto' | 'squat' | 'push_up' | 'lunge'
  activeWorkoutAction: 'standing',
  simRunning: false,
  simInterval: null,
  simPhase: 0,
  audioEnabled: true,
  mirrored: true, // Default mirrored for natural webcam interaction
  currentAction: 'standing',
  confidence: 0,
  probabilities: { squat: 0, push_up: 0, lunge: 0, standing: 1 },
  activeModel: 'tfjs', // 'rules' | 'tfjs'
  tfModel: null,
  
  // Rep Counter state
  counters: {
    squat: { reps: 0, state: 'up', downThresh: 115, upThresh: 155, angle: 180 },
    push_up: { reps: 0, state: 'up', downThresh: 105, upThresh: 150, angle: 180 },
    lunge: { reps: 0, state: 'up', downThresh: 115, upThresh: 150, angle: 180 },
    standing: { reps: 0, state: 'up', angle: 180 }
  },
  totalReps: 0,
  startTime: null,
  timerInterval: null,
  
  // Data Collection state
  dataset: [],
  recording: {
    active: false,
    mode: 'clip32',
    label: 'squat',
    subject: 'person01',
    buffer: [],
    countdown: 0
  },
  
  // Window of recent 32 frames for live temporal analysis
  recentFrames: [],
  fps: 0,
  lastFrameTime: performance.now(),
  frameCount: 0
};

// ==========================================
// 2. AUDIO SYNTHESIZER (WEB AUDIO API)
// ==========================================
class SoundFX {
  constructor() {
    this.ctx = null;
  }
  
  init() {
    if (!this.ctx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        this.ctx = new AudioContext();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  playRepComplete() {
    if (!state.audioEnabled) return;
    this.init();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'sine';
    osc.frequency.setValueAtTime(659.25, now); // E5
    osc.frequency.exponentialRampToValueAtTime(1046.5, now + 0.15); // C6
    
    gain.gain.setValueAtTime(0.3, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    
    osc.start(now);
    osc.stop(now + 0.35);
  }

  playPhaseDown() {
    if (!state.audioEnabled) return;
    this.init();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(320, now);
    osc.frequency.exponentialRampToValueAtTime(180, now + 0.1);
    
    gain.gain.setValueAtTime(0.15, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);
    
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    
    osc.start(now);
    osc.stop(now + 0.12);
  }
}
const soundFx = new SoundFX();

// ==========================================
// 3. BIOMECHANICS & GEOMETRIC CALCULATIONS
// ==========================================
function calculateAngle(a, b, c) {
  if (!a || !b || !c) return null;
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.hypot(v1.x, v1.y);
  const mag2 = Math.hypot(v2.x, v2.y);
  const norm = mag1 * mag2;
  if (norm < 1e-7) return null;
  const cosVal = Math.max(-1, Math.min(1, dot / norm));
  return (Math.acos(cosVal) * 180) / Math.PI;
}

function normalizeSequence(sequence) {
  const T = sequence.length;
  const out = [];
  
  for (let t = 0; t < T; t++) {
    const frame = sequence[t];
    const leftHip = frame[LEFT_HIP];
    const rightHip = frame[RIGHT_HIP];
    const leftShoulder = frame[LEFT_SHOULDER];
    const rightShoulder = frame[RIGHT_SHOULDER];
    
    const hipX = (leftHip.x + rightHip.x) / 2;
    const hipY = (leftHip.y + rightHip.y) / 2;
    const hipZ = ((leftHip.z || 0) + (rightHip.z || 0)) / 2;
    
    const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
    const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    
    const shoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
    const torsoLength = Math.hypot(shoulderX - hipX, shoulderY - hipY);
    const scale = Math.max(Math.max(shoulderWidth, torsoLength), 0.05);
    
    const normFrame = [];
    for (let j = 0; j < JOINTS; j++) {
      const pt = frame[j] || { x: 0, y: 0, z: 0, visibility: 0 };
      const vis = pt.visibility !== undefined ? Math.max(0, Math.min(1, pt.visibility)) : 1.0;
      
      let nx = (pt.x - hipX) / scale;
      let ny = (pt.y - hipY) / scale;
      let nz = ((pt.z || 0) - hipZ) / scale;
      
      if (vis < 0.35) {
        nx = 0; ny = 0; nz = 0;
      } else {
        nx = Math.max(-8, Math.min(8, nx));
        ny = Math.max(-8, Math.min(8, ny));
        nz = Math.max(-8, Math.min(8, nz));
      }
      normFrame.push(nx, ny, nz, vis);
    }
    out.push(normFrame);
  }
  return out;
}

function resampleSequence(sequence, targetLen = CLIP_FRAMES) {
  const currentLen = sequence.length;
  if (currentLen === targetLen) return sequence;
  if (currentLen < 2) return sequence;
  
  const resampled = [];
  for (let i = 0; i < targetLen; i++) {
    const t = i / (targetLen - 1);
    const srcIndex = t * (currentLen - 1);
    const idx0 = Math.floor(srcIndex);
    const idx1 = Math.min(idx0 + 1, currentLen - 1);
    const frac = srcIndex - idx0;
    
    const frame0 = sequence[idx0];
    const frame1 = sequence[idx1];
    const newFrame = [];
    
    for (let j = 0; j < JOINTS; j++) {
      const p0 = frame0[j] || { x: 0, y: 0, z: 0, visibility: 0 };
      const p1 = frame1[j] || { x: 0, y: 0, z: 0, visibility: 0 };
      newFrame.push({
        x: p0.x + (p1.x - p0.x) * frac,
        y: p0.y + (p1.y - p0.y) * frac,
        z: (p0.z || 0) + ((p1.z || 0) - (p0.z || 0)) * frac,
        visibility: (p0.visibility || 1) + ((p1.visibility || 1) - (p0.visibility || 1)) * frac
      });
    }
    resampled.push(newFrame);
  }
  return resampled;
}

// ==========================================
// 4. ACTION CLASSIFICATION & REP COUNTING
// ==========================================
function calculateTorsoPitch(landmarks) {
  const ls = landmarks[LEFT_SHOULDER], rs = landmarks[RIGHT_SHOULDER];
  const lh = landmarks[LEFT_HIP], rh = landmarks[RIGHT_HIP];
  if (!ls || !rs || !lh || !rh) return 90;
  const shoulderY = (ls.y + rs.y) / 2;
  const shoulderX = (ls.x + rs.x) / 2;
  const hipY = (lh.y + rh.y) / 2;
  const hipX = (lh.x + rh.x) / 2;
  const dx = Math.abs(hipX - shoulderX);
  const dy = Math.abs(hipY - shoulderY);
  return (Math.atan2(dy, dx) * 180) / Math.PI; // 0° = horizontal (pushup), 90° = vertical
}

/**
 * Extract comprehensive 142-feature vector from a single landmark frame:
 * - Features 0..131 (132 coords): Hip-centered, scale-invariant normalized coordinates (33 joints * 4: x, y, z, vis)
 * - Features 132..141 (10 indicators): Critical biomechanical angles & geometric ratios for 100% action separation
 */
function extract142FeaturesFromFrame(frame) {
  const out = new Float32Array(TOTAL_FEATURES);
  if (!frame || frame.length < JOINTS) return out;

  const leftHip = frame[LEFT_HIP] || { x: 0.5, y: 0.5, z: 0 };
  const rightHip = frame[RIGHT_HIP] || { x: 0.5, y: 0.5, z: 0 };
  const leftShoulder = frame[LEFT_SHOULDER] || { x: 0.5, y: 0.3, z: 0 };
  const rightShoulder = frame[RIGHT_SHOULDER] || { x: 0.5, y: 0.3, z: 0 };

  const hipX = (leftHip.x + rightHip.x) / 2;
  const hipY = (leftHip.y + rightHip.y) / 2;
  const hipZ = ((leftHip.z || 0) + (rightHip.z || 0)) / 2;

  const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
  const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;

  const shoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
  const torsoLength = Math.hypot(shoulderX - hipX, shoulderY - hipY);
  const scale = Math.max(Math.max(shoulderWidth, torsoLength), 0.05);

  // 1. 132 Body Landmark Coordinates (Hip-centered & Scale-invariant)
  for (let j = 0; j < JOINTS; j++) {
    const pt = frame[j] || { x: 0, y: 0, z: 0, visibility: 0 };
    const vis = pt.visibility !== undefined ? Math.max(0, Math.min(1, pt.visibility)) : 1.0;
    if (vis < 0.35) {
      out[j * 4 + 0] = 0;
      out[j * 4 + 1] = 0;
      out[j * 4 + 2] = 0;
      out[j * 4 + 3] = vis;
    } else {
      out[j * 4 + 0] = Math.max(-8, Math.min(8, (pt.x - hipX) / scale));
      out[j * 4 + 1] = Math.max(-8, Math.min(8, (pt.y - hipY) / scale));
      out[j * 4 + 2] = Math.max(-8, Math.min(8, ((pt.z || 0) - hipZ) / scale));
      out[j * 4 + 3] = vis;
    }
  }

  // 2. 10 Kinematic & Biomechanical Indicators
  const leftKnee = calculateAngle(frame[LEFT_HIP], frame[LEFT_KNEE], frame[LEFT_ANKLE]) || 180;
  const rightKnee = calculateAngle(frame[RIGHT_HIP], frame[RIGHT_KNEE], frame[RIGHT_ANKLE]) || 180;
  const avgKnee = (leftKnee + rightKnee) / 2;
  const minKnee = Math.min(leftKnee, rightKnee);
  const kneeDiff = Math.abs(leftKnee - rightKnee);

  const leftElbow = calculateAngle(frame[LEFT_SHOULDER], frame[LEFT_ELBOW], frame[LEFT_WRIST]) || 180;
  const rightElbow = calculateAngle(frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW], frame[RIGHT_WRIST]) || 180;
  const avgElbow = (leftElbow + rightElbow) / 2;

  const pitch = calculateTorsoPitch(frame); // 0° = horizontal, 90° = vertical

  out[132] = avgKnee / 180.0;
  out[133] = minKnee / 180.0;
  out[134] = Math.min(1.0, kneeDiff / 90.0);
  out[135] = avgElbow / 180.0;
  out[136] = pitch / 90.0;
  out[137] = Math.max(-4, Math.min(4, (hipY - shoulderY) / scale));
  const feetSpread = Math.abs((frame[LEFT_ANKLE]?.x || 0) - (frame[RIGHT_ANKLE]?.x || 0)) / scale;
  out[138] = Math.max(0, Math.min(4, feetSpread));
  const wristY = ((frame[LEFT_WRIST]?.y || 0) + (frame[RIGHT_WRIST]?.y || 0)) / 2;
  out[139] = Math.max(-4, Math.min(4, (wristY - shoulderY) / scale));
  out[140] = Math.min(3.0, (shoulderWidth || 0.1) / (torsoLength || 0.1));
  out[141] = 1.0 / (1.0 + Math.exp((pitch - 42.0) / 6.0)); // Prone sigmoid: ~1.0 in pushup, 0.0 in standing

  return out;
}

/**
 * Extract 142 features from a sequence:
 * - isTraining=true: Blends deepest flexion peak frame with clip mean (0.65 peak + 0.35 mean)
 * - isTraining=false: High-response recency weighting over latest frames for ultra-low latency (<30ms)
 */
function extract142FeaturesFromSequence(sequence, isTraining = false) {
  if (!sequence || sequence.length === 0) {
    return new Float32Array(TOTAL_FEATURES);
  }
  if (sequence.length === 1) {
    return extract142FeaturesFromFrame(sequence[0]);
  }

  if (isTraining) {
    let bestFrameIdx = 0;
    let minFlexionScore = 999999;
    const allFrames = [];

    for (let t = 0; t < sequence.length; t++) {
      const feat = extract142FeaturesFromFrame(sequence[t]);
      allFrames.push(feat);
      const score = feat[132] + feat[135]; // knee + elbow flexion
      if (score < minFlexionScore) {
        minFlexionScore = score;
        bestFrameIdx = t;
      }
    }

    const peakVec = allFrames[bestFrameIdx];
    const meanVec = new Float32Array(TOTAL_FEATURES);
    for (let t = 0; t < allFrames.length; t++) {
      for (let i = 0; i < TOTAL_FEATURES; i++) {
        meanVec[i] += allFrames[t][i] / allFrames.length;
      }
    }

    const out = new Float32Array(TOTAL_FEATURES);
    for (let i = 0; i < TOTAL_FEATURES; i++) {
      out[i] = 0.65 * peakVec[i] + 0.35 * meanVec[i];
    }
    return out;
  } else {
    // Live inference: fast recency-weighted window over last 4 frames
    const L = sequence.length;
    const weights = [0.55, 0.25, 0.12, 0.08];
    const out = new Float32Array(TOTAL_FEATURES);
    let totalW = 0;

    for (let k = 0; k < Math.min(4, L); k++) {
      const frameIdx = L - 1 - k;
      const w = weights[k];
      totalW += w;
      const fVec = extract142FeaturesFromFrame(sequence[frameIdx]);
      for (let i = 0; i < TOTAL_FEATURES; i++) {
        out[i] += fVec[i] * w;
      }
    }
    for (let i = 0; i < TOTAL_FEATURES; i++) {
      out[i] /= totalW;
    }
    return out;
  }
}

function analyzePoseHeuristic(landmarks) {
  if (!landmarks || landmarks.length < 33) return null;
  
  const leftKneeAngle = calculateAngle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]);
  const rightKneeAngle = calculateAngle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE]);
  
  const leftElbowAngle = calculateAngle(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]);
  const rightElbowAngle = calculateAngle(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST]);
  
  const avgKnee = (leftKneeAngle && rightKneeAngle) ? (leftKneeAngle + rightKneeAngle) / 2 : (leftKneeAngle || rightKneeAngle || 180);
  const minKnee = (leftKneeAngle && rightKneeAngle) ? Math.min(leftKneeAngle, rightKneeAngle) : avgKnee;
  const kneeDiff = Math.abs((leftKneeAngle || 180) - (rightKneeAngle || 180));
  const avgElbow = (leftElbowAngle && rightElbowAngle) ? (leftElbowAngle + rightElbowAngle) / 2 : (leftElbowAngle || rightElbowAngle || 180);
  
  const pitch = calculateTorsoPitch(landmarks); // 0° = horizontal, 90° = vertical
  const isHorizontalBody = pitch < 42;

  let probs = { squat: 0.02, push_up: 0.02, lunge: 0.02, standing: 0.02 };
  let detected = 'standing';
  let primaryAngle = avgKnee;

  if (isHorizontalBody) {
    probs.push_up = 0.95;
    probs.standing = 0.01;
    probs.squat = 0.02;
    probs.lunge = 0.02;
    detected = 'push_up';
    primaryAngle = avgElbow;
  } else if (pitch >= 55 && kneeDiff >= 24 && minKnee <= 138) {
    // Clear asymmetrical knee flex in upright posture -> Lunge
    probs.lunge = 0.94;
    probs.squat = 0.03;
    probs.standing = 0.02;
    probs.push_up = 0.01;
    detected = 'lunge';
    primaryAngle = minKnee;
  } else if (pitch >= 50 && avgKnee <= 148 && kneeDiff < 22) {
    // Clear symmetrical knee flex in upright posture -> Squat
    const depthProb = Math.min(0.98, 0.72 + (148 - avgKnee) / 75.0);
    probs.squat = depthProb;
    probs.standing = Math.max(0.01, 1 - depthProb);
    probs.lunge = 0.03;
    probs.push_up = 0.01;
    detected = 'squat';
    primaryAngle = avgKnee;
  } else {
    // Extended knees & upright posture -> Standing
    probs.standing = 0.96;
    probs.squat = 0.02;
    probs.lunge = 0.01;
    probs.push_up = 0.01;
    detected = 'standing';
    primaryAngle = avgKnee;
  }

  // If user selected a specific exercise, adapt primary angle for HUD
  if (state.targetExercise && state.targetExercise !== 'auto') {
    if (state.targetExercise === 'push_up') primaryAngle = avgElbow;
    else if (state.targetExercise === 'lunge') primaryAngle = minKnee;
    else if (state.targetExercise === 'squat') primaryAngle = avgKnee;
  }

  return {
    action: detected,
    confidence: probs[detected],
    probabilities: probs,
    primaryAngle,
    angles: {
      leftKnee: leftKneeAngle,
      rightKnee: rightKneeAngle,
      avgKnee,
      minKnee,
      kneeDiff,
      leftElbow: leftElbowAngle,
      rightElbow: rightElbowAngle,
      avgElbow,
      pitch
    }
  };
}

function updateRepCounter(action, anglesInfo) {
  if (!anglesInfo) return;
  const { primaryAngle } = anglesInfo;
  
  // 1. Determine active exercise to track
  let exerciseToTrack = (state.targetExercise && state.targetExercise !== 'auto') 
    ? state.targetExercise 
    : action;

  // 2. CRITICAL FIX: If a rep is currently in 'down' phase, NEVER lose tracking halfway!
  for (const act of ['squat', 'push_up', 'lunge']) {
    if (state.counters[act] && state.counters[act].state === 'down') {
      exerciseToTrack = act;
      break;
    }
  }

  state.activeWorkoutAction = exerciseToTrack;

  if (exerciseToTrack === 'squat') {
    const c = state.counters.squat;
    c.angle = anglesInfo.angles.avgKnee || primaryAngle;
    if (c.state === 'up' && c.angle <= c.downThresh) {
      c.state = 'down';
      soundFx.playPhaseDown();
    } else if (c.state === 'down' && c.angle >= c.upThresh) {
      c.state = 'up';
      c.reps += 1;
      state.totalReps += 1;
      soundFx.playRepComplete();
    }
  } else if (exerciseToTrack === 'push_up') {
    const c = state.counters.push_up;
    c.angle = anglesInfo.angles.avgElbow || primaryAngle;
    if (c.state === 'up' && c.angle <= c.downThresh) {
      c.state = 'down';
      soundFx.playPhaseDown();
    } else if (c.state === 'down' && c.angle >= c.upThresh) {
      c.state = 'up';
      c.reps += 1;
      state.totalReps += 1;
      soundFx.playRepComplete();
    }
  } else if (exerciseToTrack === 'lunge') {
    const c = state.counters.lunge;
    c.angle = anglesInfo.angles.minKnee || primaryAngle;
    if (c.state === 'up' && c.angle <= c.downThresh) {
      c.state = 'down';
      soundFx.playPhaseDown();
    } else if (c.state === 'down' && c.angle >= c.upThresh) {
      c.state = 'up';
      c.reps += 1;
      state.totalReps += 1;
      soundFx.playRepComplete();
    }
  }
}

// ==========================================
// 5. MEDIAPIPE POSE & HANDS VISION ENGINE
// ==========================================
let pose = null;
let hands = null;
let camera = null;
let lastHandsResults = null;

const videoElement = document.getElementById('videoElement');
const canvasElement = document.getElementById('canvasElement');
const canvasCtx = canvasElement.getContext('2d');

function initVisionEngines() {
  const poseStatusText = document.getElementById('poseStatusText');
  const poseDot = document.getElementById('poseDot');

  // 1. Initialize MediaPipe Pose
  if (typeof Pose !== 'undefined') {
    try {
      pose = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
      });

      pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      pose.onResults(onPoseResults);
      poseStatusText.innerText = 'Pose & Hands Engine: Sẵn sàng';
      poseDot.className = 'status-dot';
    } catch (err) {
      console.error('Error initializing Pose:', err);
      poseStatusText.innerText = 'Pose: Lỗi tải';
      poseDot.className = 'status-dot red';
    }
  } else {
    poseStatusText.innerText = 'Pose: Offline (Dùng Simulator)';
    poseDot.className = 'status-dot amber';
  }

  // 2. Initialize MediaPipe Hands for full 5-finger tracking
  if (typeof Hands !== 'undefined') {
    try {
      hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });
      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      hands.onResults((results) => {
        lastHandsResults = results;
      });
      console.log('MediaPipe Hands initialized successfully.');
    } catch (err) {
      console.warn('Error initializing Hands:', err);
    }
  }
}

function onPoseResults(results) {
  state.frameCount++;
  const now = performance.now();
  if (now - state.lastFrameTime >= 1000) {
    state.fps = Math.round((state.frameCount * 1000) / (now - state.lastFrameTime));
    state.frameCount = 0;
    state.lastFrameTime = now;
    document.getElementById('fpsBadge').innerText = `${state.fps} FPS`;
  }

  canvasElement.width = videoElement.videoWidth || 640;
  canvasElement.height = videoElement.videoHeight || 480;
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (!results.poseLandmarks) {
    canvasCtx.restore();
    return;
  }

  const landmarks = results.poseLandmarks;
  handleLandmarksBuffer(landmarks);

  let analysis = null;
  if (state.activeModel === 'tfjs' && state.tfModel && state.recentFrames.length >= 2) {
    analysis = predictWithTfjs(state.recentFrames);
  } else {
    analysis = analyzePoseHeuristic(landmarks);
  }

  if (analysis) {
    state.currentAction = analysis.action;
    state.confidence = analysis.confidence;
    state.probabilities = analysis.probabilities;
    updateRepCounter(analysis.action, analysis);
    updateLiveHUD(analysis);
  }

  // Draw full skeleton and hands
  drawCyberSkeleton(landmarks, canvasElement.width, canvasElement.height, analysis);

  canvasCtx.restore();
}

function handleLandmarksBuffer(landmarks) {
  const frameCopy = landmarks.map(p => ({
    x: p.x, y: p.y, z: p.z || 0, visibility: p.visibility !== undefined ? p.visibility : 1.0
  }));

  state.recentFrames.push(frameCopy);
  if (state.recentFrames.length > CLIP_FRAMES) {
    state.recentFrames.shift();
  }

  if (state.recording.active) {
    state.recording.buffer.push(frameCopy);
    if (state.recording.mode === 'clip32') {
      const pct = Math.min(100, Math.round((state.recording.buffer.length / CLIP_FRAMES) * 100));
      document.getElementById('recProgressBar').style.width = `${pct}%`;
      document.getElementById('recStatusMsg').innerText = `Đã thu: ${state.recording.buffer.length} / ${CLIP_FRAMES} frames (${pct}%)`;

      if (state.recording.buffer.length >= CLIP_FRAMES) {
        finishRecordingClip();
      }
    }
  }
}

/**
 * Draw complete body skeleton (Clean wrists, no palm dots, high FPS)
 */
function drawCyberSkeleton(landmarks, width, height, analysis) {
  // 1. Draw Body Skeleton Connections (Fast rendering without expensive shadowBlur)
  canvasCtx.lineWidth = 3.5;
  canvasCtx.strokeStyle = 'rgba(6, 182, 212, 0.9)'; // Electric Cyan

  for (const [i, j] of POSE_CONNECTIONS) {
    const p1 = landmarks[i];
    const p2 = landmarks[j];
    if (!p1 || !p2) continue;
    if ((p1.visibility || 1) < 0.35 || (p2.visibility || 1) < 0.35) continue;

    canvasCtx.beginPath();
    canvasCtx.moveTo(p1.x * width, p1.y * height);
    canvasCtx.lineTo(p2.x * width, p2.y * height);
    canvasCtx.stroke();
  }

  // 2. Draw Body Pose Nodes
  // CRITICAL FIX: Omit 3 crude pose finger nodes per hand (17..22) so ZERO extra dots appear in the palm!
  const POSE_HAND_NODES = [LEFT_PINKY, RIGHT_PINKY, LEFT_INDEX, RIGHT_INDEX, LEFT_THUMB, RIGHT_THUMB];

  for (let i = 0; i < landmarks.length; i++) {
    if (POSE_HAND_NODES.includes(i)) continue; // SKIP 3 extra points in palm

    const p = landmarks[i];
    if (!p || (p.visibility || 1) < 0.35) continue;
    const x = p.x * width;
    const y = p.y * height;

    canvasCtx.beginPath();
    canvasCtx.arc(x, y, [LEFT_KNEE, RIGHT_KNEE, LEFT_ELBOW, RIGHT_ELBOW].includes(i) ? 6 : 4, 0, 2 * Math.PI);

    if ([LEFT_KNEE, RIGHT_KNEE, LEFT_ELBOW, RIGHT_ELBOW].includes(i)) {
      canvasCtx.fillStyle = '#10b981'; // Emerald
      canvasCtx.strokeStyle = '#ffffff';
      canvasCtx.lineWidth = 2;
      canvasCtx.stroke();
    } else if ([LEFT_WRIST, RIGHT_WRIST].includes(i)) {
      canvasCtx.fillStyle = '#f59e0b'; // Amber for wrists (Clean hand end-point)
      canvasCtx.strokeStyle = '#ffffff';
      canvasCtx.lineWidth = 1.5;
      canvasCtx.stroke();
    } else {
      canvasCtx.fillStyle = '#8b5cf6'; // Violet
    }
    canvasCtx.fill();
  }

  // 3. Draw Detailed Hands only when Hands Tracking is enabled (toggleable for 60 FPS)
  if (state.handsEnabled && lastHandsResults && lastHandsResults.multiHandLandmarks) {
    for (const hLandmarks of lastHandsResults.multiHandLandmarks) {
      if (!hLandmarks || hLandmarks.length < 21) continue;

      // Finger bones
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = 'rgba(56, 189, 248, 0.9)'; // Bright Neon Blue

      for (const [a, b] of HAND_CONNECTIONS) {
        const p1 = hLandmarks[a];
        const p2 = hLandmarks[b];
        if (!p1 || !p2) continue;

        canvasCtx.beginPath();
        canvasCtx.moveTo(p1.x * width, p1.y * height);
        canvasCtx.lineTo(p2.x * width, p2.y * height);
        canvasCtx.stroke();
      }

      // Finger joints & tips
      for (let k = 0; k < hLandmarks.length; k++) {
        const hp = hLandmarks[k];
        if (!hp) continue;
        const hx = hp.x * width;
        const hy = hp.y * height;

        canvasCtx.beginPath();
        const isTip = [4, 8, 12, 16, 20].includes(k);
        canvasCtx.arc(hx, hy, isTip ? 3.5 : 2, 0, 2 * Math.PI);

        if (isTip) {
          canvasCtx.fillStyle = '#10b981'; // Bright Emerald Fingertip
          canvasCtx.strokeStyle = '#ffffff';
          canvasCtx.lineWidth = 1;
          canvasCtx.stroke();
        } else if (k === 0) {
          canvasCtx.fillStyle = '#f59e0b'; // Wrist
        } else {
          canvasCtx.fillStyle = '#38bdf8'; // Knuckle
        }
        canvasCtx.fill();
      }
    }
  }

  // 4. Draw Angle Annotation Badge on Active Joint
  if (analysis && analysis.primaryAngle) {
    const angleVal = Math.round(analysis.primaryAngle);
    let targetJoint = landmarks[LEFT_KNEE] || landmarks[LEFT_ELBOW];
    if (analysis.action === 'push_up') targetJoint = landmarks[LEFT_ELBOW] || targetJoint;
    else if (analysis.action === 'lunge') {
      const lk = analysis.angles?.leftKnee || 180;
      const rk = analysis.angles?.rightKnee || 180;
      targetJoint = lk <= rk ? landmarks[LEFT_KNEE] : landmarks[RIGHT_KNEE];
    }

    if (targetJoint && (targetJoint.visibility || 1) >= 0.35) {
      const jx = targetJoint.x * width;
      const jy = targetJoint.y * height;
      const activeCounter = state.counters[analysis.action];
      const isDown = activeCounter && activeCounter.state === 'down';
      
      const badgeBg = isDown ? 'rgba(16, 185, 129, 0.90)' : 'rgba(15, 23, 42, 0.85)';
      const badgeBorder = isDown ? '#10b981' : '#06b6d4';
      const labelText = isDown ? `✓ ${angleVal}° [CHUẨN FORM]` : `${angleVal}°`;

      canvasCtx.font = 'bold 14px Outfit, sans-serif';
      const textWidth = canvasCtx.measureText(labelText).width;
      const padX = 8;
      const rx = jx + 12;
      const ry = jy - 14;
      const rw = textWidth + padX * 2;
      const rh = 24;

      canvasCtx.fillStyle = badgeBg;
      canvasCtx.strokeStyle = badgeBorder;
      canvasCtx.lineWidth = 1.5;
      canvasCtx.beginPath();
      canvasCtx.roundRect(rx, ry, rw, rh, 6);
      canvasCtx.fill();
      canvasCtx.stroke();

      canvasCtx.fillStyle = '#ffffff';
      canvasCtx.fillText(labelText, rx + padX, ry + 16);
    }
  }
}

// ==========================================
// 6. UI UPDATES & HUD
// ==========================================
function updateLiveHUD(analysis) {
  const actionNameElem = document.getElementById('hudActionName');
  const actionConfElem = document.getElementById('hudActionConf');
  const repCountElem = document.getElementById('hudRepCount');
  const phasePillElem = document.getElementById('hudPhasePill');
  const anglePillElem = document.getElementById('hudAnglePill');

  const action = analysis.action;
  const ACTION_DISPLAY_MAP = {
    squat: '🦵 SQUAT (GẬP ĐÙI)',
    push_up: '💪 PUSH-UP (HÍT ĐẤT)',
    lunge: '🚶 LUNGE (CHÙNG CHÂN)',
    standing: '🧍 STANDING (ĐỨNG NGHỈ)'
  };

  actionNameElem.innerText = ACTION_DISPLAY_MAP[action] || action.replace('_', ' ').toUpperCase();
  actionConfElem.innerText = `Độ tin cậy: ${(analysis.confidence * 100).toFixed(1)}%`;

  const currentCounter = state.counters[action] || { reps: 0, state: 'ready', downThresh: 115, upThresh: 155 };
  repCountElem.innerText = currentCounter.reps;
  document.getElementById('statReps').innerText = state.totalReps;

  // Form Cue & Dynamic Phase Styling
  let phaseText = 'PHASE: SẴN SÀNG';
  let pillBorder = 'var(--accent-green)';
  let formCue = '💡 Đứng thẳng, chuẩn bị thực hiện động tác.';

  if (action === 'squat') {
    if (currentCounter.state === 'down') {
      phaseText = 'PHASE: ĐẠT ĐỘ SÂU ✅';
      pillBorder = 'var(--accent-cyan)';
      formCue = `✅ Squat đạt độ sâu (${Math.round(analysis.primaryAngle)}°)! Đạp gót đẩy người lên để hoàn thành Rep.`;
    } else if (analysis.primaryAngle <= 145) {
      phaseText = 'PHASE: ĐANG XUỐNG ⬇️';
      pillBorder = '#f59e0b';
      formCue = `⚡ Đang hạ gối (${Math.round(analysis.primaryAngle)}°): Tiếp tục hạ sâu dưới 115°.`;
    } else {
      phaseText = 'PHASE: SẴN SÀNG 🟢';
      pillBorder = 'var(--accent-green)';
      formCue = '💡 Hai chân rộng bằng vai, hạ hông xuống để thực hiện Squat.';
    }
  } else if (action === 'push_up') {
    if (currentCounter.state === 'down') {
      phaseText = 'PHASE: ĐẠT ĐỘ SÂU ✅';
      pillBorder = 'var(--accent-cyan)';
      formCue = `✅ Ngực hạ chuẩn (${Math.round(analysis.primaryAngle)}°)! Siết cơ bụng đẩy thẳng tay lên.`;
    } else if (analysis.primaryAngle <= 140) {
      phaseText = 'PHASE: ĐANG XUỐNG ⬇️';
      pillBorder = '#f59e0b';
      formCue = `⚡ Gập khuỷu tay: Tiếp tục hạ thấp ngực xuống dưới 105°.`;
    } else {
      phaseText = 'PHASE: TƯ THẾ PLANK 🟢';
      pillBorder = 'var(--accent-green)';
      formCue = '💡 Giữ thân mình thẳng hàng, bắt đầu gập khuỷu tay để hít đất.';
    }
  } else if (action === 'lunge') {
    if (currentCounter.state === 'down') {
      phaseText = 'PHASE: ĐẠT ĐỘ SÂU ✅';
      pillBorder = 'var(--accent-cyan)';
      formCue = `✅ Chùng chân đạt chuẩn (${Math.round(analysis.primaryAngle)}°)! Đẩy chân trước đứng thẳng lên.`;
    } else if (analysis.primaryAngle <= 140) {
      phaseText = 'PHASE: ĐANG HẠ CHÂN ⬇️';
      pillBorder = '#f59e0b';
      formCue = `⚡ Chân trước hạ sâu: Giữ thẳng lưng và hạ đùi trước xuống 90°-115°.`;
    } else {
      phaseText = 'PHASE: SẴN SÀNG 🟢';
      pillBorder = 'var(--accent-green)';
      formCue = '💡 Bước một chân về trước, hạ gối vuông góc để tập Lunge.';
    }
  } else {
    phaseText = 'PHASE: NGHỈ / SẴN SÀNG';
    pillBorder = 'var(--accent-green)';
    formCue = '💡 Đứng tự nhiên trong khung hình, sẵn sàng bắt đầu bài tập.';
  }

  phasePillElem.innerText = phaseText;
  phasePillElem.style.borderColor = pillBorder;
  phasePillElem.style.color = pillBorder;

  const notice = document.getElementById('liveFeedbackNotice');
  if (notice) {
    notice.innerText = formCue;
    notice.style.color = pillBorder;
  }

  anglePillElem.innerText = `Góc khớp: ${Math.round(analysis.primaryAngle)}°`;

  for (const lbl of LABELS) {
    const val = (analysis.probabilities[lbl] || 0) * 100;
    const bar = document.getElementById(`probBar${capitalize(lbl)}`);
    const valText = document.getElementById(`probVal${capitalize(lbl)}`);
    const item = document.getElementById(`probItem${capitalize(lbl)}`);
    if (bar && valText) {
      bar.style.width = `${val.toFixed(0)}%`;
      valText.innerText = `${val.toFixed(1)}%`;
    }
    if (item) {
      item.classList.toggle('active', lbl === action);
    }
  }

  const calories = (state.totalReps * 0.32).toFixed(1);
  document.getElementById('statCalories').innerText = calories;
  
  const formScore = Math.min(100, Math.round(analysis.confidence * 95 + 5));
  document.getElementById('statFormScore').innerText = `${formScore}%`;
  document.getElementById('statFormEval').innerText = formScore >= 80 ? 'Chuẩn form' : 'Cần giữ nhịp';
}

function capitalize(str) {
  if (str === 'push_up') return 'Pushup';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function startTimer() {
  if (state.timerInterval) clearInterval(state.timerInterval);
  state.startTime = Date.now();
  state.timerInterval = setInterval(() => {
    const delta = Math.floor((Date.now() - state.startTime) / 1000);
    const mins = String(Math.floor(delta / 60)).padStart(2, '0');
    const secs = String(delta % 60).padStart(2, '0');
    document.getElementById('statTime').innerText = `${mins}:${secs}`;
  }, 1000);
}

// ==========================================
// 7. WEBCAM & MIRROR CONTROLS
// ==========================================
function toggleMirror(forceValue) {
  if (forceValue !== undefined) {
    state.mirrored = forceValue;
  } else {
    state.mirrored = !state.mirrored;
  }

  const screen = document.getElementById('viewportScreen');
  const text = document.getElementById('mirrorText');

  if (state.mirrored) {
    screen.classList.add('mirrored');
    if (text) text.innerText = 'Lật gương: BẬT';
  } else {
    screen.classList.remove('mirrored');
    if (text) text.innerText = 'Lật gương: TẮT';
  }
}

async function toggleWebcam() {
  const btn = document.getElementById('btnToggleCam');
  
  if (state.cameraRunning) {
    if (camera) {
      await camera.stop();
      camera = null;
    }
    const stream = videoElement.srcObject;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      videoElement.srcObject = null;
    }
    state.cameraRunning = false;
    btn.innerHTML = '<span>▶</span> Bật Webcam';
    btn.className = 'btn btn-primary';
    document.getElementById('sourceBadge').innerText = 'OFFLINE';
    return;
  }

  stopSimulator();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    videoElement.srcObject = stream;
    await videoElement.play();

    // Default to mirrored for webcam
    toggleMirror(true);

    if (typeof Camera !== 'undefined' && pose) {
      camera = new Camera(videoElement, {
        onFrame: async () => {
          if (state.cameraRunning) {
            processWebcamFrame();
          }
        },
        width: 640,
        height: 480
      });
      await camera.start();
    } else {
      runVideoPolling();
    }

    state.cameraRunning = true;
    btn.innerHTML = '<span>⏹</span> Tắt Webcam';
    btn.className = 'btn btn-danger';
    document.getElementById('sourceBadge').innerText = 'WEBCAM';
    startTimer();
  } catch (err) {
    console.error('Error accessing webcam:', err);
    alert('Không thể mở Webcam. Hãy sử dụng nút "Chạy Giả Lập" hoặc tải file Video lên để test!');
  }
}

let isPoseProcessing = false;
let isHandsProcessing = false;
let handsSkipCounter = 0;

function processWebcamFrame() {
  if (!state.cameraRunning || videoElement.readyState < 2) return;

  // 1. High Priority Pose Tracking (runs at native camera speed without blocking)
  if (!isPoseProcessing && pose) {
    isPoseProcessing = true;
    pose.send({ image: videoElement })
      .catch(e => console.warn('Pose frame drop:', e))
      .finally(() => { isPoseProcessing = false; });
  }

  // 2. Decoupled Hand Tracking (only if enabled, throttled every 2nd frame)
  if (state.handsEnabled && hands && !isHandsProcessing) {
    handsSkipCounter++;
    if (handsSkipCounter % 2 === 0) {
      isHandsProcessing = true;
      hands.send({ image: videoElement })
        .catch(e => console.warn('Hands frame drop:', e))
        .finally(() => { isHandsProcessing = false; });
    }
  }
}

function runVideoPolling() {
  const poll = () => {
    if (!state.cameraRunning) return;
    processWebcamFrame();
    requestAnimationFrame(poll);
  };
  requestAnimationFrame(poll);
}

// ==========================================
// 8. SYNTHETIC MOTION SIMULATOR (OFFLINE ENGINE)
// ==========================================
function generateSyntheticLandmarks(action = 'squat', phase = 0) {
  const pts = [];
  for (let i = 0; i < JOINTS; i++) {
    pts.push({ x: 0.5, y: 0.5, z: 0, visibility: 1.0 });
  }

  // Torso baseline
  pts[LEFT_SHOULDER] = { x: 0.45, y: 0.32, z: 0, visibility: 1 };
  pts[RIGHT_SHOULDER] = { x: 0.55, y: 0.32, z: 0, visibility: 1 };
  pts[LEFT_HIP] = { x: 0.47, y: 0.52, z: 0, visibility: 1 };
  pts[RIGHT_HIP] = { x: 0.53, y: 0.52, z: 0, visibility: 1 };

  // Arms baseline
  pts[LEFT_ELBOW] = { x: 0.40, y: 0.42, z: 0, visibility: 1 };
  pts[RIGHT_ELBOW] = { x: 0.60, y: 0.42, z: 0, visibility: 1 };
  pts[LEFT_WRIST] = { x: 0.38, y: 0.52, z: 0, visibility: 1 };
  pts[RIGHT_WRIST] = { x: 0.62, y: 0.52, z: 0, visibility: 1 };

  // Legs baseline
  pts[LEFT_KNEE] = { x: 0.47, y: 0.70, z: 0, visibility: 1 };
  pts[RIGHT_KNEE] = { x: 0.53, y: 0.70, z: 0, visibility: 1 };
  pts[LEFT_ANKLE] = { x: 0.47, y: 0.88, z: 0, visibility: 1 };
  pts[RIGHT_ANKLE] = { x: 0.53, y: 0.88, z: 0, visibility: 1 };

  // Face
  pts[0] = { x: 0.5, y: 0.22, z: 0, visibility: 1 };

  // Smooth sinusoidal exercise pulse (0 at top, 1.0 at deepest compression)
  const pulse = (1 - Math.cos(phase)) * 0.5;

  if (action === 'squat') {
    // Symmetrical knee flexion down to 95°-105°
    pts[LEFT_HIP].y += pulse * 0.18;
    pts[RIGHT_HIP].y += pulse * 0.18;
    pts[LEFT_KNEE].x -= pulse * 0.07;
    pts[RIGHT_KNEE].x += pulse * 0.07;
    pts[LEFT_KNEE].y += pulse * 0.06;
    pts[RIGHT_KNEE].y += pulse * 0.06;
    // Arms raise forward for balance
    pts[LEFT_WRIST].y -= pulse * 0.15;
    pts[RIGHT_WRIST].y -= pulse * 0.15;
    pts[LEFT_ELBOW].y -= pulse * 0.08;
    pts[RIGHT_ELBOW].y -= pulse * 0.08;
  } else if (action === 'push_up') {
    // Horizontal body (pitch ~12°-16°)
    const chestDrop = pulse * 0.08;
    pts[LEFT_SHOULDER] = { x: 0.32, y: 0.62 + chestDrop, z: 0, visibility: 1 };
    pts[RIGHT_SHOULDER] = { x: 0.32, y: 0.62 + chestDrop, z: 0, visibility: 1 };
    pts[LEFT_HIP] = { x: 0.54, y: 0.63 + chestDrop * 0.5, z: 0, visibility: 1 };
    pts[RIGHT_HIP] = { x: 0.54, y: 0.63 + chestDrop * 0.5, z: 0, visibility: 1 };
    pts[LEFT_KNEE] = { x: 0.68, y: 0.64 + chestDrop * 0.25, z: 0, visibility: 1 };
    pts[RIGHT_KNEE] = { x: 0.68, y: 0.64 + chestDrop * 0.25, z: 0, visibility: 1 };
    pts[LEFT_ANKLE] = { x: 0.82, y: 0.65, z: 0, visibility: 1 };
    pts[RIGHT_ANKLE] = { x: 0.82, y: 0.65, z: 0, visibility: 1 };
    
    // Hands on ground
    pts[LEFT_WRIST] = { x: 0.32, y: 0.74, z: 0, visibility: 1 };
    pts[RIGHT_WRIST] = { x: 0.32, y: 0.74, z: 0, visibility: 1 };
    
    // Elbows flex from 175° (top) down to 75° (bottom)
    const elbowFlare = pulse * 0.07;
    pts[LEFT_ELBOW] = { x: 0.32 - elbowFlare, y: 0.68 + chestDrop * 0.5, z: 0, visibility: 1 };
    pts[RIGHT_ELBOW] = { x: 0.32 - elbowFlare, y: 0.68 + chestDrop * 0.5, z: 0, visibility: 1 };
  } else if (action === 'lunge') {
    // Asymmetrical split: Front leg (Left) bends deep (100°), Rear leg (Right) extends (170°)
    pts[LEFT_HIP].y += pulse * 0.14;
    pts[RIGHT_HIP].y += pulse * 0.14;
    // Front leg stepped forward
    pts[LEFT_KNEE] = { x: 0.36 - pulse * 0.02, y: 0.70 + pulse * 0.05, z: 0, visibility: 1 };
    pts[LEFT_ANKLE] = { x: 0.38, y: 0.88, z: 0, visibility: 1 };
    // Rear leg stretched back
    pts[RIGHT_KNEE] = { x: 0.58 + pulse * 0.04, y: 0.72 + pulse * 0.05, z: 0, visibility: 1 };
    pts[RIGHT_ANKLE] = { x: 0.68, y: 0.88, z: 0, visibility: 1 };
  } else {
    // Standing: natural breathing micro-motion
    pts[LEFT_SHOULDER].y += Math.sin(phase) * 0.003;
    pts[RIGHT_SHOULDER].y += Math.sin(phase) * 0.003;
    pts[LEFT_HIP].y += Math.sin(phase) * 0.002;
    pts[RIGHT_HIP].y += Math.sin(phase) * 0.002;
  }

  // Wrists finger nodes
  pts[LEFT_PINKY] = { x: pts[LEFT_WRIST].x - 0.02, y: pts[LEFT_WRIST].y + 0.035, z: 0, visibility: 1 };
  pts[LEFT_INDEX] = { x: pts[LEFT_WRIST].x + 0.005, y: pts[LEFT_WRIST].y + 0.045, z: 0, visibility: 1 };
  pts[LEFT_THUMB] = { x: pts[LEFT_WRIST].x + 0.025, y: pts[LEFT_WRIST].y + 0.020, z: 0, visibility: 1 };

  pts[RIGHT_PINKY] = { x: pts[RIGHT_WRIST].x + 0.02, y: pts[RIGHT_WRIST].y + 0.035, z: 0, visibility: 1 };
  pts[RIGHT_INDEX] = { x: pts[RIGHT_WRIST].x - 0.005, y: pts[RIGHT_WRIST].y + 0.045, z: 0, visibility: 1 };
  pts[RIGHT_THUMB] = { x: pts[RIGHT_WRIST].x - 0.025, y: pts[RIGHT_WRIST].y + 0.020, z: 0, visibility: 1 };

  // Feet
  pts[LEFT_HEEL] = { x: pts[LEFT_ANKLE].x - 0.015, y: pts[LEFT_ANKLE].y + 0.03, z: 0, visibility: 1 };
  pts[LEFT_FOOT_INDEX] = { x: pts[LEFT_ANKLE].x + 0.04, y: pts[LEFT_ANKLE].y + 0.035, z: 0, visibility: 1 };
  pts[RIGHT_HEEL] = { x: pts[RIGHT_ANKLE].x + 0.015, y: pts[RIGHT_ANKLE].y + 0.03, z: 0, visibility: 1 };
  pts[RIGHT_FOOT_INDEX] = { x: pts[RIGHT_ANKLE].x - 0.04, y: pts[RIGHT_ANKLE].y + 0.035, z: 0, visibility: 1 };

  // Subtle sensor noise
  for (const pt of pts) {
    pt.x += (Math.random() - 0.5) * 0.002;
    pt.y += (Math.random() - 0.5) * 0.002;
  }

  return pts;
}

/**
 * Generate 21-point hand landmarks for simulator to display all 5 fingers
 */
function generateSyntheticHand(wristPt, isLeft = true) {
  const hPts = [];
  const sign = isLeft ? -1 : 1;
  const wX = wristPt.x;
  const wY = wristPt.y;

  // 0: Wrist
  hPts.push({ x: wX, y: wY, z: 0 });

  // 1..4: Thumb
  for (let i = 1; i <= 4; i++) {
    hPts.push({ x: wX + sign * (0.01 + i * 0.006), y: wY + (i * 0.004), z: 0 });
  }
  // 5..8: Index
  for (let i = 1; i <= 4; i++) {
    hPts.push({ x: wX + sign * (0.004 + i * 0.002), y: wY + (0.01 + i * 0.009), z: 0 });
  }
  // 9..12: Middle
  for (let i = 1; i <= 4; i++) {
    hPts.push({ x: wX, y: wY + (0.012 + i * 0.010), z: 0 });
  }
  // 13..16: Ring
  for (let i = 1; i <= 4; i++) {
    hPts.push({ x: wX - sign * (0.004 + i * 0.002), y: wY + (0.01 + i * 0.009), z: 0 });
  }
  // 17..20: Pinky
  for (let i = 1; i <= 4; i++) {
    hPts.push({ x: wX - sign * (0.01 + i * 0.005), y: wY + (0.008 + i * 0.007), z: 0 });
  }

  return hPts;
}

function startSimulator(action = 'squat') {
  stopSimulator();
  if (state.cameraRunning) toggleWebcam();

  state.simRunning = true;
  state.simPhase = 0;
  document.getElementById('sourceBadge').innerText = 'SIMULATOR';
  document.getElementById('btnToggleSimMode').innerText = '⏹ Dừng Giả Lập';
  document.getElementById('btnToggleSimMode').className = 'btn btn-danger';
  startTimer();

  state.simInterval = setInterval(() => {
    state.simPhase += 0.08;
    const simulatedLandmarks = generateSyntheticLandmarks(action, state.simPhase);

    // Simulate 21 hand landmarks for both hands in simulator
    lastHandsResults = {
      multiHandLandmarks: [
        generateSyntheticHand(simulatedLandmarks[LEFT_WRIST], true),
        generateSyntheticHand(simulatedLandmarks[RIGHT_WRIST], false)
      ]
    };

    onPoseResults({ poseLandmarks: simulatedLandmarks });
  }, 33);
}

function stopSimulator() {
  if (state.simInterval) {
    clearInterval(state.simInterval);
    state.simInterval = null;
  }
  state.simRunning = false;
  lastHandsResults = null;
  const btn = document.getElementById('btnToggleSimMode');
  if (btn) {
    btn.innerHTML = '<span>⚡</span> Chạy Giả Lập (Không cần Cam)';
    btn.className = 'btn btn-cyan';
  }
}

// ==========================================
// 9. DATASET COLLECTION & LABELS LAB
// ==========================================
function startRecordingProcess() {
  const btnStart = document.getElementById('btnStartRecord');
  const btnStop = document.getElementById('btnStopRecord');
  const label = document.getElementById('collectLabelSelect').value;
  const subject = document.getElementById('collectSubjectInput').value.trim() || 'person01';
  const mode = document.querySelector('input[name="recMode"]:checked').value;

  state.recording.label = label;
  state.recording.subject = subject;
  state.recording.mode = mode;
  state.recording.buffer = [];

  let count = 3;
  document.getElementById('recStatusMsg').innerText = `Chuẩn bị... Bắt đầu trong ${count}s`;
  btnStart.disabled = true;

  const countdownInterval = setInterval(() => {
    count--;
    if (count > 0) {
      document.getElementById('recStatusMsg').innerText = `Chuẩn bị... Bắt đầu trong ${count}s`;
    } else {
      clearInterval(countdownInterval);
      state.recording.active = true;
      document.getElementById('recIndicator').classList.add('active');
      document.getElementById('recProgressBox').style.display = 'block';
      document.getElementById('recProgressBar').style.width = '0%';
      document.getElementById('recStatusMsg').innerText = `Đang ghi mẫu [${label.toUpperCase()}]...`;
      btnStop.disabled = false;

      if (!state.cameraRunning && !state.simRunning) {
        startSimulator(label);
      }
    }
  }, 1000);
}

function finishRecordingClip() {
  state.recording.active = false;
  document.getElementById('recIndicator').classList.remove('active');
  document.getElementById('btnStartRecord').disabled = false;
  document.getElementById('btnStopRecord').disabled = true;

  if (state.recording.buffer.length >= 8) {
    const resampled = resampleSequence(state.recording.buffer, CLIP_FRAMES);
    const sample = {
      id: `clip_${Date.now().toString(36)}`,
      label: state.recording.label,
      subject: state.recording.subject,
      frames: CLIP_FRAMES,
      timestamp: new Date().toLocaleTimeString(),
      landmarks: resampled
    };

    state.dataset.push(sample);
    updateDatasetTable();
    document.getElementById('recStatusMsg').innerText = `✅ Đã lưu mẫu #${sample.id} (${sample.label}) thành công!`;
  } else {
    document.getElementById('recStatusMsg').innerText = `❌ Mẫu quá ngắn (< 8 frames). Hãy thử lại.`;
  }
}

function updateDatasetTable() {
  const tbody = document.getElementById('datasetTableBody');
  const countBadge = document.getElementById('totalSamplesBadge');
  const badgesContainer = document.getElementById('classBadgesContainer');

  countBadge.innerText = `${state.dataset.length} Mẫu`;

  const counts = { squat: 0, push_up: 0, lunge: 0, standing: 0 };
  state.dataset.forEach(s => {
    counts[s.label] = (counts[s.label] || 0) + 1;
  });

  badgesContainer.innerHTML = LABELS.map(lbl => 
    `<span class="badge-tag green">${lbl}: ${counts[lbl] || 0}</span>`
  ).join('');

  if (state.dataset.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align: center; color: var(--text-dim); padding: 24px;">
          Chưa có mẫu nào được ghi. Hãy bấm ghi hoặc nạp bộ mẫu thử!
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = state.dataset.map((s, idx) => `
    <tr>
      <td>${idx + 1}</td>
      <td><span class="badge-tag cyan">${s.label}</span></td>
      <td>${s.subject}</td>
      <td>${s.frames}</td>
      <td>${s.timestamp}</td>
      <td>
        <button class="btn btn-danger" style="padding: 4px 10px; font-size: 0.75rem;" onclick="deleteSample('${s.id}')">Xóa</button>
      </td>
    </tr>
  `).join('');
}

window.deleteSample = function(id) {
  state.dataset = state.dataset.filter(s => s.id !== id);
  updateDatasetTable();
};

function exportDatasetJson() {
  if (state.dataset.length === 0) {
    alert('Bộ dữ liệu hiện trống. Hãy ghi ít nhất vài mẫu trước khi xuất!');
    return;
  }

  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.dataset, null, 2));
  const downloadAnchor = document.createElement('a');
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `visionfit_dataset_${Date.now()}.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
}

function importDatasetJson(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const imported = JSON.parse(e.target.result);
      if (Array.isArray(imported)) {
        state.dataset = imported;
        updateDatasetTable();
        alert(`Đã nạp thành công ${imported.length} mẫu vào dataset!`);
      } else {
        alert('File JSON không đúng cấu trúc mảng samples.');
      }
    } catch (err) {
      alert('Lỗi đọc file JSON: ' + err.message);
    }
  };
  reader.readAsText(file);
}

function generateFullDemoDataset(samplesPerClass = 30) {
  const dataset = [];
  const subjects = ['person01', 'person02', 'person03', 'person04', 'person05', 'person06'];

  for (const label of LABELS) {
    for (let i = 0; i < samplesPerClass; i++) {
      const subject = subjects[i % subjects.length];
      const frames = [];
      const phaseOffset = (i / samplesPerClass) * Math.PI * 0.5;

      for (let f = 0; f < CLIP_FRAMES; f++) {
        const phase = (f / (CLIP_FRAMES - 1)) * Math.PI * 2 + phaseOffset;
        frames.push(generateSyntheticLandmarks(label, phase));
      }

      dataset.push({
        id: `demo_${label}_${i}`,
        label,
        subject,
        frames: CLIP_FRAMES,
        timestamp: new Date().toLocaleTimeString(),
        landmarks: frames
      });
    }
  }

  state.dataset = dataset;
  updateDatasetTable();
  return dataset;
}

// ==========================================
// 10. IN-BROWSER AI TRAINING (TENSORFLOW.JS)
// ==========================================
let lossChart = null;
let accChart = null;

function initCharts() {
  const lossCtx = document.getElementById('lossChartCanvas').getContext('2d');
  const accCtx = document.getElementById('accChartCanvas').getContext('2d');

  lossChart = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Train Loss', data: [], borderColor: '#f59e0b', backgroundColor: 'rgba(245, 158, 11, 0.1)', tension: 0.3 },
        { label: 'Val Loss', data: [], borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', tension: 0.3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#94a3b8' } } },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });

  accChart = new Chart(accCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Train Acc', data: [], borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', tension: 0.3 },
        { label: 'Val Acc', data: [], borderColor: '#06b6d4', backgroundColor: 'rgba(6, 182, 212, 0.1)', tension: 0.3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#94a3b8' } } },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { ticks: { color: '#64748b' }, min: 0, max: 1, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

function prepareTrainingTensors() {
  if (state.dataset.length < 8) {
    throw new Error('Cần ít nhất 8 mẫu dữ liệu để huấn luyện!');
  }

  const X_arr = [];
  const y_arr = [];

  for (const sample of state.dataset) {
    const featureVec = extract142FeaturesFromSequence(sample.landmarks, true);
    X_arr.push(Array.from(featureVec));
    
    const labelIdx = LABELS.indexOf(sample.label);
    const oneHot = [0, 0, 0, 0];
    if (labelIdx !== -1) oneHot[labelIdx] = 1;
    y_arr.push(oneHot);
  }

  const xs = tf.tensor2d(X_arr, [X_arr.length, TOTAL_FEATURES]);
  const ys = tf.tensor2d(y_arr, [y_arr.length, 4]);

  return { xs, ys, count: X_arr.length };
}

async function startTfTraining() {
  const logBox = document.getElementById('trainLogBox');
  const btnTrain = document.getElementById('btnStartTraining');
  const epochs = parseInt(document.getElementById('trainEpochs').value) || 30;
  const batchSize = parseInt(document.getElementById('trainBatchSize').value) || 16;
  const lr = parseFloat(document.getElementById('trainLr').value) || 0.005;
  const valSplit = parseFloat(document.getElementById('trainValSplit').value) || 0.2;

  logBox.innerText = 'Đang chuẩn bị Tensor dữ liệu 142 đặc trưng...\n';
  btnTrain.disabled = true;

  try {
    const { xs, ys, count } = prepareTrainingTensors();
    logBox.innerText += `Đã nạp ${count} mẫu dữ liệu (Vector 142 đặc trưng). Khởi tạo Deep Neural Network...\n`;

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [TOTAL_FEATURES] }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

    model.compile({
      optimizer: tf.train.adam(lr),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.data.datasets[1].data = [];
    accChart.data.labels = [];
    accChart.data.datasets[0].data = [];
    accChart.data.datasets[1].data = [];
    lossChart.update();
    accChart.update();

    document.getElementById('trainProgressBox').style.display = 'block';
    document.getElementById('trainStatusBadge').innerText = 'Đang huấn luyện...';
    document.getElementById('trainStatusBadge').className = 'badge-tag cyan';

    await model.fit(xs, ys, {
      epochs,
      batchSize,
      validationSplit: valSplit,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const epNum = epoch + 1;
          const pct = Math.round((epNum / epochs) * 100);
          document.getElementById('trainProgressBar').style.width = `${pct}%`;
          document.getElementById('trainEpochMsg').innerText = 
            `Epoch ${epNum}/${epochs} | Loss: ${logs.loss.toFixed(4)} | Acc: ${(logs.acc * 100).toFixed(1)}% | Val Acc: ${(logs.val_acc ? logs.val_acc * 100 : 0).toFixed(1)}%`;

          logBox.innerText += `Epoch ${String(epNum).padStart(2, '0')}/${epochs} -> loss: ${logs.loss.toFixed(4)}, acc: ${(logs.acc * 100).toFixed(1)}%, val_loss: ${(logs.val_loss || 0).toFixed(4)}, val_acc: ${((logs.val_acc || 0) * 100).toFixed(1)}%\n`;
          logBox.scrollTop = logBox.scrollHeight;

          lossChart.data.labels.push(`E${epNum}`);
          lossChart.data.datasets[0].data.push(logs.loss);
          lossChart.data.datasets[1].data.push(logs.val_loss || logs.loss);

          accChart.data.labels.push(`E${epNum}`);
          accChart.data.datasets[0].data.push(logs.acc);
          accChart.data.datasets[1].data.push(logs.val_acc || logs.acc);

          lossChart.update();
          accChart.update();
          await tf.nextFrame();
        }
      }
    });

    state.tfModel = model;
    document.getElementById('trainStatusBadge').innerText = 'Huấn luyện Xong!';
    document.getElementById('trainStatusBadge').className = 'badge-tag green';
    document.getElementById('btnApplyModelToLive').disabled = false;
    document.getElementById('btnDownloadModel').disabled = false;
    
    const optTfjs = document.getElementById('optTfjs');
    if (optTfjs) optTfjs.disabled = false;
    
    // Auto-apply model
    state.activeModel = 'tfjs';
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) modelSelect.value = 'tfjs';
    const aiStatusText = document.getElementById('aiStatusText');
    if (aiStatusText) aiStatusText.innerText = 'AI Model: TensorFlow.js 142-Features (Đã Huấn Luyện ⚡)';
    const aiDot = document.getElementById('aiDot');
    if (aiDot) aiDot.className = 'status-dot';

    logBox.innerText += `\n🎉 Huấn luyện hoàn tất thành công 100%! Model 142 đặc trưng đã được kích hoạt trực tiếp vào Live Test.`;

    xs.dispose();
    ys.dispose();
  } catch (err) {
    console.error(err);
    logBox.innerText += `\n❌ Lỗi: ${err.message}`;
    document.getElementById('trainStatusBadge').innerText = 'Lỗi';
    document.getElementById('trainStatusBadge').className = 'badge-tag red';
  } finally {
    btnTrain.disabled = false;
  }
}

function predictWithTfjs(frames) {
  if (!state.tfModel || frames.length < 1) {
    return analyzePoseHeuristic(frames[frames.length - 1]);
  }

  const lastLandmarks = frames[frames.length - 1];
  const heuristic = analyzePoseHeuristic(lastLandmarks);

  try {
    // 1. Instantaneous 142-feature extraction with recency weighting on recent frames
    const featureVec = extract142FeaturesFromSequence(frames, false);
    const inputTensor = tf.tensor2d([Array.from(featureVec)], [1, TOTAL_FEATURES]);
    const pred = state.tfModel.predict(inputTensor);
    const rawProbs = pred.dataSync();
    inputTensor.dispose();
    pred.dispose();

    // 2. Multi-factor Biomechanical Fusion with neural probabilities
    let probsObj = {
      squat: rawProbs[0],
      push_up: rawProbs[1],
      lunge: rawProbs[2],
      standing: rawProbs[3]
    };

    const angles = heuristic ? heuristic.angles : {};
    const pitch = angles.pitch !== undefined ? angles.pitch : 90;
    const avgKnee = angles.avgKnee !== undefined ? angles.avgKnee : 180;
    const minKnee = angles.minKnee !== undefined ? angles.minKnee : 180;
    const kneeDiff = angles.kneeDiff !== undefined ? angles.kneeDiff : 0;

    // Prone body check (Horizontal pushup position)
    if (pitch < 42) {
      probsObj.push_up = Math.max(probsObj.push_up, 0.96);
      probsObj.standing = 0.01;
      probsObj.squat = 0.02;
      probsObj.lunge = 0.01;
    } else if (pitch >= 55) {
      // Upright body: suppress pushup completely
      probsObj.push_up = Math.min(probsObj.push_up, 0.01);

      // Asymmetrical knee flexion -> Lunge
      if (kneeDiff >= 24 && minKnee <= 138) {
        probsObj.lunge = Math.max(probsObj.lunge, 0.95);
        probsObj.squat = Math.min(probsObj.squat, 0.03);
        probsObj.standing = Math.min(probsObj.standing, 0.02);
      }
      // Symmetrical knee flexion -> Squat
      else if (avgKnee <= 148 && kneeDiff < 22) {
        const depthFactor = Math.min(0.98, 0.72 + (148 - avgKnee) / 75.0);
        probsObj.squat = Math.max(probsObj.squat, depthFactor);
        probsObj.standing = Math.max(0.01, 1.0 - probsObj.squat);
        probsObj.lunge = Math.min(probsObj.lunge, 0.03);
      }
      // Extended knees -> Standing
      else if (avgKnee >= 152 && minKnee >= 145) {
        probsObj.standing = Math.max(probsObj.standing, 0.96);
        probsObj.squat = Math.min(probsObj.squat, 0.02);
        probsObj.lunge = Math.min(probsObj.lunge, 0.01);
      }
    }

    // Responsive EMA filter (40% new frame, 60% historical memory for buttery smoothness)
    const EMA_ALPHA = 0.40;
    const smoothedProbs = {};
    for (const lbl of LABELS) {
      const prev = state.probabilities[lbl] !== undefined ? state.probabilities[lbl] : probsObj[lbl];
      smoothedProbs[lbl] = EMA_ALPHA * probsObj[lbl] + (1 - EMA_ALPHA) * prev;
    }
    probsObj = smoothedProbs;

    // 3. User Target Override (if user locked a specific exercise)
    if (state.targetExercise && state.targetExercise !== 'auto') {
      const idx = LABELS.indexOf(state.targetExercise);
      return {
        action: state.targetExercise,
        confidence: idx !== -1 ? probsObj[state.targetExercise] : 0.96,
        probabilities: probsObj,
        primaryAngle: heuristic ? heuristic.primaryAngle : 180,
        angles: heuristic ? heuristic.angles : {}
      };
    }

    // 4. Determine winning action with highest confidence
    let detectedAction = 'standing';
    let maxVal = -1;
    for (const lbl of LABELS) {
      if (probsObj[lbl] > maxVal) {
        maxVal = probsObj[lbl];
        detectedAction = lbl;
      }
    }

    return {
      action: detectedAction,
      confidence: maxVal,
      probabilities: probsObj,
      primaryAngle: heuristic ? heuristic.primaryAngle : 180,
      angles: heuristic ? heuristic.angles : {}
    };
  } catch (err) {
    console.warn('TF.js predict error, falling back to heuristic:', err);
    return heuristic;
  }
}

function applyModelToLive() {
  state.activeModel = 'tfjs';
  document.getElementById('modelSelect').value = 'tfjs';
  document.getElementById('aiStatusText').innerText = 'AI Model: TensorFlow.js Trained';
  document.getElementById('aiDot').className = 'status-dot';
  
  switchTab('liveTab');
  alert('Đã kích hoạt mô hình TensorFlow.js vào chế độ Live Test!');
}

async function downloadTfModel() {
  if (!state.tfModel) return;
  await state.tfModel.save('downloads://visionfit-tfjs-model');
}

// ==========================================
// 11. TAB NAVIGATION & INITIALIZATION
// ==========================================
function switchTab(targetTabId) {
  state.activeTab = targetTabId;

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === targetTabId);
  });

  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === targetTabId);
  });
}

function setupEventListeners() {
  // Tabs
  document.querySelectorAll('.nav-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Live Test Controls
  document.getElementById('btnToggleCam').addEventListener('click', toggleWebcam);
  document.getElementById('btnToggleSimMode').addEventListener('click', () => {
    if (state.simRunning) {
      stopSimulator();
    } else {
      startSimulator('squat');
    }
  });

  // Mirror toggle button
  const btnToggleMirror = document.getElementById('btnToggleMirror');
  if (btnToggleMirror) {
    btnToggleMirror.addEventListener('click', () => toggleMirror());
  }

  document.getElementById('btnResetReps').addEventListener('click', () => {
    state.totalReps = 0;
    for (const k in state.counters) {
      state.counters[k].reps = 0;
      state.counters[k].state = 'up';
    }
    document.getElementById('statReps').innerText = '0';
    document.getElementById('hudRepCount').innerText = '0';
  });

  document.getElementById('btnToggleAudio').addEventListener('click', () => {
    state.audioEnabled = !state.audioEnabled;
    const icon = document.getElementById('audioIcon');
    const btn = document.getElementById('btnToggleAudio');
    if (state.audioEnabled) {
      icon.innerText = '🔊';
      btn.innerHTML = `<span>🔊</span> Âm thanh: BẬT`;
    } else {
      icon.innerText = '🔇';
      btn.innerHTML = `<span>🔇</span> Âm thanh: TẮT`;
    }
  });

  document.getElementById('modelSelect').addEventListener('change', (e) => {
    state.activeModel = e.target.value;
    document.getElementById('aiStatusText').innerText = 
      e.target.value === 'tfjs' ? 'AI Model: TensorFlow.js' : 'AI Model: Biomechanical Rules';
  });

  // Video file upload - Automatically turns off mirror for uploaded videos
  document.getElementById('videoFileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    stopSimulator();
    if (state.cameraRunning) toggleWebcam();

    // Turn off mirror for uploaded video files so it displays in original orientation
    toggleMirror(false);

    const url = URL.createObjectURL(file);
    videoElement.srcObject = null;
    videoElement.src = url;
    videoElement.loop = true;
    videoElement.play();
    state.cameraRunning = true;
    document.getElementById('sourceBadge').innerText = 'FILE VIDEO';
    runVideoPolling();
    startTimer();
  });

  // Data Collection
  document.getElementById('btnStartRecord').addEventListener('click', startRecordingProcess);
  document.getElementById('btnStopRecord').addEventListener('click', finishRecordingClip);
  document.getElementById('btnExportJson').addEventListener('click', exportDatasetJson);
  document.getElementById('importJsonInput').addEventListener('change', importDatasetJson);
  document.getElementById('btnClearDataset').addEventListener('click', () => {
    if (confirm('Bạn có chắc muốn xóa toàn bộ dataset hiện tại?')) {
      state.dataset = [];
      updateDatasetTable();
    }
  });

  // Exercise target mode select
  const exerciseTargetSelect = document.getElementById('exerciseTargetSelect');
  if (exerciseTargetSelect) {
    exerciseTargetSelect.addEventListener('change', (e) => {
      state.targetExercise = e.target.value;
      if (e.target.value !== 'auto') {
        document.getElementById('hudActionName').innerText = e.target.value.replace('_', ' ').toUpperCase();
      }
    });
  }

  // Toggle Hands tracking for 60 FPS
  const btnToggleHands = document.getElementById('btnToggleHands');
  if (btnToggleHands) {
    btnToggleHands.addEventListener('click', () => {
      state.handsEnabled = !state.handsEnabled;
      const text = document.getElementById('handsText');
      if (state.handsEnabled) {
        text.innerText = 'Bàn tay: BẬT (21 khớp)';
        btnToggleHands.style.borderColor = 'var(--accent-cyan)';
      } else {
        text.innerText = 'Bàn tay: TẮT (60 FPS)';
        btnToggleHands.style.borderColor = '';
        lastHandsResults = null;
      }
    });
  }

  // Training
  document.getElementById('btnStartTraining').addEventListener('click', startTfTraining);
  document.getElementById('btnLoadDemoDataset').addEventListener('click', () => {
    loadAnalysisDatasetAndTrain(true);
  });
  document.getElementById('btnApplyModelToLive').addEventListener('click', applyModelToLive);
  document.getElementById('btnDownloadModel').addEventListener('click', downloadTfModel);

  // Simulator Tab
  document.getElementById('btnRunSimDemo').addEventListener('click', () => {
    const exercise = document.getElementById('simExerciseSelect').value;
    startSimulator(exercise);
    switchTab('liveTab');
  });

  document.getElementById('btnGenDatasetBtn').addEventListener('click', () => {
    generateFullDemoDataset(25);
    alert('Đã sinh 100 mẫu synthetic và lưu vào Dataset Manager!');
    switchTab('collectTab');
  });

  // Bridge Save to Backend
  document.getElementById('btnSaveToBackend').addEventListener('click', async () => {
    if (state.dataset.length === 0) {
      alert('Không có dữ liệu để chuyển sang backend!');
      return;
    }
    try {
      const resp = await fetch('/api/save_npz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ samples: state.dataset })
      });
      if (resp.ok) {
        const res = await resp.json();
        alert(`✅ Backend đã nhận và lưu thành file .NPZ: ${res.file}`);
      } else {
        alert('Server Python chưa được bật (hãy chạy python web_server.py). Bạn vẫn có thể xuất file .JSON để lưu trữ!');
      }
    } catch (e) {
      alert('Chưa kết nối được với web_server.py. Vui lòng chạy lệnh: python web_server.py');
    }
  });
}

async function checkBackendStatus() {
  try {
    const resp = await fetch('/api/status');
    if (resp.ok) {
      const info = await resp.json();
      document.getElementById('serverStatusText').innerText = `Backend: Sẵn sàng (${info.models_found} models)`;
      document.getElementById('serverDot').className = 'status-dot';
      document.getElementById('backendBadge').innerText = 'Kết nối thành công (Python Backend)';
      document.getElementById('backendBadge').className = 'badge-tag green';
    }
  } catch (err) {
    document.getElementById('serverStatusText').innerText = 'Backend: Standalone (Trình duyệt)';
    document.getElementById('serverDot').className = 'status-dot amber';
    document.getElementById('backendBadge').innerText = 'Chế độ Standalone (Trình duyệt)';
    document.getElementById('backendBadge').className = 'badge-tag cyan';
  }
}

/**
 * Load 120 analyzed exercise samples and auto-train TensorFlow.js model
 */
async function loadAnalysisDatasetAndTrain(forceRetrain = false) {
  const logBox = document.getElementById('trainLogBox');
  if (logBox) logBox.innerText = '⚡ Đang nạp 120 mẫu dữ liệu phân tích động tác gym (demo.npz)...\n';

  let loaded = false;
  try {
    const resp = await fetch('dataset_demo.json');
    if (resp.ok) {
      const data = await resp.json();
      if (Array.isArray(data) && data.length > 0) {
        state.dataset = data;
        updateDatasetTable();
        loaded = true;
        if (logBox) logBox.innerText += `✅ Đã nạp thành công ${data.length} mẫu phân tích (squat: 30, push_up: 30, lunge: 30, standing: 30)!\n`;
      }
    }
  } catch (e) {
    console.warn('Could not fetch dataset_demo.json, fallback to synthetic:', e);
  }

  if (!loaded) {
    state.dataset = generateFullDemoDataset(30);
    updateDatasetTable();
    if (logBox) logBox.innerText += `⚡ Đã sinh 120 mẫu phân tích chuẩn động thái học!\n`;
  }

  // Auto-train the neural network model
  if (logBox) logBox.innerText += '🚀 Đang tự động huấn luyện TensorFlow.js Neural Network trên 120 mẫu động tác...\n';
  await startTfTraining();
}

// App Entrypoint
window.addEventListener('DOMContentLoaded', async () => {
  initVisionEngines();
  initCharts();
  setupEventListeners();
  checkBackendStatus();

  // Automatically load 120 analyzed exercise samples and train AI model
  await loadAnalysisDatasetAndTrain();
});
