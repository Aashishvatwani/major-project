/*
  AERO-GUARD // Structured Mission Control Dashboard & 3D Spacecraft Digital Twin v3.5
  Features:
  1. Three.js PBR 3D Spacecraft inside dedicated Digital Twin Card container with attitude stabilization.
  2. Dual Real-time Chart.js telemetry charts (Voltage/Current and Temp/P(Anomaly)).
  3. Live AI Agent Flight Diagnostic Terminal with RAG Flight Regulations.
  4. Real-time Digital Twin Counterfactual Decision Matrix (60s forward projection).
  5. Multi-Model ML Ensemble Consensus Breakdown.
  6. Hardware HITL Arduino Uno Pin 13 LED Fixture Actuator.
  7. Authentic Web Audio API Synthesizer (beeps, arc discharge, venting hiss, klaxons, recovery chimes).
  8. Autonomous Recovery Transition Detection & Visual Frontend Reset with RL Mitigation Action Attribution.
*/

// ==============================================================================
// 1. Web Audio API Synthesizer
// ==============================================================================
class AerospaceAudioEngine {
  constructor() {
    this.ctx = null;
    this.muted = false;
    this.alarmInterval = null;
    this.initAudioContext();
  }

  initAudioContext() {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        this.ctx = new AudioContext();
      }
    } catch (e) {
      console.warn("Web Audio API not supported", e);
    }
  }

  resumeContext() {
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  playBeep(freq = 880, duration = 0.08, type = 'sine') {
    if (this.muted || !this.ctx) return;
    this.resumeContext();
    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch (e) {}
  }

  playRecoveryChime() {
    if (this.muted || !this.ctx) return;
    this.resumeContext();
    try {
      // Pleasant C5 - E5 - G5 - C6 aerospace resolution chime
      const notes = [523.25, 659.25, 783.99, 1046.50];
      notes.forEach((freq, i) => {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        const startT = this.ctx.currentTime + i * 0.07;
        const dur = 0.22;
        osc.frequency.setValueAtTime(freq, startT);
        gain.gain.setValueAtTime(0.06, startT);
        gain.gain.exponentialRampToValueAtTime(0.0001, startT + dur);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        osc.start(startT);
        osc.stop(startT + dur);
      });
    } catch (e) {}
  }

  playArcDischarge() {
    if (this.muted || !this.ctx) return;
    this.resumeContext();
    try {
      const bufferSize = this.ctx.sampleRate * 0.12;
      const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noise = this.ctx.createBufferSource();
      noise.buffer = buffer;

      const filter = this.ctx.createBiquadFilter();
      filter.type = 'bandpass';
      filter.frequency.value = 3200;
      filter.Q.value = 3.0;

      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.18, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.12);

      noise.connect(filter);
      filter.connect(gain);
      gain.connect(this.ctx.destination);
      noise.start();
    } catch (e) {}
  }

  playVentingHiss() {
    if (this.muted || !this.ctx) return;
    this.resumeContext();
    try {
      const bufferSize = this.ctx.sampleRate * 0.35;
      const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * 0.5;
      }
      const noise = this.ctx.createBufferSource();
      noise.buffer = buffer;
      const filter = this.ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(1400, this.ctx.currentTime);
      filter.frequency.linearRampToValueAtTime(300, this.ctx.currentTime + 0.35);

      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.001, this.ctx.currentTime + 0.35);

      noise.connect(filter);
      filter.connect(gain);
      gain.connect(this.ctx.destination);
      noise.start();
    } catch (e) {}
  }

  startCriticalAlarm() {
    if (this.alarmInterval || this.muted) return;
    this.alarmInterval = setInterval(() => {
      if (this.muted) return;
      this.playBeep(980, 0.12, 'sawtooth');
      setTimeout(() => {
        if (!this.muted) this.playBeep(650, 0.14, 'triangle');
      }, 140);
    }, 900);
  }

  stopCriticalAlarm() {
    if (this.alarmInterval) {
      clearInterval(this.alarmInterval);
      this.alarmInterval = null;
    }
  }

  toggleMute() {
    this.muted = !this.muted;
    if (this.muted) this.stopCriticalAlarm();
    return this.muted;
  }
}

const audioSys = new AerospaceAudioEngine();

// Safe DOM Helper Functions
function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}
function safeSetWidth(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${Math.min(Math.max(pct, 0), 100)}%`;
}
function safeSetColor(id, color) {
  const el = document.getElementById(id);
  if (el) el.style.color = color;
}

function addTerminalLog(msg, isError = false, isWarn = false, isSuccess = false) {
  const ticker = document.getElementById('terminal-ticker');
  if (!ticker) return;
  const now = new Date().toISOString().substring(11, 19);
  let color = 'text-gray-300';
  if (isError) color = 'text-red-400 font-bold';
  else if (isWarn) color = 'text-amber-300 font-semibold';
  else if (isSuccess) color = 'text-emerald-300 font-bold';
  
  ticker.innerHTML = `<span class="${color}">[${now}] ${msg}</span>`;
  if (isSuccess) {
    audioSys.playBeep(1046, 0.08, 'sine');
  } else {
    audioSys.playBeep(isError ? 440 : 1200, 0.05, 'square');
  }
}

// Global Banner Functions
function showRecoveryBanner(actionName, faultName, timeStr, customDetail = null) {
  const panel = document.getElementById('recovery-status-panel');
  const actionEl = document.getElementById('recovery-action-applied');
  const timeEl = document.getElementById('recovery-timestamp');
  const detailEl = document.getElementById('recovery-mitigation-detail');
  
  if (!panel) return;

  let detailText = "• Electrochemical & thermal equilibrium restored. Nominal ESA/NASA limits verified.";
  if (actionName.includes("LOAD_SHEDDING")) {
    detailText = "• Shed 35% non-critical scientific payloads -> Heat dissipation and cell voltage stabilized.";
  } else if (actionName.includes("SAFE_MODE")) {
    detailText = "• Emergency power bus isolated -> Battery degradation halted and thermal runaway mitigated.";
  } else if (actionName.includes("PREARM")) {
    detailText = "• High sensitivity pre-arm completed -> Sensor drift isolated and telemetry re-synchronized.";
  }

  if (actionEl) actionEl.innerText = `${actionName} (Mitigated: ${faultName})`;
  if (timeEl) timeEl.innerText = `RECOVERED AT: ${timeStr || new Date().toLocaleTimeString()}`;
  if (detailEl) detailEl.innerText = customDetail || detailText;

  panel.classList.remove('hidden');

  // Clear existing timeout
  if (SatState.recoveryBannerTimeout) {
    clearTimeout(SatState.recoveryBannerTimeout);
  }
  
  // Keep visible for 18 seconds then smoothly fade out if still nominal
  SatState.recoveryBannerTimeout = setTimeout(() => {
    if (SatState.final_severity === 'NOMINAL' && SatState.primary_fault === 'NOMINAL') {
      dismissRecoveryBanner();
    }
  }, 18000);
}

function dismissRecoveryBanner() {
  const panel = document.getElementById('recovery-status-panel');
  if (panel) panel.classList.add('hidden');
}

// ==============================================================================
// 2. Physical State Variables
// ==============================================================================
const SatState = {
  voltage: 3.72,
  current: 2.45,
  temp: 22.4,
  dtdt: 0.02,
  esr: 0.045,
  soc: 0.85,
  power: 9.11,
  
  p_ensemble: 0.012,
  p_rf: 0.01,
  p_xgboost: 0.01,
  p_extra_trees: 0.01,
  primary_fault: "NOMINAL",
  final_severity: "NOMINAL",
  risk_score: 0.05,
  rl_action_name: "NOMINAL_MONITOR",
  rl_action_id: 0,
  dynamic_threshold: 0.70,
  safety_override_active: false,
  requires_human_approval: false,
  autopilotMode: true,

  // Recovery & State Transition Tracker
  wasInAnomaly: false,
  lastMitigatingRLAction: "NOMINAL_MONITOR",
  lastMitigatedFault: "NOMINAL",
  lastRecoveryTimestamp: "STANDBY",
  recoveryBannerTimeout: null,

  // Client-side visual overrides
  faultThermalRunaway: false,
  faultMicroShort: false,
  faultDeepUndervolt: false,
  faultImpedanceSurge: false,
  faultSensorGlitch: false,

  runawayTime: 0,
  glitchPhase: 0,
  viewMode: 'orbit'
};

// ==============================================================================
// 3. Three.js 3D PBR Spacecraft Engine (Inside Card Container)
// ==============================================================================
let scene, camera, renderer, controls;
let satelliteGroup, starfieldMesh;
let batteryPackModule, solarWingLeft, solarWingRight, highGainDish;
let particleSystemThermal, arcLineMesh;
let spotSunLight, earthAlbedoLight, thermalPointLight, arcFlashLight;
let cellThermalMaterial;

function createNoiseTexture(width = 256, height = 256, intensity = 0.5) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(width, height);
  for (let i = 0; i < imgData.data.length; i += 4) {
    const val = Math.floor(Math.random() * 255 * intensity + 128 * (1 - intensity));
    imgData.data[i] = val;
    imgData.data[i + 1] = val;
    imgData.data[i + 2] = val;
    imgData.data[i + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function createSolarPanelTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#061026';
  ctx.fillRect(0, 0, 512, 1024);

  const cols = 4;
  const rows = 12;
  const cellW = 512 / cols;
  const cellH = 1024 / rows;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = c * cellW + 2;
      const y = r * cellH + 2;
      const w = cellW - 4;
      const h = cellH - 4;

      const grad = ctx.createLinearGradient(x, y, x + w, y + h);
      grad.addColorStop(0, '#0c2759');
      grad.addColorStop(0.5, '#0a1a3b');
      grad.addColorStop(1, '#07122b');
      ctx.fillStyle = grad;
      ctx.fillRect(x, y, w, h);

      ctx.strokeStyle = 'rgba(180, 220, 255, 0.28)';
      ctx.lineWidth = 1;
      for (let line = 1; line < 6; line++) {
        ctx.beginPath();
        ctx.moveTo(x + (w / 6) * line, y);
        ctx.lineTo(x + (w / 6) * line, y + h);
        ctx.stroke();
      }

      ctx.fillStyle = '#030814';
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 8, y);
      ctx.lineTo(x, y + 8);
      ctx.fill();
    }
  }

  return new THREE.CanvasTexture(canvas);
}

function initSceneEnvironment() {
  const container = document.getElementById('canvas-container');
  if (!container) return;

  const w = container.clientWidth || 380;
  const h = container.clientHeight || 260;

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x000308, 0.0005);

  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 40000);
  camera.position.set(16, 9, 22);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxDistance = 120;
  controls.minDistance = 5;

  const ambient = new THREE.AmbientLight(0x0a1428, 0.6);
  scene.add(ambient);

  spotSunLight = new THREE.DirectionalLight(0xfff8ee, 3.2);
  spotSunLight.position.set(60, 40, 50);
  spotSunLight.castShadow = true;
  scene.add(spotSunLight);

  earthAlbedoLight = new THREE.DirectionalLight(0x286ca0, 1.2);
  earthAlbedoLight.position.set(-40, -50, -30);
  scene.add(earthAlbedoLight);

  thermalPointLight = new THREE.PointLight(0xff3300, 0, 18);
  thermalPointLight.position.set(0, 0, 0);
  scene.add(thermalPointLight);

  arcFlashLight = new THREE.PointLight(0x00f0ff, 0, 15);
  arcFlashLight.position.set(0.6, 0.4, 0.8);
  scene.add(arcFlashLight);

  // Starfield
  const starGeo = new THREE.BufferGeometry();
  const starCount = 2000;
  const starPositions = new Float32Array(starCount * 3);
  const starColors = new Float32Array(starCount * 3);

  for (let i = 0; i < starCount * 3; i += 3) {
    const r = 1500 + Math.random() * 3000;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(Math.random() * 2 - 1);
    starPositions[i] = r * Math.sin(phi) * Math.cos(theta);
    starPositions[i + 1] = r * Math.sin(phi) * Math.sin(theta);
    starPositions[i + 2] = r * Math.cos(phi);

    const cVal = 0.7 + Math.random() * 0.3;
    starColors[i] = cVal;
    starColors[i + 1] = cVal * 0.9;
    starColors[i + 2] = cVal;
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
  starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

  const starMat = new THREE.PointsMaterial({
    size: 2.0,
    vertexColors: true,
    transparent: true,
    opacity: 0.85
  });
  starfieldMesh = new THREE.Points(starGeo, starMat);
  scene.add(starfieldMesh);
}

function buildDetailedSatellite() {
  satelliteGroup = new THREE.Group();

  const noiseTex = createNoiseTexture(256, 256, 0.6);
  const solarTex = createSolarPanelTexture();

  const goldMliMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xf5b700,
    emissive: 0x241700,
    roughness: 0.38,
    metalness: 0.95,
    clearcoat: 0.3,
    clearcoatRoughness: 0.25,
    bumpMap: noiseTex,
    bumpScale: 0.035
  });

  const carbonMaterial = new THREE.MeshStandardMaterial({
    color: 0x15171e,
    roughness: 0.7,
    metalness: 0.4,
    bumpMap: noiseTex,
    bumpScale: 0.015
  });

  const titaniumMaterial = new THREE.MeshStandardMaterial({
    color: 0xa8b2c2,
    roughness: 0.25,
    metalness: 0.85
  });

  // Main Bus
  const busCoreGeo = new THREE.CylinderGeometry(2.4, 2.6, 5.2, 8);
  const busCore = new THREE.Mesh(busCoreGeo, goldMliMaterial);
  busCore.castShadow = true;
  busCore.receiveShadow = true;
  satelliteGroup.add(busCore);

  // Louver Radiators
  const radLouverGeo = new THREE.BoxGeometry(0.15, 3.8, 2.2);
  const radLouverL = new THREE.Mesh(radLouverGeo, titaniumMaterial);
  radLouverL.position.set(2.4, 0, 0);
  satelliteGroup.add(radLouverL);

  const radLouverR = new THREE.Mesh(radLouverGeo, titaniumMaterial);
  radLouverR.position.set(-2.4, 0, 0);
  satelliteGroup.add(radLouverR);

  // Propulsion Deck
  const propDeckGeo = new THREE.CylinderGeometry(2.1, 2.3, 0.4, 8);
  const propDeck = new THREE.Mesh(propDeckGeo, carbonMaterial);
  propDeck.position.set(0, -2.8, 0);
  satelliteGroup.add(propDeck);

  // RCS Thrusters
  for (let i = 0; i < 4; i++) {
    const angle = (i * Math.PI) / 2;
    const nozzleGeo = new THREE.ConeGeometry(0.22, 0.5, 12, 1, true);
    const nozzle = new THREE.Mesh(nozzleGeo, titaniumMaterial);
    nozzle.rotation.x = Math.PI;
    nozzle.position.set(Math.cos(angle) * 1.8, -3.1, Math.sin(angle) * 1.8);
    satelliteGroup.add(nozzle);
  }

  // Star Trackers
  for (let i = 0; i < 2; i++) {
    const baffleGeo = new THREE.CylinderGeometry(0.18, 0.14, 0.6, 16);
    const baffle = new THREE.Mesh(baffleGeo, carbonMaterial);
    baffle.position.set(1.5, 2.2 + (i * 0.4), 1.6);
    baffle.rotation.x = 0.5;
    baffle.rotation.z = -0.3;
    satelliteGroup.add(baffle);
  }

  // Battery Pack Assembly
  batteryPackModule = new THREE.Group();
  batteryPackModule.position.set(0, 0.3, 1.95);

  const bmaCaseGeo = new THREE.BoxGeometry(2.0, 1.8, 1.0);
  const bmaCaseMat = new THREE.MeshStandardMaterial({
    color: 0x222938,
    roughness: 0.4,
    metalness: 0.7
  });
  const bmaChassis = new THREE.Mesh(bmaCaseGeo, bmaCaseMat);
  bmaChassis.castShadow = true;
  batteryPackModule.add(bmaChassis);

  const cellGroup = new THREE.Group();
  const cellGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.65, 12);
  
  cellThermalMaterial = new THREE.MeshStandardMaterial({
    color: 0x3b82f6,
    emissive: 0x001133,
    roughness: 0.3,
    metalness: 0.8
  });

  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 8; col++) {
      const cell = new THREE.Mesh(cellGeo, cellThermalMaterial);
      cell.position.set(-0.75 + col * 0.22, -0.45 + row * 0.3, 0.3);
      cell.rotation.x = Math.PI / 2;
      cellGroup.add(cell);
    }
  }
  batteryPackModule.add(cellGroup);

  const busbarGeo = new THREE.BoxGeometry(1.6, 0.06, 0.04);
  for (let b = 0; b < 4; b++) {
    const busbar = new THREE.Mesh(busbarGeo, titaniumMaterial);
    busbar.position.set(0, -0.45 + b * 0.3, 0.5);
    batteryPackModule.add(busbar);
  }
  satelliteGroup.add(batteryPackModule);

  // Solar Wings
  const arrayWidth = 9.5;
  const arrayHeight = 2.4;
  const solarMat = new THREE.MeshStandardMaterial({
    map: solarTex,
    roughness: 0.2,
    metalness: 0.65,
    bumpMap: noiseTex,
    bumpScale: 0.01
  });

  // Left Wing
  solarWingLeft = new THREE.Group();
  solarWingLeft.position.set(-2.6, 0, 0);
  const boomLeftGeo = new THREE.CylinderGeometry(0.08, 0.08, 2.0, 12);
  const boomL = new THREE.Mesh(boomLeftGeo, titaniumMaterial);
  boomL.rotation.z = Math.PI / 2;
  boomL.position.set(-1.0, 0, 0);
  solarWingLeft.add(boomL);

  const panelLeftGeo = new THREE.BoxGeometry(arrayWidth, arrayHeight, 0.1);
  const panelL = new THREE.Mesh(panelLeftGeo, solarMat);
  panelL.position.set(-1.0 - (arrayWidth / 2), 0, 0);
  panelL.castShadow = true;
  panelL.receiveShadow = true;
  solarWingLeft.add(panelL);
  satelliteGroup.add(solarWingLeft);

  // Right Wing
  solarWingRight = new THREE.Group();
  solarWingRight.position.set(2.6, 0, 0);
  const boomRightGeo = new THREE.CylinderGeometry(0.08, 0.08, 2.0, 12);
  const boomR = new THREE.Mesh(boomRightGeo, titaniumMaterial);
  boomR.rotation.z = -Math.PI / 2;
  boomR.position.set(1.0, 0, 0);
  solarWingRight.add(boomR);

  const panelRightGeo = new THREE.BoxGeometry(arrayWidth, arrayHeight, 0.1);
  const panelR = new THREE.Mesh(panelRightGeo, solarMat);
  panelR.position.set(1.0 + (arrayWidth / 2), 0, 0);
  panelR.castShadow = true;
  panelR.receiveShadow = true;
  solarWingRight.add(panelR);
  satelliteGroup.add(solarWingRight);

  // High-Gain Antenna Dish
  const antennaGimbal = new THREE.Group();
  antennaGimbal.position.set(0, 3.2, 0);

  const mastGeo = new THREE.CylinderGeometry(0.12, 0.16, 1.4, 12);
  const mast = new THREE.Mesh(mastGeo, titaniumMaterial);
  antennaGimbal.add(mast);

  const dishGeo = new THREE.SphereGeometry(1.7, 32, 16, 0, Math.PI * 2, 0, Math.PI / 3);
  const dishMat = new THREE.MeshPhysicalMaterial({
    color: 0xcca020,
    metalness: 0.95,
    roughness: 0.3,
    side: THREE.DoubleSide,
    bumpMap: noiseTex,
    bumpScale: 0.02
  });
  highGainDish = new THREE.Mesh(dishGeo, dishMat);
  highGainDish.rotation.x = -Math.PI / 2;
  highGainDish.position.set(0, 0.9, 0);
  highGainDish.castShadow = true;
  antennaGimbal.add(highGainDish);

  const hornGeo = new THREE.ConeGeometry(0.18, 0.6, 12);
  const horn = new THREE.Mesh(hornGeo, titaniumMaterial);
  horn.position.set(0, 1.4, 0);
  antennaGimbal.add(horn);

  satelliteGroup.add(antennaGimbal);

  // Thermal Venting Plumes
  const ventParticleCount = 280;
  const ventGeo = new THREE.BufferGeometry();
  const ventPositions = new Float32Array(ventParticleCount * 3);
  const ventVelocities = [];

  for (let i = 0; i < ventParticleCount; i++) {
    ventPositions[i * 3] = 0;
    ventPositions[i * 3 + 1] = 0.3;
    ventPositions[i * 3 + 2] = 2.4;
    ventVelocities.push({
      vx: (Math.random() - 0.5) * 0.08,
      vy: (Math.random() - 0.5) * 0.08 + 0.04,
      vz: 0.15 + Math.random() * 0.22,
      life: Math.random()
    });
  }
  ventGeo.setAttribute('position', new THREE.BufferAttribute(ventPositions, 3));

  const ventMat = new THREE.PointsMaterial({
    color: 0xff4411,
    size: 0.45,
    transparent: true,
    opacity: 0.0,
    blending: THREE.AdditiveBlending
  });
  particleSystemThermal = new THREE.Points(ventGeo, ventMat);
  particleSystemThermal.userData = { velocities: ventVelocities };
  satelliteGroup.add(particleSystemThermal);

  // Lightning Arcs
  const arcSegments = 24;
  const arcGeo = new THREE.BufferGeometry();
  const arcPos = new Float32Array(arcSegments * 3);
  arcGeo.setAttribute('position', new THREE.BufferAttribute(arcPos, 3));

  const arcMat = new THREE.LineBasicMaterial({
    color: 0x88ffff,
    linewidth: 3,
    transparent: true,
    opacity: 0.0
  });
  arcLineMesh = new THREE.Line(arcGeo, arcMat);
  satelliteGroup.add(arcLineMesh);

  scene.add(satelliteGroup);
}

// ==============================================================================
// 4. Real-Time Telemetry Charts (Chart.js)
// ==============================================================================
const MAX_CHART_POINTS = 50;
let viChart, tpChart;

function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 }, maxTicksLimit: 6 }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 9 } }
      }
    },
    plugins: {
      legend: {
        labels: { color: '#f8fafc', font: { family: 'Rajdhani', size: 10 }, boxWidth: 10 }
      }
    }
  };

  // Chart 1: Voltage & Current
  const canvasVI = document.getElementById("viChart");
  if (canvasVI) {
    const ctxVI = canvasVI.getContext("2d");
    viChart = new Chart(ctxVI, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Voltage (V)',
            borderColor: '#00f0ff',
            backgroundColor: 'rgba(0, 240, 255, 0.1)',
            data: [],
            borderWidth: 2,
            pointRadius: 0,
            yAxisID: 'y'
          },
          {
            label: 'Current (A)',
            borderColor: '#ffb703',
            backgroundColor: 'rgba(255, 183, 3, 0.1)',
            data: [],
            borderWidth: 2,
            pointRadius: 0,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        ...commonOptions,
        scales: {
          ...commonOptions.scales,
          y: {
            position: 'left',
            title: { display: true, text: 'V', color: '#00f0ff', font: { size: 9 } },
            grid: { color: 'rgba(255, 255, 255, 0.04)' }
          },
          y1: {
            position: 'right',
            title: { display: true, text: 'A', color: '#ffb703', font: { size: 9 } },
            grid: { drawOnChartArea: false }
          }
        }
      }
    });
  }

  // Chart 2: Temperature & Anomaly Probability
  const canvasTP = document.getElementById("tpChart");
  if (canvasTP) {
    const ctxTP = canvasTP.getContext("2d");
    tpChart = new Chart(ctxTP, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Temp (°C)',
            borderColor: '#9d4edd',
            backgroundColor: 'rgba(157, 78, 221, 0.15)',
            data: [],
            borderWidth: 2,
            pointRadius: 0,
            yAxisID: 'y'
          },
          {
            label: 'P(Anomaly)',
            borderColor: '#ff0055',
            backgroundColor: 'rgba(255, 0, 85, 0.25)',
            fill: true,
            data: [],
            borderWidth: 2,
            pointRadius: 0,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        ...commonOptions,
        scales: {
          ...commonOptions.scales,
          y: {
            position: 'left',
            title: { display: true, text: '°C', color: '#9d4edd', font: { size: 9 } },
            grid: { color: 'rgba(255, 255, 255, 0.04)' }
          },
          y1: {
            position: 'right',
            min: 0,
            max: 1.0,
            title: { display: true, text: 'P(A)', color: '#ff0055', font: { size: 9 } },
            grid: { drawOnChartArea: false }
          }
        }
      }
    });
  }
}

function appendChartData(chart, label, values) {
  if (!chart || !chart.data) return;
  if (chart.data.labels.length >= MAX_CHART_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(ds => ds.data.shift());
  }
  chart.data.labels.push(label);
  values.forEach((val, i) => {
    if (chart.data.datasets[i]) {
      chart.data.datasets[i].data.push(val);
    }
  });
  chart.update('none');
}

// ==============================================================================
// 5. Physics & Dynamic FX Update Loop
// ==============================================================================
let lastFrameTime = performance.now();
let frameCount = 0;
let fpsTimer = 0;

function updatePhysics(delta) {
  SatState.glitchPhase += delta * 12;

  const isAnomaly = (SatState.p_ensemble >= 0.30) || (SatState.primary_fault !== "NOMINAL") || (SatState.final_severity !== "NOMINAL") || SatState.safety_override_active;

  const isThermalRunaway = isAnomaly && (SatState.faultThermalRunaway || SatState.primary_fault === "THERMAL_RUNAWAY" || SatState.temp > 48.0);
  const isMicroShort = isAnomaly && (SatState.faultMicroShort || SatState.primary_fault === "INTERNAL_SHORT" || SatState.current > 4.5);
  const isUndervolt = isAnomaly && (SatState.faultDeepUndervolt || SatState.primary_fault === "UNDERVOLTAGE" || SatState.voltage < 2.9);
  const isSensorGlitch = isAnomaly && (SatState.faultSensorGlitch || SatState.primary_fault === "SENSOR_FAULT");

  // 1. 🔥 Thermal Runaway FX
  if (isThermalRunaway) {
    SatState.runawayTime += delta;
    const heatFactor = Math.min(1.0, Math.max(0.1, (SatState.temp - 20) / 60));
    
    if (cellThermalMaterial) {
      cellThermalMaterial.color.setRGB(0.2 + heatFactor * 0.8, 0.2, 0.1);
      cellThermalMaterial.emissive.setRGB(heatFactor * 1.0, heatFactor * 0.35, heatFactor * 0.05);
    }

    if (thermalPointLight) thermalPointLight.intensity = heatFactor * 4.5;
    if (particleSystemThermal) {
      particleSystemThermal.material.opacity = Math.min(0.9, heatFactor * 1.2);
      const pos = particleSystemThermal.geometry.attributes.position.array;
      const vels = particleSystemThermal.userData.velocities;
      for (let i = 0; i < vels.length; i++) {
        pos[i * 3] += vels[i].vx;
        pos[i * 3 + 1] += vels[i].vy;
        pos[i * 3 + 2] += vels[i].vz;
        vels[i].life -= delta * 0.8;

        if (vels[i].life <= 0) {
          pos[i * 3] = (Math.random() - 0.5) * 0.3;
          pos[i * 3 + 1] = 0.3 + (Math.random() - 0.5) * 0.2;
          pos[i * 3 + 2] = 2.2;
          vels[i].life = 0.8 + Math.random() * 0.4;
        }
      }
      particleSystemThermal.geometry.attributes.position.needsUpdate = true;
    }

    if (Math.random() < 0.12) {
      audioSys.playVentingHiss();
    }
  } else {
    // Guaranteed instant visual recovery to nominal cool blue
    if (thermalPointLight) thermalPointLight.intensity = 0;
    if (particleSystemThermal) particleSystemThermal.material.opacity = 0;
    if (cellThermalMaterial) {
      cellThermalMaterial.color.setHex(0x3b82f6);
      cellThermalMaterial.emissive.setHex(0x001133);
    }
    SatState.runawayTime = 0;
  }

  // 2. ⚡ Micro-Short Lightning Arcs
  if (isMicroShort) {
    if (arcLineMesh) {
      arcLineMesh.material.opacity = Math.random() > 0.3 ? 0.95 : 0.1;
      const arcPos = arcLineMesh.geometry.attributes.position.array;
      let curX = -0.6, curY = 0.4, curZ = 2.45;
      for (let i = 0; i < 24; i++) {
        arcPos[i * 3] = curX;
        arcPos[i * 3 + 1] = curY;
        arcPos[i * 3 + 2] = curZ;
        curX += 0.05 + (Math.random() - 0.5) * 0.04;
        curY += (Math.random() - 0.5) * 0.12;
        curZ += (Math.random() - 0.5) * 0.06;
      }
      arcLineMesh.geometry.attributes.position.needsUpdate = true;
    }
    if (arcFlashLight) arcFlashLight.intensity = 3.5;
    if (Math.random() < 0.2) audioSys.playArcDischarge();
  } else {
    if (arcLineMesh) arcLineMesh.material.opacity = 0;
    if (arcFlashLight) arcFlashLight.intensity = 0;
  }

  // 3. 🔋 Deep Undervoltage Tumble & Smooth Attitude Recovery
  if (isUndervolt && satelliteGroup) {
    satelliteGroup.rotation.x += delta * 0.18;
    satelliteGroup.rotation.z += delta * 0.12;
  } else if (satelliteGroup) {
    // Smooth attitude stabilization back to upright (0, 0)
    satelliteGroup.rotation.x += (0 - satelliteGroup.rotation.x) * Math.min(1.0, delta * 3.5);
    satelliteGroup.rotation.z += (0 - satelliteGroup.rotation.z) * Math.min(1.0, delta * 3.5);
  }

  // 4. 📡 Sensor Glitch Dish Jitter & Smooth Reset
  if (isSensorGlitch && highGainDish) {
    highGainDish.rotation.z = Math.sin(SatState.glitchPhase * 0.8) * 0.4;
    highGainDish.rotation.y = Math.cos(SatState.glitchPhase * 0.5) * 0.3;
  } else if (highGainDish) {
    highGainDish.rotation.z += (0 - highGainDish.rotation.z) * Math.min(1.0, delta * 5.0);
    highGainDish.rotation.y += (0 - highGainDish.rotation.y) * Math.min(1.0, delta * 5.0);
  }

  // Normal orbital rotation & solar tracking
  if (!isUndervolt && satelliteGroup) {
    satelliteGroup.rotation.y += delta * 0.04;
    if (solarWingLeft && solarWingRight) {
      solarWingLeft.rotation.x = Math.sin(Date.now() * 0.0003) * 0.25;
      solarWingRight.rotation.x = Math.sin(Date.now() * 0.0003) * 0.25;
    }
  }
}

// ==============================================================================
// 6. HUD UI Updates from Live WebSocket Stream
// ==============================================================================
function updateHUD() {
  // Gauges
  safeSetText('disp-voltage', SatState.voltage.toFixed(3));
  safeSetText('disp-current', SatState.current.toFixed(3));
  safeSetText('disp-temp', SatState.temp.toFixed(1));
  safeSetText('disp-dtdt', `dT/dt: ${(SatState.dtdt >= 0 ? '+' : '')}${SatState.dtdt.toFixed(2)}°C/s`);
  safeSetText('disp-esr', SatState.esr.toFixed(3));
  safeSetText('disp-soc', `SOC: ${(SatState.soc * 100).toFixed(1)}%`);
  safeSetText('disp-power', `P: ${SatState.power.toFixed(2)} W`);

  // Progress bars
  safeSetWidth('bar-voltage', (SatState.voltage / 4.5) * 100);
  safeSetWidth('bar-current', (SatState.current / 8.0) * 100);
  safeSetWidth('bar-temp', (SatState.temp / 80.0) * 100);
  safeSetWidth('bar-esr', (SatState.esr / 0.5) * 100);

  // Ensemble Probabilities
  safeSetText('disp-p-ens', SatState.p_ensemble.toFixed(3));
  safeSetWidth('bar-p-ens', SatState.p_ensemble * 100);
  safeSetText('ens-prob-pill', `P(Anomaly): ${SatState.p_ensemble.toFixed(3)}`);
  safeSetText('disp-rf-val', SatState.p_rf.toFixed(2));
  safeSetText('disp-xgb-val', SatState.p_xgboost.toFixed(2));
  safeSetText('disp-et-val', SatState.p_extra_trees.toFixed(2));

  // Diagnosis and Severity
  safeSetText('disp-primary-fault', SatState.primary_fault);
  safeSetText('disp-severity', SatState.final_severity);
  safeSetText('disp-risk-score', `INDEX: ${SatState.risk_score.toFixed(2)}`);
  safeSetWidth('bar-risk', SatState.risk_score * 100);

  // RL Adaptive Policy Card
  safeSetText('disp-active-rl-action', `${SatState.rl_action_name} (τ=${SatState.dynamic_threshold.toFixed(2)})`);
  safeSetText('disp-last-recovery-action', SatState.lastMitigatingRLAction || 'STANDBY');

  const rlTag = document.getElementById('rl-action-status-tag');
  if (rlTag) {
    if (SatState.rl_action_name === 'LOAD_SHEDDING') {
      rlTag.innerText = 'MITIGATING (LOAD SHED)';
      rlTag.className = 'text-[9px] font-bold px-1.5 py-0.5 rounded bg-amber-950/60 text-amber-300 border border-amber-500/50 animate-pulse';
    } else if (SatState.rl_action_name === 'TRIGGER_SAFE_MODE') {
      rlTag.innerText = 'EMERGENCY SAFE MODE';
      rlTag.className = 'text-[9px] font-bold px-1.5 py-0.5 rounded bg-red-950/60 text-red-300 border border-red-500/50 animate-pulse';
    } else if (SatState.rl_action_name === 'HIGH_SENSITIVITY_PREARM') {
      rlTag.innerText = 'SENSITIVITY PRE-ARM';
      rlTag.className = 'text-[9px] font-bold px-1.5 py-0.5 rounded bg-cyan-950/60 text-cyan-300 border border-cyan-500/50';
    } else {
      rlTag.innerText = 'NOMINAL POLICY';
      rlTag.className = 'text-[9px] font-bold px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-500/40';
    }
  }

  // Warnings
  const voltWarn = document.getElementById('volt-warning');
  if (voltWarn) {
    if (SatState.voltage < 3.0) {
      voltWarn.classList.remove('hidden');
      voltWarn.innerText = SatState.voltage < 2.5 ? 'CRITICAL UNDERVOLT' : 'LOW BUS';
    } else {
      voltWarn.classList.add('hidden');
    }
  }

  const currWarn = document.getElementById('curr-warning');
  if (currWarn) {
    if (SatState.current > 4.5) {
      currWarn.classList.remove('hidden');
    } else {
      currWarn.classList.add('hidden');
    }
  }

  const tempWarn = document.getElementById('temp-warning');
  if (tempWarn) {
    if (SatState.temp > 50.0) {
      tempWarn.classList.remove('hidden');
    } else {
      tempWarn.classList.add('hidden');
    }
  }

  // Master Badge & Beacon
  const badge = document.getElementById('overall-status-badge');
  const beaconDot = document.getElementById('header-beacon-dot');
  const beaconPulse = document.getElementById('header-pulse-beacon');

  if (SatState.final_severity === 'EMERGENCY' || SatState.safety_override_active) {
    if (badge) {
      badge.innerText = 'CRITICAL OVERRIDE';
      badge.className = 'px-2 py-0.5 text-[10px] font-mono-telemetry font-bold tracking-wider uppercase rounded bg-red-500/20 text-red-300 border border-red-500/50 animate-pulse';
    }
    if (beaconDot) beaconDot.className = 'absolute -top-1 -right-1 w-3 h-3 rounded-full bg-red-500';
    if (beaconPulse) beaconPulse.className = 'absolute -top-1 -right-1 w-3 h-3 rounded-full bg-red-400 animate-ping';
    safeSetColor('disp-severity', '#ff0055');
    safeSetColor('disp-primary-fault', '#ff0055');
    audioSys.startCriticalAlarm();
  } else if (SatState.final_severity === 'CRITICAL' || SatState.final_severity === 'WARNING') {
    if (badge) {
      badge.innerText = 'ANOMALY DETECTED';
      badge.className = 'px-2 py-0.5 text-[10px] font-mono-telemetry font-bold tracking-wider uppercase rounded bg-amber-500/20 text-amber-300 border border-amber-500/50';
    }
    if (beaconDot) beaconDot.className = 'absolute -top-1 -right-1 w-3 h-3 rounded-full bg-amber-500';
    if (beaconPulse) beaconPulse.className = 'absolute -top-1 -right-1 w-3 h-3 rounded-full bg-amber-400 animate-ping';
    safeSetColor('disp-severity', '#ffb703');
    safeSetColor('disp-primary-fault', '#ffb703');
    audioSys.stopCriticalAlarm();
  } else {
    if (badge) {
      badge.innerText = 'NOMINAL STATE';
      badge.className = 'px-2 py-0.5 text-[10px] font-mono-telemetry font-bold tracking-wider uppercase rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
    }
    if (beaconDot) beaconDot.className = 'absolute -top-1 -right-1 w-3 h-3 rounded-full bg-emerald-500';
    if (beaconPulse) beaconPulse.className = 'absolute -top-1 -right-1 w-3 h-3 rounded-full bg-emerald-400 animate-ping';
    safeSetColor('disp-severity', '#00ff88');
    safeSetColor('disp-primary-fault', '#00ff88');
    audioSys.stopCriticalAlarm();
  }

  // Hardware Pin 13 LED Fixture
  const pinDot = document.getElementById('pin13-led-dot');
  if (pinDot) {
    if (SatState.safety_override_active) {
      pinDot.className = 'w-4 h-4 rounded-full led-strobe';
      safeSetText('pin13-led-text', 'PIN 13: 50ms FAST STROBE (CRITICAL)');
    } else if (SatState.p_ensemble > 0.5 || SatState.final_severity !== 'NOMINAL') {
      pinDot.className = 'w-4 h-4 rounded-full led-active';
      safeSetText('pin13-led-text', `PIN 13: SOLID ON (${SatState.rl_action_name})`);
    } else {
      pinDot.className = 'w-4 h-4 rounded-full bg-gray-700 border border-gray-500';
      safeSetText('pin13-led-text', 'PIN 13: LOW (NOMINAL)');
    }
  }

  // Operator review gate
  const authBtn = document.getElementById('btn-operator-auth');
  if (authBtn) {
    if (SatState.requires_human_approval && !SatState.autopilotMode) {
      authBtn.classList.remove('hidden');
      safeSetText('auth-status-text', '⚠️ OPERATOR CONFIRMATION REQUIRED');
      safeSetColor('auth-status-text', '#ffb703');
    } else {
      authBtn.classList.add('hidden');
      safeSetText('auth-status-text', SatState.autopilotMode ? 'Autonomous Auto-Pilot' : 'Operator Gate Standby');
      safeSetColor('auth-status-text', '#38bdf8');
    }
  }

  // Mission Clock
  const timeSec = Math.floor(performance.now() / 1000);
  const mins = Math.floor((timeSec % 3600) / 60);
  const secs = timeSec % 60;
  safeSetText('mission-clock', `T+142:08:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`);

  // Update Dedicated RL Mitigation & Fault Recovery Engine Hub
  updateRLRecoveryHub();

  // Update Physical Subsystem Component Simulation HUD
  updatePhysicalSubsystemHUD();
}

function updateRLRecoveryHub() {
  const isAnomaly = (SatState.p_ensemble >= 0.30) || (SatState.primary_fault !== 'NOMINAL') || (SatState.final_severity !== 'NOMINAL') || SatState.safety_override_active;

  const badge = document.getElementById('rl-hub-status-badge');
  const stdRef = document.getElementById('rl-standard-ref');
  const cardState = document.getElementById('rl-card-state');
  const actionName = document.getElementById('rl-applied-action-name');
  const actionDesc = document.getElementById('rl-applied-action-desc');
  const faultStatus = document.getElementById('rl-fault-status');
  const faultName = document.getElementById('rl-mitigated-fault-name');
  const faultDesc = document.getElementById('rl-mitigated-fault-desc');
  const tempDelta = document.getElementById('rl-temp-delta');
  const panomDelta = document.getElementById('rl-panom-delta');
  const recoverySummary = document.getElementById('rl-recovery-summary');
  const execStatus = document.getElementById('recovery-exec-status');
  const procSteps = document.getElementById('recovery-procedure-steps');
  const lifecycleTag = document.getElementById('recovery-lifecycle-tag');

  const step1 = document.getElementById('step-detect');
  const step2 = document.getElementById('step-policy');
  const step3 = document.getElementById('step-verify');
  const step4 = document.getElementById('step-recover');

  if (isAnomaly) {
    const activeFault = SatState.primary_fault || 'ANOMALY';
    
    if (badge) {
      badge.innerText = 'MITIGATION ENGAGED';
      badge.className = 'px-2 py-0.5 text-[9px] font-mono-telemetry font-bold rounded bg-amber-500/20 text-amber-300 border border-amber-500/50 animate-pulse';
    }
    if (cardState) {
      cardState.innerText = 'EXECUTING';
      cardState.className = 'text-amber-400 font-bold';
    }
    if (actionName) actionName.innerText = SatState.rl_action_name || 'LOAD_SHEDDING';

    if (activeFault === 'THERMAL_RUNAWAY' || SatState.temp > 45.0) {
      if (stdRef) stdRef.innerText = 'NASA-HDBK-4008 §4.2.1 // ISRO-PCDU-EPS-04';
      if (actionDesc) actionDesc.innerText = 'PCDU Remote Power Controllers (RPC-1 & RPC-2) tripped: derated scientific payload by -35% to halt runaway Joule heat.';
      if (faultStatus) { faultStatus.innerText = 'THERMAL RUNAWAY'; faultStatus.className = 'text-red-400 font-bold'; }
      if (faultName) faultName.innerText = 'BATTERY CELL EXOTHERM';
      if (faultDesc) faultDesc.innerText = 'Cell temperature rising. Thermal gradient dT/dt > +1.2°C/s exceeding NASA 48°C flight limit.';
      if (execStatus) { execStatus.innerText = 'ENGAGED // TIER-1 LOAD SHEDDING'; execStatus.className = 'text-amber-400 font-bold animate-pulse'; }
      if (procSteps) {
        procSteps.innerHTML = `
          <div>• <span class="text-amber-300 font-semibold">[PCDU RPC Tripping]:</span> Commanded Remote Power Controllers RPC-1 & RPC-2 -> -35% Non-Essential Science Load (Current: 4.8A -> 1.8A).</div>
          <div>• <span class="text-amber-300 font-semibold">[ADCS Radiator Slew]:</span> Re-oriented satellite thermal radiator louvers +18.5° towards deep space (3K sink).</div>
          <div>• <span class="text-amber-300 font-semibold">[BCR Trickle Taper]:</span> Tapered Battery Charge Regulator charging to C/20 trickle rate until cell temperatures normalized &lt; 28°C.</div>
        `;
      }
    } else if (activeFault === 'INTERNAL_SHORT' || SatState.current > 4.5) {
      if (stdRef) stdRef.innerText = 'ISRO-URSC-PCDU-EPS-04 // ECSS-E-ST-20C';
      if (actionDesc) actionDesc.innerText = 'Sub-20ms solid-state latching relay opened to isolate faulted battery string and switch over to redundant Bus-B.';
      if (faultStatus) { faultStatus.innerText = 'CRITICAL OVERCURRENT'; faultStatus.className = 'text-red-400 font-bold'; }
      if (faultName) faultName.innerText = 'MICRO-SHORT DETECTED';
      if (faultDesc) faultDesc.innerText = 'Impedance proxy collapsed < 0.02Ω. Bus current surge demanding immediate hardware isolation.';
      if (execStatus) { execStatus.innerText = 'ENGAGED // BUS-B CROSS-STRAP SAFE MODE'; execStatus.className = 'text-red-400 font-bold animate-pulse'; }
      if (procSteps) {
        procSteps.innerHTML = `
          <div>• <span class="text-red-300 font-semibold">[Sub-20ms Bus Isolation]:</span> Opened solid-state latching relay on Battery String-A to prevent thermal propagation.</div>
          <div>• <span class="text-red-300 font-semibold">[Main Bus Cross-Strap]:</span> Switched PCDU to redundant Bus-B power rail (Hardware Pin 13 LED strobe active).</div>
          <div>• <span class="text-red-300 font-semibold">[Safe Hold Mode (SHM)]:</span> Re-routed regulated 28V bus exclusively to 9.2W essential flight computer (OBC) and TT&C.</div>
        `;
      }
    } else if (activeFault === 'UNDERVOLTAGE' || SatState.voltage < 2.9) {
      if (stdRef) stdRef.innerText = 'AIAA-S-136-2023 §5.1 // ISRO Chandrayaan UVLS';
      if (actionDesc) actionDesc.innerText = 'Under-Voltage Load Shedding (UVLS Lockout) engaged (-75% load shed) with autonomous Sun-pointing acquisition.';
      if (faultStatus) { faultStatus.innerText = 'BUS VOLTAGE COLLAPSE'; faultStatus.className = 'text-amber-400 font-bold'; }
      if (faultName) faultName.innerText = 'DEEP DISCHARGE SAG';
      if (faultDesc) faultDesc.innerText = 'Bus voltage collapsed below 2.90V. Activating Tier-2 UVLS to prevent irreversible copper dissolution.';
      if (execStatus) { execStatus.innerText = 'ENGAGED // TIER-2 UVLS & SUN POINTING'; execStatus.className = 'text-amber-400 font-bold animate-pulse'; }
      if (procSteps) {
        procSteps.innerHTML = `
          <div>• <span class="text-amber-300 font-semibold">[Tier-2 UVLS Lockout]:</span> Disconnected payload and subsystem buses (75% power demand reduction).</div>
          <div>• <span class="text-amber-300 font-semibold">[B-dot Sun Acquisition]:</span> ADCS magnetic torquers detumble spacecraft and orient solar arrays normal to Sun (1361 W/m²).</div>
          <div>• <span class="text-amber-300 font-semibold">[Constant-Current Recharge]:</span> BCR commanded solar array shunts to 2.4A CC charging until SOC &ge; 60%.</div>
        `;
      }
    } else {
      if (stdRef) stdRef.innerText = 'NASA-SP-20205003605 // CCSDS 502.0-B-3';
      if (actionDesc) actionDesc.innerText = 'Pre-armed high-sensitivity diagnostic filter bank (τ = 0.35) and switched telemetry observer channel.';
      if (faultStatus) { faultStatus.innerText = 'ANOMALY DETECTED'; faultStatus.className = 'text-amber-400 font-bold'; }
      if (faultName) faultName.innerText = activeFault;
      if (faultDesc) faultDesc.innerText = `Active anomaly on EPS bus. Safety risk index: ${SatState.risk_score.toFixed(2)}`;
      if (execStatus) { execStatus.innerText = 'ENGAGED // SENSOR CROSS-VALIDATION'; execStatus.className = 'text-amber-400 font-bold'; }
      if (procSteps) {
        procSteps.innerHTML = `
          <div>• <span class="text-cyan-300 font-semibold">[Sensor Cross-Check]:</span> Verified primary ADC vs secondary Channel-B redundant transducer.</div>
          <div>• <span class="text-cyan-300 font-semibold">[Filter Pre-Arming]:</span> Lowered decision threshold to τ = 0.35 to catch transient degradation signatures.</div>
          <div>• <span class="text-cyan-300 font-semibold">[Supercap Pulse Buffer]:</span> Engaged bus buffer capacitors to smooth transient impedance drops.</div>
        `;
      }
    }

    if (tempDelta) {
      tempDelta.innerText = `${SatState.temp.toFixed(1)}°C (${SatState.temp > 45 ? 'OVERHEAT' : 'ELEVATED'})`;
      tempDelta.className = SatState.temp > 45 ? 'text-red-400 font-bold' : 'text-amber-300 font-bold';
    }
    if (panomDelta) {
      panomDelta.innerText = `${SatState.p_ensemble.toFixed(3)} (HIGH RISK)`;
      panomDelta.className = 'text-red-400 font-bold';
    }
    if (recoverySummary) {
      recoverySummary.innerText = `Intervention: ${SatState.rl_action_name} • Forward Sim Utility: 0.95`;
    }
    if (lifecycleTag) {
      lifecycleTag.innerText = '● STEP 2: RL MITIGATION ENGAGED';
      lifecycleTag.className = 'text-amber-400 font-semibold';
    }

    if (step1) step1.className = 'p-1.5 rounded bg-red-950/60 border border-red-500/60 text-red-200 animate-pulse';
    if (step2) step2.className = 'p-1.5 rounded bg-amber-950/60 border border-amber-500/60 text-amber-200 animate-pulse';
    if (step3) step3.className = 'p-1.5 rounded bg-cyan-950/50 border border-cyan-500/40 text-cyan-200';
    if (step4) step4.className = 'p-1.5 rounded bg-black/40 border border-white/10 text-gray-500';

  } else {
    // NOMINAL / RECOVERED STATE
    if (badge) {
      badge.innerText = 'NOMINAL STABILIZED';
      badge.className = 'px-2 py-0.5 text-[9px] font-mono-telemetry font-bold rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
    }
    if (stdRef) {
      stdRef.innerText = 'NASA-HDBK-4008 // ISRO-URSC-PCDU-EPS-04';
    }
    if (cardState) {
      cardState.innerText = 'POLICY NOMINAL';
      cardState.className = 'text-emerald-400 font-bold';
    }
    if (actionName) {
      actionName.innerText = SatState.lastMitigatingRLAction ? `${SatState.lastMitigatingRLAction} (APPLIED)` : 'NOMINAL_MONITOR';
    }
    if (actionDesc) {
      if (SatState.lastMitigatingRLAction && SatState.lastMitigatingRLAction !== 'NOMINAL_MONITOR') {
        actionDesc.innerText = `Successfully executed NASA/ISRO recovery sequence for ${SatState.lastMitigatedFault || 'EPS Anomaly'}. Subsystem re-stabilized to nominal flight bounds.`;
      } else {
        actionDesc.innerText = 'Continuous telemetry polling at 2.0 Hz. Decision threshold set to nominal τ = 0.70.';
      }
    }
    if (faultStatus) {
      faultStatus.innerText = 'RESOLVED';
      faultStatus.className = 'text-emerald-400 font-bold';
    }
    if (faultName) {
      faultName.innerText = SatState.lastMitigatedFault && SatState.lastMitigatedFault !== 'NOMINAL' ? `${SatState.lastMitigatedFault} (RESOLVED)` : 'NOMINAL STATE';
    }
    if (faultDesc) {
      faultDesc.innerText = 'All electrochemical & thermal parameters operating strictly within ESA/NASA/ISRO flight envelopes.';
    }

    if (tempDelta) {
      tempDelta.innerText = `${SatState.temp.toFixed(1)}°C (STABLE)`;
      tempDelta.className = 'text-emerald-300 font-bold';
    }
    if (panomDelta) {
      panomDelta.innerText = `${SatState.p_ensemble.toFixed(3)} (LOW)`;
      panomDelta.className = 'text-emerald-300 font-bold';
    }
    if (recoverySummary) {
      recoverySummary.innerText = `PCDU Response: 0.26ms • 100% Safety Guarantee`;
    }
    if (execStatus) {
      execStatus.innerText = 'STANDBY // NOMINAL CRUISE';
      execStatus.className = 'text-emerald-400 font-bold';
    }
    if (procSteps) {
      procSteps.innerHTML = `
        <div>• <span class="text-cyan-300 font-semibold">[PCDU RPC Tripping]:</span> Commanded Remote Power Controllers to derate non-essential science payload by -35%.</div>
        <div>• <span class="text-cyan-300 font-semibold">[ADCS Thermal Slew]:</span> Re-oriented spacecraft thermal louvers towards deep space (3K radiative heat sink).</div>
        <div>• <span class="text-cyan-300 font-semibold">[BCR Trickle Taper]:</span> Tapered Battery Charge Regulator charging current to C/20 trickle rate until cell temperatures normalized &lt; 28°C.</div>
      `;
    }
    if (lifecycleTag) {
      lifecycleTag.innerText = '● STEP 4: RECOVERY VERIFIED // SUBSYSTEM NOMINAL';
      lifecycleTag.className = 'text-emerald-400 font-semibold';
    }

    if (step1) step1.className = 'p-1.5 rounded bg-emerald-950/40 border border-emerald-500/40 text-emerald-200';
    if (step2) step2.className = 'p-1.5 rounded bg-emerald-950/40 border border-emerald-500/40 text-emerald-200';
    if (step3) step3.className = 'p-1.5 rounded bg-emerald-950/40 border border-emerald-500/40 text-emerald-200';
    if (step4) step4.className = 'p-1.5 rounded bg-emerald-950/60 border border-emerald-400 text-emerald-100 font-bold shadow-sm';
  }
}

// ==============================================================================
// 6.5. Physical Spacecraft Subsystem Component Simulation (NASA & ISRO Modeling)
// ==============================================================================
function updatePhysicalSubsystemHUD() {
  const isAnomaly = (SatState.p_ensemble >= 0.30) || (SatState.primary_fault !== 'NOMINAL') || (SatState.final_severity !== 'NOMINAL') || SatState.safety_override_active;
  const isThermal = isAnomaly && (SatState.faultThermalRunaway || SatState.primary_fault === "THERMAL_RUNAWAY" || SatState.temp > 48.0);
  const isShort = isAnomaly && (SatState.faultMicroShort || SatState.primary_fault === "INTERNAL_SHORT" || SatState.current > 4.5);
  const isUndervolt = isAnomaly && (SatState.faultDeepUndervolt || SatState.primary_fault === "UNDERVOLTAGE" || SatState.voltage < 2.9);
  const isHighImp = isAnomaly && (SatState.faultHighImpedance || SatState.primary_fault === "HIGH_IMPEDANCE");
  const isSensor = isAnomaly && (SatState.faultSensorGlitch || SatState.primary_fault === "SENSOR_FAULT");
  const isLoadShedding = SatState.rl_action_name === 'LOAD_SHEDDING' || isThermal;
  const isSafeMode = SatState.rl_action_name === 'TRIGGER_SAFE_MODE' || isShort;

  // 1. PCDU Subsystem
  const pcduTag = document.getElementById('pcdu-status-tag');
  const pcduRail = document.getElementById('pcdu-rail-val');
  const pcduRpc = document.getElementById('pcdu-rpc-val');
  const pcduBusB = document.getElementById('pcdu-busb-val');
  const pcduCard = document.getElementById('subsystem-pcdu-card');

  if (pcduCard) {
    if (isShort || isSafeMode) {
      if (pcduTag) { pcduTag.innerText = 'BUS-A ISOLATED (<20ms)'; pcduTag.className = 'text-red-400 font-bold text-[9px] bg-red-950/60 px-1.5 py-0.5 rounded border border-red-500/40 animate-pulse'; }
      if (pcduRail) { pcduRail.innerText = '24.2 V (Emergency)'; pcduRail.className = 'text-red-400 font-bold'; }
      if (pcduRpc) { pcduRpc.innerText = 'LOCKED OPEN (0%)'; pcduRpc.className = 'text-red-400 font-bold'; }
      if (pcduBusB) { pcduBusB.innerText = 'ACTIVE (Bus-B 9.2W)'; pcduBusB.className = 'text-amber-300 font-bold'; }
      pcduCard.className = 'p-2.5 rounded-lg bg-red-950/30 border border-red-500/40 space-y-1.5 transition-all';
    } else if (isLoadShedding) {
      if (pcduTag) { pcduTag.innerText = 'LOAD SHED (-35%)'; pcduTag.className = 'text-amber-400 font-bold text-[9px] bg-amber-950/60 px-1.5 py-0.5 rounded border border-amber-500/40 animate-pulse'; }
      if (pcduRail) { pcduRail.innerText = '28.2 V Regulated'; pcduRail.className = 'text-amber-300 font-bold'; }
      if (pcduRpc) { pcduRpc.innerText = 'TRIPPED (-35%)'; pcduRpc.className = 'text-amber-400 font-bold'; }
      if (pcduBusB) { pcduBusB.innerText = 'HOT STANDBY'; pcduBusB.className = 'text-gray-400'; }
      pcduCard.className = 'p-2.5 rounded-lg bg-amber-950/30 border border-amber-500/40 space-y-1.5 transition-all';
    } else if (isUndervolt) {
      if (pcduTag) { pcduTag.innerText = 'TIER-2 UVLS (-75%)'; pcduTag.className = 'text-blue-400 font-bold text-[9px] bg-blue-950/60 px-1.5 py-0.5 rounded border border-blue-500/40 animate-pulse'; }
      if (pcduRail) { pcduRail.innerText = '22.8 V (Depleted)'; pcduRail.className = 'text-blue-400 font-bold'; }
      if (pcduRpc) { pcduRpc.innerText = 'TRIPPED (-75%)'; pcduRpc.className = 'text-blue-400 font-bold'; }
      if (pcduBusB) { pcduBusB.innerText = 'LOCKOUT'; pcduBusB.className = 'text-gray-400'; }
      pcduCard.className = 'p-2.5 rounded-lg bg-blue-950/30 border border-blue-500/40 space-y-1.5 transition-all';
    } else {
      if (pcduTag) { pcduTag.innerText = 'BUS-A PRIMARY'; pcduTag.className = 'text-emerald-400 font-bold text-[9px] bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-500/30'; }
      if (pcduRail) { pcduRail.innerText = '28.4 V Regulated'; pcduRail.className = 'text-cyan-300 font-bold'; }
      if (pcduRpc) { pcduRpc.innerText = 'CLOSED (100%)'; pcduRpc.className = 'text-emerald-400 font-bold'; }
      if (pcduBusB) { pcduBusB.innerText = 'HOT STANDBY'; pcduBusB.className = 'text-gray-400'; }
      pcduCard.className = 'p-2.5 rounded-lg bg-black/60 border border-cyan-500/30 space-y-1.5 transition-all';
    }
  }

  // 2. BCR Subsystem
  const bcrTag = document.getElementById('bcr-status-tag');
  const bcrMode = document.getElementById('bcr-mode-val');
  const bcrShunt = document.getElementById('bcr-shunt-val');
  const bcrTrickle = document.getElementById('bcr-trickle-val');
  const bcrCard = document.getElementById('subsystem-bcr-card');

  if (bcrCard) {
    if (isThermal) {
      if (bcrTag) { bcrTag.innerText = 'TRICKLE MODE (C/20)'; bcrTag.className = 'text-amber-400 font-bold text-[9px] bg-amber-950/60 px-1.5 py-0.5 rounded border border-amber-500/40 animate-pulse'; }
      if (bcrMode) { bcrMode.innerText = '0.50 A (Tapered)'; bcrMode.className = 'text-amber-300 font-bold'; }
      if (bcrShunt) { bcrShunt.innerText = 'HEAT BYPASS'; bcrShunt.className = 'text-amber-400 font-semibold'; }
      if (bcrTrickle) { bcrTrickle.innerText = 'ENGAGED (C/20)'; bcrTrickle.className = 'text-amber-300 font-bold'; }
      bcrCard.className = 'p-2.5 rounded-lg bg-amber-950/30 border border-amber-500/40 space-y-1.5 transition-all';
    } else if (isUndervolt) {
      if (bcrTag) { bcrTag.innerText = 'BOOST CHARGE'; bcrTag.className = 'text-blue-400 font-bold text-[9px] bg-blue-950/60 px-1.5 py-0.5 rounded border border-blue-500/40 animate-pulse'; }
      if (bcrMode) { bcrMode.innerText = '2.40 A (Max CC)'; bcrMode.className = 'text-blue-300 font-bold'; }
      if (bcrShunt) { bcrShunt.innerText = 'FULL ARRAY DIRECT'; bcrShunt.className = 'text-blue-300 font-semibold'; }
      if (bcrTrickle) { bcrTrickle.innerText = 'READY'; bcrTrickle.className = 'text-gray-400'; }
      bcrCard.className = 'p-2.5 rounded-lg bg-blue-950/30 border border-blue-500/40 space-y-1.5 transition-all';
    } else {
      if (bcrTag) { bcrTag.innerText = 'CC 2.4A CHARGE'; bcrTag.className = 'text-emerald-400 font-bold text-[9px] bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-500/30'; }
      if (bcrMode) { bcrMode.innerText = '2.45 A (CC)'; bcrMode.className = 'text-amber-300 font-bold'; }
      if (bcrShunt) { bcrShunt.innerText = 'ACTIVE REG'; bcrShunt.className = 'text-cyan-300'; }
      if (bcrTrickle) { bcrTrickle.innerText = 'STANDBY (C/20)'; bcrTrickle.className = 'text-gray-400'; }
      bcrCard.className = 'p-2.5 rounded-lg bg-black/60 border border-amber-500/30 space-y-1.5 transition-all';
    }
  }

  // 3. Secondary Li-ion Battery Chemistry
  const battTag = document.getElementById('battery-status-tag');
  const battRint = document.getElementById('batt-rint-val');
  const battJoule = document.getElementById('batt-joule-val');
  const battEff = document.getElementById('batt-eff-val');
  const battCard = document.getElementById('subsystem-battery-card');

  const rInt = isHighImp ? 0.145 : (isShort ? 0.012 : (isThermal ? 0.082 : 0.045));
  const jouleHeat = Math.pow(SatState.current, 2) * rInt;
  const coulEff = isShort ? 72.4 : (isHighImp ? 88.6 : (isThermal ? 91.2 : 99.2));

  if (battCard) {
    if (battRint) battRint.innerText = `${rInt.toFixed(3)} Ω`;
    if (battJoule) battJoule.innerText = `${jouleHeat.toFixed(2)} W`;
    if (battEff) battEff.innerText = `${coulEff.toFixed(1)} %`;

    if (isThermal) {
      if (battTag) { battTag.innerText = 'OVERHEATING'; battTag.className = 'text-red-400 font-bold text-[9px] bg-red-950/60 px-1.5 py-0.5 rounded border border-red-500/40 animate-pulse'; }
      battCard.className = 'p-2.5 rounded-lg bg-red-950/30 border border-red-500/40 space-y-1.5 transition-all';
    } else if (isShort) {
      if (battTag) { battTag.innerText = 'MICRO-SHORT'; battTag.className = 'text-red-400 font-bold text-[9px] bg-red-950/60 px-1.5 py-0.5 rounded border border-red-500/40 animate-pulse'; }
      battCard.className = 'p-2.5 rounded-lg bg-red-950/30 border border-red-500/40 space-y-1.5 transition-all';
    } else if (isHighImp) {
      if (battTag) { battTag.innerText = 'SEI AGING SURGE'; battTag.className = 'text-purple-400 font-bold text-[9px] bg-purple-950/60 px-1.5 py-0.5 rounded border border-purple-500/40'; }
      battCard.className = 'p-2.5 rounded-lg bg-purple-950/30 border border-purple-500/40 space-y-1.5 transition-all';
    } else {
      if (battTag) { battTag.innerText = 'HEALTHY'; battTag.className = 'text-emerald-400 font-bold text-[9px] bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-500/30'; }
      battCard.className = 'p-2.5 rounded-lg bg-black/60 border border-emerald-500/30 space-y-1.5 transition-all';
    }
  }

  // 4. Radiator & ADCS Thermal Subsystem
  const adcsTag = document.getElementById('adcs-status-tag');
  const adcsSlew = document.getElementById('adcs-slew-val');
  const adcsQrad = document.getElementById('adcs-qrad-val');
  const adcsCard = document.getElementById('subsystem-adcs-card');

  const qRad = 4.5 * 0.08 * (Math.max(3.0, SatState.temp) - 3.0); // h * A * (T - Tsink)

  if (adcsCard) {
    if (adcsQrad) adcsQrad.innerText = `${qRad.toFixed(1)} W (hAΔT)`;

    if (isThermal || isLoadShedding) {
      if (adcsTag) { adcsTag.innerText = 'SLEWING TO 3K SINK'; adcsTag.className = 'text-blue-300 font-bold text-[9px] bg-blue-950/60 px-1.5 py-0.5 rounded border border-blue-500/40 animate-pulse'; }
      if (adcsSlew) { adcsSlew.innerText = '+18.5° (Deep Space)'; adcsSlew.className = 'text-amber-300 font-bold'; }
      adcsCard.className = 'p-2.5 rounded-lg bg-blue-950/30 border border-blue-500/40 space-y-1.5 transition-all';
    } else {
      if (adcsTag) { adcsTag.innerText = '3K SINK TRACK'; adcsTag.className = 'text-emerald-400 font-bold text-[9px] bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-500/30'; }
      if (adcsSlew) { adcsSlew.innerText = '0.0° (Level)'; adcsSlew.className = 'text-emerald-300 font-bold'; }
      adcsCard.className = 'p-2.5 rounded-lg bg-black/60 border border-blue-500/30 space-y-1.5 transition-all';
    }
  }

  // 5. Scientific Payload Bus Subsystem
  const payloadTag = document.getElementById('payload-status-tag');
  const payloadInst = document.getElementById('payload-inst-val');
  const payloadShed = document.getElementById('payload-shed-val');
  const payloadCard = document.getElementById('subsystem-payload-card');

  if (payloadCard) {
    if (isSafeMode || isShort) {
      if (payloadTag) { payloadTag.innerText = 'ALL SHED (0W)'; payloadTag.className = 'text-red-400 font-bold text-[9px] bg-red-950/60 px-1.5 py-0.5 rounded border border-red-500/40 animate-pulse'; }
      if (payloadInst) { payloadInst.innerText = 'OFFLINE (Safe Mode)'; payloadInst.className = 'text-red-400 font-bold'; }
      if (payloadShed) { payloadShed.innerText = 'TIER-2 ACTIVE (0W)'; payloadShed.className = 'text-red-400 font-bold'; }
      payloadCard.className = 'p-2.5 rounded-lg bg-red-950/30 border border-red-500/40 space-y-1.5 transition-all';
    } else if (isLoadShedding) {
      if (payloadTag) { payloadTag.innerText = 'SHED (-35% / 91W)'; payloadTag.className = 'text-amber-400 font-bold text-[9px] bg-amber-950/60 px-1.5 py-0.5 rounded border border-amber-500/40 animate-pulse'; }
      if (payloadInst) { payloadInst.innerText = 'PARTIAL (91W Active)'; payloadInst.className = 'text-amber-300 font-bold'; }
      if (payloadShed) { payloadShed.innerText = 'TIER-1 ACTIVE (-35%)'; payloadShed.className = 'text-amber-400 font-bold'; }
      payloadCard.className = 'p-2.5 rounded-lg bg-amber-950/30 border border-amber-500/40 space-y-1.5 transition-all';
    } else {
      if (payloadTag) { payloadTag.innerText = '100% NOMINAL'; payloadTag.className = 'text-emerald-400 font-bold text-[9px] bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-500/30'; }
      if (payloadInst) { payloadInst.innerText = 'ONLINE (140W)'; payloadInst.className = 'text-purple-300 font-bold'; }
      if (payloadShed) { payloadShed.innerText = 'TIER-0 (None)'; payloadShed.className = 'text-emerald-300 font-bold'; }
      payloadCard.className = 'p-2.5 rounded-lg bg-black/60 border border-purple-500/30 space-y-1.5 transition-all';
    }
  }
}

function openComponentMatrixModal() {
  const m = document.getElementById('component-matrix-modal');
  if (m) m.classList.remove('hidden');
}

function closeComponentMatrixModal() {
  const m = document.getElementById('component-matrix-modal');
  if (m) m.classList.add('hidden');
}

function switchMatrixTab(tabName) {
  const cTab = document.getElementById('tab-content-components');
  const fTab = document.getElementById('tab-content-faults');
  const mTab = document.getElementById('tab-content-matrix');
  const bComp = document.getElementById('tab-btn-components');
  const bFault = document.getElementById('tab-btn-faults');
  const bMat = document.getElementById('tab-btn-matrix');

  if (cTab) cTab.classList.add('hidden');
  if (fTab) fTab.classList.add('hidden');
  if (mTab) mTab.classList.add('hidden');

  if (bComp) bComp.className = 'px-3 py-1.5 rounded-lg bg-black/40 border border-white/10 text-gray-400 hover:text-white transition';
  if (bFault) bFault.className = 'px-3 py-1.5 rounded-lg bg-black/40 border border-white/10 text-gray-400 hover:text-white transition';
  if (bMat) bMat.className = 'px-3 py-1.5 rounded-lg bg-black/40 border border-white/10 text-gray-400 hover:text-white transition';

  if (tabName === 'components' && cTab && bComp) {
    cTab.classList.remove('hidden');
    bComp.className = 'px-3 py-1.5 rounded-lg bg-cyan-950 border border-cyan-400 text-cyan-300 font-bold transition';
  } else if (tabName === 'faults' && fTab && bFault) {
    fTab.classList.remove('hidden');
    bFault.className = 'px-3 py-1.5 rounded-lg bg-amber-950 border border-amber-400 text-amber-300 font-bold transition';
  } else if (tabName === 'matrix' && mTab && bMat) {
    mTab.classList.remove('hidden');
    bMat.className = 'px-3 py-1.5 rounded-lg bg-purple-950 border border-purple-400 text-purple-300 font-bold transition';
  }
}

// ==============================================================================
// 7. WebSocket Live Telemetry Connection
// ==============================================================================
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    safeSetText('downlink-status', '2.0 Hz LIVE');
    addTerminalLog('WebSocket downlink connected. Telemetry synchronized.');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      
      SatState.voltage = data.voltage !== undefined ? data.voltage : SatState.voltage;
      SatState.current = data.current !== undefined ? data.current : SatState.current;
      SatState.temp = data.temperature !== undefined ? data.temperature : SatState.temp;
      SatState.esr = data.impedance_proxy !== undefined ? data.impedance_proxy : SatState.esr;
      SatState.soc = data.soc !== undefined ? data.soc : SatState.soc;
      SatState.power = data.power_watts !== undefined ? data.power_watts : SatState.power;
      SatState.dtdt = data.thermal_gradient !== undefined ? data.thermal_gradient : SatState.dtdt;

      SatState.p_ensemble = data.p_ensemble || 0.0;
      SatState.p_rf = data.p_rf || 0.0;
      SatState.p_xgboost = data.p_xgboost || 0.0;
      SatState.p_extra_trees = data.p_extra_trees || 0.0;
      SatState.primary_fault = data.primary_fault || "NOMINAL";
      SatState.final_severity = data.final_severity || "NOMINAL";
      SatState.risk_score = data.risk_score || 0.05;
      SatState.rl_action_name = data.rl_action_name || "NOMINAL_MONITOR";
      SatState.rl_action_id = data.rl_action_id !== undefined ? data.rl_action_id : 0;
      SatState.dynamic_threshold = data.dynamic_threshold !== undefined ? data.dynamic_threshold : 0.70;
      SatState.safety_override_active = !!data.safety_override_active;
      SatState.requires_human_approval = !!data.requires_human_approval;

      // Check Anomaly State Transitions
      const isCurrentlyAnomaly = (SatState.p_ensemble >= 0.30) || (SatState.primary_fault !== "NOMINAL") || (SatState.final_severity !== "NOMINAL") || SatState.safety_override_active;

      if (isCurrentlyAnomaly) {
        SatState.wasInAnomaly = true;
        if (data.rl_action_name && data.rl_action_name !== "NOMINAL_MONITOR") {
          SatState.lastMitigatingRLAction = data.rl_action_name;
        }
        if (data.primary_fault && data.primary_fault !== "NOMINAL") {
          SatState.lastMitigatedFault = data.primary_fault;
        }
      } else {
        // Telemetry is NOMINAL: Clear all visual anomaly flags immediately
        SatState.faultThermalRunaway = false;
        SatState.faultMicroShort = false;
        SatState.faultDeepUndervolt = false;
        SatState.faultImpedanceSurge = false;
        SatState.faultSensorGlitch = false;

        if (cellThermalMaterial) {
          cellThermalMaterial.color.setHex(0x3b82f6);
          cellThermalMaterial.emissive.setHex(0x001133);
        }
        if (thermalPointLight) thermalPointLight.intensity = 0;
        if (particleSystemThermal) particleSystemThermal.material.opacity = 0;

        if (SatState.wasInAnomaly) {
          SatState.wasInAnomaly = false;

          const appliedAction = SatState.lastMitigatingRLAction || "LOAD_SHEDDING";
          const previousFault = SatState.lastMitigatedFault || "THERMAL_RUNAWAY";
          const timeStr = new Date().toLocaleTimeString();
          SatState.lastRecoveryTimestamp = timeStr;

          // Stop alarm & play positive recovery chime
          audioSys.stopCriticalAlarm();
          audioSys.playRecoveryChime();

          // Log to terminal
          addTerminalLog(`✅ SATELLITE RECOVERED TO NOMINAL // Applied RL Policy: ${appliedAction} mitigated ${previousFault}!`, false, false, true);

          // Show prominent recovery banner
          showRecoveryBanner(appliedAction, previousFault, timeStr);
        }
      }

      safeSetText('eclipse-status', data.is_eclipse === 1 ? 'ECLIPSE (COOLING)' : 'DIRECT SUNLIGHT');

      // AI Agent Output
      if (data.ai_reasoning && data.ai_reasoning.agent_message) {
        safeSetText('ai-agent-text', data.ai_reasoning.agent_message);
        safeSetText('ai-rag-citation', `📖 Citation: ${data.rag_citation || 'NASA / ESA ECSS Safety Flight Standards'}`);
      }

      // Counterfactual updates
      if (data.counterfactuals && Array.isArray(data.counterfactuals)) {
        data.counterfactuals.forEach((cf) => {
          const id = cf.action_id;
          safeSetText(`cf-t${id}`, `${cf.projected_temp_60s ? cf.projected_temp_60s.toFixed(1) : '22.0'}°C`);
          safeSetText(`cf-soc${id}`, `${cf.projected_soc_60s ? (cf.projected_soc_60s * 100).toFixed(1) : '85.0'}%`);
          safeSetText(`cf-safe${id}`, `${cf.safety_score ? cf.safety_score.toFixed(2) : '1.00'}`);
          
          const card = document.getElementById(`cf-card-${id}`);
          const status = document.getElementById(`cf-status${id}`);
          if (card && status) {
            if (cf.is_recommended) {
              card.className = 'p-2 rounded-lg bg-black/45 border border-emerald-500/80 space-y-1';
              status.innerText = 'AI RECOMMENDED';
              status.className = 'text-emerald-400 text-[9px] font-bold';
              safeSetText('best-action-badge', `ACTION ${id}: ${cf.action_name}`);
            } else {
              card.className = 'p-2 rounded-lg bg-black/45 border border-white/10 space-y-1';
              status.innerText = cf.action_name;
              status.className = 'text-gray-500 text-[9px]';
            }
          }
        });
      }

      // Update Chart.js data
      const timeLabel = new Date((data.timestamp || Date.now() / 1000) * 1000).toLocaleTimeString();
      if (viChart) appendChartData(viChart, timeLabel, [SatState.voltage, SatState.current]);
      if (tpChart) appendChartData(tpChart, timeLabel, [SatState.temp, SatState.p_ensemble]);

    } catch (e) {
      console.error("Error parsing telemetry frame:", e);
    }
  };

  ws.onclose = () => {
    safeSetText('downlink-status', 'RECONNECTING...');
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = () => ws.close();
}

// ==============================================================================
// 8. Event Listeners, Operator Action Logger & Fault Injection API
// ==============================================================================
function sendOperatorAction(action, details = '', metadata = {}) {
  fetch('/api/logs/action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action: action,
      details: details,
      operator: 'MissionCommander',
      metadata: metadata
    })
  }).catch(() => {});
}

function triggerFault(faultType) {
  fetch('/api/inject_fault', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fault_type: faultType, duration_sec: 25.0 })
  }).then(() => {
    SatState.wasInAnomaly = true;
    SatState.lastMitigatedFault = faultType.toUpperCase();

    // Map initial mitigation expectations
    if (faultType === 'thermal_runaway') {
      SatState.faultThermalRunaway = true;
      SatState.lastMitigatingRLAction = "LOAD_SHEDDING";
    }
    if (faultType === 'internal_short') {
      SatState.faultMicroShort = true;
      SatState.lastMitigatingRLAction = "TRIGGER_SAFE_MODE";
    }
    if (faultType === 'undervoltage') {
      SatState.faultDeepUndervolt = true;
      SatState.lastMitigatingRLAction = "LOAD_SHEDDING";
    }
    if (faultType === 'high_impedance') {
      SatState.faultImpedanceSurge = true;
      SatState.lastMitigatingRLAction = "HIGH_SENSITIVITY_PREARM";
    }
    if (faultType === 'sensor_fault') {
      SatState.faultSensorGlitch = true;
      SatState.lastMitigatingRLAction = "HIGH_SENSITIVITY_PREARM";
    }

    addTerminalLog(`INJECTED FAULT: "${faultType.toUpperCase()}" active for 25s!`, true);
    sendOperatorAction(`INJECT_FAULT_${faultType.toUpperCase()}`, `Duration: 25s`, { fault_type: faultType });
    if (AuditLogState.isModalOpen) fetchAndRenderAuditLogs();
  }).catch(() => {});
}

function clearFaults() {
  fetch('/api/clear_fault', { method: 'POST' }).then(() => {
    const prevFault = SatState.lastMitigatedFault || "THERMAL_RUNAWAY";
    const appliedAction = SatState.lastMitigatingRLAction || "LOAD_SHEDDING";

    // Immediate complete reset of all state & visual flags
    SatState.faultThermalRunaway = false;
    SatState.faultMicroShort = false;
    SatState.faultDeepUndervolt = false;
    SatState.faultImpedanceSurge = false;
    SatState.faultSensorGlitch = false;
    SatState.wasInAnomaly = false;
    SatState.primary_fault = "NOMINAL";
    SatState.final_severity = "NOMINAL";
    SatState.p_ensemble = 0.012;
    SatState.safety_override_active = false;

    // Reset 3D visuals immediately
    if (cellThermalMaterial) {
      cellThermalMaterial.color.setHex(0x3b82f6);
      cellThermalMaterial.emissive.setHex(0x001133);
    }
    if (thermalPointLight) thermalPointLight.intensity = 0;
    if (particleSystemThermal) particleSystemThermal.material.opacity = 0;
    if (arcLineMesh) arcLineMesh.material.opacity = 0;
    if (arcFlashLight) arcFlashLight.intensity = 0;
    if (satelliteGroup) {
      satelliteGroup.rotation.x = 0;
      satelliteGroup.rotation.z = 0;
    }
    if (highGainDish) {
      highGainDish.rotation.set(0, 0, 0);
    }

    const timeStr = new Date().toLocaleTimeString();
    SatState.lastRecoveryTimestamp = timeStr;

    audioSys.stopCriticalAlarm();
    audioSys.playRecoveryChime();

    addTerminalLog(`✅ SATELLITE RECOVERED TO NOMINAL // Applied RL Policy: ${appliedAction} mitigated ${prevFault}!`, false, false, true);
    showRecoveryBanner(appliedAction, prevFault, timeStr, "• Subsystem stabilized and attitude recovered.");
    updateHUD();

    sendOperatorAction('CLEAR_ALL_FAULTS', `Mitigated fault ${prevFault} via ${appliedAction}`);
    if (AuditLogState.isModalOpen) fetchAndRenderAuditLogs();
  }).catch(() => {});
}

function manualAuthorizeMitigation() {
  fetch('/api/authorize_mitigation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ authorized: true, autopilot: SatState.autopilotMode })
  }).then(() => {
    addTerminalLog('OPERATOR CONFIRMATION: Mitigation command authorized.', false, false, true);
    const authBtn = document.getElementById('btn-operator-auth');
    if (authBtn) authBtn.classList.add('hidden');
    sendOperatorAction('AUTHORIZE_MITIGATION_MANUAL_GATE', 'Command approved for execution');
    if (AuditLogState.isModalOpen) fetchAndRenderAuditLogs();
  }).catch(() => {});
}

function setupEventListeners() {
  const btnAudio = document.getElementById('btn-audio-toggle');
  if (btnAudio) {
    btnAudio.addEventListener('click', () => {
      const muted = audioSys.toggleMute();
      safeSetText('audio-icon', muted ? '🔇' : '🔊');
      safeSetText('audio-status', muted ? 'AUDIO OFF' : 'AUDIO ON');
      addTerminalLog(`Audio sound FX ${muted ? 'muted' : 'enabled'}.`);
      sendOperatorAction('TOGGLE_AUDIO', `Sound FX set to ${muted ? 'MUTED' : 'ENABLED'}`);
    });
  }

  const btnAutopilot = document.getElementById('btn-autopilot-toggle');
  if (btnAutopilot) {
    btnAutopilot.addEventListener('click', () => {
      SatState.autopilotMode = !SatState.autopilotMode;
      safeSetText('autopilot-status', SatState.autopilotMode ? 'AUTO-PILOT' : 'MANUAL GATE');
      fetch('/api/authorize_mitigation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ authorized: false, autopilot: SatState.autopilotMode })
      }).catch(() => {});
      addTerminalLog(`Mitigation mode: ${SatState.autopilotMode ? 'Autonomous Auto-Pilot' : 'Manual Operator Gate'}.`);
      sendOperatorAction('TOGGLE_AUTOPILOT_MODE', `Mode switched to ${SatState.autopilotMode ? 'Autonomous Auto-Pilot' : 'Manual Operator Gate'}`);
    });
  }

  const btnReset = document.getElementById('btn-reset-nominal');
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      clearFaults();
    });
  }

  const btnOrbit = document.getElementById('view-orbit');
  const btnTherm = document.getElementById('view-thermal');
  const btnBatt = document.getElementById('view-battery');

  function setCameraView(mode) {
    [btnOrbit, btnTherm, btnBatt].forEach(b => {
      if (b) b.className = 'py-0.5 px-2 rounded bg-black/40 border border-gray-700 text-gray-400 text-[10px] font-mono-telemetry transition';
    });

    if (mode === 'orbit' && btnOrbit) {
      btnOrbit.className = 'py-0.5 px-2 rounded bg-cyan-950/70 border border-cyan-400 text-cyan-200 text-[10px] font-mono-telemetry font-bold';
      if (controls) controls.target.set(0, 0, 0);
      if (camera) camera.position.set(16, 9, 22);
      addTerminalLog('Camera: Orbit view perspective.');
      sendOperatorAction('CHANGE_CAMERA_VIEW', 'Orbit view active');
    } else if (mode === 'thermal' && btnTherm) {
      btnTherm.className = 'py-0.5 px-2 rounded bg-red-950/70 border border-red-400 text-red-200 text-[10px] font-mono-telemetry font-bold';
      if (controls) controls.target.set(0, 0.4, 2.0);
      if (camera) camera.position.set(4, 3, 7);
      addTerminalLog('Camera: IR Thermal mode.');
      sendOperatorAction('CHANGE_CAMERA_VIEW', 'IR Thermal mode active');
    } else if (mode === 'battery' && btnBatt) {
      btnBatt.className = 'py-0.5 px-2 rounded bg-cyan-950/70 border border-cyan-400 text-cyan-200 text-[10px] font-mono-telemetry font-bold';
      if (controls) controls.target.set(0, 0.3, 2.2);
      if (camera) camera.position.set(0.2, 0.8, 4.6);
      addTerminalLog('Camera: Battery module close-up.');
      sendOperatorAction('CHANGE_CAMERA_VIEW', 'Battery module close-up active');
    }
  }

  if (btnOrbit) btnOrbit.addEventListener('click', () => setCameraView('orbit'));
  if (btnTherm) btnTherm.addEventListener('click', () => setCameraView('thermal'));
  if (btnBatt) btnBatt.addEventListener('click', () => setCameraView('battery'));

  window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
  const container = document.getElementById('canvas-container');
  if (container && camera && renderer) {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
}

// ==============================================================================
// 9. Mission Audit Ledger & 10,000-Line FIFO Buffer Management & PDF Export
// ==============================================================================
const AuditLogState = {
  isModalOpen: false,
  isPaused: false,
  currentCategory: 'ALL',
  searchQuery: '',
  pollTimer: null,
  logs: [],
  stats: {},
  expandedSeqIds: new Set()
};

function openAuditLogModal() {
  const modal = document.getElementById('audit-logs-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  AuditLogState.isModalOpen = true;
  fetchAndRenderAuditLogs();

  if (AuditLogState.pollTimer) clearInterval(AuditLogState.pollTimer);
  AuditLogState.pollTimer = setInterval(() => {
    if (AuditLogState.isModalOpen && !AuditLogState.isPaused) {
      fetchAndRenderAuditLogs(false);
    }
  }, 1500);

  sendOperatorAction('OPEN_MISSION_AUDIT_CONSOLE', 'Opened 10k FIFO Mission Audit Ledger');
}

function closeAuditLogModal() {
  const modal = document.getElementById('audit-logs-modal');
  if (!modal) return;
  modal.classList.add('hidden');
  AuditLogState.isModalOpen = false;
  if (AuditLogState.pollTimer) {
    clearInterval(AuditLogState.pollTimer);
    AuditLogState.pollTimer = null;
  }
}

function setAuditCategoryFilter(cat) {
  AuditLogState.currentCategory = cat;
  const categories = ['all', 'action', 'predict', 'fault', 'mitigation', 'recovery', 'telemetry'];
  categories.forEach(c => {
    const btn = document.getElementById(`filter-btn-${c}`);
    if (btn) {
      btn.className = 'px-2.5 py-1 rounded-md bg-black/50 border border-white/10 text-gray-400 hover:text-white transition';
    }
  });

  const activeBtnId = cat === 'ALL' ? 'filter-btn-all' :
    cat === 'OPERATOR_ACTION' ? 'filter-btn-action' :
    cat === 'EARLY_PREDICTION' ? 'filter-btn-predict' :
    cat === 'FAULT_ANOMALY' ? 'filter-btn-fault' :
    cat === 'RL_MITIGATION' ? 'filter-btn-mitigation' :
    cat === 'RECOVERY_EVENT' ? 'filter-btn-recovery' : 'filter-btn-telemetry';

  const activeBtn = document.getElementById(activeBtnId);
  if (activeBtn) {
    activeBtn.className = 'px-2.5 py-1 rounded-md bg-cyan-950 border border-cyan-400 text-cyan-300 font-bold transition';
  }

  fetchAndRenderAuditLogs();
}

function filterAuditLogs() {
  const input = document.getElementById('audit-search-input');
  const clearBtn = document.getElementById('btn-clear-search');
  AuditLogState.searchQuery = input ? input.value.trim() : '';
  if (clearBtn) {
    if (AuditLogState.searchQuery) clearBtn.classList.remove('hidden');
    else clearBtn.classList.add('hidden');
  }
  fetchAndRenderAuditLogs();
}

function clearAuditSearch() {
  const input = document.getElementById('audit-search-input');
  if (input) input.value = '';
  filterAuditLogs();
}

function toggleAuditStreamPause() {
  AuditLogState.isPaused = !AuditLogState.isPaused;
  safeSetText('pause-log-icon', AuditLogState.isPaused ? '▶️' : '⏸️');
  safeSetText('pause-log-text', AuditLogState.isPaused ? 'RESUME STREAM' : 'PAUSE STREAM');
  const btn = document.getElementById('btn-pause-log-stream');
  if (btn) {
    btn.className = AuditLogState.isPaused ?
      'px-2.5 py-1 rounded-lg bg-amber-950/60 border border-amber-400 text-amber-300 text-[11px] transition flex items-center space-x-1' :
      'px-2.5 py-1 rounded-lg bg-black/50 border border-cyan-500/30 hover:border-cyan-400 text-cyan-300 text-[11px] transition flex items-center space-x-1';
  }
}

function refreshAuditLogs() {
  fetchAndRenderAuditLogs();
}

function clearAuditBuffer() {
  if (confirm("Are you sure you want to clear the 10,000-line Mission Audit Log buffer? This resets the ledger on disk.")) {
    fetch('/api/logs/clear', { method: 'POST' }).then(() => {
      addTerminalLog('Mission Audit buffer cleared.', false, false, true);
      fetchAndRenderAuditLogs();
    }).catch(() => {});
  }
}

function toggleLogDetails(seqId) {
  if (AuditLogState.expandedSeqIds.has(seqId)) {
    AuditLogState.expandedSeqIds.delete(seqId);
  } else {
    AuditLogState.expandedSeqIds.add(seqId);
  }
  renderAuditTableRows(AuditLogState.logs);
}

function fetchAndRenderAuditLogs(showLoading = false) {
  let url = `/api/logs/all?limit=300`;
  if (AuditLogState.currentCategory && AuditLogState.currentCategory !== 'ALL') {
    url += `&category=${encodeURIComponent(AuditLogState.currentCategory)}`;
  }
  if (AuditLogState.searchQuery) {
    url += `&search=${encodeURIComponent(AuditLogState.searchQuery)}`;
  }

  fetch(url)
    .then(res => res.json())
    .then(data => {
      AuditLogState.logs = data.logs || [];
      AuditLogState.stats = data.stats || {};
      
      // Update HUD & Modal KPI counters
      const bCount = AuditLogState.stats.buffer_count || AuditLogState.logs.length;
      const bMax = AuditLogState.stats.buffer_max_lines || 10000;
      const bPct = AuditLogState.stats.buffer_usage_pct || 0.0;
      
      safeSetText('header-log-count', `${bCount} / 10k`);
      safeSetText('audit-capacity-pill', `${bCount} / ${bMax} LINES (${bPct}%)`);
      safeSetText('kpi-total-logs', `${AuditLogState.stats.total_events_logged || bCount}`);
      safeSetText('kpi-operator-actions', `${AuditLogState.stats.operator_actions || 0}`);
      safeSetText('kpi-early-preds', `${AuditLogState.stats.early_predictions || 0}`);
      safeSetText('kpi-faults', `${AuditLogState.stats.faults_detected || 0}`);
      safeSetText('kpi-mitigations', `${AuditLogState.stats.mitigations_executed || 0}`);
      safeSetText('kpi-recoveries', `${AuditLogState.stats.recoveries || 0}`);
      safeSetText('audit-showing-count', `Showing ${AuditLogState.logs.length} of ${bCount} stored in FIFO buffer`);

      renderAuditTableRows(AuditLogState.logs);
    })
    .catch(err => {
      console.warn("Error fetching audit logs:", err);
    });
}

function renderAuditTableRows(logs) {
  const tbody = document.getElementById('audit-log-tbody');
  if (!tbody) return;

  if (!logs || logs.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" class="p-8 text-center text-gray-500 font-mono-telemetry">
          No structured log events match the current filter criteria.
        </td>
      </tr>
    `;
    return;
  }

  let html = '';
  logs.forEach(item => {
    const seq = item.seq_id || 0;
    const met = item.met || "T+00:00:00";
    const lvl = item.level || "INFO";
    const cat = item.event_type || "SYSTEM";
    const sub = item.subsystem || "EPS_CORE";
    const msg = item.message || "";
    const telem = item.telemetry || {};
    const models = item.models || {};
    const meta = item.metadata || {};
    const isExpanded = AuditLogState.expandedSeqIds.has(seq);

    // Color Badges
    let lvlBadge = 'bg-gray-800 text-gray-300 border-gray-600';
    if (lvl === 'CRITICAL' || lvl === 'EMERGENCY') lvlBadge = 'bg-red-950/80 text-red-300 border-red-500 animate-pulse';
    else if (lvl === 'FAULT' || lvl === 'WARNING') lvlBadge = 'bg-amber-950/80 text-amber-300 border-amber-500';
    else if (lvl === 'ACTION') lvlBadge = 'bg-cyan-950/80 text-cyan-300 border-cyan-400 font-bold';
    else if (lvl === 'PREDICT') lvlBadge = 'bg-orange-950/80 text-orange-300 border-orange-400 font-bold';
    else if (lvl === 'RL-POLICY') lvlBadge = 'bg-emerald-950/80 text-emerald-300 border-emerald-400 font-bold';
    else if (lvl === 'RECOVER') lvlBadge = 'bg-purple-950/80 text-purple-300 border-purple-400 font-bold';
    else if (lvl === 'SAFETY') lvlBadge = 'bg-red-950/60 text-red-200 border-red-400 font-bold';

    // Subsystem Badge
    let subBadge = 'text-gray-400';
    if (sub.includes('PCDU')) subBadge = 'text-cyan-300 font-bold';
    else if (sub.includes('BCR')) subBadge = 'text-amber-300 font-bold';
    else if (sub.includes('BATTERY')) subBadge = 'text-emerald-300 font-bold';
    else if (sub.includes('OPERATOR')) subBadge = 'text-cyan-400 font-bold';
    else if (sub.includes('AI')) subBadge = 'text-purple-300 font-bold';
    else if (sub.includes('PREDICT')) subBadge = 'text-orange-300 font-bold';

    // Telemetry Snapshot String
    const telemPills = [];
    if (telem.voltage !== undefined) telemPills.push(`<span class="text-cyan-300">${telem.voltage.toFixed(2)}V</span>`);
    if (telem.current !== undefined) telemPills.push(`<span class="text-amber-300">${telem.current.toFixed(2)}A</span>`);
    if (telem.temperature !== undefined) telemPills.push(`<span class="text-purple-300">${telem.temperature.toFixed(1)}°C</span>`);
    if (telem.soc !== undefined) {
      const socVal = telem.soc <= 1.0 ? telem.soc * 100 : telem.soc;
      telemPills.push(`<span class="text-emerald-300">${socVal.toFixed(0)}%</span>`);
    }
    if (models.p_ensemble !== undefined) telemPills.push(`<span class="text-red-300">P_ens:${models.p_ensemble.toFixed(2)}</span>`);
    const telemStr = telemPills.length > 0 ? telemPills.join(' • ') : '<span class="text-gray-600">N/A</span>';

    html += `
      <tr class="hover:bg-white/5 cursor-pointer transition ${isExpanded ? 'bg-cyan-950/20' : ''}" onclick="toggleLogDetails(${seq})">
        <td class="p-2.5 font-bold text-gray-400">#${seq}</td>
        <td class="p-2.5 text-cyan-300">${met}</td>
        <td class="p-2.5 text-center">
          <span class="px-2 py-0.5 text-[9px] font-bold rounded border ${lvlBadge}">
            ${lvl}
          </span>
        </td>
        <td class="p-2.5 font-semibold text-gray-200">${cat}</td>
        <td class="p-2.5 ${subBadge}">${sub}</td>
        <td class="p-2.5 text-[10px]">${telemStr}</td>
        <td class="p-2.5 text-gray-200 leading-snug">${msg}</td>
        <td class="p-2.5 text-center text-cyan-400 font-bold">
          ${isExpanded ? '▲' : '▼'}
        </td>
      </tr>
    `;

    if (isExpanded) {
      html += `
        <tr class="bg-black/80 border-b border-cyan-500/30">
          <td colspan="8" class="p-3.5 space-y-2">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-[10px]">
              
              <!-- Telemetry Detailed Metrics -->
              <div class="p-2.5 rounded-lg bg-black/60 border border-cyan-500/30 space-y-1">
                <div class="text-cyan-300 font-bold uppercase pb-1 border-b border-white/10">Full Telemetry Snapshot</div>
                <div class="space-y-0.5 text-gray-300">
                  <div>• Voltage: <strong class="text-white">${telem.voltage !== undefined ? telem.voltage.toFixed(3) : 'N/A'} V</strong></div>
                  <div>• Current: <strong class="text-white">${telem.current !== undefined ? telem.current.toFixed(3) : 'N/A'} A</strong></div>
                  <div>• Temperature: <strong class="text-white">${telem.temperature !== undefined ? telem.temperature.toFixed(2) : 'N/A'} °C</strong></div>
                  <div>• State of Charge (SOC): <strong class="text-white">${telem.soc !== undefined ? (telem.soc * 100).toFixed(1) : 'N/A'} %</strong></div>
                  <div>• Power: <strong class="text-white">${telem.power_watts !== undefined ? telem.power_watts.toFixed(2) : 'N/A'} W</strong></div>
                  <div>• Impedance Proxy: <strong class="text-white">${telem.impedance_proxy !== undefined ? telem.impedance_proxy.toFixed(4) : 'N/A'} Ω</strong></div>
                </div>
              </div>

              <!-- Multi-Model Consensus & Thresholds -->
              <div class="p-2.5 rounded-lg bg-black/60 border border-amber-500/30 space-y-1">
                <div class="text-amber-300 font-bold uppercase pb-1 border-b border-white/10">AI & Ensemble Consensus</div>
                <div class="space-y-0.5 text-gray-300">
                  <div>• P(Ensemble): <strong class="text-white">${models.p_ensemble !== undefined ? models.p_ensemble.toFixed(3) : 'N/A'}</strong></div>
                  <div>• Random Forest (35%): <strong class="text-white">${models.p_rf !== undefined ? models.p_rf.toFixed(3) : 'N/A'}</strong></div>
                  <div>• XGBoost (40%): <strong class="text-white">${models.p_xgboost !== undefined ? models.p_xgboost.toFixed(3) : 'N/A'}</strong></div>
                  <div>• Extra Trees (25%): <strong class="text-white">${models.p_extra_trees !== undefined ? models.p_extra_trees.toFixed(3) : 'N/A'}</strong></div>
                  <div>• Dynamic Threshold (Tau): <strong class="text-white">${models.dynamic_threshold !== undefined ? models.dynamic_threshold.toFixed(2) : '0.70'}</strong></div>
                </div>
              </div>

              <!-- Metadata & ISO Timestamps -->
              <div class="p-2.5 rounded-lg bg-black/60 border border-purple-500/30 space-y-1">
                <div class="text-purple-300 font-bold uppercase pb-1 border-b border-white/10">Event Metadata & Flight Standards</div>
                <div class="space-y-0.5 text-gray-300">
                  <div>• ISO Timestamp: <strong class="text-white">${item.timestamp || 'N/A'}</strong></div>
                  <div>• UNIX Timestamp: <strong class="text-white">${item.unix_ts || 'N/A'}</strong></div>
                  <div>• Standard Compliance: <strong class="text-emerald-300">NASA-HDBK-4008 / ECSS</strong></div>
                  <div>• Metadata Payload: <strong class="text-gray-300">${JSON.stringify(meta)}</strong></div>
                </div>
              </div>

            </div>
          </td>
        </tr>
      `;
    }
  });

  tbody.innerHTML = html;
}

function exportAuditPDF() {
  const exportBtn = document.getElementById('btn-export-pdf');
  const icon = document.getElementById('export-pdf-icon');
  const text = document.getElementById('export-pdf-text');

  if (icon) icon.innerText = '⏳';
  if (text) text.innerText = 'GENERATING PDF...';
  if (exportBtn) exportBtn.disabled = true;

  fetch('/api/logs/export_pdf')
    .then(response => {
      if (!response.ok) throw new Error("Failed to generate PDF");
      return response.blob();
    })
    .then(blob => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `AERO_GUARD_MISSION_AUDIT_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      audioSys.playRecoveryChime();
      addTerminalLog('Mission Audit PDF report generated and downloaded.', false, false, true);
    })
    .catch(err => {
      alert("Error generating PDF: " + err.message);
    })
    .finally(() => {
      if (icon) icon.innerText = '📥';
      if (text) text.innerText = 'EXPORT AUDIT PDF';
      if (exportBtn) exportBtn.disabled = false;
    });
}

// ==============================================================================
// 10. Main Render Loop & Initialization
// ==============================================================================
function animateLoop(now) {
  requestAnimationFrame(animateLoop);

  const delta = Math.min(0.1, (now - lastFrameTime) / 1000);
  lastFrameTime = now;

  frameCount++;
  fpsTimer += delta;
  if (fpsTimer >= 1.0) {
    safeSetText('fps-counter', `${frameCount} FPS`);
    frameCount = 0;
    fpsTimer = 0;
  }

  updatePhysics(delta);
  updateHUD();

  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

window.onload = function() {
  initCharts();
  initSceneEnvironment();
  buildDetailedSatellite();
  setupEventListeners();
  connectWebSocket();
  fetchAndRenderAuditLogs(false);
  addTerminalLog('AERO-GUARD Mission Control Dashboard online. Telemetry sync active.');
  animateLoop(performance.now());
};

