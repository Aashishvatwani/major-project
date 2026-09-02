/*
  AERO-GUARD // Structured Mission Control Dashboard & 3D Spacecraft Digital Twin v3.4
  Features:
  1. Three.js PBR 3D Spacecraft inside dedicated Digital Twin Card container.
  2. Dual Real-time Chart.js telemetry charts (Voltage/Current and Temp/P(Anomaly)).
  3. Live AI Agent Flight Diagnostic Terminal with RAG Flight Regulations.
  4. Real-time Digital Twin Counterfactual Decision Matrix (60s forward projection).
  5. Multi-Model ML Ensemble Consensus Breakdown.
  6. Hardware HITL Arduino Uno Pin 13 LED Fixture Actuator.
  7. Authentic Web Audio API Synthesizer (beeps, arc discharge, venting hiss, klaxons).
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

function addTerminalLog(msg, isError = false, isWarn = false) {
  const ticker = document.getElementById('terminal-ticker');
  if (!ticker) return;
  const now = new Date().toISOString().substring(11, 19);
  let color = 'text-gray-300';
  if (isError) color = 'text-red-400 font-bold';
  else if (isWarn) color = 'text-amber-300 font-semibold';
  
  ticker.innerHTML = `<span class="${color}">[${now}] ${msg}</span>`;
  audioSys.playBeep(isError ? 440 : 1200, 0.05, 'square');
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
  safety_override_active: false,
  requires_human_approval: false,
  autopilotMode: true,

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

  const isThermalRunaway = SatState.faultThermalRunaway || SatState.primary_fault === "THERMAL_RUNAWAY";
  const isMicroShort = SatState.faultMicroShort || SatState.primary_fault === "INTERNAL_SHORT";
  const isUndervolt = SatState.faultDeepUndervolt || SatState.primary_fault === "UNDERVOLTAGE";
  const isSensorGlitch = SatState.faultSensorGlitch || SatState.primary_fault === "SENSOR_FAULT";

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
    if (thermalPointLight) thermalPointLight.intensity = Math.max(0, thermalPointLight.intensity - delta * 2);
    if (particleSystemThermal) particleSystemThermal.material.opacity = Math.max(0, particleSystemThermal.material.opacity - delta);
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

  // 3. 🔋 Deep Undervoltage Tumble
  if (isUndervolt && satelliteGroup) {
    satelliteGroup.rotation.x += delta * 0.18;
    satelliteGroup.rotation.z += delta * 0.12;
  }

  // 4. 📡 Sensor Glitch Dish Jitter
  if (isSensorGlitch && highGainDish) {
    highGainDish.rotation.z = Math.sin(SatState.glitchPhase * 0.8) * 0.4;
    highGainDish.rotation.y = Math.cos(SatState.glitchPhase * 0.5) * 0.3;
  } else if (highGainDish) {
    highGainDish.rotation.z = 0;
    highGainDish.rotation.y = 0;
  }

  // Normal orbital rotation
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
      SatState.safety_override_active = !!data.safety_override_active;
      SatState.requires_human_approval = !!data.requires_human_approval;

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
// 8. Event Listeners & Fault Injection API
// ==============================================================================
function triggerFault(faultType) {
  fetch('/api/inject_fault', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fault_type: faultType, duration_sec: 25.0 })
  }).then(() => {
    addTerminalLog(`INJECTED FAULT: "${faultType.toUpperCase()}" active for 25s!`, true);
    if (faultType === 'thermal_runaway') SatState.faultThermalRunaway = true;
    if (faultType === 'internal_short') SatState.faultMicroShort = true;
    if (faultType === 'undervoltage') SatState.faultDeepUndervolt = true;
    if (faultType === 'high_impedance') SatState.faultImpedanceSurge = true;
    if (faultType === 'sensor_fault') SatState.faultSensorGlitch = true;
  }).catch(() => {});
}

function clearFaults() {
  fetch('/api/clear_fault', { method: 'POST' }).then(() => {
    SatState.faultThermalRunaway = false;
    SatState.faultMicroShort = false;
    SatState.faultDeepUndervolt = false;
    SatState.faultImpedanceSurge = false;
    SatState.faultSensorGlitch = false;
    if (satelliteGroup) satelliteGroup.rotation.set(0, 0, 0);
    addTerminalLog('All injected faults cleared. Returning to nominal orbit.');
  }).catch(() => {});
}

function manualAuthorizeMitigation() {
  fetch('/api/authorize_mitigation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ authorized: true, autopilot: SatState.autopilotMode })
  }).then(() => {
    addTerminalLog('OPERATOR CONFIRMATION: Mitigation command authorized.');
    const authBtn = document.getElementById('btn-operator-auth');
    if (authBtn) authBtn.classList.add('hidden');
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
    } else if (mode === 'thermal' && btnTherm) {
      btnTherm.className = 'py-0.5 px-2 rounded bg-red-950/70 border border-red-400 text-red-200 text-[10px] font-mono-telemetry font-bold';
      if (controls) controls.target.set(0, 0.4, 2.0);
      if (camera) camera.position.set(4, 3, 7);
      addTerminalLog('Camera: IR Thermal mode.');
    } else if (mode === 'battery' && btnBatt) {
      btnBatt.className = 'py-0.5 px-2 rounded bg-cyan-950/70 border border-cyan-400 text-cyan-200 text-[10px] font-mono-telemetry font-bold';
      if (controls) controls.target.set(0, 0.3, 2.2);
      if (camera) camera.position.set(0.2, 0.8, 4.6);
      addTerminalLog('Camera: Battery module close-up.');
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
// 9. Main Render Loop & Initialization
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
  addTerminalLog('AERO-GUARD Mission Control Dashboard online. Telemetry sync active.');
  animateLoop(performance.now());
};
