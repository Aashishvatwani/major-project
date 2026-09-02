/*
  ==============================================================================
  Satellite Telemetry Anomaly Detection — Hardware-in-the-Loop (HITL) Controller
  Target Platform: Arduino Uno / Nano / Mega (ATmega328P / ATmega2560)
  
  Controls:
  - Digital Pin 13: Built-in LED & Actuation Relay
    * State 0 (Nominal): LED OFF
    * State 1 (ML/RL Anomaly Alert): LED SOLID ON
    * Severity 4 (Critical Safety Override): High-frequency STROBE (50ms)
    * Action 2 (Load Shedding Warning): Moderate PULSE (250ms)
  
  Serial Protocol:
  - Baud Rate: 115200 baud
  - Packet Format: <ALERT_STATE,SEVERITY_CODE,RL_ACTION>
  - Telemetry Response: <ACK,PIN13_STATE,SEVERITY,RL_ACTION,UPTIME_MS>
  ==============================================================================
*/

const int PIN_LED = 13;       // Built-in LED (Digital Pin 13)
const int PIN_RELAY = 12;     // Optional auxiliary power relay / buzzer
const long BAUD_RATE = 115200;

// Internal state variables
int alertState = 0;           // 0: OFF, 1: ON
int severityCode = 0;         // 0: Normal, 1: Elevated, 2: Warning, 3: Critical, 4: Override
int rlAction = 0;             // 0: Nominal, 1: Prearm, 2: LoadShed, 3: SafeMode

unsigned long lastBlinkTime = 0;
bool strobePinState = LOW;
String inputBuffer = "";

void setup() {
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_RELAY, OUTPUT);
  
  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_RELAY, LOW);
  
  Serial.begin(BAUD_RATE);
  inputBuffer.reserve(64);
  
  // Quick startup flash (3 pulses) to signal ready
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_LED, HIGH);
    delay(80);
    digitalWrite(PIN_LED, LOW);
    delay(80);
  }
}

void loop() {
  // Read Serial Commands non-blocking
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += inChar;
      if (inputBuffer.length() > 60) {
        inputBuffer = ""; // Overflow guard
      }
    }
  }

  // Actuation logic for Digital Pin 13
  updateHardwareActuation();
}

void processCommand(String cmd) {
  cmd.trim();
  if (cmd.startsWith("<") && cmd.endsWith(">")) {
    String payload = cmd.substring(1, cmd.length() - 1);
    
    int firstComma = payload.indexOf(',');
    int secondComma = payload.indexOf(',', firstComma + 1);
    
    if (firstComma > 0) {
      alertState = payload.substring(0, firstComma).toInt();
      
      if (secondComma > 0) {
        severityCode = payload.substring(firstComma + 1, secondComma).toInt();
        rlAction = payload.substring(secondComma + 1).toInt();
      } else {
        severityCode = payload.substring(firstComma + 1).toInt();
        rlAction = 0;
      }
      
      // Send Acknowledgement telemetry
      Serial.print("<ACK,");
      Serial.print(alertState);
      Serial.print(",");
      Serial.print(severityCode);
      Serial.print(",");
      Serial.print(rlAction);
      Serial.print(",");
      Serial.print(millis());
      Serial.println(">");
    }
  }
}

void updateHardwareActuation() {
  unsigned long currentMillis = millis();

  if (severityCode == 4) {
    // CRITICAL SAFETY OVERRIDE: Fast 50ms Emergency Strobe
    if (currentMillis - lastBlinkTime >= 50) {
      lastBlinkTime = currentMillis;
      strobePinState = !strobePinState;
      digitalWrite(PIN_LED, strobePinState);
      digitalWrite(PIN_RELAY, HIGH);
    }
  } 
  else if (rlAction == 2 && alertState == 1) {
    // LOAD SHEDDING / WARNING: Moderate 250ms Alert Pulse
    if (currentMillis - lastBlinkTime >= 250) {
      lastBlinkTime = currentMillis;
      strobePinState = !strobePinState;
      digitalWrite(PIN_LED, strobePinState);
    }
  } 
  else if (alertState == 1) {
    // SOLID ON (ML Anomaly Alert / Safe Mode)
    digitalWrite(PIN_LED, HIGH);
    digitalWrite(PIN_RELAY, HIGH);
  } 
  else {
    // NORMAL / SAFE: LED OFF
    digitalWrite(PIN_LED, LOW);
    digitalWrite(PIN_RELAY, LOW);
  }
}
