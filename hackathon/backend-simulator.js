const { Server } = require("socket.io");
const io = new Server(3001, { cors: { origin: "*" } });

console.log("Infant Signal Simulator running on port 3001...");

let time = 0;

// Emits data every 1 second
setInterval(() => {
  time += 1;
  
  // Base metrics with slight sinusoidal physiological variance (respiratory sinus arrhythmia)
  // Infant ranges: BPM 120-160, SysBP ~80, DiaBP ~45, SpO2 95-100%
  const bpm = 140 + Math.sin(time / 2) * 5 + (Math.random() * 2 - 1); 
  const sysBP = 85 + Math.sin(time / 3) * 3;
  const diaBP = 45 + Math.sin(time / 3) * 2;
  const spo2 = 98 + (Math.random() > 0.8 ? 1 : 0); // Mostly 98, occasionally 99
  
  io.emit("vitals", {
    timestamp: Date.now(),
    bpm: Math.round(bpm),
    bp: { systolic: Math.round(sysBP), diastolic: Math.round(diaBP) },
    spo2: Math.round(spo2)
  });
}, 1000);