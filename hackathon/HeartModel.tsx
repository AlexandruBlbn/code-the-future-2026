import React, { useState, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
// import { getCardiacFrame } from '../lib/cardiacAnimator'; // Uncomment and point to your math file

export function HeartModel({ bpm = 140, lvGeometry, rvGeometry }: { bpm?: number, lvGeometry: any, rvGeometry: any }) {
  const [isAnimating, setIsAnimating] = useState(true);
  
  // Keep track of accumulated time so pausing doesn't instantly reset the heart to frame 0
  const accumulatedTime = useRef(0);
  
  const lvRef = useRef<THREE.Mesh>(null);
  const rvRef = useRef<THREE.Mesh>(null);

  useFrame((state, delta) => {
    if (isAnimating) {
      accumulatedTime.current += delta;
    }
    
    // Convert accumulated time to a 0-1 cycle based on infant BPM
    const t = (accumulatedTime.current * bpm / 60) % 1.0;
    
    /* Uncomment once your getCardiacFrame is imported
    const frame = getCardiacFrame(t);
    
    if (lvRef.current) {
      lvRef.current.scale.set(...frame.leftVentricle.scale);
      lvRef.current.rotation.set(...frame.leftVentricle.rotation);
    }
    if (rvRef.current) {
      rvRef.current.scale.set(...frame.rightVentricle.scale);
      rvRef.current.rotation.set(...frame.rightVentricle.rotation);
    }
    */
  });

  return (
    <>
      {/* UI Overlay */}
      <div style={{ position: 'absolute', zIndex: 10, top: 20, left: 20, color: 'white' }}>
        <label style={{ cursor: 'pointer', fontFamily: 'sans-serif' }}>
          <input 
            type="checkbox" 
            checked={isAnimating} 
            onChange={(e) => setIsAnimating(e.target.checked)} 
            style={{ marginRight: '8px' }}
          />
          Enable Heart Pumping
        </label>
      </div>

      <mesh ref={lvRef} castShadow receiveShadow geometry={lvGeometry}>
        <meshStandardMaterial attach="material" color="#ff4444" roughness={0.4} />
      </mesh>
      
      <mesh ref={rvRef} castShadow receiveShadow geometry={rvGeometry}>
        <meshStandardMaterial attach="material" color="#4444ff" roughness={0.4} />
      </mesh>
    </>
  );
}