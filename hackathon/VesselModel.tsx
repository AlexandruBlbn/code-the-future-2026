
import React, { useRef, useState, useEffect } from 'react';
import * as THREE from 'three';
import { ThreeEvent } from '@react-three/fiber';

export function VesselModel({ geometry }: { geometry: any }) {
  // Canvas for painting the localized stenosis regions
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const textureRef = useRef<THREE.CanvasTexture | null>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const [textureInitialized, setTextureInitialized] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'white'; // White = healthy vessel (no displacement)
      ctx.fillRect(0, 0, 512, 512);
    }
    textureRef.current = new THREE.CanvasTexture(canvas);
    setTextureInitialized(true);
  }, []);

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    // Raycast hit to paint stenosis map directly onto the mesh
    if (e.uv && textureRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const x = e.uv.x * canvas.width;
        const y = (1 - e.uv.y) * canvas.height;
        
        // Draw dark spot for stenosis (shrinks the vertices inward based on displacementScale)
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(x, y, 15, 0, Math.PI * 2); // 15px radius for the narrow area
        ctx.fill();
        textureRef.current.needsUpdate = true;
      }
    }
  };

  // Note on Physics: 7 Pa is the high Wall Shear Stress (WSS) threshold, not physical transmural rupture pressure.
  // Note on RBCs: Removed buggy RBC instancedMesh implementation to dramatically improve performance.

  return (
    <mesh ref={meshRef} castShadow receiveShadow geometry={geometry} onPointerDown={handlePointerDown}>
      {textureInitialized && textureRef.current ? (
        <meshStandardMaterial attach="material" color="#aa0000" roughness={0.6} displacementMap={textureRef.current} displacementScale={-0.2} />
      ) : (
        <meshStandardMaterial attach="material" color="#aa0000" roughness={0.6} />
      )}
    </mesh>
  );
}