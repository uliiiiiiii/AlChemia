import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/Addons.js";

// Define color mapping for common elements
const elementColors: { [key: number]: string } = {
  1: "#FFFFFF", // H - White
  6: "#5A5A5A", // C - Black
  7: "#0000FF", // N - Blue
  8: "#FF0000", // O - Red
  9: "#00FF00", // F - Green
  17: "#00FF00", // Cl - Green
  35: "#8B0000", // Br - Dark Red
  53: "#800080", // I - Dark Violet
  2: "#00FFFF", // He - Cyan
  10: "#00FFFF", // Ne - Cyan
  18: "#00FFFF", // Ar - Cyan
  36: "#00FFFF", // Kr - Cyan
  54: "#00FFFF", // Xe - Cyan
  86: "#00FFFF", // Rn - Cyan
  15: "#FFA500", // P - Orange
  16: "#FFFF00", // S - Yellow
  5: "#C0C0C0", // B - Beige
  3: "#800080", // Li - Violet
  11: "#800080", // Na - Violet
  19: "#800080", // K - Violet
  37: "#800080", // Rb - Violet
  55: "#800080", // Cs - Violet
  87: "#800080", // Fr - Violet
  4: "#008000", // Be - Dark Green
  12: "#008000", // Mg - Dark Green
  20: "#008000", // Ca - Dark Green
  38: "#008000", // Sr - Dark Green
  56: "#008000", // Ba - Dark Green
  88: "#008000", // Ra - Dark Green
  22: "#808080", // Ti - Grey
  26: "#FF8C00", // Fe - Dark Orange
};
interface Atom {
  id: number;
  element: number;
  x: number;
  y: number;
  z: number;
}

interface Bond {
  aid1: number;
  aid2: number;
  order: number;
}

interface MoleculeData {
  atoms: Atom[];
  bonds: Bond[];
}

interface MoleculeViewerProps {
  moleculeData: MoleculeData;
}

export default React.memo(function MoleculeViewer({
  moleculeData,
}: MoleculeViewerProps) {
  console.log("MoleculeViewer rendering with data:", moleculeData);
  const containerRef = useRef<HTMLDivElement>(null);
  const mountRef = useRef(false);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    // set up Three.js only once
    if (mountRef.current) return;
    mountRef.current = true;

    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 15);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });

    const width = containerRef.current.clientWidth || 800;
    const height = containerRef.current.clientHeight || 600;
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 20;
    controlsRef.current = controls;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0xffffff, 1.5);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    return () => {
      console.log("Cleaning up...");
      if (rendererRef.current) {
        rendererRef.current.dispose();
        containerRef.current?.removeChild(rendererRef.current.domElement);
      }
    };
  }, []);

  useEffect(() => {
    if (!sceneRef.current || !moleculeData) {
      console.log("No scene or molecule data available");
      return;
    }

    // actually clear previous objects in the scene
    const objectsToRemove: any = [];
    sceneRef.current.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        objectsToRemove.push(object);
        console.log(objectsToRemove);
      }
    });
    objectsToRemove.forEach((object: any) => sceneRef.current?.remove(object));

    const atomGeometry = new THREE.SphereGeometry(0.4, 32, 32);
    moleculeData.atoms.forEach((atom) => {
      const material = new THREE.MeshStandardMaterial({
        color: elementColors[atom.element] || "#FFC0CB",
        metalness: 0.5,
        roughness: 0.2,
      });

      const ambientLight = new THREE.AmbientLight(0xffffff, 0.05);
      sceneRef.current?.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.05);
      directionalLight.position.set(5, 5, 5);
      sceneRef.current?.add(directionalLight);

      const sphere = new THREE.Mesh(atomGeometry, material);
      sphere.position.set(atom.x, atom.y, atom.z);
      sceneRef.current?.add(sphere);
      console.log(`Added atom at position:`, {
        x: atom.x,
        y: atom.y,
        z: atom.z,
      });
    });

    moleculeData.bonds.forEach((bond) => {
      const atom1 = moleculeData.atoms[bond.aid1];
      const atom2 = moleculeData.atoms[bond.aid2];

      const start = new THREE.Vector3(atom1.x, atom1.y, atom1.z);
      const end = new THREE.Vector3(atom2.x, atom2.y, atom2.z);

      const direction = end.clone().sub(start);
      const length = direction.length();

      const cylinderGeometry = new THREE.CylinderGeometry(0.1, 0.1, length, 12);
      const material = new THREE.MeshPhongMaterial({
        color: 0xcccccc,
        shininess: 100,
      });
      const cylinder = new THREE.Mesh(cylinderGeometry, material);

      cylinder.position.copy(start);
      cylinder.position.lerp(end, 0.5);
      cylinder.lookAt(end);
      cylinder.rotateX(Math.PI / 2);

      sceneRef.current?.add(cylinder);
      console.log(`Added bond between atoms:`, { start: start, end: end });
    });

    if (cameraRef.current && moleculeData.atoms.length > 0) {
      const center = new THREE.Vector3();
      moleculeData.atoms.forEach((atom) => {
        center.add(new THREE.Vector3(atom.x, atom.y, atom.z));
      });
      center.divideScalar(moleculeData.atoms.length);

      if (controlsRef.current) {
        controlsRef.current.target.copy(center);
        controlsRef.current.update();
      }

      const maxDistance = moleculeData.atoms.reduce((max, atom) => {
        const distance = new THREE.Vector3(atom.x, atom.y, atom.z).distanceTo(
          center
        );
        return Math.max(max, distance);
      }, 0);

      cameraRef.current.position.copy(center);
      cameraRef.current.position.z += Math.max(10, maxDistance * 2);
      cameraRef.current.lookAt(center);
    }
  }, [moleculeData]);

  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current || !cameraRef.current || !rendererRef.current)
        return;

      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;

      console.log("Resizing to:", { width, height });

      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(width, height);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
      style={{
        minHeight: "500px",
        height: "100%",
        width: "100%",
        position: "absolute",
      }}
    />
  );
});
