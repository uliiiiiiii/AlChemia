import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Define color mapping for common elements
const elementColors = {
  1: 0xffffff, // H - White
  6: 0x808080, // C - Grey
  7: 0x0000ff, // N - Blue
  8: 0xff0000, // O - Red
  9: 0x90ee90, // F - Light green
  11: 0x800080, // Na - Purple
  15: 0xffa500, // P - Orange
  16: 0xffff00, // S - Yellow
  17: 0x00ff00, // Cl - Green
  35: 0x8b0000, // Br - Dark red
  53: 0x4b0082, // I - Indigo
};

// Define atomic radii (in Angstroms)
const atomicRadii = {
  1: 0.25, // H
  6: 0.7, // C
  7: 0.65, // N
  8: 0.6, // O
  9: 0.5, // F
  11: 1.8, // Na
  15: 1.0, // P
  16: 1.0, // S
  17: 1.0, // Cl
  35: 1.15, // Br
  53: 1.4, // I
};

const MoleculeViewer = ({ moleculeData }: any) => {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!moleculeData || !moleculeData[0]) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.z = 5;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 1);
    scene.add(directionalLight);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const molecule = moleculeData[0];
    const { atoms, bonds } = molecule;

    const atomGroup = new THREE.Group();
    atoms.aid.forEach((id, index) => {
      const element = atoms.element[index];
      const radius = atomicRadii[element] || 0.5; //WTF? change to use data from API
      const color = elementColors[element] || 0x808080; //same if possible

      const geometry = new THREE.SphereGeometry(radius, 32, 32);
      const material = new THREE.MeshPhongMaterial({
        color: color,
        shininess: 100,
        specular: 0x444444,
      });
      const sphere = new THREE.Mesh(geometry, material);

      // I need to get actual coordinates from my data??
      sphere.position.set((index - atoms.aid.length / 2) * 2, 0, 0);

      atomGroup.add(sphere);
    });
    scene.add(atomGroup);

    // Add bonds
    if (bonds && bonds.aid1) {
      bonds.aid1.forEach((aid1, index) => {
        const aid2 = bonds.aid2[index];
        const order = bonds.order[index];

        // Find positions of connected atoms
        const atom1Index = atoms.aid.indexOf(aid1);
        const atom2Index = atoms.aid.indexOf(aid2);

        if (atom1Index !== -1 && atom2Index !== -1) {
          const pos1 = atomGroup.children[atom1Index].position;
          const pos2 = atomGroup.children[atom2Index].position;

          // Create bond cylinder
          const bondMaterial = new THREE.MeshPhongMaterial({
            color: 0xcccccc,
            shininess: 100,
          });

          const distance = pos1.distanceTo(pos2);
          const bondGeometry = new THREE.CylinderGeometry(
            0.1,
            0.1,
            distance,
            8
          );
          const bond = new THREE.Mesh(bondGeometry, bondMaterial);

          // Position bond between atoms
          const midpoint = new THREE.Vector3()
            .addVectors(pos1, pos2)
            .multiplyScalar(0.5);
          bond.position.copy(midpoint);
          bond.lookAt(pos2);
          bond.rotateX(Math.PI / 2);

          scene.add(bond);
        }
      });
    }

    // Center camera on molecule
    const box = new THREE.Box3().setFromObject(atomGroup);
    const center = box.getCenter(new THREE.Vector3());
    atomGroup.position.sub(center);

    // Adjust camera distance based on molecule size
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.z = maxDim * 3;

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      mountRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, [moleculeData]);

  return <div ref={mountRef} className="w-full h-screen" />;
};

export default MoleculeViewer;
