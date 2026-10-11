import { Canvas, useFrame } from '@react-three/fiber'
import { Line, OrbitControls, RoundedBox, Text } from '@react-three/drei'
import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import labelFont from '@fontsource/barlow-condensed/files/barlow-condensed-latin-600-normal.woff?url'
import type { Assignment, Decision } from '../api'
import { shortName } from '../api'

// Depot layout in world units. Position 1 on each stabling line is nearest the exit (east, +x).
const TRACKS = ['SL1', 'SL2', 'SL3', 'SL4', 'SL5']
const TRACK_Z = (track: string) => -4 + TRACKS.indexOf(track) * 2
const SLOT_X = (pos: number) => 7 - (pos - 1) * 3.3
const IBL_Z = -8.2
const CLEAN_Z = 8.2
const BAY_X = [-1.5, 2, 5.5]

const COLORS: Record<Assignment, string> = {
  Service: '#4cc38a',
  Standby: '#cdd6de',
  'Inspection Bay': '#e8b54a',
  Cleaning: '#6fb2e6',
  Held: '#e5604d',
}
const IVORY = '#e6e1d3'
const RULE = '#3b5664'

function targetFor(d: Decision, bayIndex: number): [number, number, number] {
  if (d.assignment === 'Inspection Bay') return [BAY_X[bayIndex % 3], 0.35, IBL_Z]
  if (d.assignment === 'Cleaning') return [BAY_X[bayIndex % 3], 0.35, CLEAN_Z]
  return [SLOT_X(d.stabling_position), 0.35, TRACK_Z(d.stabling_track)]
}

interface TrainProps {
  decision: Decision
  target: [number, number, number]
  selected: boolean
  changed: boolean
  onSelect: (name: string) => void
}

function Train({ decision, target, selected, changed, onSelect }: TrainProps) {
  const group = useRef<THREE.Group>(null)
  const [hovered, setHovered] = useState(false)
  const goal = useMemo(() => new THREE.Vector3(...target), [target])
  const reduceMotion = useMemo(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches, [])

  useFrame((_, delta) => {
    const g = group.current
    if (!g) return
    const lift = selected ? 0.45 : hovered ? 0.2 : 0
    const want = goal.clone().setY(goal.y + lift)
    if (reduceMotion) g.position.copy(want)
    else g.position.lerp(want, 1 - Math.exp(-delta * 3.2))
  })

  const color = COLORS[decision.assignment]
  return (
    <group
      ref={group}
      position={target}
      onClick={(e) => { e.stopPropagation(); onSelect(decision.train_name) }}
      onPointerOver={(e) => { e.stopPropagation(); setHovered(true); document.body.style.cursor = 'pointer' }}
      onPointerOut={() => { setHovered(false); document.body.style.cursor = 'auto' }}
    >
      <RoundedBox args={[2.9, 0.55, 0.8]} radius={0.16} smoothness={3}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={selected ? 0.55 : hovered ? 0.35 : 0.16}
          roughness={0.55}
        />
      </RoundedBox>
      {/* cab windows mark the leading end, facing the exit */}
      <mesh position={[1.36, 0.08, 0]}>
        <boxGeometry args={[0.2, 0.22, 0.62]} />
        <meshStandardMaterial color="#13212a" roughness={0.3} />
      </mesh>
      <Text
        font={labelFont}
        position={[0, 0.55, 0]}
        rotation={[-Math.PI / 2.6, 0, 0]}
        fontSize={0.5}
        color={selected ? '#ffffff' : IVORY}
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.02}
        outlineColor="#13212a"
      >
        {shortName(decision.train_name) + (changed ? ' •' : '')}
      </Text>
      {selected && (
        <mesh position={[0, -0.33, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[1.75, 1.95, 48]} />
          <meshBasicMaterial color="#1aa39c" />
        </mesh>
      )}
    </group>
  )
}

function Yard() {
  const label = (text: string, position: [number, number, number], size = 0.5, color = '#8fa2a8') => (
    <Text font={labelFont} position={position} rotation={[-Math.PI / 2, 0, 0]} fontSize={size} color={color} anchorX="left" anchorY="middle">
      {text}
    </Text>
  )
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[400, 400]} />
        <meshBasicMaterial color="#13212a" />
      </mesh>
      {/* stabling lines run into the exit throat on the east side */}
      {TRACKS.map((t) => {
        const z = TRACK_Z(t)
        return (
          <group key={t}>
            <Line points={[[-8.5, 0, z], [9.5, 0, z], [12.5, 0, 0]]} color={IVORY} lineWidth={1.6} transparent opacity={0.55} />
            {label(t, [-11, 0, z], 0.7)}
            {[1, 2, 3, 4, 5].map((p) => (
              <Line key={p} points={[[SLOT_X(p) - 1.55, 0, z - 0.5], [SLOT_X(p) - 1.55, 0, z + 0.5]]} color={RULE} lineWidth={1} />
            ))}
          </group>
        )
      })}
      <Line points={[[12.5, 0, 0], [17, 0, 0]]} color={IVORY} lineWidth={2.2} />
      {label('To mainline', [13, 0, 0.8], 0.65, IVORY)}

      {/* Inspection Bay Line shed and the deep-cleaning bay */}
      {[{ z: IBL_Z, name: 'Inspection bay line', color: '#e8b54a' }, { z: CLEAN_Z, name: 'Deep-cleaning bay', color: '#6fb2e6' }].map((s) => (
        <group key={s.name}>
          <Line
            points={[[-3.6, 0, s.z - 1], [7.6, 0, s.z - 1], [7.6, 0, s.z + 1], [-3.6, 0, s.z + 1], [-3.6, 0, s.z - 1]]}
            color={s.color} lineWidth={1.2} dashed dashSize={0.4} gapSize={0.25} transparent opacity={0.7}
          />
          {BAY_X.map((x) => (
            <Line key={x} points={[[x - 1.6, 0, s.z], [x + 1.6, 0, s.z]]} color={IVORY} lineWidth={1.2} transparent opacity={0.45} />
          ))}
          {label(s.name, [-11, 0, s.z], 0.62, s.color)}
        </group>
      ))}
    </group>
  )
}

interface DepotProps {
  decisions: Decision[]
  selected: string | null
  changed: Set<string>
  onSelect: (name: string | null) => void
}

export function Depot3D({ decisions, selected, changed, onSelect }: DepotProps) {
  const targets = useMemo(() => {
    let ibl = 0
    let clean = 0
    return Object.fromEntries(
      decisions.map((d) => {
        const idx = d.assignment === 'Inspection Bay' ? ibl++ : d.assignment === 'Cleaning' ? clean++ : 0
        return [d.train_name, targetFor(d, idx)]
      }),
    )
  }, [decisions])

  return (
    <Canvas
      camera={{ position: [2.5, 16.5, 12.5], fov: 44 }}
      dpr={[1, 2]}
      onPointerMissed={() => onSelect(null)}
      aria-label="3D view of the Muttom depot showing every trainset coloured by its planned assignment"
    >
      <color attach="background" args={['#13212a']} />
      <fog attach="fog" args={['#13212a', 30, 55]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[8, 18, 6]} intensity={1.1} />
      <Yard />
      {decisions.map((d) => (
        <Train
          key={d.train_name}
          decision={d}
          target={targets[d.train_name]}
          selected={selected === d.train_name}
          changed={changed.has(d.train_name)}
          onSelect={onSelect}
        />
      ))}
      <OrbitControls
        target={[2.5, 0, 0.6]}
        enablePan
        minDistance={10}
        maxDistance={34}
        minPolarAngle={0.25}
        maxPolarAngle={Math.PI / 2.4}
      />
    </Canvas>
  )
}
