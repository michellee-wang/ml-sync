/**
 * Segment Templates - Reusable level segment patterns
 *
 * These templates serve dual purposes:
 * 1. Procedural generation building blocks
 * 2. Training data for ML models to learn obstacle patterns
 *
 * ML Integration Notes:
 * - Each template is encoded with numerical features that ML models can learn
 * - Templates can be used to generate synthetic training data
 * - Feature vectors can be extracted for supervised/reinforcement learning
 */

import { LevelSegment, GameObject, GameObjectType, Vector2D } from '../types';

/**
 * Numerical feature encoding for ML training
 * These features describe a segment in a way ML models can process
 */
export interface SegmentFeatures {
  difficulty: number;           // 0-1 scale
  density: number;              // Obstacles per unit length
  verticalComplexity: number;   // Height variation measure
  gapFrequency: number;         // Number of gaps/jumps
  platformRatio: number;        // Platform coverage ratio
  obstacleTypes: number[];      // One-hot encoding of obstacle types
  rhythmPattern: number[];      // Temporal spacing pattern
}

/**
 * Template definition with ML-ready features
 */
export interface SegmentTemplate {
  name: string;
  difficulty: number;
  length: number;
  features: SegmentFeatures;
  generator: (startX: number, seed: number) => GameObject[];
}

/**
 * Generate a unique ID for game objects
 */
function generateId(prefix: string, index: number, seed: number): string {
  return `${prefix}_${seed}_${index}_${Date.now()}`;
}

/**
 * Seeded random number generator (for reproducibility)
 */
class SeededRandom {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed;
  }

  next(): number {
    this.seed = (this.seed * 9301 + 49297) % 233280;
    return this.seed / 233280;
  }

  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  integer(min: number, max: number): number {
    return Math.floor(this.range(min, max + 1));
  }
}

// ============================================================================
// BASIC TEMPLATES (Difficulty 0.1 - 0.3)
// ============================================================================

export const FLAT_GROUND_SPIKE: SegmentTemplate = {
  name: 'flat_ground_spike',
  difficulty: 0.1,
  length: 200,
  features: {
    difficulty: 0.1,
    density: 0.15,
    verticalComplexity: 0.1,
    gapFrequency: 0,
    platformRatio: 1.0,
    obstacleTypes: [1, 0, 0],
    rhythmPattern: [0, 0, 1, 0, 0, 1],
  },
  generator: (startX: number, seed: number): GameObject[] => {
    const objects: GameObject[] = [];
    const groundY = 500;
    const segmentLength = 200;

    // Ground platform
    objects.push({
      id: generateId('platform', 0, seed),
      position: { x: startX, y: groundY },
      velocity: { x: 0, y: 0 },
      size: { x: segmentLength, y: 20 },
      type: GameObjectType.PLATFORM,
      active: true,
    });

    // Two spikes with plenty of room between them
    const spikePositions = [60, 140];
    spikePositions.forEach((offset, idx) => {
      objects.push({
        id: generateId('spike', idx, seed),
        position: { x: startX + offset, y: groundY - 20 },
        velocity: { x: 0, y: 0 },
        size: { x: 15, y: 20 },
        type: GameObjectType.OBSTACLE_SPIKE,
        active: true,
      });
    });

    return objects;
  },
};

export const SIMPLE_GAP: SegmentTemplate = {
  name: 'simple_gap',
  difficulty: 0.2,
  length: 200,
  features: {
    difficulty: 0.2,
    density: 0.1,
    verticalComplexity: 0.2,
    gapFrequency: 1,
    platformRatio: 0.8,
    obstacleTypes: [0, 0, 1],
    rhythmPattern: [1, 0, 0, 0, 1],
  },
  generator: (startX: number, seed: number): GameObject[] => {
    const objects: GameObject[] = [];
    const groundY = 500;

    // Wide left platform
    objects.push({
      id: generateId('platform', 0, seed),
      position: { x: startX, y: groundY },
      velocity: { x: 0, y: 0 },
      size: { x: 80, y: 20 },
      type: GameObjectType.PLATFORM,
      active: true,
    });

    // 40px gap (easily jumpable)

    // Wide right platform
    objects.push({
      id: generateId('platform', 1, seed),
      position: { x: startX + 120, y: groundY },
      velocity: { x: 0, y: 0 },
      size: { x: 80, y: 20 },
      type: GameObjectType.PLATFORM,
      active: true,
    });

    return objects;
  },
};

// ============================================================================
// INTERMEDIATE TEMPLATES (Difficulty 0.4 - 0.6)
// ============================================================================

export const STAIRCASE_BLOCKS: SegmentTemplate = {
  name: 'staircase_blocks',
  difficulty: 0.4,
  length: 200,
  features: {
    difficulty: 0.4,
    density: 0.5,
    verticalComplexity: 0.6,
    gapFrequency: 0,
    platformRatio: 0.8,
    obstacleTypes: [0, 1, 0],
    rhythmPattern: [1, 1, 1, 1, 1],
  },
  generator: (startX: number, seed: number): GameObject[] => {
    const objects: GameObject[] = [];
    const groundY = 500;

    // Ground platform
    objects.push({
      id: generateId('platform', 0, seed),
      position: { x: startX, y: groundY },
      velocity: { x: 0, y: 0 },
      size: { x: 200, y: 20 },
      type: GameObjectType.PLATFORM,
      active: true,
    });

    // 3-step staircase with jumpable heights (each step ~40px, max height 120px)
    const stepConfigs = [
      { x: 40, y: groundY - 40, w: 50, h: 40 },
      { x: 100, y: groundY - 80, w: 50, h: 80 },
      { x: 160, y: groundY - 40, w: 50, h: 40 },
    ];

    stepConfigs.forEach((step, i) => {
      objects.push({
        id: generateId('block', i, seed),
        position: { x: startX + step.x, y: step.y },
        velocity: { x: 0, y: 0 },
        size: { x: step.w, y: step.h },
        type: GameObjectType.OBSTACLE_BLOCK,
        active: true,
      });
    });

    return objects;
  },
};

export const SPIKE_RHYTHM: SegmentTemplate = {
  name: 'spike_rhythm',
  difficulty: 0.5,
  length: 250,
  features: {
    difficulty: 0.5,
    density: 0.4,
    verticalComplexity: 0.3,
    gapFrequency: 0,
    platformRatio: 1.0,
    obstacleTypes: [1, 0, 0],
    rhythmPattern: [1, 0, 1, 0, 1, 0],
  },
  generator: (startX: number, seed: number): GameObject[] => {
    const objects: GameObject[] = [];
    const groundY = 500;

    // Ground platform
    objects.push({
      id: generateId('platform', 0, seed),
      position: { x: startX, y: groundY },
      velocity: { x: 0, y: 0 },
      size: { x: 250, y: 20 },
      type: GameObjectType.PLATFORM,
      active: true,
    });

    // Spaced-out spike pairs with room to jump between them
    const pattern = [40, 100, 160, 220];
    pattern.forEach((offset, idx) => {
      objects.push({
        id: generateId('spike', idx, seed),
        position: { x: startX + offset, y: groundY - 20 },
        velocity: { x: 0, y: 0 },
        size: { x: 15, y: 20 },
        type: GameObjectType.OBSTACLE_SPIKE,
        active: true,
      });
    });

    return objects;
  },
};

export const PLATFORM_JUMPS: SegmentTemplate = {
  name: 'platform_jumps',
  difficulty: 0.6,
  length: 280,
  features: {
    difficulty: 0.6,
    density: 0.3,
    verticalComplexity: 0.5,
    gapFrequency: 3,
    platformRatio: 0.6,
    obstacleTypes: [0, 0, 1],
    rhythmPattern: [1, 0, 0, 1, 0, 0],
  },
  generator: (startX: number, seed: number): GameObject[] => {
    const objects: GameObject[] = [];
    const groundY = 500;

    // Wider platforms with reasonable gaps and heights
    const platforms = [
      { x: 0, y: 0, width: 70 },
      { x: 100, y: -30, width: 60 },
      { x: 190, y: -50, width: 60 },
      { x: 270, y: 0, width: 70 },
    ];

    platforms.forEach((plat, idx) => {
      objects.push({
        id: generateId('platform', idx, seed),
        position: { x: startX + plat.x, y: groundY + plat.y },
        velocity: { x: 0, y: 0 },
        size: { x: plat.width, y: 20 },
        type: GameObjectType.PLATFORM,
        active: true,
      });
    });

    // Single spike on the ground-level entry platform
    objects.push({
      id: generateId('spike', 0, seed),
      position: { x: startX + 40, y: groundY - 20 },
      velocity: { x: 0, y: 0 },
      size: { x: 15, y: 20 },
      type: GameObjectType.OBSTACLE_SPIKE,
      active: true,
    });

    return objects;
  },
};

// ============================================================================
// ADVANCED TEMPLATES (Difficulty 0.7 - 1.0)
// ============================================================================

export const MIXED_OBSTACLES: SegmentTemplate = {
  name: 'mixed_obstacles',
  difficulty: 0.7,
  length: 300,
  features: {
    difficulty: 0.7,
    density: 0.5,
    verticalComplexity: 0.5,
    gapFrequency: 1,
    platformRatio: 0.7,
    obstacleTypes: [0.5, 0.3, 0.2],
    rhythmPattern: [1, 0, 1, 0, 1, 0],
  },
  generator: (startX: number, seed: number): GameObject[] => {
    const objects: GameObject[] = [];
    const groundY = 500;

    // Long ground platform with a gap in the middle
    objects.push({
      id: generateId('platform', 0, seed),
      position: { x: startX, y: groundY },
      velocity: { x: 0, y: 0 },
      size: { x: 120, y: 20 },
      type: GameObjectType.PLATFORM,
      active: true,
    });

    // Spike on first section
    objects.push({
      id: generateId('spike', 0, seed),
      position: { x: startX + 70, y: groundY - 20 },
      velocity: { x: 0, y: 0 },
      size: { x: 15, y: 20 },
      type: GameObjectType.OBSTACLE_SPIKE,
      active: true,
    });

    // Block you can jump on or over
    objects.push({
      id: generateId('block', 0, seed),
      position: { x: startX + 140, y: groundY - 40 },
      velocity: { x: 0, y: 0 },
      size: { x: 40, y: 40 },
      type: GameObjectType.OBSTACLE_BLOCK,
      active: true,
    });

    // Second ground section
    objects.push({
      id: generateId('platform', 1, seed),
      position: { x: startX + 190, y: groundY },
      velocity: { x: 0, y: 0 },
      size: { x: 110, y: 20 },
      type: GameObjectType.PLATFORM,
      active: true,
    });

    // Spike on second section
    objects.push({
      id: generateId('spike', 1, seed),
      position: { x: startX + 250, y: groundY - 20 },
      velocity: { x: 0, y: 0 },
      size: { x: 15, y: 20 },
      type: GameObjectType.OBSTACLE_SPIKE,
      active: true,
    });

    return objects;
  },
};

export const EXTREME_CHALLENGE: SegmentTemplate = {
  name: 'extreme_challenge',
  difficulty: 0.9,
  length: 400,
  features: {
    difficulty: 0.9,
    density: 1.0,
    verticalComplexity: 1.0,
    gapFrequency: 4,
    platformRatio: 0.4,
    obstacleTypes: [0.7, 0.6, 0.5], // All types heavily used
    rhythmPattern: [1, 1, 1, 0, 1, 1],
  },
  generator: (startX: number, seed: number): GameObject[] => {
    const objects: GameObject[] = [];
    const groundY = 500;

    // Challenging but possible: wider platforms, jumpable heights (max 60px up)
    const platformConfigs = [
      { x: 0, y: 0, width: 60 },
      { x: 90, y: -40, width: 50 },
      { x: 170, y: -60, width: 50 },
      { x: 250, y: -30, width: 50 },
      { x: 330, y: 0, width: 60 },
    ];

    platformConfigs.forEach((config, idx) => {
      objects.push({
        id: generateId('platform', idx, seed),
        position: { x: startX + config.x, y: groundY + config.y },
        velocity: { x: 0, y: 0 },
        size: { x: config.width, y: 20 },
        type: GameObjectType.PLATFORM,
        active: true,
      });
    });

    // Spikes on a couple of platforms
    [0, 2, 4].forEach((platIdx, i) => {
      const plat = platformConfigs[platIdx];
      objects.push({
        id: generateId('spike', i, seed),
        position: { x: startX + plat.x + plat.width / 2, y: groundY + plat.y - 20 },
        velocity: { x: 0, y: 0 },
        size: { x: 15, y: 20 },
        type: GameObjectType.OBSTACLE_SPIKE,
        active: true,
      });
    });

    return objects;
  },
};

// ============================================================================
// TEMPLATE REGISTRY
// ============================================================================

/**
 * All available templates organized by difficulty
 * ML models can use this for training data generation
 */
export const TEMPLATE_REGISTRY = {
  easy: [FLAT_GROUND_SPIKE, SIMPLE_GAP],
  medium: [STAIRCASE_BLOCKS, SPIKE_RHYTHM, PLATFORM_JUMPS],
  hard: [MIXED_OBSTACLES],
  extreme: [EXTREME_CHALLENGE],
};

/**
 * Get all templates as a flat array
 */
export function getAllTemplates(): SegmentTemplate[] {
  return [
    ...TEMPLATE_REGISTRY.easy,
    ...TEMPLATE_REGISTRY.medium,
    ...TEMPLATE_REGISTRY.hard,
    ...TEMPLATE_REGISTRY.extreme,
  ];
}

/**
 * Get templates filtered by difficulty range
 */
export function getTemplatesByDifficulty(minDiff: number, maxDiff: number): SegmentTemplate[] {
  return getAllTemplates().filter(
    t => t.difficulty >= minDiff && t.difficulty <= maxDiff
  );
}

/**
 * Extract feature vector from a template (for ML training)
 * Returns a normalized feature array that can be fed to ML models
 */
export function extractFeatureVector(template: SegmentTemplate): number[] {
  const features = template.features;
  return [
    features.difficulty,
    features.density,
    features.verticalComplexity,
    features.gapFrequency,
    features.platformRatio,
    ...features.obstacleTypes,
    ...features.rhythmPattern,
  ];
}

/**
 * Generate synthetic training data from templates
 * Useful for creating datasets for ML model training
 */
export function generateTrainingData(
  template: SegmentTemplate,
  numSamples: number
): Array<{ features: number[]; template: string }> {
  const samples = [];
  for (let i = 0; i < numSamples; i++) {
    samples.push({
      features: extractFeatureVector(template),
      template: template.name,
    });
  }
  return samples;
}
