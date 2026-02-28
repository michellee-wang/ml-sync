/**
 * Level Generation System - Main Exports
 *
 * This module provides a complete, ML-ready level generation system for
 * a Geometry Dash clone.
 *
 * QUICK START:
 * ============
 * ```typescript
 * import { LevelGenerator } from '@/game/levels';
 *
 * const generator = new LevelGenerator();
 *
 * const level = generator.generate({
 *   difficulty: 0.5,
 *   length: 1000,
 *   seed: 12345,
 *   style: 'classic',
 * });
 * ```
 *
 * ARCHITECTURE:
 * =============
 * - LevelGenerator: Main interface, handles both procedural and ML generation
 * - ProceduralGenerator: Algorithm-based generation (baseline)
 * - SegmentTemplates: Reusable building blocks with ML-ready features
 *
 * ML INTEGRATION:
 * ===============
 * 1. Implement MLModelInterface
 * 2. Train your model on collected gameplay data
 * 3. Register: generator.setMLModel(yourModel)
 * 4. Generate: generator.generate(config) - automatically uses ML
 */

// ============================================================================
// MAIN GENERATOR
// ============================================================================

export { LevelGenerator, StubMLModel } from './LevelGenerator';
export type { MLModelInterface, MLModelOutput, TrainingExample } from './LevelGenerator';

// ============================================================================
// TEST LEVEL
// ============================================================================

export { createTestLevel } from './TestLevel';

// ============================================================================
// PROCEDURAL GENERATION
// ============================================================================

export {
  ProceduralGenerator,
  SeededRandom,
  TemplateBasedStrategy,
  NoiseBasedStrategy,
  WaveBasedStrategy,
  applyConstraints,
  analyzeSegmentDifficulty,
} from './ProceduralGenerator';
export type { GenerationStrategy } from './ProceduralGenerator';

// ============================================================================
// SEGMENT TEMPLATES
// ============================================================================

export {
  FLAT_GROUND_SPIKE,
  SIMPLE_GAP,
  STAIRCASE_BLOCKS,
  SPIKE_RHYTHM,
  PLATFORM_JUMPS,
  MIXED_OBSTACLES,
  EXTREME_CHALLENGE,
  TEMPLATE_REGISTRY,
  getAllTemplates,
  getTemplatesByDifficulty,
  extractFeatureVector,
  generateTrainingData,
} from './SegmentTemplates';
export type { SegmentTemplate, SegmentFeatures } from './SegmentTemplates';

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

import { LevelGenerator } from './LevelGenerator';
import { LevelGeneratorConfig, Level, GameObjectType } from '../types';

/**
 * Create a default level generator instance
 */
export function createLevelGenerator(): LevelGenerator {
  return new LevelGenerator();
}

/**
 * Quick generate function for simple use cases
 */
export function generateLevel(
  difficulty: number,
  length: number,
  seed?: number
): Level {
  const generator = new LevelGenerator();
  return generator.generate({
    difficulty,
    length,
    seed,
    style: 'classic',
  });
}

/**
 * Generate a level with ML if model is provided
 */
export function generateLevelWithML(
  config: LevelGeneratorConfig,
  model?: any
): Level {
  const generator = new LevelGenerator();

  if (model) {
    generator.setMLModel(model);
  }

  return generator.generate(config);
}

/**
 * Batch generate multiple levels (useful for testing/training data)
 */
export function generateLevelBatch(
  baseConfig: LevelGeneratorConfig,
  count: number
): Level[] {
  const generator = new LevelGenerator();
  const levels: Level[] = [];

  for (let i = 0; i < count; i++) {
    const config = {
      ...baseConfig,
      seed: (baseConfig.seed ?? 0) + i,
    };

    levels.push(generator.generate(config));
  }

  return levels;
}

/**
 * Create a very long procedurally-generated level for endless/survival gameplay.
 * Difficulty ramps gradually from easy to hard over the course of the level.
 */
export function createInfiniteLevel(seed?: number): Level {
  const generator = new LevelGenerator();
  const totalLength = 100000;
  const chunkLength = 500;
  const numChunks = totalLength / chunkLength;
  const groundY = 500;

  const segments: import('../types').LevelSegment[] = [];

  // Safe starting zone: flat ground with no obstacles so the player can land
  segments.push({
    id: 'spawn-zone',
    startX: 0,
    length: 300,
    difficulty: 0,
    objects: [{
      id: 'spawn-platform',
      position: { x: 0, y: groundY },
      velocity: { x: 0, y: 0 },
      size: { x: 300, y: 50 },
      type: GameObjectType.PLATFORM,
      active: true,
    }],
  });

  for (let i = 0; i < numChunks; i++) {
    const globalOffset = 300 + i * chunkLength;
    // Very gradual ramp: stays easy for the first ~20 chunks (~45s), then slowly climbs
    const progress = i / numChunks;
    const difficulty = Math.min(1, 0.05 + progress * progress * 0.95);

    const generated = generator.generate({
      difficulty,
      length: chunkLength,
      seed: (seed ?? 42) + i,
      style: 'classic',
    });

    for (const seg of generated.segments) {
      segments.push({
        ...seg,
        id: `inf-seg-${i}-${seg.id}`,
        startX: globalOffset + seg.startX,
        objects: seg.objects.map(obj => ({
          ...obj,
          position: { x: obj.position.x + globalOffset, y: obj.position.y },
        })),
      });
    }
  }

  return {
    id: `infinite-level-${seed ?? 42}`,
    name: 'Infinite Level',
    segments,
    totalLength: 300 + totalLength,
    difficulty: 0.5,
    generatedBy: 'procedural',
  };
}

/**
 * Generate levels across difficulty spectrum (for playtesting)
 */
export function generateDifficultySpectrum(
  length: number = 1000,
  steps: number = 5
): Level[] {
  const generator = new LevelGenerator();
  const levels: Level[] = [];

  for (let i = 0; i < steps; i++) {
    const difficulty = i / (steps - 1); // 0.0 to 1.0

    levels.push(
      generator.generate({
        difficulty,
        length,
        seed: i,
        style: 'classic',
      })
    );
  }

  return levels;
}

// ============================================================================
// DEFAULT INSTANCE
// ============================================================================

/**
 * Singleton instance for convenience
 * Use this if you don't need multiple generators
 */
export const defaultLevelGenerator = new LevelGenerator();
