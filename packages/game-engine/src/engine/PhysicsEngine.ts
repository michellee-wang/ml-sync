// Physics simulation system for the game
import { GameObject, Player, PhysicsConfig, Vector2D, GameObjectType } from '../types';
import { CollisionDetection } from './CollisionDetection';

export class PhysicsEngine {
  private config: PhysicsConfig;

  constructor(config?: Partial<PhysicsConfig>) {
    // Default physics configuration (tuned for Geometry Dash-like gameplay)
    this.config = {
      gravity: 1800,
      jumpForce: -650,
      maxVelocity: { x: 400, y: 900 },
      friction: 0.8,
      ...config,
    };
  }

  /**
   * Get the current physics configuration
   */
  getConfig(): PhysicsConfig {
    return { ...this.config };
  }

  /**
   * Update physics configuration
   */
  updateConfig(config: Partial<PhysicsConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Apply gravity to an object
   * @param obj The game object
   * @param deltaTime Time elapsed since last update (in seconds)
   */
  applyGravity(obj: GameObject, deltaTime: number): void {
    obj.velocity.y += this.config.gravity * deltaTime;

    // Clamp to max velocity
    if (obj.velocity.y > this.config.maxVelocity.y) {
      obj.velocity.y = this.config.maxVelocity.y;
    }
  }

  /**
   * Update object position based on velocity
   * @param obj The game object
   * @param deltaTime Time elapsed since last update (in seconds)
   */
  updatePosition(obj: GameObject, deltaTime: number): void {
    obj.position.x += obj.velocity.x * deltaTime;
    obj.position.y += obj.velocity.y * deltaTime;
  }

  /**
   * Make the player jump
   * @param player The player object
   */
  jump(player: Player): void {
    if (player.isOnGround && !player.isJumping) {
      player.velocity.y = this.config.jumpForce;
      player.isJumping = true;
      player.isOnGround = false;
    }
  }

  /**
   * Set the player's horizontal velocity (auto-run)
   * @param player The player object
   * @param speed Speed in pixels per second
   */
  setPlayerRunSpeed(player: Player, speed: number): void {
    player.velocity.x = Math.min(speed, this.config.maxVelocity.x);
  }

  /**
   * Update player physics state
   * @param player The player object
   * @param platforms List of platform objects
   * @param deltaTime Time elapsed since last update (in seconds)
   */
  updatePlayer(player: Player, platforms: GameObject[], deltaTime: number): void {
    // Apply gravity
    this.applyGravity(player, deltaTime);

    // Update position
    this.updatePosition(player, deltaTime);

    // Prevent jumping through platforms from below
    if (player.velocity.y < 0) {
      for (const platform of platforms) {
        if (!platform.active) continue;
        if (!CollisionDetection.checkAABBCollision(player, platform)) continue;

        const platformBottom = platform.position.y + platform.size.y;
        if (player.position.y + player.size.y > platformBottom) {
          player.position.y = platformBottom;
          player.velocity.y = 0;
        }
      }
    }

    // Check ground collision
    const wasOnGround = player.isOnGround;
    player.isOnGround = CollisionDetection.isOnGround(player, platforms, 5);

    if (player.isOnGround) {
      // Snap to platform
      const groundPlatform = this.findGroundPlatform(player, platforms);
      if (groundPlatform) {
        player.position.y = groundPlatform.position.y - player.size.y;
        player.velocity.y = 0;
        player.isJumping = false;
      }
    } else if (wasOnGround && !player.isOnGround) {
      // Just left the ground (falling)
      player.isJumping = false;
    }

    // Apply friction to horizontal movement (optional)
    // player.velocity.x *= this.config.friction;
  }

  /**
   * Find the platform the player is standing on
   * @param player The player object
   * @param platforms List of platform objects
   * @returns The platform object or null
   */
  private findGroundPlatform(player: Player, platforms: GameObject[]): GameObject | null {
    const bottomY = player.position.y + player.size.y;

    for (const platform of platforms) {
      if (!platform.active) continue;

      const platformTop = platform.position.y;
      const horizontalOverlap =
        player.position.x + player.size.x > platform.position.x &&
        player.position.x < platform.position.x + platform.size.x;

      if (horizontalOverlap && Math.abs(bottomY - platformTop) <= 5) {
        return platform;
      }
    }

    return null;
  }

  /**
   * Handle collision response for the player
   * @param player The player object
   * @param collidedObject The object the player collided with
   * @returns true if the collision was fatal
   */
  handlePlayerCollision(player: Player, collidedObject: GameObject): boolean {
    switch (collidedObject.type) {
      case GameObjectType.OBSTACLE_SPIKE:
        player.health = 0;
        return true;

      case GameObjectType.OBSTACLE_BLOCK:
        // Blocks are solid surfaces -- resolve like a platform collision.
        // Only push the player out along the axis of least penetration.
        this.resolveBlockCollision(player, collidedObject);
        return false;

      case GameObjectType.PLATFORM:
        return false;

      case GameObjectType.PORTAL:
        return false;

      default:
        return false;
    }
  }

  private resolveBlockCollision(player: Player, block: GameObject): void {
    const overlapLeft = (player.position.x + player.size.x) - block.position.x;
    const overlapRight = (block.position.x + block.size.x) - player.position.x;
    const overlapTop = (player.position.y + player.size.y) - block.position.y;
    const overlapBottom = (block.position.y + block.size.y) - player.position.y;

    const minOverlap = Math.min(overlapLeft, overlapRight, overlapTop, overlapBottom);

    if (minOverlap === overlapTop && player.velocity.y >= 0) {
      // Landing on top of the block
      player.position.y = block.position.y - player.size.y;
      player.velocity.y = 0;
      player.isOnGround = true;
      player.isJumping = false;
    } else if (minOverlap === overlapBottom && player.velocity.y < 0) {
      // Hitting the block from below
      player.position.y = block.position.y + block.size.y;
      player.velocity.y = 0;
    } else if (minOverlap === overlapLeft) {
      // Running into the block's left side -- fatal (wall collision)
      player.health = 0;
    } else if (minOverlap === overlapRight) {
      player.position.x = block.position.x + block.size.x;
    }
  }

  /**
   * Reset player physics state
   * @param player The player object
   */
  resetPlayer(player: Player): void {
    player.velocity = { x: 0, y: 0 };
    player.isJumping = false;
    player.isOnGround = false;
  }

  /**
   * Check if an object is out of bounds
   * @param obj The game object
   * @param bounds The game bounds
   * @returns true if object is out of bounds
   */
  isOutOfBounds(obj: GameObject, bounds: { width: number; height: number }): boolean {
    return (
      obj.position.y > bounds.height || // Fell off the bottom
      obj.position.y + obj.size.y < 0 || // Above the screen
      obj.position.x + obj.size.x < 0 // Left of the screen
    );
  }
}
