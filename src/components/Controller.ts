/**
 * Copyright (C) - All Rights Reserved
 * Written by Noah Mattia Bussinger, October 2023
 */

import { float, MapString } from "../../types/utils.type.js";
import { toRadian, clamp } from "../utilities/utils.js";
import { Vec3 } from "../utilities/Vec3.js";
import { Mat4 } from "../utilities/Mat4.js";

type CameraRequirements = {
    position: Vec3;
    direction: Vec3;
};

export class Controller {
    private isLocked: boolean;
    private readonly canvas: HTMLCanvasElement;
    private readonly activeKeys: MapString<boolean>;

    private direction: Vec3;
    private readonly globalUp: Vec3;
    private left: Vec3;
    private localUp: Vec3;

    private readonly camera: CameraRequirements;
    private transform: Mat4;
    private velocity: float;
    private static readonly MinVelocity: float = 0.01;
    private static readonly DefaultVelocity: float = 0.35;
    private static readonly MaxVelocity: float = 5;

    public constructor(canvas: HTMLCanvasElement, camera: CameraRequirements) {
        this.isLocked = false;
        this.canvas = canvas;
        this.activeKeys = new MapString<boolean>();
        this.direction = new Vec3();
        this.globalUp = new Vec3(0, 1, 0);
        this.left = new Vec3();
        this.localUp = new Vec3();
        this.camera = camera;
        this.transform = new Mat4();
        this.initializeLocking();
        this.initializekeyboard();
        this.initializeMouse();
        this.reset();
    }

    private initializeLocking(): void {
        this.canvas.addEventListener("click", () => {
            this.lock(true);
        });
        document.addEventListener("pointerlockchange", (_event: Event) => {
            if (document.pointerLockElement === this.canvas) {
                this.lock();
            } else {
                this.unlock();
            }
        });
    }

    private lock(force: boolean = false): void {
        if (this.isLocked && !force) {
            return;
        }
        this.isLocked = true;
        this.canvas.requestPointerLock();
        this.reset();
    }

    private unlock(): void {
        if (!this.isLocked) {
            return;
        }
        this.isLocked = false;
        this.reset();
    }

    private initializekeyboard(): void {
        document.addEventListener("keydown", (event: KeyboardEvent) => {
            if (!this.isLocked) {
                return;
            }
            event.preventDefault();
            this.activeKeys.set(event.key.toLowerCase(), true);
        });
        document.addEventListener("keyup", (event: KeyboardEvent) => {
            if (!this.isLocked) {
                return;
            }
            event.preventDefault();
            this.activeKeys.set(event.key.toLowerCase(), false);
        });
    }

    private initializeMouse(): void {
        document.addEventListener("mousemove", (event: MouseEvent) => {
            if (!this.isLocked) {
                return;
            }
            this.updateDirection(event.movementX, event.movementY);
        });
        document.addEventListener("wheel", (event: WheelEvent) => {
            if (!this.isLocked) {
                return;
            }
            this.velocity -= event.deltaY * 0.001;
            this.velocity = clamp(
                this.velocity,
                Controller.MinVelocity,
                Controller.MaxVelocity,
            );
        });
    }

    public update(): void {
        if (!this.isLocked) {
            return;
        }
        this.updatePosition();
    }

    private updatePosition(): void {
        this.direction.set(0, 0, 0);
        this.left.copy(this.camera.direction).cross(this.globalUp);
        this.localUp.copy(this.left).cross(this.camera.direction);
        if (this.activeKeys.get("w") === true) {
            this.direction.sub(this.camera.direction);
        } else if (this.activeKeys.get("s") === true) {
            this.direction.add(this.camera.direction);
        }
        if (this.activeKeys.get("a") === true) {
            this.direction.add(this.left);
        } else if (this.activeKeys.get("d") === true) {
            this.direction.sub(this.left);
        }
        if (this.activeKeys.get(" ") === true) {
            this.direction.add(this.localUp);
        } else if (this.activeKeys.get("control") === true) {
            this.direction.sub(this.localUp);
        }
        if (!this.direction.x && !this.direction.y && !this.direction.z) {
            return;
        }
        this.camera.position.add(
            this.direction.normalize().scale(this.velocity),
        );
    }

    private updateDirection(x: float, y: float): void {
        this.transform.reset();
        this.transform.rotateY(-x * 0.1 * toRadian);
        this.camera.direction.applyMat(this.transform);
        this.left.copy(this.camera.direction).cross(this.globalUp);
        this.transform.reset();
        this.transform.rotateAxis(this.left.normalize(), y * 0.1 * toRadian);
        this.camera.direction.applyMat(this.transform).normalize();
    }

    private reset(): void {
        this.activeKeys.clear();
        this.velocity = Controller.DefaultVelocity;
    }
}
