import type { Vec3 } from "./types";

export class VectorMath { 
    static add(a: Vec3, b: Vec3): Vec3 {
        return {
            x: a.x + b.x,
            y: a.y + b.y,
            z: a.z + b.z,
        }
    }

    static sub(a: Vec3, b: Vec3): Vec3 {
        return {
            x: a.x - b.x,
            y: a.y - b.y,
            z: a.z - b.z,
        }
    }
    
    static scale(v: Vec3, s: number): Vec3 {
        return {
            x: v.x * s,
            y: v.y * s,
            z: v.z * s,
        }
    }
    
    static length(v: Vec3): number {
        return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }       
    
    static dot(a: Vec3, b: Vec3): number {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    
    static normalizeSafe(v: Vec3, eps = 1e-6): Vec3 {
        const len = VectorMath.length(v);
        if (len <= eps) return VectorMath.zero();
        const inv = 1 / len;
        return { x: v.x * inv, y: v.y * inv, z: v.z * inv };
    }
    
    static zero(): Vec3 {
        return {
            x: 0,
            y: 0,
            z: 0,
        }
    }

    static distance(a: Vec3, b: Vec3): number {
        return VectorMath.length(VectorMath.sub(a, b));
    }
    
}

