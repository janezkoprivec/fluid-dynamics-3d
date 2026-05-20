export class SphKernels {
  static poly6(r: number, h: number): number {
    if (h <= 0) return 0; 
    if (r >= 0 && r <= h ) { 
      return (315/(64*Math.PI*Math.pow(h, 9)))*Math.pow((Math.pow(h, 2) - Math.pow(r, 2)), 3); 
    } else { 
      return 0;
    }
  }

  static spikyGradient(r: number, h: number): number {
    if (h <= 0) return 0;
    if (r < 0 || r > h) return 0;
    return (45 / (Math.PI * Math.pow(h, 6))) * Math.pow(h - r, 2);
  }

  static viscosityLaplacian(r: number, h: number): number {
    if (h <= 0) return 0;
    if (r < 0 || r > h) return 0;
    return (45 / (Math.PI * Math.pow(h, 6))) * (h - r);
  }
}
