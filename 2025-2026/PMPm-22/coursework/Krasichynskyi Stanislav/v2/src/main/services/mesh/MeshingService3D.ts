import { IMeshingService } from "./IMeshingService";
import { IMeshingOptions } from "./IMeshingOptions";

export class MeshingService3D implements IMeshingService {
  generateMesh(options?: Partial<IMeshingOptions>): any {
    console.log('In Development', options);
  }
}
