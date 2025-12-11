import { IMeshingService } from "./IMeshingService";
import { IMeshingOptions } from "./IMeshingOptions";

export class MeshingService2D implements IMeshingService {
  generateMesh(options?: Partial<IMeshingOptions>): any {
    console.log(options);
  }
}
