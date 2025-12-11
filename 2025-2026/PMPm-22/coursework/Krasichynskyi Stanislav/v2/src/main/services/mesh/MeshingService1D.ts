import { IMeshingService } from "./IMeshingService";
import { IMeshingOptions } from "./IMeshingOptions";

export class MeshingService1D implements IMeshingService {
  generateMesh(options?: Partial<IMeshingOptions>): object {
    console.log(options);
    return {}
  }

}
