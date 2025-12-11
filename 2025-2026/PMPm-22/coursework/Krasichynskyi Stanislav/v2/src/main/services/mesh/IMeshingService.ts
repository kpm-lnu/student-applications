import { IMeshingOptions } from "./IMeshingOptions";

export interface IMeshingService {
  generateMesh(options?: Partial<IMeshingOptions>): object;
}
