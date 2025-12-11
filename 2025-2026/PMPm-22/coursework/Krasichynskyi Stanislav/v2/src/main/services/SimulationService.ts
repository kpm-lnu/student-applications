import { ModelType } from '../../shared/enums'
import { IMeshingOptions } from './mesh/IMeshingOptions'
import { IMeshingService } from './mesh/IMeshingService'
import { IFEMService } from './fem/IFEMService'
import { MeshingService1D } from "./mesh/MeshingService1D";
import { ISimulationSolver } from "./simulation/ISimulationSolver";
import { SimulationSolver1D } from "./simulation/SimulationSolver1D";
import { MeshingService2D } from "./mesh/MeshingService2D";
import { SimulationSolver2D } from "./simulation/SimulationSolver2D";
import { MeshingService3D } from "./mesh/MeshingService3D";
import { SimulationSolver3D } from './simulation/SimulationSolver3D'
import { join } from 'path'
import * as fs from 'fs'

export class SimulationService {
  private static _instance: SimulationService
  private stateFilePath: string;

  // DATA TYPES
  public modelType: ModelType = ModelType.None
  public domainData: object | null = null // Holes + Areas
  public borderConditions: object | null = null
  public meshData: object | null = null
  public meshOptions: IMeshingOptions | null = null

  // SERVICES
  public meshService: IMeshingService | null = null
  public femService: IFEMService | null = null
  public simulationService: ISimulationSolver | null = null

  private constructor() {
    this.stateFilePath = join(process.cwd(), 'simulationState.json');
    this.loadState();

    return new Proxy(this, {
      set: (target, prop, value) => {
        const result = Reflect.set(target, prop, value)
        target.saveState()
        return result;
      }
    })
  }


  //FUNCTIONS
  private setOneDimensionalServices(): void {
    this.meshService = new MeshingService1D();
    this.femService = new MeshingService1D();
    this.simulationService = new SimulationSolver1D();
  }

  private setTwoDimensionalServices(): void {
    this.meshService = new MeshingService2D();
    this.femService = new MeshingService2D();
    this.simulationService = new SimulationSolver2D();
  }

  private setThreeDimensionalServices(): void {
    this.meshService = new MeshingService3D();
    this.femService = new MeshingService3D();
    this.simulationService = new SimulationSolver3D();
  }

  public setModelType(newModelType: ModelType): void {
    this.modelType = newModelType;
    switch (newModelType) {
      case ModelType.OneDimensional:
        this.setOneDimensionalServices()
        break

      case ModelType.TwoDimensional:
        this.setTwoDimensionalServices()
        break

      case ModelType.ThreeDimensional:
        this.setThreeDimensionalServices()
        break

      default:
        console.log('Error: Could not get model type')
    }
  }

  public static getInstance(): SimulationService {
    if (SimulationService._instance == null) {
      SimulationService._instance = new SimulationService()
    }

    return SimulationService._instance
  }

  public reset() {
    console.log('Simulation Data was reset.')
    this.borderConditions = null
    this.meshData = null
    this.meshOptions = null
    this.meshService = null
    this.modelType = ModelType.None
    this.domainData = null
    this.femService = null
    this.simulationService = null
  }

  private loadState(): void {
    if (fs.existsSync(this.stateFilePath)) {
      try {
        const state = JSON.parse(fs.readFileSync(this.stateFilePath, 'utf-8'));
        Object.assign(this, state);
      } catch (error) {
        console.error('❌ Error loading state:', error);
      }
    }
  }

  public saveState(): void {
    const state = { ...this };

    delete (state as any)._instance;
    delete (state as any).stateFilePath;

    try {
      fs.writeFileSync(this.stateFilePath, JSON.stringify(state, null, 2));
    } catch (e) {
      console.error(`Error: Could not save state file ${e}`)
    }
  }

  public toString(): string {
    return `Model Type: ${this.modelType}`
  }
}
