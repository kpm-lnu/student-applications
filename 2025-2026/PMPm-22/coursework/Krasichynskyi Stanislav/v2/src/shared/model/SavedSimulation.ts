export class SavedSimulation {
  private _id: number;
  private _simulationName: string;

  constructor(id: number, simulationName: string) {
    this._id = id;
    this._simulationName = simulationName;
  }

  public get id(): number { return this._id; }
  public set id(id: number) { this._id = id; }

  public get simulationName(): string { return this._simulationName; }
  public set simulationName(simulationName: string) { this._simulationName = simulationName; }



}
