import { ApplicationSettings } from "../../settings/ApplicationSettings";
import { AreaData } from "./AreaData";
import { MeshData } from "./MeshData";
import { MathematicalModelData } from "./MathematicalModelData";

export class ApplicationData {

  /* === Data === */
  private _applicationSettings: ApplicationSettings;
  private _areaData: AreaData;
  private _meshDate: MeshData;
  private _mathematicalModelData: MathematicalModelData;

  /* === Services === */


  constructor(areaData: AreaData, meshDate: MeshData, mathematicalModelData: MathematicalModelData) {
    this._applicationSettings = ApplicationSettings.getInstance();
    this._areaData = areaData;
    this._meshDate = meshDate;
    this._mathematicalModelData = mathematicalModelData;
  }

  public get applicationSettings(): ApplicationSettings {
    return this._applicationSettings;
  }
  public set applicationSettings(value: ApplicationSettings) {
    this._applicationSettings = value;
  }

  public get areaData(): AreaData {
    return this._areaData;
  }
  public set areaData(value: AreaData) {
    this._areaData = value;
  }

  public get meshData(): MeshData {
    return this._meshDate;
  }
  public set meshDate(value: MeshData) {
    this._meshDate = value;
  }

  public get mathematicalModelData(): MathematicalModelData {
    return this._mathematicalModelData;
  }
  public set mathematicalModelData(value: MathematicalModelData) {
    this._mathematicalModelData = value;
  }


}
