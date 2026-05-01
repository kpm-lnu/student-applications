from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    StepRequest,
    RunRequest,
    CreateFromPointsRequest,
)
from .geometry import load_area_from_image_bytes, load_area_from_points
from .simulation_service import (
    serialize_state,
    build_area_from_state,
    do_single_step,
)
from .fem_service import solve_fields

app = FastAPI(title="Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/simulation/from-image")
async def create_simulation_from_image(
    file: UploadFile = File(...),
    threshold: int = 2,
    sampling_rate: float = 0.02,
):
    try:
        content = await file.read()
        area = load_area_from_image_bytes(
            content,
            threshold=threshold,
            sampling_rate=sampling_rate,
        )
        area.triangulate_polygon()
        vertices, triangles, concentration, pressure = solve_fields(area)
        return serialize_state(
            area,
            vertices,
            triangles,
            concentration,
            pressure,
            delta_t=0.1,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/simulation/from-points")
def create_simulation_from_points(request: CreateFromPointsRequest):
    try:
        area = load_area_from_points([p.model_dump() for p in request.points])
        area.triangulate_polygon(
            min_angle=request.params.triangulation_min_angle,
            max_area=request.params.triangulation_max_area,
            max_steiner_points=request.params.triangulation_max_steiner_points,
        )

        vertices, triangles, concentration, pressure = solve_fields(
            area,
            a_11=request.params.a_11,
            a_22=request.params.a_22,
            force_value=request.params.force_value,
            boundary_value=request.params.boundary_value,
            pressure_a=request.params.pressure_a,
            pressure_g=request.params.pressure_g,
            pressure_chi=request.params.pressure_chi,
            pressure_k=request.params.pressure_k,
            pressure_dimension=request.params.pressure_dimension,
        )

        return serialize_state(
            area,
            vertices,
            triangles,
            concentration,
            pressure,
            delta_t=request.params.delta_t,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/simulation/step")
def simulation_step(request: StepRequest):
    try:
        area = build_area_from_state(request.state)
        area, vertices, triangles, concentration, pressure, delta_t = do_single_step(
            area,
            request.params,
            request.state.delta_t,
        )
        return serialize_state(
            area,
            vertices,
            triangles,
            concentration,
            pressure,
            delta_t,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/simulation/run")
def simulation_run(request: RunRequest):
    try:
        area = build_area_from_state(request.state)
        delta_t = request.state.delta_t

        vertices = []
        triangles = []
        concentration = []
        pressure = []

        for _ in range(request.steps):
            area, vertices, triangles, concentration, pressure, delta_t = do_single_step(
                area,
                request.params,
                delta_t,
            )

        return serialize_state(
            area,
            vertices,
            triangles,
            concentration,
            pressure,
            delta_t,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
