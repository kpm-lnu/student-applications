from fastapi import FastAPI
from pydantic import BaseModel
from dmsir_core.dmsir_dde import simulate, Params
import numpy as np

app = FastAPI(title="D-MSIR COVID-UA mini-API")


class Req(BaseModel):
    days: int = 180
    step: float = .25
    beta: float | None = None
    tau:  float | None = None


@app.post("/forecast")
def forecast(req: Req):
    p = Params(beta=req.beta or .28, tau=req.tau or 3.7)
    m = simulate(req.days, req.step, p)
    return dict(
        t=m[:, 0].tolist(), D=m[:, 1].tolist(), M=m[:, 2].tolist(),
        I=m[:, 3].tolist(), R=m[:, 4].tolist(),
        peak=int(m[:, 3].max()),
        day=float(m[:, 0][np.argmax(m[:, 3])])
    )
