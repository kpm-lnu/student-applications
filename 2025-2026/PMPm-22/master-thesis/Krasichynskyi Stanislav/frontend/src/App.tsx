import { BrowserRouter, Route, Routes } from "react-router-dom";
import { SimulationPage } from "./pages/SimulationPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<SimulationPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;