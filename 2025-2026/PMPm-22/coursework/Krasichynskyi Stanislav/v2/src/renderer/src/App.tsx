import { Route, Routes } from 'react-router-dom'
import { Home } from '@renderer/pages/Home'
import { Settings } from '@renderer/pages/Settings'
import { TopBar } from '@renderer/components/top-bar/TopBar'
import { SavedSimulations } from "@renderer/pages/SavedSimulations";
import { NewSimulation } from "@renderer/pages/NewSimulation";

function App(): React.JSX.Element {
  return (
    <>
      <TopBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/saved-simulations" element={<SavedSimulations />} />
        <Route path="/new-simulation" element={<NewSimulation />} />
      </Routes>
    </>
  )
}

export default App
