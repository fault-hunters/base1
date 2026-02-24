import { Route, Routes } from "react-router-dom"
import Detection from "./pages/Detection"
import History from "./pages/History"
import AIGCGenerator from "./pages/AIGCGenerator"

export default function App() {
    return (
        <Routes>
            <Route path="/" element={<AIGCGenerator />} />
            <Route path="/detect" element={<Detection />} />
            <Route path="/history" element={<History />} />
            <Route path="/aigc" element={<AIGCGenerator />} />
        </Routes>
    )
}
