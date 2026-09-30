import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";


export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
