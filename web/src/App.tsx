import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Home } from '@/pages/Home';
import { Game } from '@/pages/Game';
import { Stats } from '@/pages/Stats';
import { Replay } from '@/pages/Replay';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/game/:id" element={<Game />} />
          <Route path="/replay/:id" element={<Replay />} />
          <Route path="/stats" element={<Stats />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
