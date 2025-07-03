import logo from './logo.svg';
import React, {useState} from 'react';
import './App.css';

function App() {

  const [username, setUsername] = useState('');

  const [recommendations, setRecommendations] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!username){
      setError('Username tidak boleh kosong.');
      return;
    }

    setLoading(true);
    setError('');
    setRecommendations(null);

    try {
      const response = await fetch(`http://localhost:8000/recommend/${username}`);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Terjadi kesalahan pada server.' );
      }

      const data = await response.json();
      setRecommendations(data.recommendations);

    }catch (err) {
      setError(err.message);
    }finally {
      setLoading(false);
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Git Graph</h1>
        <p>Dapatkan Rekomendasi Repositori berbasis GNN</p>

        <form onSubmit={handleSubmit}>
          <input
            type='text'
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Masukkan username GitHub..."
          />
          <button type='submit' disabled={loading}>
            {loading ? 'Mencari...' : 'Dapatkan Rekomendasi'}
          </button>
        </form>

        {error && <p className='error-message'>{error}</p>}

        {recommendations && (
          <div className='results'>
            <h2>Rekomendasi untuk "{username}"</h2>
            <ul>
              {recommendations.map((repo, index) => (
                <li key={index}>
                  <a href={`https://github.com/${repo}`} target='_blank' rel='noopener noreferrer'>
                    {repo}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
