import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import { checkHealth, fetchTrips, reconstructRoute, BASE_URL } from './services/api';

export default function App() {
  const [backendConnected, setBackendConnected] = useState(false);
  const [checkingHealth, setCheckingHealth] = useState(false);
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Ping backend health & load trips
  const verifyBackend = useCallback(async () => {
    setCheckingHealth(true);
    setError(null);
    try {
      await checkHealth();
      setBackendConnected(true);

      const availableTrips = await fetchTrips();
      setTrips(availableTrips);
      if (availableTrips.length > 0 && !selectedTrip) {
        setSelectedTrip(availableTrips[0]);
      }
    } catch (err) {
      setBackendConnected(false);
      setError(`Cannot connect to Laptop 1 backend at ${BASE_URL}. Ensure server is running.`);
    } finally {
      setCheckingHealth(false);
    }
  }, [selectedTrip]);

  useEffect(() => {
    verifyBackend();
  }, []);

  // Handle route reconstruction execution
  const handleRunReconstruction = async () => {
    if (!selectedTrip || loading) return;

    setLoading(true);
    setError(null);
    try {
      const data = await reconstructRoute(selectedTrip, 'gradient_boosting');
      setResult(data);
    } catch (err) {
      console.error('Reconstruction error:', err);
      setError(err.message || 'Route reconstruction failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Header
        backendConnected={backendConnected}
        backendUrl={BASE_URL}
        checkingHealth={checkingHealth}
        onRefreshHealth={verifyBackend}
      />

      <Dashboard
        trips={trips}
        selectedTrip={selectedTrip}
        onSelectTrip={setSelectedTrip}
        onRunReconstruction={handleRunReconstruction}
        loading={loading}
        backendConnected={backendConnected}
        backendUrl={BASE_URL}
        result={result}
        error={error}
      />
    </div>
  );
}
