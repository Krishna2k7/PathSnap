/**
 * api.js - Robust Client for PathSnap API (Laptop 1 communication).
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Checks server health and model readiness.
 */
export async function checkHealth() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3000);

  try {
    const res = await fetch(`${BASE_URL}/health`, {
      signal: controller.signal,
      headers: { 'Accept': 'application/json' },
    });
    clearTimeout(timeoutId);
    if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Connection timed out. Check Laptop 1 IP and port.');
    }
    throw new Error(err.message || 'Cannot connect to Laptop 1 backend.');
  }
}

/**
 * Retrieves the list of available evaluation trips from the backend registry.
 */
export async function fetchTrips() {
  try {
    const res = await fetch(`${BASE_URL}/trips`, {
      headers: { 'Accept': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status} fetching trips`);
    const data = await res.json();
    return data.trips || [];
  } catch (err) {
    console.error('Failed to load trips from backend:', err);
    throw err;
  }
}

/**
 * Triggers full map-matching and route reconstruction for the selected trip.
 * @param {string} tripId - The trip identifier (e.g. "00000007")
 * @param {string} modelType - ML model ("gradient_boosting")
 */
export async function reconstructRoute(tripId, modelType = 'gradient_boosting') {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout for large trajectories

  try {
    const res = await fetch(`${BASE_URL}/reconstruct/${tripId}?model_type=${modelType}`, {
      method: 'POST',
      signal: controller.signal,
      headers: { 'Accept': 'application/json' },
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.error || `Server returned error HTTP ${res.status}`);
    }

    return await res.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Inference timed out. The trajectory may be exceptionally large.');
    }
    throw err;
  }
}

export { BASE_URL };
