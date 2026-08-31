import { useEffect, useState } from "react";

const cache = new Map<string, unknown>();
const DATA_BASE = `${import.meta.env.BASE_URL}data/10k/`;

export async function loadJson<T>(file: string): Promise<T> {
  if (cache.has(file)) {
    return cache.get(file) as T;
  }
  const response = await fetch(`${DATA_BASE}${file}`);
  if (!response.ok) {
    throw new Error(`Failed to load /data/10k/${file}`);
  }
  const value = (await response.json()) as T;
  cache.set(file, value);
  return value;
}

export function useJsonData<T>(file: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    loadJson<T>(file)
      .then((value) => {
        if (!alive) return;
        setData(value);
        setError(null);
      })
      .catch((err: Error) => {
        if (!alive) return;
        setError(err.message);
      })
      .finally(() => {
        if (!alive) return;
        setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [file]);

  return { data, error, loading };
}
