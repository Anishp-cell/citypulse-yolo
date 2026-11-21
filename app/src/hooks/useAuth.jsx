import { createContext, useCallback, useContext, useMemo } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { setLanguage } from '../i18n';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const qc = useQueryClient();
  const me = useQuery({
    queryKey: ['auth', 'me'],
    queryFn: async () => {
      try {
        return await api.get('/api/auth/me');
      } catch (e) {
        if (e.status === 401) return null;
        throw e;
      }
    },
    staleTime: 5 * 60_000,
    retry: 1,
  });

  const setUser = useCallback((user) => {
    qc.setQueryData(['auth', 'me'], user);
    if (user?.language) setLanguage(user.language);
    return user;
  }, [qc]);

  const value = useMemo(() => ({
    user: me.data ?? null,
    loading: me.isPending,
    login: async (email, password) => setUser(await api.post('/api/auth/login', { email, password })),
    register: async (data) => setUser(await api.post('/api/auth/register', data)),
    logout: async () => {
      await api.post('/api/auth/logout').catch(() => {});
      qc.clear();
      qc.setQueryData(['auth', 'me'], null);
    },
    updateProfile: async (patch) => setUser(await api.patch('/api/auth/me', patch)),
    deleteAccount: async () => {
      await api.del('/api/auth/me');
      qc.clear();
      qc.setQueryData(['auth', 'me'], null);
    },
    refresh: () => me.refetch(),
  }), [me, qc, setUser]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}

// eslint-disable-next-line react-refresh/only-export-components
export const isStaff = (user) => user?.role === 'officer' || user?.role === 'admin';
