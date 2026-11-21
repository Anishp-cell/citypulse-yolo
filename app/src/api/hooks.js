import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';

export const keys = {
  hazards: (bbox, status) => ['hazards', bbox, status],
  hazard: (id) => ['hazard', String(id)],
  complaint: (id, lang) => ['complaint', String(id), lang],
  myReports: ['me', 'reports'],
  following: ['me', 'following'],
  impact: ['me', 'impact'],
  scorecard: (city) => ['scorecard', city],
};

export function useHazards(bbox, status = 'active', options = {}) {
  return useQuery({
    queryKey: keys.hazards(bbox, status),
    queryFn: () => api.get(`/api/hazards?bbox=${bbox}&status=${status}`),
    enabled: Boolean(bbox),
    placeholderData: (prev) => prev,
    staleTime: 30_000,
    ...options,
  });
}

export const useHazard = (id) => useQuery({
  queryKey: keys.hazard(id), queryFn: () => api.get(`/api/hazards/${id}`), retry: (n, e) => e?.status !== 404 && n < 2,
});

export const useComplaint = (id, lang, enabled = true) => useQuery({
  queryKey: keys.complaint(id, lang), queryFn: () => api.get(`/api/hazards/${id}/complaint?lang=${lang}`), enabled,
});

export const useMyReports = (enabled = true) => useQuery({
  queryKey: keys.myReports, queryFn: () => api.get('/api/me/reports'), enabled,
});
export const useFollowing = (enabled = true) => useQuery({
  queryKey: keys.following, queryFn: () => api.get('/api/me/following'), enabled,
});
export const useImpact = (enabled = true) => useQuery({
  queryKey: keys.impact, queryFn: () => api.get('/api/me/impact'), enabled,
});
export const useScorecard = (city) => useQuery({
  queryKey: keys.scorecard(city), queryFn: () => api.get(`/api/scorecard?city=${encodeURIComponent(city)}`),
});

function useHazardMutation(id, fn) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.hazard(id) });
      qc.invalidateQueries({ queryKey: ['me'] });
      qc.invalidateQueries({ queryKey: ['hazards'] });
    },
  });
}

export const useFollow = (id) => useHazardMutation(id, (follow) =>
  (follow ? api.post(`/api/hazards/${id}/follow`) : api.del(`/api/hazards/${id}/follow`)));

export const useFiled = (id) => useHazardMutation(id, (body) => api.post(`/api/hazards/${id}/filed`, body));

export const useVerify = (id) => useHazardMutation(id, ({ isFixed, photo }) => {
  const form = new FormData();
  form.append('is_fixed', isFixed ? 'true' : 'false');
  if (photo) form.append('photo', photo);
  return api.form(`/api/hazards/${id}/verify`, form);
});

export const useFlag = (id) => useHazardMutation(id, (reason) => api.post(`/api/hazards/${id}/flag`, { reason }));

// ---- reporting ----
export function analyzePhoto({ file, lat, lng, accuracy, source }) {
  const form = new FormData();
  form.append('file', file, file.name || 'photo.jpg');
  form.append('lat', String(lat));
  form.append('lng', String(lng));
  if (accuracy != null) form.append('accuracy_m', String(Math.round(accuracy)));
  form.append('location_source', source);
  return api.form('/api/reports/analyze', form, { idempotent: true });
}

export function submitReport({ uploadId, type, severity, note, attachTo, idempotencyKey }) {
  return api.post('/api/reports', {
    upload_id: uploadId, hazard_type: type, severity, note: note || null, attach_to_hazard_id: attachTo ?? null,
  }, { idempotent: true, headers: { 'Idempotency-Key': idempotencyKey } });
}

// ---- admin ----
export const useAdminQueue = (params) => useQuery({
  queryKey: ['admin', 'queue', params],
  queryFn: () => api.get(`/api/admin/queue?${new URLSearchParams(Object.entries(params).filter(([, v]) => v))}`),
  placeholderData: (prev) => prev,
});
export const useModeration = () => useQuery({ queryKey: ['admin', 'moderation'], queryFn: () => api.get('/api/admin/moderation') });
export const useAdminStats = () => useQuery({ queryKey: ['admin', 'stats'], queryFn: () => api.get('/api/admin/stats'), refetchInterval: 60_000 });
export const useUsers = (q, offset) => useQuery({
  queryKey: ['admin', 'users', q, offset],
  queryFn: () => api.get(`/api/admin/users?${new URLSearchParams({ q: q || '', offset: String(offset || 0) })}`),
  placeholderData: (prev) => prev,
});

export function useAdminAction() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ path, body, form }) => (form ? api.form(path, form) : api.post(path, body)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin'] });
      qc.invalidateQueries({ queryKey: ['hazard'] });
      qc.invalidateQueries({ queryKey: ['hazards'] });
    },
  });
}

export function useSetRole() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, role }) => api.patch(`/api/admin/users/${id}/role`, { role }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['admin', 'users'] }),
  });
}
