import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Optional bearer token support (for Clerk-integrated deployments)
client.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const authApi = {
  me: async () => {
    const response = await client.get('/auth/me');
    return response.data;
  },
};

export const scoringApi = {
  mergePlan: async (files) => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));

    const response = await client.post('/merge-plan', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  train: async (modelName, files, targetCol = null, mode = "supervised") => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));

    let url = `/train?model_name=${encodeURIComponent(modelName)}&mode=${encodeURIComponent(mode)}`;
    if (targetCol) url += `&target_column=${encodeURIComponent(targetCol)}`;

    const response = await client.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  score: async (modelName, files) => {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));

    const url = `/score-csv?model_name=${encodeURIComponent(modelName)}`;
    const response = await client.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  feedback: async (modelName, file, outcomeColumn = null, autoRetrain = false, feedbackWeight = 2) => {
    const formData = new FormData();
    formData.append('file', file);

    let url = `/feedback?model_name=${encodeURIComponent(modelName)}`;
    if (outcomeColumn) url += `&outcome_column=${encodeURIComponent(outcomeColumn)}`;
    url += `&auto_retrain=${encodeURIComponent(autoRetrain)}`;
    url += `&feedback_weight=${encodeURIComponent(feedbackWeight)}`;

    const response = await client.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  retrainFromFeedback: async (modelName, feedbackWeight = 2) => {
    const url = `/retrain-from-feedback?model_name=${encodeURIComponent(modelName)}&feedback_weight=${encodeURIComponent(feedbackWeight)}`;
    const response = await client.post(url);
    return response.data;
  },
  retrainSegmentFromFeedback: async (modelName, segmentDimension, segmentValue, feedbackWeight = 2) => {
    const url = `/retrain-segment-feedback?model_name=${encodeURIComponent(modelName)}&segment_dimension=${encodeURIComponent(segmentDimension)}&segment_value=${encodeURIComponent(segmentValue)}&feedback_weight=${encodeURIComponent(feedbackWeight)}`;
    const response = await client.post(url);
    return response.data;
  },
};

export const modelsApi = {
  list: async () => {
    const response = await client.get('/models');
    return response.data;
  },
  get: async (modelName) => {
    const response = await client.get(`/models/${encodeURIComponent(modelName)}`);
    return response.data;
  },
  delete: async (modelName) => {
    const response = await client.delete(`/models/${encodeURIComponent(modelName)}`);
    return response.data;
  },
};

export default client;
