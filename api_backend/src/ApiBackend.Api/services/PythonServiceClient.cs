using System;
using System.Net.Http.Headers;

namespace ApiBackend.Api.Services
{
    public class PythonServiceClient
    {
        private readonly HttpClient _httpClient;

        public PythonServiceClient(IHttpClientFactory httpClientFactory)
        {
            _httpClient = httpClientFactory.CreateClient("PythonService");
        }

        // `token`: token de sesión del gateway (Bearer). Va por petición, no en DefaultRequestHeaders,
        // para que cada llamada lleve el token de SU cuenta MT5.
        public async Task<T?> GetAsync<T>(string endpoint, string? token = null)
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, endpoint);
            return await SendAsync<T>(request, token);
        }

        public async Task<TResponse?> PostAsync<TRequest, TResponse>(string endpoint, TRequest? body = default, string? token = null)
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, endpoint);
            if (body is not null)
            {
                request.Content = JsonContent.Create(body);
            }
            return await SendAsync<TResponse>(request, token);
        }

        private async Task<T?> SendAsync<T>(HttpRequestMessage request, string? token)
        {
            if (!string.IsNullOrEmpty(token))
            {
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
            }
            using var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadFromJsonAsync<T>();
        }
    }
}
