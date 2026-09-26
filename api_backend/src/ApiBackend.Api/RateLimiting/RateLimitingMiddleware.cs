using System;

namespace ApiBackend.Api.RateLimiting
{
    public class RateLimitingMiddleware
    {
        private readonly RequestDelegate _next;

        public RateLimitingMiddleware(RequestDelegate next)
        {
            _next = next;
        }

        public async Task InvokeAsync(
            HttpContext context,
            EndpointRateLimitResolver resolver,
            IRateLimitingCounterStore store)
        {
            if (!resolver.HasRateLimitConfigured(context))
            {
                await _next(context);
                return;
            }

            var limit = resolver.Resolve(context);
            var key = resolver.BuildPartitionKey(context);
            var window = TimeSpan.FromSeconds(limit.WindowSeconds);

            var allowed = await store.TryAcquireAsync(key, limit.LimitRequests, window);

            // Cabeceras informativas para el cliente
            context.Response.Headers["X-RateLimit-Limit"] = limit.LimitRequests.ToString();
            context.Response.Headers["X-RateLimit-Window-Seconds"] = limit.WindowSeconds.ToString();

            if (!allowed)
            {
                context.Response.StatusCode = StatusCodes.Status429TooManyRequests;
                context.Response.Headers["Retry-After"] = limit.WindowSeconds.ToString();
                context.Response.ContentType = "application/json";
                await context.Response.WriteAsync(
                    $"{{\"error\":\"Rate limit exceeded\",\"retryAfterSeconds\":{limit.WindowSeconds}}}");
                return;
            }

            await _next(context);
        }
    }

}