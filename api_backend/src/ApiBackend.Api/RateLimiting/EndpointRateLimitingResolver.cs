using System;
using System.Security.Claims;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.RateLimiting
{

    public class ResolvedRateLimit
    {
        public int LimitRequests { get; set; }
        public int WindowSeconds { get; set; }
    }

    public class EndpointRateLimitResolver
    {
        private readonly ResolvedRateLimit _globalDefault = new()
        {
            LimitRequests = 10,
            WindowSeconds = 10
        };

        public bool HasRateLimitConfigured(HttpContext context)
        {
            var endpoint = context.GetEndpoint();
            if (endpoint is null)
            {
                return false;
            }

            return endpoint.Metadata.GetMetadata<RateLimitAttribute>() != null
              || endpoint.Metadata.GetOrderedMetadata<RateLimitPerRoleAttribute>().Count > 0;
        }

        public ResolvedRateLimit Resolve(HttpContext context)
        {
            var endpoint = context.GetEndpoint();
            if (endpoint is null)
            {
                return _globalDefault;
            }

            var roleAttrs = endpoint.Metadata.GetOrderedMetadata<RateLimitPerRoleAttribute>();
            var baseAttr = endpoint.Metadata.GetMetadata<RateLimitAttribute>();

            var userRoles = GetUserRoles(context);

            if (roleAttrs.Count > 0 && userRoles.Count > 0)
            {
                var matches = roleAttrs.Where(a => userRoles.Contains(a.Role)).ToList();

                if (matches.Count > 0)
                {
                    var best = matches.OrderByDescending(a => a.LimitRequests).First();
                    return new ResolvedRateLimit
                    {
                        LimitRequests = best.LimitRequests,
                        WindowSeconds = best.WindowSeconds
                    };
                }
            }

            if (baseAttr is not null)
            {
                return new ResolvedRateLimit
                {
                    LimitRequests = baseAttr.LimitRequests,
                    WindowSeconds = baseAttr.WindowSeconds
                };
            }

            return _globalDefault;
        }

        public string BuildPartitionKey(HttpContext context)
        {
            var endpoint = context.GetEndpoint()?.DisplayName ?? context.Request.Path.ToString();
            var identity = context.User.Identity?.IsAuthenticated == true
              ? context.User.FindFirstValue(ClaimTypes.NameIdentifier)
              : null;
            identity ??= context.Connection.RemoteIpAddress?.ToString() ?? "anon";

            return $"ratelimit:{endpoint}:{identity}";
        }

        private static List<Roles> GetUserRoles(HttpContext context)
        {
            if (context.User.Identity?.IsAuthenticated != true)
            {
                return new List<Roles>();
            }

            return context.User.Claims
              .Where(c => c.Type == ClaimTypes.Role)
              .Select(c => Enum.TryParse<Roles>(c.Value, ignoreCase: true, out var role)
                ? (Roles?)role : null)
              .Where(r => r.HasValue)
              .Select(r => r!.Value)
              .ToList();
        }
    }

}