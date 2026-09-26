namespace ApiBackend.Api.RateLimiting;

public interface IRateLimitingCounterStore {
  Task<bool> TryAcquireAsync(string key, int limit, TimeSpan window);
}
