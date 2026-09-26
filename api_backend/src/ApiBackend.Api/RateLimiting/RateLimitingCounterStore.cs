using System;
using System.Collections.Concurrent;
using ApiBackend.Api.RateLimiting;

public class RateLimitingCounterStore : IRateLimitingCounterStore
{
    private class Counter
    {
        public int Count;
        public DateTime WindowStart;
    }

    private readonly ConcurrentDictionary<string, Counter> _counters = new();

    public Task<bool> TryAcquireAsync(string key, int limit, TimeSpan window)
    {
        var now = DateTime.UtcNow;
        var counter = _counters.GetOrAdd(key, _ => new Counter { Count = 0, WindowStart = now });

        lock (counter)
        {
            if (now - counter.WindowStart >= window)
            {
                counter.WindowStart = now;
                counter.Count = 0;
            }
            if (counter.Count >= limit)
            {
                return Task.FromResult(false);
            }
            counter.Count++;
            return Task.FromResult(true);
        }
    }

    public void PurgeExpired(TimeSpan maxIdleTime)
    {
        var now = DateTime.UtcNow;
        foreach (var kvp in _counters)
        {
            var counter = kvp.Value;
            bool isStale;

            lock (counter)
            {
                isStale = now - counter.WindowStart >= maxIdleTime;
            }
            if (isStale)
            {
                _counters.TryRemove(kvp.Key, out _);
            }
        }
    }

    public int ActiveCountersCount
    {
        get { return _counters.Count; }
    }
}
