using System;
using StackExchange.Redis;

namespace ApiBackend.Api.Data
{
    public class CacheService
    {
        private readonly IDatabase _redis;
        public CacheService(IConnectionMultiplexer redis)
        {
            _redis = redis.GetDatabase();
        }

        public async Task SetAsync(string key, string value, TimeSpan? expiry = null)
        {
            await _redis.StringSetAsync(key, value, expiry);
        }

        public async Task<string?> GetAsync(string key)
        {
            var value = await _redis.StringGetAsync(key);
            return value.HasValue ? value.ToString() : null;
        }

        public async Task<bool> DeleteAsync(string key)
        {
            return await _redis.KeyDeleteAsync(key);
        }
    }
}