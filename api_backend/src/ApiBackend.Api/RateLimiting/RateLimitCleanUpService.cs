using System;

namespace ApiBackend.Api.RateLimiting
{
    public class RateLimitCleanUpService : BackgroundService
    {
        private readonly RateLimitingCounterStore _store;
        private readonly TimeSpan _interval = TimeSpan.FromMinutes(5);
        private readonly TimeSpan _maxIdleTime = TimeSpan.FromMinutes(10);

        public RateLimitCleanUpService(RateLimitingCounterStore store)
        {
            _store = store;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    _store.PurgeExpired(_maxIdleTime);
                }
                catch
                {

                }

                await Task.Delay(_interval, stoppingToken);
            }
        }
    }

}