using System;
using ApiBackend.Api.Data;
using ApiBackend.Api.Models;
using Microsoft.EntityFrameworkCore;

namespace ApiBackend.Api.Services
{
    public class PlanExpirationService : BackgroundService
    {
        private readonly IServiceScopeFactory _scopeFactory;

        public PlanExpirationService(IServiceScopeFactory scopeFactory)
        {
            _scopeFactory = scopeFactory;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                using var scope = _scopeFactory.CreateScope();

                var context = scope.ServiceProvider
                    .GetRequiredService<ApplicationDBContext>();

                var now = DateTime.UtcNow;

                var users = await context.User
                    .Include(u => u.Plan)
                    .Include(u => u.PendingPlan)
                    .Where(u =>
                        u.PlanPaymentDate != null &&
                        u.PlanPaymentDate <= now &&
                        u.PendingPlanId != null)
                    .ToListAsync(stoppingToken);

                foreach (var user in users)
                {
                    user.PlanId = user.PendingPlanId;
                    user.PendingPlanId = null;
                    user.PlanPaymentDate = now;

                    if (user.Plan != null)
                    {
                        user.Role = user.Plan.PlanType switch
                        {
                            PlanTypes.PLUS => Roles.PLUS_USER,
                            PlanTypes.GOLDEN => Roles.GOLDEN_USER,
                            _ => user.Role
                        };
                    }
                }

                await context.SaveChangesAsync(stoppingToken);

                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
            }
        }
    }
}