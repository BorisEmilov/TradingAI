using System;
using ApiBackend.Api.Data;
using ApiBackend.Api.Models;
using Microsoft.EntityFrameworkCore;

namespace ApiBackend.Api.Services
{
    public class PlanExpirationService : BackgroundService
    {
        private readonly IServiceScopeFactory _scopeFactory;
        public const int PlanDurationDays = 30;  // también la usa UserRepository.CancellPlan

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
                        u.PlanPaymentDate <= now.AddDays(-PlanDurationDays) &&
                        u.PendingPlanId != null)
                    .ToListAsync(stoppingToken);

                foreach (var user in users)
                {
                    var newPlan = user.PendingPlan;  // user.Plan sigue siendo el plan VIEJO
                    user.PlanId = user.PendingPlanId;
                    user.PendingPlanId = null;
                    user.PlanPaymentDate = now;

                    if (newPlan != null && user.Role is not (Roles.ADMIN or Roles.SUPER_ADMIN))
                    {
                        user.Role = newPlan.PlanType switch
                        {
                            PlanTypes.PLUS => Roles.PLUS_USER,
                            PlanTypes.GOLDEN => Roles.GOLDEN_USER,
                            _ => Roles.FREE_USER
                        };
                    }
                }

                await context.SaveChangesAsync(stoppingToken);

                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
            }
        }
    }
}