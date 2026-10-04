using System;
using ApiBackend.Api.Data;
using ApiBackend.Api.Dtos.Plan;
using ApiBackend.Api.Interfaces;
using ApiBackend.Api.Models;
using Microsoft.EntityFrameworkCore;


namespace ApiBackend.Api.Repositories
{
    public class PlanRepository : IPlanRepository
    {
        private readonly ApplicationDBContext _context;
        public PlanRepository(ApplicationDBContext context)
        {
            _context = context;
        }

        public async Task<Plan?> CreatePlan(Plan dto)
        {
            var existingPlan = await _context.Plan
                .FirstOrDefaultAsync(p => p.Name == dto.Name || p.Price == dto.Price || p.PlanType == dto.PlanType);

            if(existingPlan != null)
            {
                return null;
            }

            var newPlan = await _context.Plan.AddAsync(dto);
            await _context.SaveChangesAsync();

            return newPlan.Entity;
        }

        public async Task<Plan?> GetPlanById(Guid planId)
        {
            return await _context.Plan.FirstOrDefaultAsync(p => p.Id == planId);
        }

        public async Task<(Plan, List<User>)?> GetUsersPlan(Guid planId)
        {
            var response = await _context.Plan
                .Include(p => p.Users)
                .FirstOrDefaultAsync(p => p.Id == planId);
            
            if(response == null)
            {
                return null;
            }

            return (response, response.Users);
        }

        public async Task<Plan?> UpdatePlan(Guid planId, Plan dto)
        {
            var plan = await _context.Plan.FirstOrDefaultAsync(p => p.Id == planId);
            if(plan == null)
            {
                return null;
            }

            plan.Name = dto.Name;
            plan.Price = dto.Price;
            await _context.SaveChangesAsync();

            return plan;
        }

        public async Task<bool> DeletePlan(Guid planId)
        {
            var plan = await _context.Plan.FirstOrDefaultAsync(p => p.Id == planId);
            if(plan == null)
            {
                return false;
            }

            var inUse = await _context.User.AnyAsync(u => u.PlanId == planId || u.PendingPlanId == planId);
            if(inUse)
            {
                return false;
            }

            _context.Plan.Remove(plan);
            await _context.SaveChangesAsync();

            return true;
        }
    }
}