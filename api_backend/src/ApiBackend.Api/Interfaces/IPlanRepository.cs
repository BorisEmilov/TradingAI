using System;
using ApiBackend.Api.Dtos.Plan;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.Interfaces
{
    public interface IPlanRepository
    {
        Task<Plan?> CreatePlan(Plan dto);
        Task<Plan?> GetPlanById(Guid planId);
        Task<(Plan, List<User>)?> GetUsersPlan(Guid planId);
        Task<Plan?> UpdatePlan(Guid planId, Plan dto);
        Task<bool> DeletePlan(Guid planId);
    }
}