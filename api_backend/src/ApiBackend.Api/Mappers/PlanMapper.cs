using System;
using ApiBackend.Api.Dtos.Plan;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.Mappers
{
    public static class PlanMapper
    {
        public static GetPlanDto FromPlanToGetPlanDto(this Plan plan)
        {
            return new GetPlanDto
            {
                Id = plan.Id,
                Price = plan.Price,
                Name = plan.Name,
                PlanType = plan.PlanType
            };
        }

        public static Plan FromCreatePlanDtoToPlan(this CreatePlanDto dto)
        {
            return new Plan
            {
                Price = dto.Price,
                Name = dto.Name,
                PlanType = dto.PlanType
            };
        }

        public static Plan FromUpdatePlanDtoToPlan(this UpdatePlanDto dto)
        {
            return new Plan
            {
                Price = dto.Price,
                Name = dto.Name,
            };
        }
    }
}