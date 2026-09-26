using System;
using ApiBackend.Api.Models;


namespace ApiBackend.Api.Dtos.Plan
{
    public class GetPlanDto
    {
        public Guid Id { get; set; }
        public float Price { get; set; }
        public string Name { get; set; }
        public PlanTypes PlanType { get; set; }
    }
}