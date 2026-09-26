using System;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.Dtos.Plan
{
    public class UpdatePlanDto
    {
        public float Price { get; set; }
        public string Name { get; set; }
    }
}