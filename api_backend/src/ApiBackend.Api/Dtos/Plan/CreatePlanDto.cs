using System;
using System.ComponentModel.DataAnnotations;
using ApiBackend.Api.Models;


namespace ApiBackend.Api.Dtos.Plan
{
    public class CreatePlanDto
    {
        [Required]
        public float Price { get; set; }
        [Required, MaxLength(50)]
        public string Name { get; set; }
        [Required]
        public PlanTypes PlanType { get; set; }
    }
}