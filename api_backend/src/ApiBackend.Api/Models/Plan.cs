using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ApiBackend.Api.Models
{
    public enum PlanTypes
    {
        FREE,
        PLUS,
        GOLDEN,
    }
    public class Plan
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public float Price { get; set; }
        public string Name { get; set; }
        public PlanTypes PlanType { get; set; }
        public List<User> Users { get; set; } = new();
    }
}

