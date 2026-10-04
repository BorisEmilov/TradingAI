using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;
using ApiBackend.Api.Models;


namespace ApiBackend.Api.Dtos.User
{
    public class GetUserDto
    {
        public Guid Id { get; set; }
        public string Email { get; set; }
        public string FullName { get; set; }
        public Roles? Role { get; set; }
        public Guid? PlanId { get; set; }
        public DateTime? PlanPaymentDate { get; set; }
        public Boolean Verified { get; set; }
        public DateTime CreatedAt { get; set; }
        public Guid? PendingPlanId { get; set; }
    }
}