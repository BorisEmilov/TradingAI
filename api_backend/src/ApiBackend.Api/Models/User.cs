using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;


namespace ApiBackend.Api.Models
{
    public enum Roles
    {
        ADMIN,
        SUPER_ADMIN,
        FREE_USER,
        PLUS_USER,
        GOLDEN_USER,
    }
    public class User
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public string Email { get; set; }
        public string FullName { get; set; }
        public string HashedPassword { get; set; }
        public Roles? Role { get; set; } = Roles.FREE_USER;
        public List<RefreshDoc> RefreshDocs { get; set; } = new();
        public List<MtAccount> MtAccounts { get; set; } = new();
        public Guid? PlanId { get; set; }
        public Plan? Plan { get; set; }
        public DateTime? PlanPaymentDate { get; set; }
        public Boolean Verified { get; set; } = false;
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public Guid? PendingPlanId { get; set; }
        public Plan? PendingPlan { get; set; }
    }

}