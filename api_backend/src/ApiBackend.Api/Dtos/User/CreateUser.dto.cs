using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;
using ApiBackend.Api.Models;


namespace ApiBackend.Api.Dtos.User
{
    public class CreateUserDto
    {
        [Required, EmailAddress]
        public string Email { get; set; }
        [Required, MaxLength(100), MinLength(5)]
        public string FullName { get; set; }
        [Required, MaxLength(100), MinLength(6)]
        public string HashedPassword { get; set; }
        public Roles? Role { get; set; } 
        public Guid? PlanId { get; set; }
    }
}