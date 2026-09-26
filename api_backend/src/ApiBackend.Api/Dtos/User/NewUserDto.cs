using System;
using System.ComponentModel.DataAnnotations;

namespace ApiBackend.Api.Dtos.User
{
    public class NewUserDto
    {
        [Required, EmailAddress]
        public string Email { get; set; }
        [Required, MaxLength(100), MinLength(5)]
        public string FullName { get; set; }
        [Required, MaxLength(100), MinLength(6)]
        public string Password { get; set; }
        [Required, MaxLength(100), MinLength(6)]
        public string RepeatPassword { get; set; }
    }
}