using System;
using System.ComponentModel.DataAnnotations;

namespace ApiBackend.Api.Dtos.User
{
    public class LoginDto
    {
        [Required, EmailAddress]
        public string Email { get; set; } = String.Empty;
        [Required]
        public string Password { get; set; } = String.Empty;
    }
}
