using System;
using System.ComponentModel.DataAnnotations;


namespace ApiBackend.Api.Dtos.User
{
    public class UpdatePasswordDto
    {
        [Required, MinLength(6), MaxLength(100)]
        public string Password {get; set;}
        [Required, MinLength(6), MaxLength(100)]
        public string RepeatPassword {get; set;}
    }
}