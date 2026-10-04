using System;
using System.ComponentModel.DataAnnotations;

namespace ApiBackend.Api.Dtos.MtAccount
{
    public class CreateMtAccountDto
    {
        [Required]
        public long Login { get; set; }
        [Required]
        public string Password { get; set; } = String.Empty;
        [Required]
        public string Server { get; set; } = String.Empty;
    }
}