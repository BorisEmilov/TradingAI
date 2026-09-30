using System;
using System.ComponentModel.DataAnnotations;

namespace ApiBackend.Api.Dtos.MtAuthentication
{
    public class MtLoginDto
    {
        [Required]
        public long Login { get; set; }
        [Required]
        public string Password { get; set; }
        [Required]
        public string Server { get; set; }
    }
}