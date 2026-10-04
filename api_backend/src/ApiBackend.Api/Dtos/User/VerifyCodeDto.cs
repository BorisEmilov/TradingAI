using System;
using System.ComponentModel.DataAnnotations;

namespace ApiBackend.Api.Dtos.User
{
    public class VerifyCodeDto
    {
        [Required, RegularExpression(@"^\d{6}$", ErrorMessage = "Code must be 6 digits")]
        public string Code { get; set; } = String.Empty;
    }
}
