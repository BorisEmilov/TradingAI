using System;
using System.ComponentModel.DataAnnotations;
using ApiBackend.Api.Models;


namespace ApiBackend.Api.Dtos.RefreshDoc
{
    public class CreateRefreshDocDto
    {
        [Required]
        public Guid UserId { get; set; }
        [Required]
        public string Token { get; set; }
        [Required]
        public DateTime ExpiresAt { get; set; }
    }
}