using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;


namespace ApiBackend.Api.Models
{
    public class MtAuthentication
    {
        public Guid Id { get; set; } = Guid.NewGuid();

        public long Login { get; set; }
        public string Password { get; set; }
        public string Server { get; set; } = string.Empty;

        public string? Currency { get; set; }
        public double? Balance { get; set; }
        public double? Equity { get; set; }
        public int? TradeMode { get; set; }
        public string? TradeModeName { get; set; }
        public bool? TradeAllowed { get; set; }

        public string? SlotId { get; set; }
        public string? Token { get; set; }
        public bool IsSessionActive { get; set; } = false;

        public DateTime? SessionCreatedAt { get; set; }
        public DateTime? SessionLastSeen { get; set; }
        public DateTime? SessionExpiresAt { get; set; }
        public long? TtlSeconds { get; set; }
        public long? IdleTimeoutSeconds { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime? UpdatedAt { get; set; }

        public bool? Reused { get; set; }

        public Guid UserId { get; set; }
        public User User { get; set; } = null!;
    }

}