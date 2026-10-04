using System;
using System.Text.Json.Serialization;

namespace ApiBackend.Api.Models
{
    public class MtAccount
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public long Login { get; set; }
        public string Password { get; set; } = String.Empty;
        public long? TradeMode { get; set; }
        public long? Leverage { get; set; }
        public long? LimitOrders { get; set; }
        public long? MarginSoMode { get; set; }
        public bool? TradeAllowed { get; set; }
        public bool? TradeExpert { get; set; }
        public long? MarginMode { get; set; }
        public long? CurrencyDigits { get; set; }
        public bool? FifoClose { get; set; }
        public double? Balance { get; set; }
        public double? Credit { get; set; }
        public double? Profit { get; set; }
        public double? Equity { get; set; }
        public double? Margin { get; set; }
        public double? FreeMargin { get; set; }
        public double? MarginLevel { get; set; }
        public double? MarginSoCall { get; set; }
        public double? MarginSoSo { get; set; }
        public double? MarginInitial { get; set; }
        public double? MarginMaintenance { get; set; }
        public double? Assets { get; set; }
        public double? Liabilities { get; set; }
        public double? CommissionBlocked { get; set; }
        public string? Name { get; set; } = String.Empty;
        public string Server { get; set; } = String.Empty;
        public string? Currency { get; set; } = String.Empty;
        public string? Company { get; set; } = String.Empty;
        public string? TradeModeName { get; set; } = String.Empty;
        public string? MarginModeName { get; set; } = String.Empty;
        public string? Token { get; set; } = String.Empty;

        public DateTime? updatedAt { get; set; }
        public Guid UserId { get; set; }
        [JsonIgnore]
        public User User { get; set; } = default!;
    }

}