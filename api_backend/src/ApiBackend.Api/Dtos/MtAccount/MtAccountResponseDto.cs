using System;
using System.Text.Json.Serialization;

namespace ApiBackend.Api.Dtos.MtAccount
{
    public class MtAccountResponseDto
    {
        [JsonPropertyName("login")]
        public long Login { get; set; }
        [JsonPropertyName("trade_mode")]
        public long TradeMode { get; set; }
        [JsonPropertyName("leverage")]
        public long Leverage { get; set; }
        [JsonPropertyName("limit_orders")]
        public long LimitOrders { get; set; }
        [JsonPropertyName("margin_so_mode")]
        public long MarginSoMode { get; set; }
        [JsonPropertyName("trade_allowed")]
        public bool TradeAllowed { get; set; }
        [JsonPropertyName("trade_expert")]
        public bool TradeExpert { get; set; }
        [JsonPropertyName("margin_mode")]
        public long MarginMode { get; set; }
        [JsonPropertyName("currency_digits")]
        public long CurrencyDigits { get; set; }
        [JsonPropertyName("fifo_close")]
        public bool FifoClose { get; set; }
        [JsonPropertyName("balance")]
        public double Balance { get; set; }
        [JsonPropertyName("credit")]
        public double? Credit { get; set; }
        [JsonPropertyName("profit")]
        public double? Profit { get; set; }
        [JsonPropertyName("equity")]
        public double? Equity { get; set; }
        [JsonPropertyName("margin")]
        public double? Margin { get; set; }
        [JsonPropertyName("margin_level")]
        public double? MarginLevel { get; set; }
        [JsonPropertyName("margin_so_so")]
        public double? MarginSoSo { get; set; }
        [JsonPropertyName("margin_initial")]
        public double? MarginInitial { get; set; }
        [JsonPropertyName("margin_maintenance")]
        public double? MarginMaintenance { get; set; }
        [JsonPropertyName("assets")]
        public double? Assets { get; set; }
        [JsonPropertyName("liabilities")]
        public double? Liabilities { get; set; }
        [JsonPropertyName("commission_blocked")]
        public double? CommissionBlocked { get; set; }
        [JsonPropertyName("name")]
        public string? Name { get; set; }
        [JsonPropertyName("server")]
        public string Server { get; set; }
        [JsonPropertyName("currency")]
        public string? Currency { get; set; }
        [JsonPropertyName("company")]
        public string? Company { get; set; }
        [JsonPropertyName("trade_mode_name")]
        public string? TradeModeName { get; set; }
        [JsonPropertyName("margin_mode_name")]
        public string? MarginModeName { get; set; }
    }
}