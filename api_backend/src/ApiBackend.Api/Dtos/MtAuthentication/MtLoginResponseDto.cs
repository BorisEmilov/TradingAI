using System;
using System.Text.Json.Serialization;

namespace ApiBackend.Api.Dtos.MtAuthentication
{
    public class MtLoginResponseDto
    {
        [JsonPropertyName("slot_id")]
        public string SlotId { get; set; } = string.Empty;

        [JsonPropertyName("login")]
        public long Login { get; set; }

        [JsonPropertyName("server")]
        public string Server { get; set; } = string.Empty;

        [JsonPropertyName("account")]
        public MtAccountSummaryDto? Account { get; set; }

        [JsonPropertyName("created_at")]
        public long CreatedAt { get; set; }

        [JsonPropertyName("last_seen")]
        public long LastSeen { get; set; }

        [JsonPropertyName("expires_at")]
        public long ExpiresAt { get; set; }

        [JsonPropertyName("expires_in")]
        public long ExpiresIn { get; set; }

        [JsonPropertyName("ttl_seconds")]
        public long TtlSeconds { get; set; }

        [JsonPropertyName("idle_timeout_seconds")]
        public long IdleTimeoutSeconds { get; set; }

        [JsonPropertyName("token")]
        public string Token { get; set; } = string.Empty;

        [JsonPropertyName("reused")]
        public bool Reused { get; set; }
    }


    public class MtAccountSummaryDto
    {
        [JsonPropertyName("login")]
        public long? Login { get; set; }

        [JsonPropertyName("server")]
        public string? Server { get; set; } = string.Empty;

        [JsonPropertyName("currency")]
        public string? Currency { get; set; } = string.Empty;

        [JsonPropertyName("balance")]
        public double? Balance { get; set; }

        [JsonPropertyName("equity")]
        public double? Equity { get; set; }

        [JsonPropertyName("trade_mode")]
        public int? TradeMode { get; set; }

        [JsonPropertyName("trade_mode_name")]
        public string? TradeModeName { get; set; } = string.Empty;

        [JsonPropertyName("trade_allowed")]
        public bool? TradeAllowed { get; set; }
    }
}