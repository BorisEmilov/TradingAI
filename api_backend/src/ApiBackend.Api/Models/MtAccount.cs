using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;


namespace ApiBackend.Api.Models
{
    public class MtAccount
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public int MtId { get; set; }
        public string MtPassword { get; set; }
        public int Leverage { get; set; }
        public float Balance { get; set; }
        public float Credit { get; set; }
        public float Profit { get; set; }
        public float Equity { get; set; }
        public float Margin { get; set; }
        public float MarginFree { get; set; }
        public float MarginLevel { get; set; }
        public string Name { get; set; }
        public string Server { get; set; }
        public string Currency { get; set; }
        public string Company { get; set; }
        public string TradeMode { get; set; }
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }

}