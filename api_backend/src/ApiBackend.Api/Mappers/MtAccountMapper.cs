using System;
using ApiBackend.Api.Dtos.MtAccount;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.Mappers
{
    public static class MtAccountMapper
    {
        public static MtAccount FromCreateAccountDtoToMtAccount(this CreateMtAccountDto dto, Guid userId){
            return new MtAccount
            {
                Login = dto.Login,
                Password = dto.Password,
                Server = dto.Server,
                UserId = userId
            };
        }

        public static MtAccount FomMtAccountResponseToMtAccount(this MtAccountResponseDto dto)
        {
            return new MtAccount
            {
                Login = dto.Login,
                TradeMode = dto.TradeMode,
                Leverage = dto.Leverage,
                LimitOrders = dto.LimitOrders,
                MarginSoMode = dto.MarginSoMode,
                TradeAllowed = dto.TradeAllowed,
                TradeExpert = dto.TradeExpert,
                MarginMode = dto.MarginMode,
                CurrencyDigits = dto.CurrencyDigits,
                FifoClose = dto.FifoClose,
                Balance = dto.Balance,
                Credit = dto.Credit,
                Profit = dto.Profit,
                Equity = dto.Equity,
                Margin = dto.Margin,
                MarginLevel = dto.MarginLevel,
                MarginSoSo = dto.MarginSoSo,
                MarginInitial = dto.MarginInitial,
                MarginMaintenance = dto.MarginMaintenance,
                Assets = dto.Assets,
                Liabilities = dto.Liabilities,
                CommissionBlocked = dto.CommissionBlocked,
                Name = dto.Name,
                Server = dto.Server,
                Currency = dto.Currency,
                Company = dto.Company,
                TradeModeName = dto.TradeModeName,
                MarginModeName = dto.MarginModeName,
                updatedAt = DateTime.UtcNow
            };
        }
    }
}