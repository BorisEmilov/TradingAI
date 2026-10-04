using System;
using ApiBackend.Api.Dtos.MtAccount;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.Interfaces
{
    public interface IMtAccountRepository
    {
        Task UpdateAccountInfo(MtAccount account, MtAccount updatedAccount); 
        Task CacheAccount(Guid id, MtAccount account);
        Task<MtAccount?> GetAccountFromDb(Guid userId);
        Task<MtAccount?> CreateAccount(CreateMtAccountDto dto, Guid userId);
        Task<MtAccount?> GetAccountLive(Guid userId);
    }
}