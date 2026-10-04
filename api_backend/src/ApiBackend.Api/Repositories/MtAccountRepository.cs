using System;
using System.Text.Json;
using ApiBackend.Api.Data;
using ApiBackend.Api.Dtos.MtAccount;
using ApiBackend.Api.Interfaces;
using ApiBackend.Api.Mappers;
using ApiBackend.Api.Models;
using ApiBackend.Api.Services;
using Microsoft.EntityFrameworkCore;

namespace ApiBackend.Api.Repositories
{
    public class MtAccountRepository : IMtAccountRepository
    {
        private readonly ApplicationDBContext _context;
        private readonly CacheService _redis;
        private readonly PythonServiceClient _getaway;

        public MtAccountRepository(
            ApplicationDBContext context,
            CacheService redis,
            PythonServiceClient getaway
        )
        {
            _context = context;
            _redis = redis;
            _getaway = getaway;
        }

        public async Task UpdateAccountInfo(MtAccount account, MtAccount updatedAccount)
        {
            // el objeto que viene del gateway no trae identidad ni credenciales: se conservan las guardadas
            updatedAccount.Id = account.Id;
            updatedAccount.UserId = account.UserId;
            updatedAccount.Password = account.Password;
            updatedAccount.Token = account.Token;
            _context.Entry(account).CurrentValues.SetValues(updatedAccount);
            await _context.SaveChangesAsync();
        }

        public async Task CacheAccount(Guid userId, MtAccount account)
        {
            await _redis.SetAsync($"MtCachedAccounts:{userId}", JsonSerializer.Serialize(account));
        }

        public async Task<MtAccount?> GetAccountFromDb(Guid userId)
        {
            return await _context.MtAccount.FirstOrDefaultAsync(a => a.UserId == userId);
        }

        public async Task<MtAccount?> CreateAccount(CreateMtAccountDto dto, Guid userId)
        {
            var response = await _context.MtAccount.FirstOrDefaultAsync(a => a.UserId == userId && a.Login == dto.Login);
            if (response != null) return null;

            var newAccount = await _context.MtAccount.AddAsync(dto.FromCreateAccountDtoToMtAccount(userId));
            await _context.SaveChangesAsync();
            return newAccount.Entity;
        }

        public async Task<MtAccount?> GetAccountLive(Guid userId)
        {
            var account = await _context.MtAccount.FirstOrDefaultAsync(a => a.UserId == userId);
            if (account == null) return null;

            if (string.IsNullOrEmpty(account.Token)) return null;  
            var getawayRes = await _getaway.GetAsync<MtAccountResponseDto>("/account", account.Token);
            if (getawayRes == null) return null;

            await UpdateAccountInfo(account, getawayRes.FomMtAccountResponseToMtAccount());

            await CacheAccount(userId, account);

            return account;
        }

        public async Task<MtAccount?> GetAccount(Guid userId, string token)
        {
            var cached = await _redis.GetAsync($"MtCachedAccounts:{userId}");
            var response = cached == null ? null : JsonSerializer.Deserialize<MtAccount>(cached);
            if (response == null || response.Token != token)
            {
                return await _context.MtAccount.FirstOrDefaultAsync(a => a.UserId == userId && a.Token == token);
            }
            return response;
        }

        public async Task<bool> DeleteAccount(Guid userId, Guid accountId)
        {
            var account = await _context.MtAccount.FirstOrDefaultAsync(a => a.UserId == userId && a.Id == accountId);

            if (account == null) return false;

            _context.MtAccount.Remove(account);
            await _context.SaveChangesAsync();

            return true;
        }

    }
}