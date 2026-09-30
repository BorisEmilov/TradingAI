using System;
using ApiBackend.Api.Data;
using ApiBackend.Api.Dtos.MtAuthentication;
using ApiBackend.Api.Interfaces;
using ApiBackend.Api.Models;
using ApiBackend.Api.Services;
using Microsoft.EntityFrameworkCore;
using StackExchange.Redis;

namespace ApiBackend.Api.Repositories
{
    public class MtAuthenticationRepository : IMtAuthenticationRepository
    {
        private readonly ApplicationDBContext _context;
        private readonly PythonServiceClient _getaway;
        private readonly CacheService _redis;

        public MtAuthenticationRepository(
            ApplicationDBContext context,
            PythonServiceClient getaway,
            CacheService redis
        )
        {
            _context = context;
            _getaway = getaway;
            _redis = redis;
        }
        public async Task<MtAuthentication?> MtLogin(Guid userId, MtLoginDto dto)
        {
            var loginExists = await _context.MtAuthentication.FirstOrDefaultAsync(u => u.UserId == userId && u.Login == dto.Login);

            if (loginExists != null)
            {
                var getawayResponse = await _getaway.PostAsync<MtLoginDto, MtLoginResponseDto>("/auth/login", dto);
                if (getawayResponse == null)
                {
                    return null;
                }
                TimeSpan remainingTime = DateTimeOffset.FromUnixTimeSeconds(getawayResponse.ExpiresAt) - DateTime.UtcNow;
                await _redis.SetAsync($"MtTokens:{userId}", getawayResponse.Token, remainingTime);

                loginExists.Balance = getawayResponse.Account?.Balance;
                loginExists.Equity = getawayResponse.Account?.Equity;
                loginExists.TradeMode = getawayResponse.Account?.TradeMode;
                loginExists.TradeModeName = getawayResponse.Account?.TradeModeName;
                loginExists.SessionLastSeen = DateTimeOffset.FromUnixTimeSeconds(getawayResponse.LastSeen).UtcDateTime;
                loginExists.SessionExpiresAt = DateTimeOffset.FromUnixTimeSeconds(getawayResponse.LastSeen).UtcDateTime;
                loginExists.TtlSeconds = getawayResponse.TtlSeconds;
                loginExists.IdleTimeoutSeconds = getawayResponse.IdleTimeoutSeconds;
                loginExists.Token = getawayResponse.Token;
                loginExists.UpdatedAt = DateTime.UtcNow;

                await _context.SaveChangesAsync();

                return loginExists;
            }

            var linkedlogin = await _context.MtAuthentication.FirstOrDefaultAsync(u => u.Login == dto.Login);
            if (linkedlogin != null)
            {
                return null;
            }

            var tryLogin = await _getaway.PostAsync<MtLoginDto, MtLoginResponseDto>("/auth/login", dto);
            if (tryLogin == null)
            {
                return null;
            }

            var newAccountObj = new MtAuthentication
            {
                Login = tryLogin.Login,
                Password = dto.Password,
                Server = tryLogin.Server,
                Currency = tryLogin?.Account?.Currency,
                Balance = tryLogin?.Account?.Balance,
                Equity = tryLogin?.Account?.Equity,
                TradeMode = tryLogin?.Account?.TradeMode,
                TradeModeName = tryLogin?.Account?.TradeModeName,
                TradeAllowed = tryLogin?.Account?.TradeAllowed,
                SlotId = tryLogin?.SlotId,
                Token = tryLogin?.Token,
                IsSessionActive = true,
                SessionCreatedAt = DateTimeOffset.FromUnixTimeSeconds(tryLogin.CreatedAt).UtcDateTime,
                SessionLastSeen = DateTimeOffset.FromUnixTimeSeconds(tryLogin.LastSeen).UtcDateTime,
                SessionExpiresAt = DateTimeOffset.FromUnixTimeSeconds(tryLogin.ExpiresAt).UtcDateTime,
                TtlSeconds = tryLogin?.TtlSeconds,
                IdleTimeoutSeconds = tryLogin?.IdleTimeoutSeconds,
                Reused = tryLogin?.Reused,
                UserId = userId
            };

            var addNewMtAccount = await _context.AddAsync(newAccountObj);
            if (addNewMtAccount != null && addNewMtAccount.Entity.Token != null)
            {
                TimeSpan expirationTime = DateTimeOffset.FromUnixTimeSeconds(tryLogin.ExpiresAt) - DateTimeOffset.UtcNow;
                await _redis.SetAsync($"MtTokens:{userId}", addNewMtAccount.Entity.Token, expirationTime);
                await _context.SaveChangesAsync();
            }

            return addNewMtAccount?.Entity;
        }

        public async Task<MtSessionDto?> MtRefresh(Guid userId)
        {
            var token = await _redis.GetAsync($"MtTokens:{userId}");
            if (token == null) return null;

            var efResponse = await _context.MtAuthentication.FirstOrDefaultAsync(u => u.Id == userId && u.Token == token);



            if (efResponse == null)
            {
                var mtAuth = await _context.MtAuthentication.FirstOrDefaultAsync(m => m.UserId == userId);
                
                if (mtAuth == null) return null;
                var getawayRes = await _getaway.PostAsync<string, MtSessionDto>($"/auth/refresh", token);
                
                if (getawayRes == null) return null;
                
                mtAuth.Token = token;
                mtAuth.SessionLastSeen = DateTime.UtcNow;
                mtAuth.SessionExpiresAt = DateTimeOffset.FromUnixTimeSeconds(getawayRes.ExpiresAt).UtcDateTime;
                
                await _context.SaveChangesAsync();
                return getawayRes;
            }

            var getawayRes1 = await _getaway.PostAsync<string, MtSessionDto>($"/auth/refresh", token);
            if (getawayRes1 == null) return null;

            efResponse.SessionLastSeen = DateTime.UtcNow;
            efResponse.SessionExpiresAt = DateTimeOffset.FromUnixTimeSeconds(getawayRes1.ExpiresAt).UtcDateTime;
            await _context.SaveChangesAsync();

            return getawayRes1;
        }


    }
}