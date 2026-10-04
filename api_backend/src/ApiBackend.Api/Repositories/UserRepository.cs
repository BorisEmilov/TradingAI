using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ApiBackend.Api.Data;
using ApiBackend.Api.Dtos.RefreshDoc;
using ApiBackend.Api.Dtos.User;
using ApiBackend.Api.Interfaces;
using ApiBackend.Api.Models;
using ApiBackend.Api.Services;
using Microsoft.EntityFrameworkCore;
using ApiBackend.Api.Mappers;
using System.Security.Cryptography;
using System.Text;


namespace ApiBackend.Api.Repositories
{
    public class UserRepository : IUserRepository
    {
        private readonly ApplicationDBContext _context;
        private readonly TokenService _tokenService;
        private readonly CacheService _redis;
        private readonly SendEmailService _emailSender;
        public UserRepository(
            ApplicationDBContext context,
            TokenService tokenService,
            CacheService redis,
            SendEmailService emailSender
            )
        {
            _context = context;
            _tokenService = tokenService;
            _redis = redis;
            _emailSender = emailSender;
        }

        private static readonly TimeSpan CodeTtl = TimeSpan.FromMinutes(10);
        private const int MaxCodeAttempts = 5;

        
        private static string NewCode() => RandomNumberGenerator.GetInt32(0, 1_000_000).ToString("D6");

        private async Task StoreCode(string key, string code)
        {
            await _redis.SetAsync(key, code, CodeTtl);
            await _redis.DeleteAsync($"{key}:attempts");
        }

       
        private async Task<bool> ConsumeCode(string key, string code)
        {
            var stored = await _redis.GetAsync(key);
            if (stored == null)
            {
                return false;
            }
            if (CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(stored), Encoding.UTF8.GetBytes(code ?? "")))
            {
                await _redis.DeleteAsync(key);
                await _redis.DeleteAsync($"{key}:attempts");
                return true;
            }
            var attempts = int.Parse(await _redis.GetAsync($"{key}:attempts") ?? "0") + 1;
            if (attempts >= MaxCodeAttempts)
            {
                await _redis.DeleteAsync(key);
                await _redis.DeleteAsync($"{key}:attempts");
            }
            else
            {
                await _redis.SetAsync($"{key}:attempts", attempts.ToString(), CodeTtl);
            }
            return false;
        }

        public async Task<string?> SendVerificationCode(Guid userId)
        {
            var user = await _context.User.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null || user.Verified == true)
            {
                return null;
            }

            var code = NewCode();
            await StoreCode($"email-verification:{userId}", code);
            await _emailSender.SendVerificationEmailAsync(user.Email, code);

            return $"Verification code sent to {user.Email}";
        }

        public async Task<string?> VerifyEmail(Guid userId, string code)
        {
            if (!await ConsumeCode($"email-verification:{userId}", code))
            {
                return null;
            }

            var user = await _context.User.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null)
            {
                return null;
            }
            user.Verified = true;
            await _context.SaveChangesAsync();

            return "Email verified";
        }

        public async Task<(User, string)?> CreateUser(User dto)
        {
            var existingUser = await _context.User.FirstOrDefaultAsync(u => u.Email == dto.Email);
            if (existingUser != null)
            {
                return null;
            }

            var newUser = await _context.User.AddAsync(dto);
            await _context.SaveChangesAsync();

            var verification = await SendVerificationCode(newUser.Entity.Id);
            if (verification == null)
            {
                return null;
            }
            return (newUser.Entity, verification);
        }

        public async Task<User?> GetUserById(Guid id)
        {
            return await _context.User.FirstOrDefaultAsync(u => u.Id == id);

        }

        public async Task<(User, string, string, DateTime)?> Login(string email, string password)
        {
            var user = await _context.User.FirstOrDefaultAsync(u => u.Email == email);
            if (user == null)
            {
                return null;
            }
            var passwordMatch = BCrypt.Net.BCrypt.Verify(password, user.HashedPassword);

            if (passwordMatch != true)
            {
                return null;
            }

            DateTime ExpirityDate = DateTime.UtcNow.AddDays(7);
            string AccessToken = _tokenService.GenerateAccessToken(user);
            string RefreshToken = _tokenService.GenerateRefreshToken();

            CreateRefreshDocDto dto = new CreateRefreshDocDto
            {
                UserId = user.Id,
                Token = RefreshToken,
                ExpiresAt = ExpirityDate
            };

            await _context.RefreshDoc.AddAsync(dto.FromRefreshTokenDtoToRefreshToken());
            await _context.SaveChangesAsync();

            return (user, AccessToken, RefreshToken, ExpirityDate);
        }

        public async Task<(string AccessToken, string RefreshToken, DateTime ExpiresAt)?> RefreshAsync(string token)
        {
            var refreshDoc = await _context.RefreshDoc
                .Include(rd => rd.User)
                .FirstOrDefaultAsync(rd => rd.Token == token && rd.Revoked == false && rd.ExpiresAt > DateTime.UtcNow);

            if (refreshDoc == null || refreshDoc.User == null)
            {
                return null;
            }

            refreshDoc.Revoked = true;
            var newAccessToken = _tokenService.GenerateAccessToken(refreshDoc.User);
            var newRefreshToken = _tokenService.GenerateRefreshToken();

            CreateRefreshDocDto dto = new CreateRefreshDocDto
            {
                UserId = refreshDoc.User.Id,
                Token = newRefreshToken,
                ExpiresAt = DateTime.UtcNow.AddDays(7)
            };
            await _context.RefreshDoc.AddAsync(dto.FromRefreshTokenDtoToRefreshToken());
            await _context.SaveChangesAsync();

            return (newAccessToken, newRefreshToken, dto.ExpiresAt);
        }

        public async Task<string?> SendChangePasswordCode(Guid userId)
        {
            var user = await _context.User.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null)
            {
                return null;
            }

            var code = NewCode();
            await StoreCode($"password-update:{userId}", code);
            await _emailSender.SendPasswordChangeCodeAsync(user.Email, code);

            return $"Password change code sent to {user.Email}";
        }

        public async Task<string?> ChangePassword(Guid userId, string password, string code)
        {
            if (!await ConsumeCode($"password-update:{userId}", code))
            {
                return null;
            }

            var user = await _context.User.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null)
            {
                return null;
            }

            user.HashedPassword = password;
            await _context.SaveChangesAsync();

            return "Password Updated";
        }

        public async Task<string?> AssignPlan(Guid planId, Guid userId)
        {
            var user = await _context.User
                .Include(p => p.Plan)
                .FirstOrDefaultAsync(u => u.Id == userId);
            var plan = await _context.Plan.FirstOrDefaultAsync(p => p.Id == planId);
            if (user == null || plan == null || user.PlanId == planId)
            {
                return null;
            }

            PlanTypes userPlan = user.Plan?.PlanType ?? PlanTypes.FREE;

            // ! si el plan seleccionado es inferior esperar a que termine el plan actual y luego hacer update
            if (userPlan > plan.PlanType)
            {
                user.PendingPlanId = planId;
                await _context.SaveChangesAsync();
                return $"Plan {plan.Name} will be activated when your current plan expires.";
            }

            user.PlanId = planId;
            user.PlanPaymentDate = DateTime.UtcNow;
            switch (plan.PlanType)
            {
                case PlanTypes.PLUS:
                    user.Role = Roles.PLUS_USER;
                    break;
                case PlanTypes.GOLDEN:
                    user.Role = Roles.GOLDEN_USER;
                    break;
                default:
                    break;
            }
            await _context.SaveChangesAsync();

            return $"Updated to plan {plan.Name}";
        }

        public async Task<string?> CancellPlan(Guid userId)
        {
            var user = await _context.User.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null || user.PlanId == null || user.PlanPaymentDate == null)
            {
                return null;
            }

            // cancelar = bajar al plan FREE (tiene que existir un Plan con PlanType FREE)
            var freePlan = await _context.Plan.FirstOrDefaultAsync(p => p.PlanType == PlanTypes.FREE);
            if (freePlan == null || user.PlanId == freePlan.Id)
            {
                return null;
            }

            var expirationDate = user.PlanPaymentDate.Value.AddDays(PlanExpirationService.PlanDurationDays);
            var remaining = expirationDate - DateTime.UtcNow;

            if (remaining > TimeSpan.Zero)
            {
                // el plan pagado sigue activo hasta vencer; PlanExpirationService aplica la baja ese momento
                user.PendingPlanId = freePlan.Id;
                await _context.SaveChangesAsync();
                return $"Plan cancelled, it stays active until {expirationDate:yyyy-MM-dd HH:mm} UTC " +
                       $"({remaining.Days} days {remaining.Hours} hours left)";
            }

            user.PlanId = freePlan.Id;
            user.PendingPlanId = null;
            user.PlanPaymentDate = DateTime.UtcNow;
            if (user.Role is not (Roles.ADMIN or Roles.SUPER_ADMIN))
            {
                user.Role = Roles.FREE_USER;
            }
            await _context.SaveChangesAsync();
            return "Plan cancelled";
        }

        public async Task<Plan?> GetUserPlan(Guid userId)
        {
            var user = await _context.User
                .Include(u => u.Plan)
                .FirstOrDefaultAsync(u => u.Id == userId);

            if (user == null || user.Plan == null || !Enum.IsDefined(typeof(PlanTypes), user.Plan.PlanType))
            {
                return null;
            }

            return user.Plan;
        }

        public async Task<bool> DeleteUser(Guid userId)
        {
            var user = await _context.User.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null)
            {
                return false;
            }

            _context.User.Remove(user);
            await _context.SaveChangesAsync();

            return true;
        }
    }
}

