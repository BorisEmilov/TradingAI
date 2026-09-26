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
using Microsoft.AspNetCore.Routing.Template;
using System.Security.Cryptography;


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

        public async Task<string?> SendVerificationLink(Guid userId)
        {
            var randomBytes = new byte[64];
            RandomNumberGenerator.Fill(randomBytes);
            var token = Convert.ToBase64String(randomBytes).ToString();

            var user = await _context.User.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null || user.Verified == true)
            {
                return null;
            }

            string verificationUrl = $"/verify/{token}";

            await _emailSender.SendVerificationEmailAsync(user.Email, verificationUrl);

            await _redis.SetAsync($"email-verification:{userId}", token, TimeSpan.FromMinutes(10));

            return $"Verification email sent to {user.Email}";
        }

        public async Task<string?> VerifyEmail(Guid userId, string token)
        {
            var verificationInfo = await _redis.GetAsync($"email-verification:{userId}");
            if (verificationInfo == null || verificationInfo != token)
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

            await _redis.DeleteAsync($"email-verification:{userId}");

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
            var verification = await SendVerificationLink(newUser.Entity.Id);
            if (newUser == null || verification == null)
            {
                return null;
            }
            await _context.SaveChangesAsync();
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
            user.RefreshDocs.Add(dto.FromRefreshTokenDtoToRefreshToken());
            await _context.SaveChangesAsync();

            return (user, AccessToken, RefreshToken, ExpirityDate);
        }

        public async Task<string?> RefreshAsync(string token)
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

            return newAccessToken;
        }

        public async Task<string?> SendChangePasswordLink(Guid userId)
        {
            var user = await _context.User.FirstOrDefaultAsync(u => u.Id == userId);
            if (user == null)
            {
                return null;
            }

            var randomBytes = new byte[64];
            RandomNumberGenerator.Fill(randomBytes);
            var token = Convert.ToBase64String(randomBytes).ToString();

            var url = $"/change-password/{token}";

            await _emailSender.SendPasswordChangeLink(user.Email, url);

            await _redis.SetAsync($"password-update:{userId}", token, TimeSpan.FromMinutes(10));

            return $"Password change link sent to ${user.Email}";
        }

        public async Task<string?> ChangePassword(Guid userId, string password, string token)
        {
            var storedToken = await _redis.GetAsync($"password-update:{userId}");
            if (storedToken == null || storedToken != token)
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

            await _redis.DeleteAsync($"password-update:{userId}");

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

            PlanTypes userPlan = user.Plan.PlanType;

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

            var today = DateTime.UtcNow;
            var expirationDate = user.PlanPaymentDate?.AddDays(30);

            var reminingDays = expirationDate.Value - today;

            if (reminingDays.Days > 0 || reminingDays.Hours > 0)
            {
                // ! esperar al dia para cortar los servicios
                // ! dejar cron job que corte los servicios ese dia a esa hora

                return $"Plan Cancelled, you have {reminingDays.Days} - {reminingDays.Hours} of plan";
            }

            return "Plan is alredy Inactive";
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

