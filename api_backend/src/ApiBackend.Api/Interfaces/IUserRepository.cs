using System;
using ApiBackend.Api.Dtos.User;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.Interfaces
{
    public interface IUserRepository
    {
        Task<(User, string)?> CreateUser(User dto);
        Task<string?> SendVerificationCode(Guid userId);
        Task<string?> VerifyEmail(Guid userId, string code);
        Task<User?> GetUserById(Guid id);
        Task<(User, string, string, DateTime)?> Login(string email, string password);
        Task<(string AccessToken, string RefreshToken, DateTime ExpiresAt)?> RefreshAsync(string token);
        Task<string?> SendChangePasswordCode(Guid userId);
        Task<string?> ChangePassword(Guid userId, string password, string code);
        Task<string?> AssignPlan(Guid planId, Guid userId);
        Task<string?> CancellPlan(Guid userId);
        Task<Plan?> GetUserPlan(Guid userId);
        Task<bool> DeleteUser(Guid userId);
    }
}