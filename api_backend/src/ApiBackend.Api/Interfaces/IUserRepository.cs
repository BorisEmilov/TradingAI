using System;
using ApiBackend.Api.Dtos.User;
using ApiBackend.Api.Models;
using Microsoft.VisualBasic;

namespace ApiBackend.Api.Interfaces
{
    public interface IUserRepository
    {
        Task<(User, string)?> CreateUser(User dto);
        Task<string?> SendVerificationLink(Guid userId);
        Task<string?> VerifyEmail(Guid userId, string token);
        Task<User?> GetUserById(Guid id);
        Task<(User, string, string, DateTime)?> Login(string email, string password);
        Task<string?> RefreshAsync(string token);
        Task<string?> SendChangePasswordLink(Guid userId);
        Task<string?> ChangePassword(Guid userId, string password, string token);
        Task<string?> AssignPlan(Guid planId, Guid userId);
        Task<string?> CancellPlan(Guid userId);
        Task<Plan?> GetUserPlan(Guid userId);
        Task<bool> DeleteUser(Guid userId);
    }
}