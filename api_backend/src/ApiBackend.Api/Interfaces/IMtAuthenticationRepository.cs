using System;
using ApiBackend.Api.Dtos.MtAuthentication;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.Interfaces
{
    public interface IMtAuthenticationRepository
    {
        Task<MtAuthentication?> MtLogin(Guid userId, MtLoginDto dto);
        Task<MtSessionDto?> MtRefresh(Guid userId);
    }
}