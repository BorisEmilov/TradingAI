using System;
using ApiBackend.Api.Dtos.RefreshDoc;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.Mappers
{
    public static class RefreshDocMapper
    {
        public static RefreshDoc FromRefreshTokenDtoToRefreshToken(this CreateRefreshDocDto dto)
        {
            return new RefreshDoc
            {
                UserId = dto.UserId,
                Token = dto.Token,
                ExpiresAt = dto.ExpiresAt
            };
        }
    }
}