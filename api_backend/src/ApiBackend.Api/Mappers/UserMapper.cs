using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ApiBackend.Api.Dtos.User;
using ApiBackend.Api.Models;


namespace ApiBackend.Api.Mappers
{
    public static class UserMapper
    {
        public static GetUserDto FromUserToGetUserDto(this User user)
        {
            return new GetUserDto
            {
                Id = user.Id,
                Email = user.Email,
                FullName = user.FullName,
                HashedPassword = user.HashedPassword,
                Role = user.Role,
                PlanId = user.PlanId,
                PlanPaymentDate = user.PlanPaymentDate,
                Verified = user.Verified,
                CreatedAt = user.CreatedAt,
                PendingPlanId = user.PendingPlanId
            };
        }

        public static User FromCreateUserDtoToUser(this CreateUserDto dto)
        {
            return new User
            {
                Email = dto.Email,
                FullName = dto.FullName,
                HashedPassword = dto.HashedPassword,
                Role = dto.Role,
                PlanId = dto.PlanId
            };
        }
    }
}